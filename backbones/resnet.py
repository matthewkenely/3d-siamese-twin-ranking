import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]

def conv3x3x3(inplanes, out_planes, stride=1, dilation=1):
    return nn.Conv3d(inplanes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     dilation=dilation,
                     padding=dilation,
                     bias=False)

def conv1x1x1(inplanes, out_planes, stride=1):
    return nn.Conv3d(inplanes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class SpatialAttention3D(nn.Module):
    """3D Spatial Attention Module"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention


class ChannelAttention3D(nn.Module):
    """3D Channel Attention Module"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _, _, _ = x.size()
        
        # Global average pooling and max pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        out = avg_out + max_out
        attention = self.sigmoid(out).view(b, c, 1, 1, 1)
        return x * attention


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                            out.size(3), out.size(4))
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = torch.cat([out.data, zero_pads], dim=1)
    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, use_attention=False):
        super().__init__()

        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        
        # Add attention modules
        self.use_attention = use_attention
        if use_attention:
            self.channel_attention = ChannelAttention3D(planes)
            self.spatial_attention = SpatialAttention3D(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply attention before residual connection
        if self.use_attention:
            out = self.channel_attention(out)
            out = self.spatial_attention(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, use_attention=False):
        super().__init__()

        self.conv1_dropout = nn.Dropout3d(p=0.1)
        self.conv2_dropout = nn.Dropout3d(p=0.15)
        self.conv3_dropout = nn.Dropout3d(p=0.2)

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, 
                              dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        
        # Add attention modules
        self.use_attention = use_attention
        if use_attention:
            self.channel_attention = ChannelAttention3D(planes * self.expansion)
            self.spatial_attention = SpatialAttention3D(planes * self.expansion)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.conv1_dropout(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.conv2_dropout(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # out = self.conv3_dropout(out)
        out = self.bn3(out)
        
        # Apply attention before residual connection
        if self.use_attention:
            out = self.channel_attention(out)
            out = self.spatial_attention(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 n_input_channels=1,
                 embedding_dim=256,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 dropout_rate=0.3,
                 shortcut_type='B',
                 widen_factor=1.0,
                 sample_input_D=None,
                 sample_input_H=None,
                 sample_input_W=None,
                 num_seg_classes=None,
                 no_cuda=False,
                 use_attention=True,
                 attention_layers=[2, 3, 4]):  # Which layers to apply attention to
        super().__init__()

        self.threshold = None

        self.inplanes = 64
        self.no_cuda = no_cuda
        self.no_max_pool = no_max_pool
        self.use_attention = use_attention
        self.attention_layers = attention_layers

        if widen_factor != 1.0:
            self.inplanes = int(self.inplanes * widen_factor)

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.inplanes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        layer_planes = [int(x * widen_factor) for x in [64, 128, 256, 512]]
        
        # Build layers with attention flags
        self.layer1 = self._make_layer(block, layer_planes[0], layers[0], shortcut_type, use_attention=1 in attention_layers)
        self.layer2 = self._make_layer(block, layer_planes[1], layers[1], shortcut_type, stride=2, use_attention=2 in attention_layers)
        self.layer3 = self._make_layer(block, layer_planes[2], layers[2], shortcut_type, stride=1, dilation=2, use_attention=3 in attention_layers)
        self.layer4 = self._make_layer(block, layer_planes[3], layers[3], shortcut_type, stride=1, dilation=4, use_attention=4 in attention_layers)

        # Dropout layers
        self.layer1_dropout = nn.Dropout3d(p=0.1)
        self.layer2_dropout = nn.Dropout3d(p=0.15)
        self.layer3_dropout = nn.Dropout3d(p=0.2)
        self.layer4_dropout = nn.Dropout3d(p=0.25)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(layer_planes[3] * block.expansion, embedding_dim * 2)
        self.bn_fc1 = nn.BatchNorm1d(embedding_dim * 2)
        self.fc2 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.bn_fc2 = nn.BatchNorm1d(embedding_dim)

        # self.fc = nn.Linear(layer_planes[3] * block.expansion, embedding_dim)

        if num_seg_classes is not None:
            self.conv_seg = nn.Sequential(
                nn.ConvTranspose3d(layer_planes[3] * block.expansion, 32, 2, stride=2),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 32, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False), 
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, num_seg_classes, kernel_size=1, stride=(1, 1, 1), bias=False) 
            )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_threshold(self, threshold):
        self.threshold = threshold

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1, use_attention=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride,
                                     no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, 
                             kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(inplanes=self.inplanes,
                           planes=planes,
                           stride=stride,
                           dilation=dilation,
                           downsample=downsample,
                           use_attention=use_attention))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, use_attention=use_attention))

        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        # x = self.layer1_dropout(x)
        
        x = self.layer2(x)
        # x = self.layer2_dropout(x)
        
        x = self.layer3(x)
        # x = self.layer3_dropout(x)
        
        x = self.layer4(x)
        # x = self.layer4_dropout(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc_dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)

        return x

    def forward(self, x):
        return self.forward_once(x)


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)

    return model