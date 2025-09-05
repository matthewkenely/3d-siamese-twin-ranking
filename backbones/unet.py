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


class SingleConv(nn.Module):
    """Basic convolutional module with attention support"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, use_attention=False):
        super().__init__()
        
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.use_attention = use_attention
        if use_attention:
            self.channel_attention = ChannelAttention3D(out_channels)
            self.spatial_attention = SpatialAttention3D(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        if self.use_attention:
            x = self.channel_attention(x)
            x = self.spatial_attention(x)
            
        return x


class DoubleConv(nn.Module):
    """Double convolution block with attention support"""
    def __init__(self, in_channels, out_channels, use_attention=False):
        super().__init__()
        
        self.conv1 = SingleConv(in_channels, out_channels, use_attention=False)
        self.conv2 = SingleConv(out_channels, out_channels, use_attention=use_attention)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResNetBlock(nn.Module):
    """Residual block matching ResNet implementation"""
    def __init__(self, in_channels, out_channels, stride=1, use_attention=False):
        super().__init__()
        
        self.conv1 = conv3x3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv1x1x1(in_channels, out_channels, stride=stride),
                nn.BatchNorm3d(out_channels)
            )
        
        self.use_attention = use_attention
        if use_attention:
            self.channel_attention = ChannelAttention3D(out_channels)
            self.spatial_attention = SpatialAttention3D(out_channels)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_attention:
            out = self.channel_attention(out)
            out = self.spatial_attention(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class Encoder(nn.Module):
    """Encoder module with pooling and convolution blocks"""
    def __init__(self, in_channels, out_channels, apply_pooling=True, 
                 use_attention=False, block_type='double'):
        super().__init__()
        
        self.pooling = None
        if apply_pooling:
            self.pooling = nn.MaxPool3d(kernel_size=2, stride=2)
        
        if block_type == 'resnet':
            self.block = ResNetBlock(in_channels, out_channels, use_attention=use_attention)
        else:
            self.block = DoubleConv(in_channels, out_channels, use_attention=use_attention)
    
    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.block(x)
        return x


class UNet(nn.Module):
    """3D UNet with embedding head for twin identification"""
    
    def __init__(self,
                 n_input_channels=1,
                 f_maps=[64, 128, 256, 512],  # Reduced from 5 to 4 layers
                 embedding_dim=256,
                 dropout_rate=0.3,
                 use_attention=True,
                 attention_layers=[2, 3, 4],
                 block_type='resnet'):
        super().__init__()
        
        self.threshold = None
        self.use_attention = use_attention
        self.attention_layers = attention_layers
        
        # Initial convolution - less aggressive stride
        self.conv1 = nn.Conv3d(n_input_channels, f_maps[0], 
                              kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(f_maps[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Encoder layers - limit to 4 layers to prevent over-pooling
        self.encoders = nn.ModuleList()
        for i, out_channels in enumerate(f_maps):
            if i == 0:
                # First encoder (no pooling, already handled by maxpool)
                encoder = Encoder(f_maps[0], f_maps[0], apply_pooling=False,
                                use_attention=1 in attention_layers, block_type=block_type)
            else:
                # Only pool for the first 2 additional layers
                apply_pool = i <= 2
                encoder = Encoder(f_maps[i-1], out_channels, apply_pooling=apply_pool,
                                use_attention=(i+1) in attention_layers, block_type=block_type)
            self.encoders.append(encoder)
        
        # Dropout layers
        self.layer_dropouts = nn.ModuleList([
            nn.Dropout3d(p=0.1),
            nn.Dropout3d(p=0.15),
            nn.Dropout3d(p=0.2),
            nn.Dropout3d(p=0.25)
        ])
        
        # Global average pooling and embedding head
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(f_maps[-1], embedding_dim * 2)
        self.bn_fc1 = nn.BatchNorm1d(embedding_dim * 2)
        self.fc2 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.bn_fc2 = nn.BatchNorm1d(embedding_dim)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def set_threshold(self, threshold):
        self.threshold = threshold
    
    def forward_once(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Encoder path
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            # Apply dropout (commented out like in ResNet)
            # if i < len(self.layer_dropouts):
            #     x = self.layer_dropouts[i](x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Embedding head
        x = self.fc_dropout(x)
        x = self.fc1(x)
        # x = self.bn_fc1(x)  # commented out like in ResNet
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.bn_fc2(x)  # commented out like in ResNet
        x = F.normalize(x, p=2, dim=1)
        
        return x
    
    def forward(self, x):
        return self.forward_once(x)


def generate_model(model_depth=None, **kwargs):
    """Generate UNet model with different configurations"""
    
    # Define feature maps for different depths
    depth_configs = {
        'small': [32, 64, 128, 256],
        'medium': [64, 128, 256, 512],
        'large': [64, 128, 256, 512, 1024]
    }
    
    if model_depth is None:
        f_maps = depth_configs['medium']
    elif isinstance(model_depth, str):
        f_maps = depth_configs.get(model_depth, depth_configs['medium'])
    else:
        # For backward compatibility with ResNet depth numbers
        if model_depth <= 34:
            f_maps = depth_configs['small']
        elif model_depth <= 50:
            f_maps = depth_configs['medium']
        else:
            f_maps = depth_configs['large']
    
    model = UNet(f_maps=f_maps, **kwargs)
    return model