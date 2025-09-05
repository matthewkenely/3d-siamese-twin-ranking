import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


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


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module(
            'conv1',
            nn.Conv3d(num_input_features,
                      bn_size * growth_rate,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module(
            'conv2',
            nn.Conv3d(bn_size * growth_rate,
                      growth_rate,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate, use_attention=False):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer{}'.format(i + 1), layer)
        
        # Add attention modules
        self.use_attention = use_attention
        if use_attention:
            final_features = num_input_features + num_layers * growth_rate
            self.channel_attention = ChannelAttention3D(final_features)
            self.spatial_attention = SpatialAttention3D(final_features)

    def forward(self, x):
        features = super().forward(x)
        
        if self.use_attention:
            features = self.channel_attention(features)
            features = self.spatial_attention(features)
            
        return features


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv3d(num_input_features,
                      num_output_features,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densenet-BC model class with embedding head and attention
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        embedding_dim (int) - dimension of output embedding
        use_attention (bool) - whether to use attention mechanisms
        attention_layers (list) - which layers to apply attention to
    """

    def __init__(self,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 embedding_dim=256,
                 dropout_rate=0.3,
                 num_seg_classes=None,
                 no_cuda=False,
                 use_attention=True,
                 attention_layers=[2, 3, 4]):

        super().__init__()

        self.threshold = None
        self.use_attention = use_attention
        self.attention_layers = attention_layers

        # First convolution
        self.features = [('conv1',
                          nn.Conv3d(n_input_channels,
                                    num_init_features,
                                    kernel_size=(conv1_t_size, 7, 7),
                                    stride=(conv1_t_stride, 2, 2),
                                    padding=(conv1_t_size // 2, 3, 3),
                                    bias=False)),
                         ('norm1', nn.BatchNorm3d(num_init_features)),
                         ('relu1', nn.ReLU(inplace=True))]
        if not no_max_pool:
            self.features.append(
                ('pool1', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)))
        self.features = nn.Sequential(OrderedDict(self.features))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,
                                use_attention=(i + 1) in attention_layers)
            self.features.add_module('denseblock{}'.format(i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition{}'.format(i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        # Dropout layers (matching ResNet structure)
        self.layer1_dropout = nn.Dropout3d(p=0.1)
        self.layer2_dropout = nn.Dropout3d(p=0.15)
        self.layer3_dropout = nn.Dropout3d(p=0.2)
        self.layer4_dropout = nn.Dropout3d(p=0.25)

        # Embedding head (matching ResNet structure)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(num_features, embedding_dim * 2)
        self.bn_fc1 = nn.BatchNorm1d(embedding_dim * 2)
        self.fc2 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.bn_fc2 = nn.BatchNorm1d(embedding_dim)
        self.relu = nn.ReLU(inplace=True)

        # Segmentation head (if needed)
        if num_seg_classes is not None:
            self.conv_seg = nn.Sequential(
                nn.ConvTranspose3d(num_features, 32, 2, stride=2),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 32, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False), 
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, num_seg_classes, kernel_size=1, stride=(1, 1, 1), bias=False) 
            )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def set_threshold(self, threshold):
        self.threshold = threshold

    def forward_once(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        
        out = self.fc_dropout(out)
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = F.normalize(out, p=2, dim=1)

        return out

    def forward(self, x):
        return self.forward_once(x)


# Mapping from ResNet depths to DenseNet configurations
RESNET_TO_DENSENET_MAPPING = {
    10: (121, (6, 12, 24, 16)),    # Simple mapping
    18: (121, (6, 12, 24, 16)),    # DenseNet-121
    34: (169, (6, 12, 32, 32)),    # DenseNet-169  
    50: (201, (6, 12, 48, 32)),    # DenseNet-201
    101: (264, (6, 12, 64, 48)),   # DenseNet-264
    152: (264, (6, 12, 64, 48)),   # DenseNet-264
    200: (264, (6, 12, 64, 48))    # DenseNet-264
}


def generate_model(model_depth, **kwargs):
    """Generate DenseNet model with ResNet-compatible depth mapping"""
    
    # Handle both DenseNet depths and ResNet depths
    if model_depth in [121, 169, 201, 264]:
        # Direct DenseNet depth specification
        densenet_configs = {
            121: (64, 32, (6, 12, 24, 16)),
            169: (64, 32, (6, 12, 32, 32)), 
            201: (64, 32, (6, 12, 48, 32)),
            264: (64, 32, (6, 12, 64, 48))
        }
        num_init_features, growth_rate, block_config = densenet_configs[model_depth]
        
    elif model_depth in RESNET_TO_DENSENET_MAPPING:
        # ResNet depth mapping to DenseNet
        densenet_depth, block_config = RESNET_TO_DENSENET_MAPPING[model_depth]
        densenet_configs = {
            121: (64, 32),
            169: (64, 32), 
            201: (64, 32),
            264: (64, 32)
        }
        num_init_features, growth_rate = densenet_configs[densenet_depth]
        
    else:
        raise ValueError(f"Unsupported model depth: {model_depth}. "
                        f"Supported depths: {list(RESNET_TO_DENSENET_MAPPING.keys())} or [121, 169, 201, 264]")

    model = DenseNet(num_init_features=num_init_features,
                     growth_rate=growth_rate,
                     block_config=block_config,
                     **kwargs)

    return model