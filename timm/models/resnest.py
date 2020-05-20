##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNeSt models"""

import torch
from .registry import register_model
import torch.nn as nn
from .resnet import ResNet
from .layers import SplAtConv2d, ConvBnAct, create_attn, Mish, Swish
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


__all__ = ['resnest10_2s1', 'resnest14_2s1', 'resnest18_2s1', 'resnest10_2s2', 'resnest14_2s2', 'resnest18_2s2',
           'resnest10_4s1', 'resnest14_4s1', 'resnest18_4s1', 'resnest10_1s1', 'resnest10_1s1_mish', 'resnest10_2s1_mish',
          'resnest10_2s2_mish', 'resnest10_4s1_mish']


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }

default_cfgs = {
    'resnest10': _cfg(),
    'resnest14': _cfg(),
    'resnest18': _cfg(),
}


class SplitAtteBasic(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 sa_kwargs=None, reduce_first=1, dilation=1, first_dilation=None,
                 drop_block=None, drop_path=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, attn_layer=None):
        super(SplitAtteBasic, self).__init__()

        sa_kwargs = sa_kwargs or {}
        conv_kwargs = dict(drop_block=drop_block, act_layer=act_layer, norm_layer=norm_layer)
        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock doest not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        self.conv1 = SplAtConv2d(
            inplanes, first_planes, kernel_size=3,
            stride=stride, padding=dilation,
            dilation=first_dilation, bias=False,
            rectify=False,
            rectify_avg=False,
            norm_layer=norm_layer,
            dropblock_prob=drop_block if drop_block else 0., **sa_kwargs)
        conv_kwargs['act_layer'] = None
        self.conv2 = ConvBnAct(
            first_planes, outplanes, kernel_size=3, dilation=dilation, **conv_kwargs)
        self.se = create_attn(attn_layer, outplanes)
        self.act = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv2.bn.weight)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.se is not None:
            x = self.se(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.act(x)
        return x

    
@register_model
def resnest10_1s1(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['resnest10']
    sa_kwargs = dict(
        groups=1,
        radix=1
    )
    model = ResNet(
        SplitAtteBasic, [1,1,1,1], num_classes=num_classes, in_chans=in_chans,
        block_args=dict(sa_kwargs=sa_kwargs), zero_init_last_bn=False, **kwargs)
    model.default_cfg = default_cfg
    return model    
    
@register_model
def resnest10_2s1(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['resnest10']
    sa_kwargs = dict(
        groups=1,
        radix=2
    )
    model = ResNet(
        SplitAtteBasic, [1,1,1,1], num_classes=num_classes, in_chans=in_chans,
        block_args=dict(sa_kwargs=sa_kwargs), zero_init_last_bn=False, **kwargs)
    model.default_cfg = default_cfg
    return model

@register_model
def resnest14_2s1(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['resnest14']
    sa_kwargs = dict(
        groups=1,
        radix=2
    )
    model = ResNet(
        SplitAtteBasic, [1,2,2,1], num_classes=num_classes, in_chans=in_chans,
        block_args=dict(sa_kwargs=sa_kwargs), zero_init_last_bn=False, **kwargs)
    model.default_cfg = default_cfg
    return model

@register_model
def resnest18_2s1(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['resnest18']
    sa_kwargs = dict(
        groups=1,
        radix=2
    )
    model = ResNet(
        SplitAtteBasic, [2,2,2,2], num_classes=num_classes, in_chans=in_chans,
        block_args=dict(sa_kwargs=sa_kwargs), zero_init_last_bn=False, **kwargs)
    model.default_cfg = default_cfg
    return model

@register_model
def resnest10_2s2(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['resnest10']
    sa_kwargs = dict(
        groups=2,
        radix=2
    )
    model = ResNet(
        SplitAtteBasic, [1,1,1,1], num_classes=num_classes, in_chans=in_chans,
        block_args=dict(sa_kwargs=sa_kwargs), zero_init_last_bn=False, **kwargs)
    model.default_cfg = default_cfg
    return model

@register_model
def resnest14_2s2(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['resnest10']
    sa_kwargs = dict(
        groups=2,
        radix=2
    )
    model = ResNet(
        SplitAtteBasic, [1,2,2,1], num_classes=num_classes, in_chans=in_chans,
        block_args=dict(sa_kwargs=sa_kwargs), zero_init_last_bn=False, **kwargs)
    model.default_cfg = default_cfg
    return model

@register_model
def resnest18_2s2(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['resnest10']
    sa_kwargs = dict(
        groups=2,
        radix=2
    )
    model = ResNet(
        SplitAtteBasic, [2,2,2,2], num_classes=num_classes, in_chans=in_chans,
        block_args=dict(sa_kwargs=sa_kwargs), zero_init_last_bn=False, **kwargs)
    model.default_cfg = default_cfg
    return model

@register_model
def resnest10_4s1(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['resnest10']
    sa_kwargs = dict(
        groups=1,
        radix=4
    )
    model = ResNet(
        SplitAtteBasic, [1,1,1,1], num_classes=num_classes, in_chans=in_chans,
        block_args=dict(sa_kwargs=sa_kwargs), zero_init_last_bn=False, **kwargs)
    model.default_cfg = default_cfg
    return model

@register_model
def resnest14_4s1(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['resnest10']
    sa_kwargs = dict(
        groups=1,
        radix=4
    )
    model = ResNet(
        SplitAtteBasic, [1,2,2,1], num_classes=num_classes, in_chans=in_chans,
        block_args=dict(sa_kwargs=sa_kwargs), zero_init_last_bn=False, **kwargs)
    model.default_cfg = default_cfg
    return model

@register_model
def resnest18_4s1(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['resnest10']
    sa_kwargs = dict(
        groups=1,
        radix=4
    )
    model = ResNet(
        SplitAtteBasic, [2,2,2,2], num_classes=num_classes, in_chans=in_chans,
        block_args=dict(sa_kwargs=sa_kwargs), zero_init_last_bn=False, **kwargs)
    model.default_cfg = default_cfg
    return model

@register_model
def resnest10_2s4(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['resnest10']
    sa_kwargs = dict(
        groups=2,
        radix=4
    )
    model = ResNet(
        SplitAtteBasic, [1,1,1,1], num_classes=num_classes, in_chans=in_chans,
        block_args=dict(sa_kwargs=sa_kwargs), zero_init_last_bn=False, **kwargs)
    model.default_cfg = default_cfg
    return model

@register_model
def resnest10_1s1_mish(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['resnest10']
    sa_kwargs = dict(
        groups=1,
        radix=1
    )
    model = ResNet(
        SplitAtteBasic, [1,1,1,1], num_classes=num_classes, in_chans=in_chans,
        block_args=dict(sa_kwargs=sa_kwargs), zero_init_last_bn=False, act_layer=Mish, **kwargs)
    model.default_cfg = default_cfg
    return model

@register_model
def resnest10_2s1_mish(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['resnest10']
    sa_kwargs = dict(
        groups=1,
        radix=2
    )
    model = ResNet(
        SplitAtteBasic, [1,1,1,1], num_classes=num_classes, in_chans=in_chans,
        block_args=dict(sa_kwargs=sa_kwargs), zero_init_last_bn=False, act_layer=Mish, **kwargs)
    model.default_cfg = default_cfg
    return model  

@register_model
def resnest10_2s2_mish(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['resnest10']
    sa_kwargs = dict(
        groups=2,
        radix=2
    )
    model = ResNet(
        SplitAtteBasic, [1,1,1,1], num_classes=num_classes, in_chans=in_chans,
        block_args=dict(sa_kwargs=sa_kwargs), zero_init_last_bn=False, act_layer=Mish, **kwargs)
    model.default_cfg = default_cfg
    return model    

@register_model
def resnest10_4s1_mish(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['resnest10']
    sa_kwargs = dict(
        groups=1,
        radix=4
    )
    model = ResNet(
        SplitAtteBasic, [1,1,1,1], num_classes=num_classes, in_chans=in_chans,
        block_args=dict(sa_kwargs=sa_kwargs), zero_init_last_bn=False, act_layer=Mish, **kwargs)
    model.default_cfg = default_cfg
    return model