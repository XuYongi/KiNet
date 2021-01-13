##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNeSt models"""

import torch
import torch.nn as nn
from .resnet import ResNet, ResNet_for_middle, ResNet_for_last, ResNet_middle_all, Bottleneck
from .resnet import Bottleneck_half 
from torch.nn import init
__all__ = ['resnest50','resnest50_middle','resnest50_last', 'resnest50_middle_16','resnest101', 'resnest200', 'resnest269']

_url_format = 'https://s3.us-west-1.wasabisys.com/resnest/torch/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ('528c19ca', 'resnest50'),
    ('22405ba7', 'resnest101'),
    ('75117900', 'resnest200'),
    ('0cc87c48', 'resnest269'),
    ]}

def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]

resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for
    name in _model_sha256.keys()
}

def resnest50_middle(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet_for_middle(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    #kaiming初始化
    # for key in model.state_dict():
    #     if key.split('.')[-1] == 'weight':
    #         if 'conv' in key:
    #             init.kaiming_normal(model.state_dict()[key], mode='fan_in', nonlinearity='leaky_relu')
    #         if 'bn' in key:
    #             model.state_dict()[key][...] = 1
    #     elif key.split('.')[-1] == 'bias':
    #         model.state_dict()[key][...] = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    if pretrained:
        # model.load_state_dict(torch.hub.load_state_dict_from_url(
        #     resnest_model_urls['resnest50'], progress=True, check_hash=True))
        pretrained_dict = torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest50'], progress=True, check_hash=True)
        model_dict = model.state_dict()
        # 筛除不加载的层结构
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and k not in ['conv1.0.weight',
                                                            'fc.weight','fc.bias','layer4.0.conv1.weight','layer4.0.downsample.1.weight']
                                                                        and k[:7] != 'layer3.' and k[:7]!='layer4.' and k[:6]!='conv1.' and k[:4]!='bn1.'  )}
        # 更新当前网络的结构字典
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model
def resnest50_middle_16(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet_middle_all(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    #kaiming初始化
    # for key in model.state_dict():
    #     if key.split('.')[-1] == 'weight':
    #         if 'conv' in key:
    #             init.kaiming_normal(model.state_dict()[key], mode='fan_in', nonlinearity='leaky_relu')
    #         if 'bn' in key:
    #             model.state_dict()[key][...] = 1
    #     elif key.split('.')[-1] == 'bias':
    #         model.state_dict()[key][...] = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    if pretrained:
        # model.load_state_dict(torch.hub.load_state_dict_from_url(
        #     resnest_model_urls['resnest50'], progress=True, check_hash=True))
        pretrained_dict = torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest50'], progress=True, check_hash=True)
        model_dict = model.state_dict()
        # 筛除不加载的层结构
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and k not in ['conv1.0.weight',
                                                            'fc.weight','fc.bias','layer4.0.conv1.weight','layer4.0.downsample.1.weight']
                                                                    and k[:6]!='conv1.' and k[:4]!='bn1.'  )}
        # 更新当前网络的结构字典
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model

def resnest50_last(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet_for_last(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    #kaiming初始化
    # for key in model.state_dict():
    #     if key.split('.')[-1] == 'weight':
    #         if 'conv' in key:
    #             init.kaiming_normal(model.state_dict()[key], mode='fan_in', nonlinearity='leaky_relu')
    #         if 'bn' in key:
    #             model.state_dict()[key][...] = 1
    #     elif key.split('.')[-1] == 'bias':
    #         model.state_dict()[key][...] = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    if pretrained:
        # model.load_state_dict(torch.hub.load_state_dict_from_url(
        #     resnest_model_urls['resnest50'], progress=True, check_hash=True))
        pretrained_dict = torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest50'], progress=True, check_hash=True)
        model_dict = model.state_dict()
        # 筛除不加载的层结构
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and k not in ['conv1.0.weight',
                                                            'fc.weight','fc.bias','layer4.0.conv1.weight','layer4.0.downsample.1.weight']
                                                                        and k[:7] != 'layer3.' and k[:7]!='layer4.' )}
        # 更新当前网络的结构字典
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model

def resnest50(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck_half, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    #kaiming初始化
    # for key in model.state_dict():
    #     if key.split('.')[-1] == 'weight':
    #         if 'conv' in key:
    #             init.kaiming_normal(model.state_dict()[key], mode='fan_in', nonlinearity='leaky_relu')
    #         if 'bn' in key:
    #             model.state_dict()[key][...] = 1
    #     elif key.split('.')[-1] == 'bias':
    #         model.state_dict()[key][...] = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    if pretrained:
        # model.load_state_dict(torch.hub.load_state_dict_from_url(
        #     resnest_model_urls['resnest50'], progress=True, check_hash=True))
        pretrained_dict = torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest50'], progress=True, check_hash=True)
        model_dict = model.state_dict()
        # 筛除不加载的层结构
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and k not in ['conv1.0.weight',
                                                            'fc.weight','fc.bias','layer4.0.conv1.weight','layer4.0.downsample.1.weight']  )}
        # 更新当前网络的结构字典
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model

def resnest101(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest101'], progress=True, check_hash=True))
    return model

def resnest200(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 24, 36, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest200'], progress=True, check_hash=True))
    return model

def resnest269(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 30, 48, 8],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            resnest_model_urls['resnest269'], progress=True, check_hash=True))
    return model
