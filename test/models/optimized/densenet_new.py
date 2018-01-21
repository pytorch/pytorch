import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb, math
from torch.autograd import Variable, Function
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from collections import OrderedDict
import os, time, sys
from subprocess import Popen, PIPE

__all__ = ['DenseNet', 'densenet100', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'densenet264']

def densenet100(pretrained=False, **kwargs):
    r"""Densenet-100 model"""
    model = DenseNet(num_init_features=64, growth_rate=12, block_config=(6, 12, 24, 16),
                     **kwargs)
    return model

def densenet121(pretrained=False, **kwargs):
    r"""Densenet-121 model"""
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    return model


def densenet161(pretrained=False, **kwargs):
    r"""Densenet-161 model"""
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                     **kwargs)
    return model


def densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model"""
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    return model


def densenet201(pretrained=False, **kwargs):
    r"""Densenet-201 model"""
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
    return model


def densenet264(pretrained=False, **kwargs):
    r"""Densenet-264 model"""
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 64, 48),
                     **kwargs)
    return model


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class"""
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            num_input_features = num_features
            for j in range(num_layers):
                layer = _DenseLayer(
                    num_input_features + j * growth_rate, growth_rate, bn_size, drop_rate)
                self.features.add_module('denseblock{}_layer{}'.format((i + 1), (j + 1)), layer)

            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x, chunks=None):
        modules = [module for k, module in self._modules.items()][0]
        input_var = Variable(x.data, requires_grad=True)
        input_var = checkpoint_sequential(modules, chunks, input_var)
        input_var = F.relu(input_var, inplace=True)
        input_var = F.avg_pool2d(input_var, kernel_size=7, stride=1).view(input_var.size(0), -1)
        input_var = self.classifier(input_var)
        return input_var
