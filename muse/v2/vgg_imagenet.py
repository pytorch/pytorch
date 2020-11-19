'''
Custom vgg for MUSEv2 design
'''

import math
import torch
import torch.nn as nn

from quant_layer import QuantLayer


__all__ = ['vgg']

defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class BasicBlock(nn.Module):
    def __init__(self, in_channels, v, batch_norm=False, stride=1):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, v, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(v)
        self.relu = nn.ReLU(inplace=True)
        self.quant = QuantLayer()

    def forward(self, x):
        return nn.Sequential(
            *[self.conv, self.bn, self.relu, self.quant])(x)


class vgg(nn.Module):
    def __init__(self, num_classes=10, depth=16, init_weights=True, cfg=None):
        super(vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.layers = self.make_layers(cfg, True)

        # self.extra_layer = BasicBlock(512, 512, stride=2)

        self.avgpool_1 = nn.AvgPool2d(2, stride=2)
        self.avgpool_2 = nn.AvgPool2d((4, 4))

        # self.classifier = nn.Linear(cfg[-1], num_classes)
        self.quant_avg1 = QuantLayer()
        self.quant_avg2 = QuantLayer()
        self.quant_fc1 = QuantLayer()
        self.quant_fc2 = QuantLayer()
        self.quant_fc3 = QuantLayer()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            self.quant_fc1,
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            self.quant_fc2,
            nn.Linear(512, num_classes),
            self.quant_fc3,
        )
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        # layers += [nn.BatchNorm2d(3)]
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [BasicBlock(in_channels, v, batch_norm=batch_norm)]

                in_channels = v
        return nn.Sequential(*layers)

    def padding(self, x):
        assert x.dim() == 4
        assert x.size(2) == x.size(3) == 7
        sz = list(x.size())
        sz[2] = 1
        x = torch.cat([x, torch.zeros(sz, device=x.device)], dim=2)

        sz = list(x.size())
        sz[3] = 1
        x = torch.cat([x, torch.zeros(sz, device=x.device)], dim=3)
        assert x.size(2) == x.size(3) == 8
        return x

    def forward(self, x):
        x = self.layers(x)
        x = self.padding(x)
        x = self.avgpool_1(x)
        x = self.quant_avg1(x)
        x = self.avgpool_2(x)
        x = self.quant_avg2(x)
        print("before x.view", x.dim(), x.size())
        x = x.view(x.size(0), -1)
        print("after x.view", x.dim(), x.size())
        x = self.classifier(x)
        print("after class", x.dim(), x.size())
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
m = vgg(depth=16, init_weights=True, cfg=None)
for name, W in m.named_parameters():
    print(name, W.size())
print(m)
print(m(torch.rand(1,3,224,224)))
