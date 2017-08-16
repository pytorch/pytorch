import torch
import torch.jit
import torch.nn as nn
import unittest
import math
import torch.nn.functional as F
from torch.autograd import Variable, Function
from common import TestCase, run_tests


class TestJit(TestCase):
    maxDiff = None

    def test_simple(self):
        a = x = Variable(torch.Tensor([0.4]), requires_grad=True)
        b = y = Variable(torch.Tensor([0.7]), requires_grad=True)

        trace, (x, y) = torch._C._tracer_enter((x, y))
        z = torch.sigmoid(torch.tanh(x * (x + y)))
        z, = torch._C._tracer_exit((z,))
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_init(trace)
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_fuse(trace)
        torch._C._jit_pass_lint(trace)

        self.assertExpected(str(trace))

    def test_export(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        trace, (x, y) = torch._C._tracer_enter((x, y))
        z = -torch.sigmoid(torch.tanh(x * (x + y)))
        torch._C._tracer_exit((z,))
        torch._C._jit_pass_lint(trace)
        self.assertExpected(torch._C._jit_pass_export(trace))

    def test_lstm(self):
        # Careful: don't use fused backend (enabled with CUDA)
        # Pasted from test_LSTM_cell
        input = Variable(torch.randn(3, 10))
        hx = Variable(torch.randn(3, 20))
        cx = Variable(torch.randn(3, 20))
        trace, _ = torch.jit.record_trace(
            nn.LSTMCell(10, 20), input, (hx, cx))
        print(str(trace))
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_init(trace)
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_fuse(trace)
        torch._C._jit_pass_lint(trace)
        self.assertExpected(str(trace))

    def test_function_as_argument(self):
        # Careful: don't use fused backend (enabled with CUDA)
        # Pasted from test_LSTM_cell
        input = Variable(torch.randn(3, 10))
        hx = Variable(torch.randn(3, 20))
        cx = Variable(torch.randn(3, 20))
        lstm = nn.LSTMCell(10, 20)

        def a_function(a, b):
            return lstm(a, b)
        trace, _ = torch.jit.record_trace(
            a_function, input, (hx, cx), parameters=lstm.parameters())
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_init(trace)
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_fuse(trace)
        torch._C._jit_pass_lint(trace)
        self.assertExpected(str(trace))

    def test_verify(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        def doit(x, y):
            return torch.sigmoid(torch.tanh(x * (x + y)))

        traced = torch.jit.traced(
            doit, enabled=True, verify=True, time=True, optimize=False)
        z = traced(x, y)
        z2 = traced(x, y)
        self.assertEqual(z, torch.sigmoid(torch.tanh(x * (x + y))))
        self.assertEqual(z, z2)

    def test_traced_function(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        def doit(x, y):
            return torch.sigmoid(torch.tanh(x * (x + y)))

        traced = torch.jit.traced(doit)
        z = traced(x, y)
        z2 = traced(x, y)
        self.assertEqual(z, torch.sigmoid(torch.tanh(x * (x + y))))
        self.assertEqual(z, z2)

    def test_disabled_traced_function(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        def doit(x, y):
            return torch.sigmoid(torch.tanh(x * (x + y)))

        traced = torch.jit.traced(doit, enabled=False)
        z = traced(x, y)
        z2 = traced(x, y)
        self.assertEqual(z, torch.sigmoid(torch.tanh(x * (x + y))))
        self.assertEqual(z, z2)

    def test_traced_module(self):
        input = Variable(torch.randn(3, 10))
        hx = Variable(torch.randn(3, 20))
        cx = Variable(torch.randn(3, 20))
        lstm = nn.LSTMCell(10, 20)
        lstm = torch.jit.traced(lstm, verify=True)

        out = lstm(input, (hx, cx))
        out2 = lstm(input, (hx, cx))
        self.assertEqual(out, out2)

    def test_alexnet(self):

        inplace = False
        class AlexNet(nn.Module):

            def __init__(self, num_classes=1000):
                super(AlexNet, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                    nn.ReLU(inplace=inplace),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(64, 192, kernel_size=5, padding=2),
                    nn.ReLU(inplace=inplace),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(192, 384, kernel_size=3, padding=1),
                    nn.ReLU(inplace=inplace),
                    nn.Conv2d(384, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=inplace),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=inplace),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                )
                self.classifier = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(256 * 6 * 6, 4096),
                    nn.ReLU(inplace=inplace),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=inplace),
                    nn.Linear(4096, num_classes),
                )

            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), 256 * 6 * 6)
                x = self.classifier(x)
                return x

        x = Variable(torch.randn(10, 3, 224, 224).fill_(1.0), requires_grad=True)
        trace, _ = torch.jit.record_trace(AlexNet(), x)
        print(str(trace))
        self.assertExpected(str(trace))
        self.assertExpected(torch._C._jit_pass_export(trace), "pbtxt")

    def test_vgg(self):
        inplace = False
        cfg = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }

        class VGG(nn.Module):

            def __init__(self, features, num_classes=1000):
                super(VGG, self).__init__()
                self.features = features
                self.classifier = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(inplace=inplace),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=inplace),
                    nn.Dropout(),
                    nn.Linear(4096, num_classes),
                )
                self._initialize_weights()

            def _initialize_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        m.weight.data.normal_(0, math.sqrt(2. / n))
                        if m.bias is not None:
                            m.bias.data.zero_()
                    elif isinstance(m, nn.BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()
                    elif isinstance(m, nn.Linear):
                        n = m.weight.size(1)
                        m.weight.data.normal_(0, 0.01)
                        m.bias.data.zero_()

            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x

        def make_layers(cfg, batch_norm=False):
            layers = []
            in_channels = 3
            for v in cfg:
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=inplace)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=inplace)]
                    in_channels = v
            return nn.Sequential(*layers)

        # VGG 16-layer model (configuration "D")
        x = Variable(torch.randn(10, 3, 224, 224).fill_(1.0), requires_grad=True)
        vgg16 = VGG(make_layers(cfg['D']))
        trace, _ = torch.jit.record_trace(vgg16, x)
        print(str(trace))
        self.assertExpected(str(trace), "16")
        self.assertExpected(torch._C._jit_pass_export(trace), "16-pbtxt")

        # VGG 16-layer model (configuration "D") with batch normalization
        x = Variable(torch.randn(10, 3, 224, 224).fill_(1.0), requires_grad=True)
        vgg16_bn = VGG(make_layers(cfg['D'], batch_norm=True))
        trace, _ = torch.jit.record_trace(vgg16_bn, x)
        print(str(trace))
        self.assertExpected(str(trace), "16_bn")
        # self.assertExpected(torch._C._jit_pass_export(trace), "16_bn-pbtxt")

        # VGG 19-layer model (configuration "E")
        x = Variable(torch.randn(10, 3, 224, 224).fill_(1.0), requires_grad=True)
        vgg19 = VGG(make_layers(cfg['E']))
        trace, _ = torch.jit.record_trace(vgg19, x)
        print(str(trace))
        self.assertExpected(str(trace), "19")
        self.assertExpected(torch._C._jit_pass_export(trace), "19-pbtxt")

        # VGG 19-layer model (configuration 'E') with batch normalization
        x = Variable(torch.randn(10, 3, 224, 224).fill_(1.0), requires_grad=True)
        vgg19_bn = VGG(make_layers(cfg['E'], batch_norm=True))
        trace, _ = torch.jit.record_trace(vgg19_bn, x)
        print(str(trace))
        self.assertExpected(str(trace), "19_bn")
        # self.assertExpected(torch._C._jit_pass_export(trace), "19_bn-pbtxt")

    def test_resnet(self):
        inplace = False

        def conv3x3(in_planes, out_planes, stride=1):
            "3x3 convolution with padding"
            return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=True)

        class BasicBlock(nn.Module):
            expansion = 1

            def __init__(self, inplanes, planes, stride=1, downsample=None):
                super(BasicBlock, self).__init__()
                self.conv1 = conv3x3(inplanes, planes, stride)
                self.bn1 = nn.BatchNorm2d(planes)
                self.relu = nn.ReLU(inplace=inplace)
                self.conv2 = conv3x3(planes, planes)
                self.bn2 = nn.BatchNorm2d(planes)
                self.downsample = downsample
                self.stride = stride

            def forward(self, x):
                residual = x

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)

                if self.downsample is not None:
                    residual = self.downsample(x)

                out_sum = out + residual
                out = self.relu(out_sum)

                return out

        class Bottleneck(nn.Module):
            expansion = 4

            def __init__(self, inplanes, planes, stride=1, downsample=None):
                super(Bottleneck, self).__init__()
                self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
                self.bn1 = nn.BatchNorm2d(planes)
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                       padding=1, bias=True)
                self.bn2 = nn.BatchNorm2d(planes)
                self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
                self.bn3 = nn.BatchNorm2d(planes * 4)
                self.relu = nn.ReLU(inplace=inplace)
                self.downsample = downsample
                self.stride = stride

            def forward(self, x):
                residual = x

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)
                out = self.relu(out)

                out = self.conv3(out)
                out = self.bn3(out)

                if self.downsample is not None:
                    residual = self.downsample(x)

                out_sum = out + residual
                out = self.relu(out_sum)

                return out

        class ResNet(nn.Module):

            def __init__(self, block, layers, num_classes=1000):
                self.inplanes = 64
                super(ResNet, self).__init__()
                self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                       bias=True)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU(inplace=inplace)
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                self.layer1 = self._make_layer(block, 64, layers[0])
                self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
                self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
                self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
                self.avgpool = nn.AvgPool2d(7)
                self.fc = nn.Linear(512 * block.expansion, num_classes)

                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        m.weight.data.normal_(0, math.sqrt(2. / n))
                    elif isinstance(m, nn.BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()

            def _make_layer(self, block, planes, blocks, stride=1):
                downsample = None
                if stride != 1 or self.inplanes != planes * block.expansion:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=stride, bias=True),
                        nn.BatchNorm2d(planes * block.expansion),
                    )

                layers = []
                layers.append(block(self.inplanes, planes, stride, downsample))
                self.inplanes = planes * block.expansion
                for _ in range(1, blocks):
                    layers.append(block(self.inplanes, planes))

                return nn.Sequential(*layers)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)

                return x

        # ResNet50 model
        x = Variable(torch.randn(10, 3, 224, 224).fill_(1.0), requires_grad=True)
        resnet50 = ResNet(Bottleneck, [3, 4, 6, 3])
        trace, _ = torch.jit.record_trace(resnet50, x)
        print(str(trace))
        self.assertExpected(str(trace), "50")
        # self.assertExpected(torch._C._jit_pass_export(trace), "50-pbtxt")

    def test_inception(self):
        inplace = False

        class BasicConv2d(nn.Module):

            def __init__(self, in_channels, out_channels, **kwargs):
                super(BasicConv2d, self).__init__()
                self.conv = nn.Conv2d(
                    in_channels, out_channels, bias=True, **kwargs)
                self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return F.relu(x, inplace=inplace)

        class InceptionA(nn.Module):

            def __init__(self, in_channels, pool_features):
                super(InceptionA, self).__init__()
                self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

                self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
                self.branch5x5_2 = BasicConv2d(
                    48, 64, kernel_size=5, padding=2)

                self.branch3x3dbl_1 = BasicConv2d(
                    in_channels, 64, kernel_size=1)
                self.branch3x3dbl_2 = BasicConv2d(
                    64, 96, kernel_size=3, padding=1)
                self.branch3x3dbl_3 = BasicConv2d(
                    96, 96, kernel_size=3, padding=1)

                self.branch_pool = BasicConv2d(
                    in_channels, pool_features, kernel_size=1)

            def forward(self, x):
                branch1x1 = self.branch1x1(x)

                branch5x5 = self.branch5x5_1(x)
                branch5x5 = self.branch5x5_2(branch5x5)

                branch3x3dbl = self.branch3x3dbl_1(x)
                branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
                branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

                branch_pool = F.avg_pool2d(
                    x, kernel_size=3, stride=1, padding=1)
                branch_pool = self.branch_pool(branch_pool)

                outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
                return torch.cat(outputs, 1)

        class InceptionB(nn.Module):

            def __init__(self, in_channels):
                super(InceptionB, self).__init__()
                self.branch3x3 = BasicConv2d(
                    in_channels, 384, kernel_size=3, stride=2)

                self.branch3x3dbl_1 = BasicConv2d(
                    in_channels, 64, kernel_size=1)
                self.branch3x3dbl_2 = BasicConv2d(
                    64, 96, kernel_size=3, padding=1)
                self.branch3x3dbl_3 = BasicConv2d(
                    96, 96, kernel_size=3, stride=2)

            def forward(self, x):
                branch3x3 = self.branch3x3(x)

                branch3x3dbl = self.branch3x3dbl_1(x)
                branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
                branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

                branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

                outputs = [branch3x3, branch3x3dbl, branch_pool]
                return torch.cat(outputs, 1)

        class InceptionC(nn.Module):

            def __init__(self, in_channels, channels_7x7):
                super(InceptionC, self).__init__()
                self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

                c7 = channels_7x7
                self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
                self.branch7x7_2 = BasicConv2d(
                    c7, c7, kernel_size=(1, 7), padding=(0, 3))
                self.branch7x7_3 = BasicConv2d(
                    c7, 192, kernel_size=(7, 1), padding=(3, 0))

                self.branch7x7dbl_1 = BasicConv2d(
                    in_channels, c7, kernel_size=1)
                self.branch7x7dbl_2 = BasicConv2d(
                    c7, c7, kernel_size=(7, 1), padding=(3, 0))
                self.branch7x7dbl_3 = BasicConv2d(
                    c7, c7, kernel_size=(1, 7), padding=(0, 3))
                self.branch7x7dbl_4 = BasicConv2d(
                    c7, c7, kernel_size=(7, 1), padding=(3, 0))
                self.branch7x7dbl_5 = BasicConv2d(
                    c7, 192, kernel_size=(1, 7), padding=(0, 3))

                self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

            def forward(self, x):
                branch1x1 = self.branch1x1(x)

                branch7x7 = self.branch7x7_1(x)
                branch7x7 = self.branch7x7_2(branch7x7)
                branch7x7 = self.branch7x7_3(branch7x7)

                branch7x7dbl = self.branch7x7dbl_1(x)
                branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
                branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
                branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
                branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

                branch_pool = F.avg_pool2d(
                    x, kernel_size=3, stride=1, padding=1)
                branch_pool = self.branch_pool(branch_pool)

                outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
                return torch.cat(outputs, 1)

        class InceptionD(nn.Module):

            def __init__(self, in_channels):
                super(InceptionD, self).__init__()
                self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
                self.branch3x3_2 = BasicConv2d(
                    192, 320, kernel_size=3, stride=2)
                self.branch7x7x3_1 = BasicConv2d(
                    in_channels, 192, kernel_size=1)
                self.branch7x7x3_2 = BasicConv2d(
                    192, 192, kernel_size=(1, 7), padding=(0, 3))
                self.branch7x7x3_3 = BasicConv2d(
                    192, 192, kernel_size=(7, 1), padding=(3, 0))
                self.branch7x7x3_4 = BasicConv2d(
                    192, 192, kernel_size=3, stride=2)

            def forward(self, x):
                branch3x3 = self.branch3x3_1(x)
                branch3x3 = self.branch3x3_2(branch3x3)

                branch7x7x3 = self.branch7x7x3_1(x)
                branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
                branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
                branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

                branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
                outputs = [branch3x3, branch7x7x3, branch_pool]
                return torch.cat(outputs, 1)

        class InceptionE(nn.Module):

            def __init__(self, in_channels):
                super(InceptionE, self).__init__()
                self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

                self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
                self.branch3x3_2a = BasicConv2d(
                    384, 384, kernel_size=(1, 3), padding=(0, 1))
                self.branch3x3_2b = BasicConv2d(
                    384, 384, kernel_size=(3, 1), padding=(1, 0))
                self.branch3x3dbl_1 = BasicConv2d(
                    in_channels, 448, kernel_size=1)
                self.branch3x3dbl_2 = BasicConv2d(
                    448, 384, kernel_size=3, padding=1)
                self.branch3x3dbl_3a = BasicConv2d(
                    384, 384, kernel_size=(1, 3), padding=(0, 1))
                self.branch3x3dbl_3b = BasicConv2d(
                    384, 384, kernel_size=(3, 1), padding=(1, 0))

                self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

            def forward(self, x):
                branch1x1 = self.branch1x1(x)

                branch3x3 = self.branch3x3_1(x)
                branch3x3 = [
                    self.branch3x3_2a(branch3x3),
                    self.branch3x3_2b(branch3x3),
                ]
                branch3x3 = torch.cat(branch3x3, 1)

                branch3x3dbl = self.branch3x3dbl_1(x)
                branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
                branch3x3dbl = [
                    self.branch3x3dbl_3a(branch3x3dbl),
                    self.branch3x3dbl_3b(branch3x3dbl),
                ]
                branch3x3dbl = torch.cat(branch3x3dbl, 1)

                branch_pool = F.avg_pool2d(
                    x, kernel_size=3, stride=1, padding=1
                )
                branch_pool = self.branch_pool(branch_pool)

                outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
                return torch.cat(outputs, 1)

        class InceptionAux(nn.Module):

            def __init__(self, in_channels, num_classes):
                super(InceptionAux, self).__init__()
                self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
                self.conv1 = BasicConv2d(128, 768, kernel_size=3)
                self.conv1.stddev = 0.01
                self.fc = nn.Linear(768, num_classes)
                self.fc.stddev = 0.001

            def forward(self, x):
                # 17 x 17 x 768
                x = F.avg_pool2d(x, kernel_size=5, stride=3)
                # 5 x 5 x 768
                x = self.conv0(x)
                # 5 x 5 x 128
                x = self.conv1(x)
                # 1 x 1 x 768
                x = x.view(x.size(0), -1)
                # 768
                x = self.fc(x)
                # 1000
                return x

        class Inception3(nn.Module):

            def __init__(
                self, num_classes=1000, aux_logits=True, transform_input=False
            ):
                super(Inception3, self).__init__()
                self.aux_logits = aux_logits
                self.transform_input = transform_input
                self.Conv2d_1a_3x3 = BasicConv2d(
                    3, 32, kernel_size=3, stride=2
                )
                self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
                self.Conv2d_2b_3x3 = BasicConv2d(
                    32, 64, kernel_size=3, padding=1
                )
                self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
                self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
                self.Mixed_5b = InceptionA(192, pool_features=32)
                self.Mixed_5c = InceptionA(256, pool_features=64)
                self.Mixed_5d = InceptionA(288, pool_features=64)
                self.Mixed_6a = InceptionB(288)
                self.Mixed_6b = InceptionC(768, channels_7x7=128)
                self.Mixed_6c = InceptionC(768, channels_7x7=160)
                self.Mixed_6d = InceptionC(768, channels_7x7=160)
                self.Mixed_6e = InceptionC(768, channels_7x7=192)
                if aux_logits:
                    self.AuxLogits = InceptionAux(768, num_classes)
                self.Mixed_7a = InceptionD(768)
                self.Mixed_7b = InceptionE(1280)
                self.Mixed_7c = InceptionE(2048)
                self.fc = nn.Linear(2048, num_classes)

                for m in self.modules():
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                        import scipy.stats as stats
                        stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                        X = stats.truncnorm(-2, 2, scale=stddev)
                        values = torch.Tensor(X.rvs(m.weight.data.numel()))
                        m.weight.data.copy_(values)
                    elif isinstance(m, nn.BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()

            def forward(self, x):
                if self.transform_input:
                    x = x.clone()
                    x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
                    x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
                    x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
                # 299 x 299 x 3
                x = self.Conv2d_1a_3x3(x)
                # 149 x 149 x 32
                x = self.Conv2d_2a_3x3(x)
                # 147 x 147 x 32
                x = self.Conv2d_2b_3x3(x)
                # 147 x 147 x 64
                x = F.max_pool2d(x, kernel_size=3, stride=2)
                # 73 x 73 x 64
                x = self.Conv2d_3b_1x1(x)
                # 73 x 73 x 80
                x = self.Conv2d_4a_3x3(x)
                # 71 x 71 x 192
                x = F.max_pool2d(x, kernel_size=3, stride=2)
                # 35 x 35 x 192
                x = self.Mixed_5b(x)
                # 35 x 35 x 256
                x = self.Mixed_5c(x)
                # 35 x 35 x 288
                x = self.Mixed_5d(x)
                # 35 x 35 x 288
                x = self.Mixed_6a(x)
                # 17 x 17 x 768
                x = self.Mixed_6b(x)
                # 17 x 17 x 768
                x = self.Mixed_6c(x)
                # 17 x 17 x 768
                x = self.Mixed_6d(x)
                # 17 x 17 x 768
                x = self.Mixed_6e(x)
                # 17 x 17 x 768
                if self.training and self.aux_logits:
                    aux = self.AuxLogits(x)
                # 17 x 17 x 768
                x = self.Mixed_7a(x)
                # 8 x 8 x 1280
                x = self.Mixed_7b(x)
                # 8 x 8 x 2048
                x = self.Mixed_7c(x)
                # 8 x 8 x 2048
                x = F.avg_pool2d(x, kernel_size=5)
                # 1 x 1 x 2048
                x = F.dropout(x, training=self.training)
                # 1 x 1 x 2048
                x = x.view(x.size(0), -1)
                # 2048
                x = self.fc(x)
                # 1000 (num_classes)
                if self.training and self.aux_logits:
                    return x, aux
                return x

        x = Variable(
            torch.randn(10, 3, 224, 224).fill_(1.0), requires_grad=True)
        trace, _ = torch.jit.record_trace(Inception3(), x)
        print(str(trace))
        self.assertExpected(str(trace), "3")
        # self.assertExpected(torch._C._jit_pass_export(trace), "3-pbtxt")

    def test_squeezenet(self):
        inplace = False

        class Fire(nn.Module):

            def __init__(self, inplanes, squeeze_planes,
                         expand1x1_planes, expand3x3_planes):
                super(Fire, self).__init__()
                self.inplanes = inplanes
                self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
                self.squeeze_activation = nn.ReLU(inplace=inplace)
                self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                           kernel_size=1)
                self.expand1x1_activation = nn.ReLU(inplace=inplace)
                self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                           kernel_size=3, padding=1)
                self.expand3x3_activation = nn.ReLU(inplace=inplace)

            def forward(self, x):
                x = self.squeeze_activation(self.squeeze(x))
                return torch.cat([
                    self.expand1x1_activation(self.expand1x1(x)),
                    self.expand3x3_activation(self.expand3x3(x))
                ], 1)


        class SqueezeNet(nn.Module):

            def __init__(self, version=1.0, num_classes=1000):
                super(SqueezeNet, self).__init__()
                if version not in [1.0, 1.1]:
                    raise ValueError("Unsupported SqueezeNet version {version}:"
                                     "1.0 or 1.1 expected".format(version=version))
                self.num_classes = num_classes
                if version == 1.0:
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 96, kernel_size=7, stride=2),
                        nn.ReLU(inplace=inplace),
                        nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                        Fire(96, 16, 64, 64),
                        Fire(128, 16, 64, 64),
                        Fire(128, 32, 128, 128),
                        nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                        Fire(256, 32, 128, 128),
                        Fire(256, 48, 192, 192),
                        Fire(384, 48, 192, 192),
                        Fire(384, 64, 256, 256),
                        nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                        Fire(512, 64, 256, 256),
                    )
                else:
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3, stride=2),
                        nn.ReLU(inplace=inplace),
                        nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                        Fire(64, 16, 64, 64),
                        Fire(128, 16, 64, 64),
                        nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                        Fire(128, 32, 128, 128),
                        Fire(256, 32, 128, 128),
                        nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                        Fire(256, 48, 192, 192),
                        Fire(384, 48, 192, 192),
                        Fire(384, 64, 256, 256),
                        Fire(512, 64, 256, 256),
                    )
                # Final convolution is initialized differently form the rest
                final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
                self.classifier = nn.Sequential(
                    nn.Dropout(p=0.5),
                    final_conv,
                    nn.ReLU(inplace=inplace),
                    nn.AvgPool2d(13)
                )

                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        gain = 2.0
                        if m is final_conv:
                            m.weight.data.normal_(0, 0.01)
                        else:
                            fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                            u = math.sqrt(3.0 * gain / fan_in)
                            m.weight.data.uniform_(-u, u)
                        if m.bias is not None:
                            m.bias.data.zero_()

            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x.view(x.size(0), self.num_classes)

        # SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and
        # <0.5MB model size
        x = Variable(torch.randn(10, 3, 224, 224).fill_(1.0), requires_grad=True)
        sqnet_v1_0 = SqueezeNet(version=1.1)
        trace, _ = torch.jit.record_trace(sqnet_v1_0, x)
        print(str(trace))
        self.assertExpected(str(trace), "1_0")
        # self.assertExpected(torch._C._jit_pass_export(trace), "1_0-pbtxt")

        # SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
        # than SqueezeNet 1.0, without sacrificing accuracy.
        x = Variable(torch.randn(10, 3, 224, 224).fill_(1.0), requires_grad=True)
        sqnet_v1_1 = SqueezeNet(version=1.1)
        trace, _ = torch.jit.record_trace(sqnet_v1_1, x)
        print(str(trace))
        self.assertExpected(str(trace), "1_1")
        # self.assertExpected(torch._C._jit_pass_export(trace), "1_1-pbtxt")

    def test_autograd_closure(self):
        a = x = Variable(torch.Tensor([0.4]), requires_grad=True)
        b = y = Variable(torch.Tensor([0.7]), requires_grad=True)

        trace, (x, y) = torch._C._tracer_enter((x, y))

        z, _ = torch.max(x * (x + y), 0)
        w = torch.abs(x * x * x + y)

        z, w = torch._C._tracer_exit((z, w))
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_init(trace)
        torch._C._jit_pass_lint(trace)
        closure = torch._C._jit_createAutogradClosure(trace)
        z2, w2 = Variable._execution_engine.run_forward(closure, (a, b))
        self.assertEqual(z, z2)
        self.assertEqual(w, w2)

    def test_constant(self):
        a = x = Variable(torch.randn(2, 2), requires_grad=True)

        trace, (x,) = torch._C._tracer_enter((x,))

        y = Variable(torch.diag(torch.Tensor([2, 2])))
        z = x.matmul(y)

        z, = torch._C._tracer_exit((z,))
        closure = torch._C._jit_createAutogradClosure(trace)

        z2, = Variable._execution_engine.run_forward(closure, (a,))
        self.assertEqual(z, z2)

        y.data.fill_(1000)  # make sure the data has been cloned

        a2 = Variable(torch.ones(2, 2) * 2, requires_grad=True)
        z3, = Variable._execution_engine.run_forward(closure, (a2,))
        self.assertEqual(z3.data, torch.ones(2, 2) * 4)

    def test_c_function(self):
        x = Variable(torch.randn(1, 3, 10, 10))
        m = nn.Conv2d(3, 8, 3, 1)

        trace, new_vars = torch._C._tracer_enter((x,) + tuple(m.parameters()))
        x = new_vars[0]
        y = m(x)
        _ = torch._C._tracer_exit((y,))
        self.assertExpected(str(trace))

    def test_legacy_fail(self):

        class Legacy(Function):
            def forward(self, x):
                return x

            def backward(self, grad_output):
                return grad_output
        a = x = Variable(torch.Tensor([0]), requires_grad=True)
        trace, (x,) = torch._C._tracer_enter((x,))
        self.assertRaises(RuntimeError, lambda: Legacy()(x))
        x, = torch._C._tracer_exit((x,))

    @unittest.skip("in-place is not supported")
    def test_inplace_transplant(self):
        a = x = Variable(torch.Tensor([0]), requires_grad=True)
        trace, (x,) = torch._C._tracer_enter((x,))
        y = x.clone()
        y.add_(2)
        y.add_(3)
        y, = torch._C._tracer_exit((y,))
        self.assertExpected(str(trace))

    def test_backward(self):
        a = Variable(torch.randn(2, 2), requires_grad=True)
        b = Variable(torch.randn(2, 2), requires_grad=True)

        x = a
        y = a * b

        trace, (x, y) = torch._C._tracer_enter((x, y))
        z = y * 2 * x
        z, = torch._C._tracer_exit((z,))
        torch._C._jit_pass_lint(trace)

        grad, = torch.autograd.grad(z, x, Variable(
            torch.ones(2, 2), requires_grad=True), create_graph=True)
        torch._C._jit_pass_lint(trace)

        # Run dead code elimination to remove unused trace nodes
        torch._C._jit_pass_dco(trace)
        self.assertExpected(str(trace))

    def test_cpp(self):
        torch._C._jit_run_cpp_tests()


if __name__ == '__main__':

    run_tests()
