from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.mobile as mobile
import torch.quantization.fuse_modules as fuse

import torchvision

def mobilenetv2():
    model = torchvision.models.mobilenet_v2(pretrained=True).eval()

    # Fold in the batch norm in mobilenet.ConvBNReLUs into the preceding
    # convolution, and fuse the resulting Conv with the subsequent ReLU.
    for module in model.modules():
        if type(module) == torchvision.models.mobilenet.ConvBNReLU:
            module._modules['2'] = nn.ReLU6(inplace=False).eval()
            fuse(module, ['0', '1', '2'], inplace=True)

    class Residual1(nn.Module):
        def __init__(self, inverted_residual):
            super().__init__()

            self.conv = inverted_residual.conv

        def forward(self, input):
            return self.conv(input)

    class Residual2(nn.Module):
        def __init__(self, inverted_residual):
            super().__init__()

            self.conv = inverted_residual.conv
            self.add = mobile.Add()

        def forward(self, input):
            output = self.conv(input)
            return self.add(input, output, output)

    block = 1

    # Fold in the batch norm in mobilenet.InvertedResidual into the preceding
    # convolution and patch the forward function to use mobile.Add.
    for module in model.modules():
        if type(module) == torchvision.models.mobilenet.InvertedResidual:
            conv = module.conv

            for index in range(len(conv)):
                if type(conv[index]) == nn.Conv2d:
                    fuse(conv, [str(index), str(index + 1)], inplace=True)

            if not module.use_res_connect:
                model.features[block] = Residual1(module)
            else:
                model.features[block] = Residual2(module)

            block += 1

    return mobile.modules.freeze(model, inplace=True)

def resnet(model):
    assert type(model) == torchvision.models.ResNet, "Invalid model!  Expected a ResNet!"
    fuse(model, ['conv1', 'bn1', 'relu'], inplace=True)

    class Basic1(nn.Module):
        def __init__(self, basic):
            super().__init__()

            self.conv1 = basic.conv1
            self.conv2 = basic.conv2
            self.add = mobile.AddReLU(nn.ReLU())

        def forward(self, input):
            output = self.conv1(input)
            output = self.conv2(output)
            return self.add(input, output, output)

    class Basic2(nn.Module):
        def __init__(self, basic):
            super().__init__()

            self.conv1 = basic.conv1
            self.conv2 = basic.conv2
            self.downsample = basic.downsample
            self.add = mobile.AddReLU(nn.ReLU())

        def forward(self, input):
            output = self.conv1(input)
            output = self.conv2(output)
            return self.add(self.downsample(input), output, output)

    class Bottleneck1(nn.Module):
        def __init__(self, bottleneck):
            super().__init__()

            self.conv1 = bottleneck.conv1
            self.conv2 = bottleneck.conv2
            self.conv3 = bottleneck.conv3
            self.add = mobile.AddReLU(nn.ReLU())

        def forward(self, input):
            output = self.conv1(input)
            output = self.conv2(output)
            output = self.conv3(output)
            return self.add(input, output, output)

    class Bottleneck2(nn.Module):
        def __init__(self, bottleneck):
            super().__init__()

            self.conv1 = bottleneck.conv1
            self.conv2 = bottleneck.conv2
            self.conv3 = bottleneck.conv3
            self.downsample = bottleneck.downsample
            self.add = mobile.AddReLU(nn.ReLU())

        def forward(self, input):
            output = self.conv1(input)
            output = self.conv2(output)
            output = self.conv3(output)
            return self.add(self.downsample(input), output, output)

    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        block = 0

        for module in layer.children():
            module._modules['relu'] = nn.Identity()
            if module.downsample:
                fuse(module.downsample, ['0', '1'], inplace=True)

            if (type(module) == torchvision.models.resnet.BasicBlock):
                module._modules['relu1'] = nn.ReLU(inplace=False).eval()
                fuse(module, [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2']], inplace=True)
                layer[block] = Basic1(module) if not module.downsample else Basic2(module)

            elif (type(module) == torchvision.models.resnet.Bottleneck):
                module._modules['relu1'] = nn.ReLU(inplace=False).eval()
                module._modules['relu2'] = nn.ReLU(inplace=False).eval()
                fuse(module, [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2'], ['conv3', 'bn3']], inplace=True)
                layer[block] = Bottleneck1(module) if not module.downsample else Bottleneck2(module)

            else:
                assert False, "Unknown block!"

            block +=1

    return mobile.modules.freeze(model, inplace=True)

def resnet18():
    return resnet(torchvision.models.resnet18(pretrained=True).eval())

def resnet34():
    return resnet(torchvision.models.resnet34(pretrained=True).eval())

def resnet50():
    return resnet(torchvision.models.resnet50(pretrained=True).eval())

def resnext50():
    return resnet(torchvision.models.resnext50_32x4d(pretrained=True).eval())

def resnet101():
    return resnet(torchvision.models.resnet101(pretrained=True).eval())

def resnext101():
    return resnet(torchvision.models.resnext101_32x8d(pretrained=True).eval())

def test():
    input = torch.rand(1, 3, 224, 224)

    suite = {
        torchvision.models.mobilenet_v2 : mobilenetv2,
        torchvision.models.resnet18 : resnet18,
        torchvision.models.resnet34 : resnet34,
        torchvision.models.resnet50 : resnet50,
        torchvision.models.resnext50_32x4d : resnext50,
        torchvision.models.resnet101 : resnet101,
        torchvision.models.resnext101_32x8d : resnext101,
    }

    for original, optimized in suite.items():
        original = original(pretrained=True).eval()(input)
        optimized = optimized()(input)
        assert ((original - optimized).abs().max() < 2e-5), "Error!"

def script(path):
    models = [
        mobilenetv2, resnet18, resnet34, resnet50, resnext50, resnet101, resnext101
    ]

    for model in models:
        torch.jit.script(
            model(),
            torch.rand(1, 3, 224, 224)).save(path + str(model.__name__) + "_mobile_optimized.pt")
