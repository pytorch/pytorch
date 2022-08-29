import torch
from torchvision.models import resnet18
from torch.fx import subgraph_rewriter
from torch.fx import symbolic_trace

m = resnet18()
traced = symbolic_trace(m)

traced.print_readable()

# class ResNet(torch.nn.Module):
#     def forward(self, x : torch.Tensor) -> torch.Tensor:

#         # No stacktrace found for following nodes
#         conv1 = self.conv1(x);  x = None
#         bn1 = self.bn1(conv1);  conv1 = None
#         relu = self.relu(bn1);  bn1 = None
#         maxpool = self.maxpool(relu);  relu = None
#         layer1_0_conv1 = getattr(self.layer1, "0").conv1(maxpool)
#         layer1_0_bn1 = getattr(self.layer1, "0").bn1(layer1_0_conv1);  layer1_0_conv1 = None
#         layer1_0_relu = getattr(self.layer1, "0").relu(layer1_0_bn1);  layer1_0_bn1 = None
#         layer1_0_conv2 = getattr(self.layer1, "0").conv2(layer1_0_relu);  layer1_0_relu = None
#         layer1_0_bn2 = getattr(self.layer1, "0").bn2(layer1_0_conv2);  layer1_0_conv2 = None
#         add = layer1_0_bn2 + maxpool;  layer1_0_bn2 = maxpool = None
#         layer1_0_relu_1 = getattr(self.layer1, "0").relu(add);  add = None
#         layer1_1_conv1 = getattr(self.layer1, "1").conv1(layer1_0_relu_1)
#         layer1_1_bn1 = getattr(self.layer1, "1").bn1(layer1_1_conv1);  layer1_1_conv1 = None
#         layer1_1_relu = getattr(self.layer1, "1").relu(layer1_1_bn1);  layer1_1_bn1 = None
#         layer1_1_conv2 = getattr(self.layer1, "1").conv2(layer1_1_relu);  layer1_1_relu = None
#         layer1_1_bn2 = getattr(self.layer1, "1").bn2(layer1_1_conv2);  layer1_1_conv2 = None
#         add_1 = layer1_1_bn2 + layer1_0_relu_1;  layer1_1_bn2 = layer1_0_relu_1 = None
#         layer1_1_relu_1 = getattr(self.layer1, "1").relu(add_1);  add_1 = None
#         layer2_0_conv1 = getattr(self.layer2, "0").conv1(layer1_1_relu_1)
#         layer2_0_bn1 = getattr(self.layer2, "0").bn1(layer2_0_conv1);  layer2_0_conv1 = None
#         layer2_0_relu = getattr(self.layer2, "0").relu(layer2_0_bn1);  layer2_0_bn1 = None
#         layer2_0_conv2 = getattr(self.layer2, "0").conv2(layer2_0_relu);  layer2_0_relu = None
#         layer2_0_bn2 = getattr(self.layer2, "0").bn2(layer2_0_conv2);  layer2_0_conv2 = None
#         layer2_0_downsample_0 = getattr(getattr(self.layer2, "0").downsample, "0")(layer1_1_relu_1);  layer1_1_relu_1 = None
#         layer2_0_downsample_1 = getattr(getattr(self.layer2, "0").downsample, "1")(layer2_0_downsample_0);  layer2_0_downsample_0 = None
#         add_2 = layer2_0_bn2 + layer2_0_downsample_1;  layer2_0_bn2 = layer2_0_downsample_1 = None
#         layer2_0_relu_1 = getattr(self.layer2, "0").relu(add_2);  add_2 = None
#         layer2_1_conv1 = getattr(self.layer2, "1").conv1(layer2_0_relu_1)
#         layer2_1_bn1 = getattr(self.layer2, "1").bn1(layer2_1_conv1);  layer2_1_conv1 = None
#         layer2_1_relu = getattr(self.layer2, "1").relu(layer2_1_bn1);  layer2_1_bn1 = None
#         layer2_1_conv2 = getattr(self.layer2, "1").conv2(layer2_1_relu);  layer2_1_relu = None
#         layer2_1_bn2 = getattr(self.layer2, "1").bn2(layer2_1_conv2);  layer2_1_conv2 = None
#         add_3 = layer2_1_bn2 + layer2_0_relu_1;  layer2_1_bn2 = layer2_0_relu_1 = None
#         layer2_1_relu_1 = getattr(self.layer2, "1").relu(add_3);  add_3 = None
#         layer3_0_conv1 = getattr(self.layer3, "0").conv1(layer2_1_relu_1)
#         layer3_0_bn1 = getattr(self.layer3, "0").bn1(layer3_0_conv1);  layer3_0_conv1 = None
#         layer3_0_relu = getattr(self.layer3, "0").relu(layer3_0_bn1);  layer3_0_bn1 = None
#         layer3_0_conv2 = getattr(self.layer3, "0").conv2(layer3_0_relu);  layer3_0_relu = None
#         layer3_0_bn2 = getattr(self.layer3, "0").bn2(layer3_0_conv2);  layer3_0_conv2 = None
#         layer3_0_downsample_0 = getattr(getattr(self.layer3, "0").downsample, "0")(layer2_1_relu_1);  layer2_1_relu_1 = None
#         layer3_0_downsample_1 = getattr(getattr(self.layer3, "0").downsample, "1")(layer3_0_downsample_0);  layer3_0_downsample_0 = None
#         add_4 = layer3_0_bn2 + layer3_0_downsample_1;  layer3_0_bn2 = layer3_0_downsample_1 = None
#         layer3_0_relu_1 = getattr(self.layer3, "0").relu(add_4);  add_4 = None
#         layer3_1_conv1 = getattr(self.layer3, "1").conv1(layer3_0_relu_1)
#         layer3_1_bn1 = getattr(self.layer3, "1").bn1(layer3_1_conv1);  layer3_1_conv1 = None
#         layer3_1_relu = getattr(self.layer3, "1").relu(layer3_1_bn1);  layer3_1_bn1 = None
#         layer3_1_conv2 = getattr(self.layer3, "1").conv2(layer3_1_relu);  layer3_1_relu = None
#         layer3_1_bn2 = getattr(self.layer3, "1").bn2(layer3_1_conv2);  layer3_1_conv2 = None
#         add_5 = layer3_1_bn2 + layer3_0_relu_1;  layer3_1_bn2 = layer3_0_relu_1 = None
#         layer3_1_relu_1 = getattr(self.layer3, "1").relu(add_5);  add_5 = None
#         layer4_0_conv1 = getattr(self.layer4, "0").conv1(layer3_1_relu_1)
#         layer4_0_bn1 = getattr(self.layer4, "0").bn1(layer4_0_conv1);  layer4_0_conv1 = None
#         layer4_0_relu = getattr(self.layer4, "0").relu(layer4_0_bn1);  layer4_0_bn1 = None
#         layer4_0_conv2 = getattr(self.layer4, "0").conv2(layer4_0_relu);  layer4_0_relu = None
#         layer4_0_bn2 = getattr(self.layer4, "0").bn2(layer4_0_conv2);  layer4_0_conv2 = None
#         layer4_0_downsample_0 = getattr(getattr(self.layer4, "0").downsample, "0")(layer3_1_relu_1);  layer3_1_relu_1 = None
#         layer4_0_downsample_1 = getattr(getattr(self.layer4, "0").downsample, "1")(layer4_0_downsample_0);  layer4_0_downsample_0 = None
#         add_6 = layer4_0_bn2 + layer4_0_downsample_1;  layer4_0_bn2 = layer4_0_downsample_1 = None
#         layer4_0_relu_1 = getattr(self.layer4, "0").relu(add_6);  add_6 = None
#         layer4_1_conv1 = getattr(self.layer4, "1").conv1(layer4_0_relu_1)
#         layer4_1_bn1 = getattr(self.layer4, "1").bn1(layer4_1_conv1);  layer4_1_conv1 = None
#         layer4_1_relu = getattr(self.layer4, "1").relu(layer4_1_bn1);  layer4_1_bn1 = None
#         layer4_1_conv2 = getattr(self.layer4, "1").conv2(layer4_1_relu);  layer4_1_relu = None
#         layer4_1_bn2 = getattr(self.layer4, "1").bn2(layer4_1_conv2);  layer4_1_conv2 = None
#         add_7 = layer4_1_bn2 + layer4_0_relu_1;  layer4_1_bn2 = layer4_0_relu_1 = None
#         layer4_1_relu_1 = getattr(self.layer4, "1").relu(add_7);  add_7 = None
#         avgpool = self.avgpool(layer4_1_relu_1);  layer4_1_relu_1 = None
#         flatten = torch.flatten(avgpool, 1);  avgpool = None
#         fc = self.fc(flatten);  flatten = None
#         return fc


class ReluPattern(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(x)


class BatchNormPattern(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(num_features=10)

    def forward(self, x):
        return self.bn(x)

# Define the replacement (same rules as the pattern)
def replacement(x):
    return x

relu_pattern = symbolic_trace(ReluPattern())
bn_pattern = symbolic_trace(BatchNormPattern())

# Replace `pattern` with `replacement` in `traced`
matches = subgraph_rewriter.replace_pattern(traced, bn_pattern, replacement)
matches = subgraph_rewriter.replace_pattern(traced, relu_pattern, replacement)

traced.print_readable()
# All batchnorm and relu are removed.

# class ResNet(torch.nn.Module):
#     def forward(self, x : torch.Tensor) -> torch.Tensor:

#         # No stacktrace found for following nodes
#         conv1 = self.conv1(x);  x = None
#         maxpool = self.maxpool(conv1);  conv1 = None
#         layer1_0_conv1 = getattr(self.layer1, "0").conv1(maxpool)
#         layer1_0_conv2 = getattr(self.layer1, "0").conv2(layer1_0_conv1);  layer1_0_conv1 = None
#         add = layer1_0_conv2 + maxpool;  layer1_0_conv2 = maxpool = None
#         layer1_1_conv1 = getattr(self.layer1, "1").conv1(add)
#         layer1_1_conv2 = getattr(self.layer1, "1").conv2(layer1_1_conv1);  layer1_1_conv1 = None
#         add_1 = layer1_1_conv2 + add;  layer1_1_conv2 = add = None
#         layer2_0_conv1 = getattr(self.layer2, "0").conv1(add_1)
#         layer2_0_conv2 = getattr(self.layer2, "0").conv2(layer2_0_conv1);  layer2_0_conv1 = None
#         layer2_0_downsample_0 = getattr(getattr(self.layer2, "0").downsample, "0")(add_1);  add_1 = None
#         layer2_0_downsample_1 = getattr(getattr(self.layer2, "0").downsample, "1")(layer2_0_downsample_0);  layer2_0_downsample_0 = None
#         add_2 = layer2_0_conv2 + layer2_0_downsample_1;  layer2_0_conv2 = layer2_0_downsample_1 = None
#         layer2_1_conv1 = getattr(self.layer2, "1").conv1(add_2)
#         layer2_1_conv2 = getattr(self.layer2, "1").conv2(layer2_1_conv1);  layer2_1_conv1 = None
#         add_3 = layer2_1_conv2 + add_2;  layer2_1_conv2 = add_2 = None
#         layer3_0_conv1 = getattr(self.layer3, "0").conv1(add_3)
#         layer3_0_conv2 = getattr(self.layer3, "0").conv2(layer3_0_conv1);  layer3_0_conv1 = None
#         layer3_0_downsample_0 = getattr(getattr(self.layer3, "0").downsample, "0")(add_3);  add_3 = None
#         layer3_0_downsample_1 = getattr(getattr(self.layer3, "0").downsample, "1")(layer3_0_downsample_0);  layer3_0_downsample_0 = None
#         add_4 = layer3_0_conv2 + layer3_0_downsample_1;  layer3_0_conv2 = layer3_0_downsample_1 = None
#         layer3_1_conv1 = getattr(self.layer3, "1").conv1(add_4)
#         layer3_1_conv2 = getattr(self.layer3, "1").conv2(layer3_1_conv1);  layer3_1_conv1 = None
#         add_5 = layer3_1_conv2 + add_4;  layer3_1_conv2 = add_4 = None
#         layer4_0_conv1 = getattr(self.layer4, "0").conv1(add_5)
#         layer4_0_conv2 = getattr(self.layer4, "0").conv2(layer4_0_conv1);  layer4_0_conv1 = None
#         layer4_0_downsample_0 = getattr(getattr(self.layer4, "0").downsample, "0")(add_5);  add_5 = None
#         layer4_0_downsample_1 = getattr(getattr(self.layer4, "0").downsample, "1")(layer4_0_downsample_0);  layer4_0_downsample_0 = None
#         add_6 = layer4_0_conv2 + layer4_0_downsample_1;  layer4_0_conv2 = layer4_0_downsample_1 = None
#         layer4_1_conv1 = getattr(self.layer4, "1").conv1(add_6)
#         layer4_1_conv2 = getattr(self.layer4, "1").conv2(layer4_1_conv1);  layer4_1_conv1 = None
#         add_7 = layer4_1_conv2 + add_6;  layer4_1_conv2 = add_6 = None
#         avgpool = self.avgpool(add_7);  add_7 = None
#         flatten = torch.flatten(avgpool, 1);  avgpool = None
#         fc = self.fc(flatten);  flatten = None
#         return fc