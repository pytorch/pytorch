from typing import Tuple, List, Dict

import torch
import torch.nn as nn
from torch import Tensor


class Simple(torch.nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, input):
        output = self.weight + input
        return output


def load_library():
    torch.ops.load_library("my_so.so")


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
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

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
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


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


class BatchedModel(nn.Module):
    def forward(self, input1: Tensor, input2: Tensor) -> Tuple[Tensor, Tensor]:
        return (input1 * -1, input2 * -1)

    def make_prediction(
        self, input: List[Tuple[Tensor, Tensor]]
    ) -> List[Tuple[Tensor, Tensor]]:
        return [self.forward(i[0], i[1]) for i in input]

    def make_batch(
        self, mega_batch: List[Tuple[Tensor, Tensor, int]], goals: Dict[str, str]
    ) -> List[List[Tuple[Tensor, Tensor, int]]]:
        max_bs = int(goals["max_bs"])
        return [
            mega_batch[start_idx : start_idx + max_bs]
            for start_idx in range(0, len(mega_batch), max_bs)
        ]


class MultiReturn(torch.nn.Module):
    def __init__(self):
        super(MultiReturn, self).__init__()

    def forward(self, t: Tuple[Tensor, Tensor]) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        a, b = t
        result = ((a.masked_fill_(b, 0.1), b), (torch.ones_like(a), b))
        return result


multi_return_metadata = r"""
{
 "metadata_container": {
  "forward": {
   "named_input_metadata": {
    "t": {
     "argument_type": {
      "tuple": {
       "tuple_elements": [
        {
         "tensor": 1
        },
        {
         "tensor": 6
        }
       ]
      }
     },
     "optional_argument": false,
     "metadata": {
      "dense_features": {
       "feature_desc": [
        {
          "feature_name": "test_feature_1",
          "feature_id": 1
        }
       ],
       "expected_shape": {
        "dims": [
         -1,
         1
        ],
        "unknown_rank": false
       },
       "data_type": 1,
       "feature_store_feature_type": 0
      }
     }
    }
   },
   "positional_output_metadata": [
    {
     "argument_type": {
      "tuple": {
       "tuple_elements": [
        {
         "tensor": 1
        },
        {
         "tensor": 6
        }
       ]
      }
     },
     "optional_argument": false,
     "metadata": {
      "dense_features": {
       "feature_desc": [
        {
          "feature_name": "test_feature_1",
          "feature_id": 1
        }
       ],
       "expected_shape": {
        "dims": [
         -1,
         1
        ],
        "unknown_rank": false
       },
       "data_type": 1,
       "feature_store_feature_type": 0
      }
     }
    },
    {
     "argument_type": {
      "tuple": {
       "tuple_elements": [
        {
         "tensor": 1
        },
        {
         "tensor": 6
        }
       ]
      }
     },
     "optional_argument": false,
     "metadata": {
      "dense_features": {
       "feature_desc": [
        {
          "feature_name": "test_feature_3",
          "feature_id": 3
        }
       ],
       "expected_shape": {
        "dims": [
         -1,
         1
        ],
        "unknown_rank": false
       },
       "data_type": 1,
       "feature_store_feature_type": 0
      }
     }
    }
   ]
  }
 }
}
"""
