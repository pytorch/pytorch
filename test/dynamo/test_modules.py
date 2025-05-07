# Owner(s): ["module: dynamo"]
# ruff: noqa: F841

import collections
import copy
import itertools
import os
import tempfile
import traceback
import types
import unittest
from copy import deepcopy
from functools import partial
from typing import NamedTuple
from unittest.mock import patch

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch.nn.functional as F
from torch._dynamo.debug_utils import same_two_models
from torch._dynamo.eval_frame import unsupported
from torch._dynamo.mutation_guard import GenerationTracker
from torch._dynamo.testing import expectedFailureDynamic, same
from torch._dynamo.utils import ifdynstaticdefault
from torch._dynamo.variables.torch_function import TensorWithTFOverrideVariable
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import skipIfHpu


try:
    from . import test_functions
except ImportError:
    import test_functions


_variable = 0
_variable1 = 0


def update_global():
    global _variable, _variable1
    _variable += 1
    _variable1 += 1


class BasicModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        return F.relu(self.linear1(x)) * self.scale


class FnMember(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.activation = F.relu

    def forward(self, x):
        x = self.linear1(x)
        if self.activation:
            x = self.activation(x)
        return x


class FnMemberCmp(torch.nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.activation = activation

    def forward(self, x):
        x = self.linear1(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.activation is None:
            x = torch.sigmoid(x)
        return x


class SubmoduleExample(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x * self.scale


class IsTrainingCheck(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.linear2 = torch.nn.Linear(10, 10)
        self.train(True)

    def forward(self, x):
        if self.training:
            mod = self.linear1
        else:
            mod = self.linear2
        return F.relu(mod(x))


class IsEvalCheck(IsTrainingCheck):
    def __init__(self) -> None:
        super().__init__()
        self.train(False)


class ModuleMethodCall(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.scale = torch.randn(1, 10)

    def call_and_scale(self, mod, x):
        x = mod(x)
        return x * self.scale

    def forward(self, x):
        x1 = self.call_and_scale(self.layer1, x)
        x2 = self.call_and_scale(self.layer2, x)
        return x1 + x2


class UnsupportedMethodCall(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = BasicModule()
        self.scale = torch.randn(1, 10)

    def call_and_scale(self, mod, x):
        x = mod(x)
        x = x * self.scale
        return unsupported(x, x)

    def forward(self, x):
        x1 = self.call_and_scale(self.layer1, x)
        return x + x1


class UnsupportedModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = BasicModule()
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        x = self.layer1(x) * self.scale
        return unsupported(x, x)


class UnsupportedModuleCall(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mod = UnsupportedModule()

    def forward(self, x):
        return 1 + self.mod(x * 1.5)


class ModuleWithStaticForward(torch.nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class ModuleCallModuleWithStaticForward(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mod = ModuleWithStaticForward()

    def forward(self, x):
        return self.mod(x)


class ModuleStaticMethodCall(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.scale = torch.randn(1, 10)

    @staticmethod
    def call_and_scale(scale, mod, x):
        x = mod(x)
        return x * scale

    def forward(self, x):
        x1 = self.call_and_scale(self.scale, self.layer1, x)
        x2 = self.call_and_scale(self.scale, self.layer2, x)
        return x1 + x2


class ModuleClassMethodCall(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.scale = torch.randn(1, 10)

    @classmethod
    def call_and_scale(cls, scale, mod, x):
        x = mod(x)
        return x * scale

    def forward(self, x):
        x1 = self.call_and_scale(self.scale, self.layer1, x)
        x2 = self.call_and_scale(self.scale, self.layer2, x)
        return x1 + x2


class ModuleProperty(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.randn(1, 10)

    @property
    def scale_alias(self):
        return self.scale

    def forward(self, x):
        return x * self.scale_alias


class NestedModuleList(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(3):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        torch.nn.Linear(10, 10),
                        torch.nn.ReLU(),
                    ]
                )
            )

    def forward(self, x):
        for layer, act in self.layers:
            x = act(layer(x))
        return x


class ConstLoop(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.count = 3

    def forward(self, x):
        for i in range(self.count):
            x = torch.sigmoid(self.linear1(x))
        return x


class ViaModuleCall(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)

    def forward(self, x):
        return test_functions.constant3(torch.sigmoid(self.linear1(x)), x)


class IsNoneLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)
        self.layer2 = None
        self.train(True)

    def forward(self, x):
        if self.layer1 is not None:
            x = self.layer1(x)
        if self.layer2 is not None:
            x = self.layer2(x)
        return x


class LayerList(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = [
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ModuleList(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
            ]
        )

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)

        for layer in self.layers:
            x = layer(x)

        for layer, val in zip(self.layers, (x, x, x, x)):
            x = layer(x) + val

        for layer, val in zip(self.layers, (1, 2, 3, 4)):
            x = layer(x) + val

        for idx, layer in enumerate(self.layers):
            x = layer(x) * idx

        for idx, layer in enumerate(self.layers[::-1]):
            x = layer(x) * idx

        return x


class CustomGetItemModuleList(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
            ]
        )

    def __getitem__(self, idx: int):
        return self.layers[idx]

    def __len__(self) -> int:
        return len(self.layers)

    def forward(self, x):
        for i in range(len(self)):
            x = self[i](x)

        return x


class ModuleDict(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleDict(
            {
                "0": torch.nn.Linear(10, 10),
            }
        )

    def forward(self, x):
        # TODO(future PR): handle more logic
        x = self.layers["0"](x)
        return x


class ParameterDict(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ParameterDict(
            {
                "0": torch.nn.Parameter(torch.randn(10, 10)),
            }
        )

    def forward(self, x):
        x = self.layers["0"].mm(x)
        return x


class CustomGetItemParameterDict(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ParameterDict(
            {
                "0": torch.nn.Parameter(torch.randn(10, 10)),
            }
        )

    def __getitem__(self, key: str) -> torch.nn.Module:
        return self.layers[key]

    def forward(self, x):
        x = self["0"].mm(x)
        return x


class CustomGetItemModuleDict(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleDict(
            {
                "0": torch.nn.Linear(10, 10),
            }
        )

    def __getitem__(self, key: str) -> torch.nn.Module:
        return self.layers[key]

    def forward(self, x):
        x = self["0"](x)
        return x


class TensorList(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = (
            torch.randn((1, 10)),
            torch.randn((10, 1)),
            torch.randn((1, 10)),
            torch.randn((10, 1)),
        )

    def forward(self, x):
        for layer in self.layers:
            x = x * layer
        return x


class Children(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(10, 10)
        self.l2 = torch.nn.ReLU()
        self.l3 = torch.nn.Linear(10, 10)
        self.l4 = torch.nn.ReLU()

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class NamedChildren(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(10, 10)
        self.l2 = torch.nn.ReLU()
        self.l3 = torch.nn.Linear(10, 10)
        self.l4 = torch.nn.ReLU()

    def forward(self, x):
        for _, block in self.named_children():
            x = block(x)
        return x


class IntArg(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)

    def forward(self, x, offset=1):
        x = F.relu(self.layer1(x)) + offset
        return x


class Seq(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class Cfg:
    def __init__(self) -> None:
        self.val = 0.5
        self.count = 3


class CfgModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = Cfg()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        for i in range(self.cfg.count):
            x = self.layer(x + self.cfg.val)
        return x


class StringMember(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.mode = "some_string"

    def forward(self, x):
        if self.mode == "some_string":
            return F.relu(self.linear1(x))


class _Block(torch.nn.Module):
    def forward(self, x):
        return 1.5 * torch.cat(x, 1)


class _DenseBlock(torch.nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            self.add_module(f"denselayer{i + 1:d}", _Block())

    def forward(self, init_features):
        features = [init_features]
        for layer in self.values():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNetBlocks(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = _DenseBlock()

    def forward(self, x):
        return self.layers(x)


class MaterializedModule(torch.nn.Module):
    """Once the below lazy module is initialized with its first input,
    it is transformed into this module."""

    param: Parameter

    def __init__(self) -> None:
        super().__init__()
        self.register_parameter("param", None)

    def forward(self, x):
        return x


class LazyModule(LazyModuleMixin, MaterializedModule):
    param: UninitializedParameter
    cls_to_become = MaterializedModule

    def __init__(self) -> None:
        super().__init__()
        self.param = UninitializedParameter()

    def initialize_parameters(self, x):
        # force graph break to ensure this was not inlined
        torch._dynamo.graph_break()
        self.param.materialize(x.shape)


class LazyMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.LazyLinear(10)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.LazyLinear(1)
        self.relu2 = torch.nn.ReLU()

    def forward(self, input):
        x = self.relu1(self.fc1(input))
        y = self.relu2(self.fc2(x))
        return y


class MyInput(NamedTuple):
    x: dict[str, dict[str, torch.Tensor]]
    y: torch.Tensor


class LazyLayerWithNamedTupleInput(LazyModuleMixin, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def initialize_parameters(self, input):
        with torch.no_grad():
            self._param = torch.nn.Parameter(
                torch.empty(input.x["a"][0].shape).fill_(0.5)
            )

    def forward(self, input):
        input = input.x["a"]
        x = 0
        for i in range(len(input)):
            x = x + input[i]
        return x


class LazyModuleWithNamedTupleInput(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = LazyLayerWithNamedTupleInput()

    def forward(self, input):
        return self.layer(input)


class LazyLayerWithListInput(LazyModuleMixin, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def initialize_parameters(self, input):
        with torch.no_grad():
            self._param = torch.nn.Parameter(torch.empty(input[0].shape).fill_(0.5))

    def forward(self, input):
        x = 0
        for i in range(len(input)):
            x = x + input[i]
        return x


class LazyModuleWithListInput(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = LazyLayerWithListInput()

    def forward(self, input):
        return self.layer(input[:-1])


class LazyModuleWithLazySubmodule(LazyModuleMixin, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def initialize_parameters(self, input):
        with torch.no_grad():
            self.layer = LazyLayerWithListInput()

    def forward(self, x):
        return self.layer(x)


class LazyLayerWithInputs(LazyModuleMixin, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def initialize_parameters(self, x, y):
        with torch.no_grad():
            self._param_x = torch.nn.Parameter(torch.empty(x[0].shape).fill_(0.5))
            self._param_y = torch.nn.Parameter(torch.empty(y[0].shape).fill_(0.5))

    def forward(self, x, y):
        res_x = 0
        for i in range(len(x)):
            res_x = res_x + x[i]
        res_y = 0
        for i in range(len(y)):
            res_y = res_y + y[i]
        return res_x + res_y


class LazyModuleKwArgs(LazyModuleMixin, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def initialize_parameters(self, *args, **kwargs):
        with torch.no_grad():
            self.layer = LazyLayerWithInputs()

    def forward(self, x, y):
        return self.layer(x, y=y)


class LazyParentModule(LazyModuleMixin, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def impl(self, x):
        return x.cos() + self._val


class LazyChildModuleNoClsToBecome(LazyParentModule):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return super().impl(x.sin())

    def initialize_parameters(self, input):
        self._val = torch.nn.Parameter(torch.ones(2, 2))


def requires_grad1(module: torch.nn.Module, recurse: bool = False) -> bool:
    requires_grad = any(p.requires_grad for p in module.parameters(recurse))
    return requires_grad


def requires_grad2(module: torch.nn.Module, recurse: bool = False) -> bool:
    requires_grad = any(p.requires_grad for p in module.parameters(recurse))
    return requires_grad


class ParametersModule1(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.scale = torch.nn.Parameter(torch.randn(1, 10))

    def forward(self, x):
        if not requires_grad1(self):
            return F.relu(self.linear1(x)) * self.scale
        else:
            return x + 1


class ParametersModule2(ParametersModule1):
    def forward(self, x):
        if not requires_grad2(self):
            return F.relu(self.linear1(x)) * self.scale
        else:
            return x + 1


class ParametersModule3(ParametersModule1):
    def forward(self, x):
        ones = torch.ones(10, dtype=next(self.parameters()).dtype)
        return F.relu(self.linear1(x)) * self.scale + ones


class ParametersModule4(ParametersModule1):
    def forward(self, x):
        ones = torch.ones(10, dtype=next(self.parameters(recurse=False)).dtype)
        return F.relu(self.linear1(x)) * self.scale + ones


class ParametersModule5(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.scale = torch.nn.Parameter(torch.randn(10, 10))
        self.scale_dup = self.scale

    def forward(self, x):
        counter = 0
        for param in self.parameters():
            counter += 1

        return x * self.scale * counter


class SuperModule(BasicModule):
    def forward(self, x):
        x = super().forward(x)
        return x + 10.0


class SuperModule2(BasicModule):
    def forward(self, x):
        return BasicModule.forward(self, x)


class ComplicatedSuperParent(torch.nn.Module):
    @classmethod
    def custom_add(cls, x):
        x = x + x
        return x


class SuperChildCallsClassMethod(ComplicatedSuperParent):
    @classmethod
    def child_func(cls, x):
        x = super().custom_add(x)
        return x

    def forward(self, x):
        x = self.child_func(x)
        return x


class HasAttrModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.randn(1, 10))

    def forward(self, x):
        x = F.relu(x)
        if hasattr(self, "scale"):
            x *= self.scale
        if hasattr(self, "scale2"):
            x *= self.scale2
        return x


class EnumValues(torch.nn.ModuleDict):
    def __init__(
        self,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            self.add_module(f"denselayer{i + 1:d}", _Block())

    def forward(self, init_features):
        features = [init_features]
        for idx, layer in enumerate(self.values()):
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class AccessByKeys(torch.nn.ModuleDict):
    def __init__(
        self,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            self.add_module(f"denselayer{i + 1:d}", _Block())

    def forward(self, init_features):
        features = [init_features]
        for k in self.keys():
            new_features = self[k](features)
            features.append(new_features)
        return torch.cat(features, 1)


class CallForwardDirectly(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.layer1.forward(x)
        x = self.layer2.forward(x)
        return x


class ConvCallForwardDirectly(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)

    def forward(self, x):
        return self.layer.forward(x)


class ConvTransposeCallForwardDirectly(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.ConvTranspose2d(4, 4, 4)

    def forward(self, x):
        return self.layer.forward(x)


class ConvCallSuperForwardDirectly(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            **kwargs,
        )

    def forward(self, inputs, mask=None):
        outputs = super().forward(inputs)
        return outputs


class ConvTransposeCallSuperForwardDirectly(torch.nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            **kwargs,
        )

    def forward(self, x):
        if x.numel() > 0:
            return super().forward(x)
        output_shape = [
            ((i - 1) * d - 2 * p + (di * (k - 1) + 1) + op)
            for i, p, di, k, d, op in zip(
                x.shape[-2:],
                self.padding,
                self.dilation,
                self.kernel_size,
                self.stride,
                self.output_padding,
            )
        ]
        output_shape = [x.shape[0], self.bias.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)  # noqa: F821


class ModuleNameString(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)

    def forward(self, x):
        if self.__class__.__name__ == "ABC":
            return 10
        if self.linear1.__class__.__name__ == "Linear":
            return F.relu(self.linear1(x) + 10)
        return 11


class SelfMutatingModule(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.counter = 0

    def forward(self, x):
        result = self.layer(x) + self.counter
        self.counter += 1
        return F.relu(result)


class ModuleAttributePrecedenceBase(torch.nn.Module):
    def linear(self, x, flag=None):
        if flag:
            return x * 2.0
        return x * 3.0


class ModuleAttributePrecedence(ModuleAttributePrecedenceBase):
    def __init__(self) -> None:
        super().__init__()
        self.activation = torch.nn.ReLU()
        self.linear = torch.nn.Linear(10, 10)
        self.initializer = torch.ones([10, 10])
        self.scale = 0.5

    def activation(self, x):
        return x * 1.2

    def initializer(self):
        return torch.zeros([10, 10])

    def scale(self):
        return 2.0

    def forward(self, x):
        # object attribute takes precedence unless it's a nn.Module
        return self.activation(self.linear(self.initializer + x)) * self.scale


class ModuleForwardHasGraphBreak(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.layer3 = torch.nn.Sequential(BasicModule(), BasicModule())
        self.layer4 = torch.nn.ModuleList(
            [
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
            ]
        )
        self.layer5 = torch.nn.ModuleDict(
            {
                "0": torch.nn.Linear(10, 10),
            }
        )
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        """
        This is used to test if the results of functions like `named_parameters`
        can be reconstructed correctly after graph break.

        https://github.com/pytorch/torchdynamo/issues/1931
        """
        x = self.layer1(x)
        params1 = dict(self.named_parameters())
        params2 = list(self.parameters())
        buffers1 = dict(self.named_buffers())
        buffers2 = list(self.buffers())
        modules1 = dict(self.named_modules())
        modules2 = list(self.modules())
        torch._dynamo.graph_break()
        y = modules2
        y = modules1
        y = buffers2
        y = buffers1
        y = params2
        y = params1
        x = (
            self.layer2(x)
            + y["layer3.1.linear1.weight"]
            + y["layer4.2.weight"]
            + y["layer5.0.weight"]
        )
        return x * self.scale


class ModuleGuardNameIsValid(torch.nn.ModuleDict):
    # Guard names should be valid python identifier as we use eval() to get
    # corresponding guard value. Some guard names come from source(module path)
    # where special symbols are valid. But they are not valid python identifier,
    # we should identify these pattern and rewrite them with getattr.
    def __init__(self) -> None:
        super().__init__()
        for i in range(2):
            self.add_module(f"l@yer-{i + 1:d}", BasicModule())

    def forward(self, x):
        for layer in self.values():
            x = layer(x)
        return x


class SequentialWithDuplicatedModule(torch.nn.Module):
    # Sequential module(self.layer) contains three duplicated ReLU module.
    def __init__(self) -> None:
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            self.relu,
            torch.nn.Linear(20, 20),
            self.relu,
            torch.nn.Linear(20, 10),
            self.relu,
        )

    def forward(self, x):
        return self.layer(x)


class SequentialWithDuplicatedModule2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.layer = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    ("linear1", torch.nn.Linear(10, 20)),
                    ("relu1", self.relu),
                    ("linear2", torch.nn.Linear(20, 20)),
                    ("relu2", self.relu),
                    ("linear3", torch.nn.Linear(20, 10)),
                    ("relu3", self.relu),
                ]
            )
        )

    def forward(self, x):
        return self.layer(x)


class ModuleComparison(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer0 = torch.nn.Linear(10, 10)
        self.layer1 = torch.nn.Linear(10, 10)
        self.layer2 = torch.nn.Linear(10, 10)

    @property
    def encoder_layers(self):
        return [self.layer0, self.layer1, self.layer2]

    def forward(self, x):
        for layer in self.encoder_layers:
            output = layer(x)
            if layer is None or layer == self.layer0:
                output = F.relu6(output)
            else:
                output = F.relu(output)
        return output


class ModulePatch1(torch.nn.Module):
    pass


class ModulePatch2(torch.nn.Module):
    def forward(self, x):
        return x - 1


class UnspecNonInlinableModule(torch.nn.Module):
    torchdynamo_force_dynamic = True  # forced to be a UnspecializedNNModule

    def forward(self, x):
        if x.sum() > 0:
            return x + 1
        else:
            return x - 1


class UnspecNonInlinableToplevelModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.m = UnspecNonInlinableModule()

    def forward(self, x):
        return self.m(x)


class ModuleWithIntAttr(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 4)
        self.step = 10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 1
        return self.layer(x) + self.step


class UnspecInlinableModule(torch.nn.Module):
    torchdynamo_force_dynamic = True  # forced to be a UnspecializedNNModule

    def forward(self, x):
        return torch.sin(x)


class UnspecModuleWithIntAttr(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = UnspecInlinableModule()
        self.step = 10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 1
        return self.layer(x) + self.step


def make_test(fn, expected_ops=None):
    def test_fn(self):
        return torch._dynamo.testing.standard_test(
            self, fn=fn, nargs=1, expected_ops=expected_ops
        )

    fn.eval()
    return test_fn


def temporary_tensor_subclass(torch_function=None):
    class TensorProxy(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if torch_function is not None:
                torch_function()
            return super().__torch_function__(func, types, args, kwargs)

    return TensorProxy


class NNModuleTests(torch._dynamo.test_case.TestCase):
    test_seq = make_test(Seq())
    test_basicmodule1 = make_test(BasicModule())
    test_basicmodule2 = make_test(BasicModule())
    test_submodules1 = make_test(SubmoduleExample())
    test_submodules2 = make_test(SubmoduleExample())
    test_modulemethod1 = make_test(ModuleMethodCall())
    test_modulemethod2 = make_test(ModuleMethodCall())
    test_module_call_module_with_static_forward = make_test(
        ModuleCallModuleWithStaticForward()
    )
    test_module_static_method = make_test(ModuleStaticMethodCall())
    test_fnmember = make_test(FnMember())
    test_fnmembercmp1 = make_test(FnMemberCmp(F.relu))
    test_fnmembercmp2 = make_test(FnMemberCmp(None))
    test_constloop = make_test(ConstLoop())
    test_istraining1 = make_test(IsTrainingCheck())
    test_istraining2 = make_test(IsTrainingCheck())
    test_iseval1 = make_test(IsEvalCheck())
    test_iseval2 = make_test(IsEvalCheck())
    test_viamodulecall = make_test(ViaModuleCall())
    test_isnonelayer = make_test(IsNoneLayer())
    test_layerlist = make_test(LayerList())
    test_tensorlist = make_test(TensorList())
    test_intarg = make_test(IntArg())
    test_cfgmod = make_test(CfgModule())
    test_stringmember = make_test(StringMember())
    test_modulelist = make_test(ModuleList())
    test_modulelist_nested = make_test(NestedModuleList())
    test_modulelist_custom = make_test(CustomGetItemModuleList())
    test_moduledict = make_test(ModuleDict())
    test_moduledict_custom = make_test(CustomGetItemModuleDict())
    test_parameterdict = make_test(ParameterDict())
    test_parameterdict_custom = make_test(CustomGetItemParameterDict())
    test_super1 = make_test(SuperModule())
    test_super2 = make_test(SuperModule2())
    test_super_class_method = make_test(SuperChildCallsClassMethod())
    test_children = make_test(Children())
    test_named_children = make_test(NamedChildren())
    test_densenet = make_test(DenseNetBlocks())
    test_parameters1 = make_test(ParametersModule1())
    test_parameters2 = make_test(ParametersModule2())
    test_parameters3 = make_test(ParametersModule3(), expected_ops=5)
    test_parameters4 = make_test(ParametersModule4())
    test_parameters5 = make_test(ParametersModule5())
    test_hasattr = make_test(HasAttrModule())
    test_enumvalues = make_test(EnumValues())
    test_access_by_keys = make_test(AccessByKeys())
    test_module_class_method = make_test(ModuleClassMethodCall())
    test_module_property = make_test(ModuleProperty())
    test_forward_directly = make_test(CallForwardDirectly())
    test_module_name_string = make_test(ModuleNameString())
    test_module_attribute_precedence = make_test(ModuleAttributePrecedence())
    test_module_guard_name_is_valid = make_test(ModuleGuardNameIsValid())
    test_sequential_with_duplicated_module = make_test(SequentialWithDuplicatedModule())
    test_sequential_with_duplicated_module2 = make_test(
        SequentialWithDuplicatedModule2()
    )
    test_module_comparison = make_test(ModuleComparison())

    def test_inject_module_parameters(self):
        from collections import OrderedDict

        class ZeROOrderedDict(OrderedDict):
            def __init__(self, parent_module=None, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._parent_module = parent_module

            def __getitem__(self, key):
                param = super().__getitem__(key)
                return param

        def inject_parameters(module, cls):
            for m in module.modules():
                if cls == ZeROOrderedDict:
                    new_param = cls(parent_module=m)
                else:
                    new_param = cls()

                for key, param in m._parameters.items():
                    new_param[key] = param
                m._parameters = new_param

        model = ParametersModule5()
        inject_parameters(model, ZeROOrderedDict)
        model = torch.compile(model, backend="inductor")
        x = torch.ones(10)
        # model can be compiled without error
        y = model(x)

    def test_module_forward_has_graph_break(self):
        m = ModuleForwardHasGraphBreak()
        x = torch.rand([10, 10])
        ref = m(x)
        opt_m = torch.compile(m, backend="eager")
        res = opt_m(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_unsupportedmethod(self):
        m = UnsupportedMethodCall()
        i = torch.randn(10)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_m = torch.compile(m, backend=cnt)
        r = opt_m(i)
        self.assertTrue(torch._dynamo.testing.same(r, m(i)))
        self.assertEqual(cnt.op_count, 5)

    def test_unsupportedmodule(self):
        m = UnsupportedModuleCall()
        i = torch.randn(10)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_m = torch.compile(m, backend=cnt)
        r = opt_m(i)
        self.assertTrue(torch._dynamo.testing.same(r, m(i)))
        self.assertEqual(cnt.op_count, 6)

    def test_self_mutating1(self):
        m1 = torch.nn.Linear(10, 10)
        m2 = SelfMutatingModule(m1)
        m3 = SelfMutatingModule(m1)
        m4 = SelfMutatingModule(m1)
        i = torch.randn(10)
        out2 = [m2(i), m2(i), m2(i)]
        cnt = torch._dynamo.testing.CompileCounter()
        opt_m3 = torch._dynamo.optimize_assert(cnt)(m3)
        opt_m4 = torch._dynamo.optimize_assert(cnt)(m4)
        out3 = [opt_m3(i), opt_m3(i), opt_m3(i)]
        out4 = [opt_m4(i), opt_m4(i), opt_m4(i)]
        self.assertTrue(torch._dynamo.testing.same(out2, out3))
        self.assertTrue(torch._dynamo.testing.same(out2, out4))
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """2""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")

    def test_nn_module_setattr(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.var = 0

        @torch.compile(backend="eager", dynamic=False)
        def f(x, m):
            return x + m.var

        inp = torch.ones(3)
        m = Mod()

        self.assertEqual(f(inp, m), inp)
        # In 3.13.0, setattr will not fire a __dict__'s watchers,
        # so guards may not be invalidated.
        m.var = 1
        # should trigger a recompile
        self.assertEqual(f(inp, m), inp + 1)

    @patch.object(torch._dynamo.config, "raise_on_ctx_manager_usage", False)
    def test_generation_tag(self):
        cnt = torch._dynamo.testing.CompileCounter()

        # guarantee that we have installed
        # the generation tagging function
        with torch._dynamo.optimize_assert(cnt):
            pass

        m1 = torch.nn.Linear(10, 10)
        prev_generation = GenerationTracker.get_generation_value(m1)
        cur_generation = prev_generation + 1

        with torch._dynamo.optimize_assert(cnt):
            m2 = torch.nn.Linear(10, 10)

        self.assertEqual(GenerationTracker.get_generation_value(m1), prev_generation)
        self.assertEqual(GenerationTracker.get_generation_value(m2), cur_generation)
        # check that newly constructed instances
        # also have the same generation (even if copied from an old instance)
        m3 = deepcopy(m1)
        self.assertEqual(GenerationTracker.get_generation_value(m3), cur_generation)

    def test_simple_torch_function(self):
        def foo(x):
            # function call, twice to test wrapping
            x = F.sigmoid(x)
            x = F.sigmoid(x)
            # method call, twice to test wrapping
            x = x.sigmoid()
            x = x.sigmoid()
            return x

        TensorProxy = temporary_tensor_subclass()
        x = torch.randn(1).as_subclass(TensorProxy)
        cnt = torch._dynamo.testing.CompileCounter()
        out1 = foo(x)
        opt_foo = torch.compile(foo, backend=cnt, fullgraph=True)
        out2 = opt_foo(x)

        self.assertEqual(cnt.op_count, 4)
        self.assertTrue(torch._dynamo.testing.same(out1, out2))

    def test_torch_function_with_closure(self):
        def run():
            def foo(x):
                # function call, twice to test wrapping
                x = F.sigmoid(x)
                x = F.sigmoid(x)
                # method call, twice to test wrapping
                x = x.sigmoid()
                x = x.sigmoid()
                return x

            counter = 0

            def function():
                nonlocal counter
                # for now, only support reads from closure cells
                # TODO(future PR): support writes as well
                counter + 1

            TensorProxy = temporary_tensor_subclass(function)
            x = torch.randn(1).as_subclass(TensorProxy)
            x = torch.randn(1)
            cnt = torch._dynamo.testing.CompileCounter()
            out1 = foo(x)
            opt_foo = torch.compile(foo, backend=cnt, fullgraph=True)
            out2 = opt_foo(x)

            self.assertEqual(cnt.op_count, 4)
            self.assertTrue(torch._dynamo.testing.same(out1, out2))

        run()

    def test_torch_mangled_class_name(self):
        original = TensorWithTFOverrideVariable.global_mangled_class_name
        results = []

        def instrumented(self, tx):
            result = original(self, tx)
            results.append(result)
            return result

        TensorWithTFOverrideVariable.global_mangled_class_name = instrumented

        def one_break(x):
            x = F.sigmoid(x)
            print()  # force break
            x = x.sigmoid()
            return x

        try:
            TensorProxy = temporary_tensor_subclass()
            x = torch.randn(1).as_subclass(TensorProxy)
            x1 = one_break(x)

            cnt = torch._dynamo.testing.CompileCounter()
            opt_one_break = torch.compile(one_break, backend=cnt)
            x2 = opt_one_break(x)

            self.assertTrue(torch._dynamo.testing.same(x1, x2))
            self.assertEqual(cnt.frame_count, 2)
            self.assertEqual(cnt.op_count, 2)

            compile_ids = set()
            for r in results:
                # A mangled classname looks like __subclass_TensorProxy_94524181138240_c0
                # where the last segment contains the compile_id.
                prefix = "__subclass_TensorProxy_"
                before, sep, after = r.partition(prefix)
                self.assertEqual(before, "")
                self.assertEqual(sep, prefix)

                class_type_id, compile_id = after.split("_")
                self.assertTrue(class_type_id.isnumeric())
                self.assertTrue(compile_id.startswith("c"))

                cid = compile_id[1:]
                self.assertTrue(cid.isnumeric())
                compile_ids.add(cid)

            self.assertEqual(len(compile_ids), 3)

        finally:
            TensorWithTFOverrideVariable.global_mangled_class_name = original

    def test_nn_moduledict_contains(self):
        class M(torch.nn.Module):
            def __init__(self, module_dict):
                super().__init__()
                self.module_dict = module_dict

            def forward(self, x):
                if "foo" in self.module_dict:
                    x = torch.mul(x, 1.0)
                x = torch.add(x, 1.0)
                return x

        module_dict = torch.nn.ModuleDict({"foo": torch.nn.Conv2d(1, 1, 1)})
        m = M(module_dict)
        data = torch.randn(1)
        out1 = m(data)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_m = torch._dynamo.optimize(cnt, nopython=True)(m)
        out2 = opt_m(data)
        self.assertEqual(cnt.op_count, 2)
        self.assertTrue(torch._dynamo.testing.same(out1, out2))

        module_dict = torch.nn.ModuleDict({"bar": torch.nn.Conv2d(1, 1, 1)})
        m = M(module_dict)
        data = torch.randn(1)
        out1 = m(data)
        cnt = torch._dynamo.testing.CompileCounter()
        torch._dynamo.reset()
        opt_m = torch._dynamo.optimize(cnt, nopython=True)(m)
        out2 = opt_m(data)

        self.assertEqual(cnt.op_count, 1)
        self.assertTrue(torch._dynamo.testing.same(out1, out2))

    # RuntimeError: SymIntArrayRef expected to contain only concrete integers
    @expectedFailureDynamic
    def test_lazy_module1(self):
        input_shape = (16, 3, 6, 7, 8)

        cnt = torch._dynamo.testing.CompileCounter()
        module = LazyModule()

        def test_static_module():
            input = torch.ones(*input_shape)
            module(input)

        # test no graph break
        opt_test_static_module = torch.compile(
            test_static_module, backend=cnt, fullgraph=True
        )
        opt_test_static_module()

        self.assertTrue(
            isinstance(module, MaterializedModule),
            "Module should be transformed to an instance of MaterializedModule.",
        )
        self.assertEqual(module.param.shape, input_shape)

        # test when mapped to UnspecializedNNModule
        module = LazyModule()

        def test_unspecialized():
            nonlocal module
            module = LazyModule()
            input = torch.ones(*input_shape)
            module(input)

        opt_test_unspecialized = torch.compile(test_unspecialized, backend=cnt)
        opt_test_unspecialized()

        self.assertTrue(
            isinstance(module, MaterializedModule),
            "Module should be transformed to an instance of MaterializedModule.",
        )
        self.assertEqual(module.param.shape, input_shape)

        # test with a static module in torch.*
        module = torch.nn.modules.LazyBatchNorm3d(
            affine=False, track_running_stats=False
        )

        cnt = torch._dynamo.testing.CompileCounter()

        torch._dynamo.reset()

        def test_torch_static():
            input = torch.ones(*input_shape)
            return module(input)  # fully materialized

        # test no graph break
        opt_test_torch_static = torch.compile(
            test_torch_static, backend=cnt, fullgraph=True
        )
        opt_test_torch_static()
        out = opt_test_torch_static()

        self.assertTrue(same(out, module(torch.ones(*input_shape))))

        self.assertTrue(
            isinstance(module, torch.nn.modules.batchnorm.BatchNorm3d),
            "Module should be transformed to an instance of BatchNorm3d.",
        )
        self.assertEqual(cnt.frame_count, 1, "No guards should have triggered.")

    # RuntimeError: SymIntArrayRef expected to contain only concrete integers
    @expectedFailureDynamic
    def test_lazy_module2(self):
        # Test FX graph 'call_module' works well if argument is lazy module
        m = LazyMLP()
        x = torch.rand([10, 10])
        opt_m = torch.compile(m, backend="eager", fullgraph=True)
        # We should run compile mode firstly, otherwise the module
        # would be initialized when running eager mode.
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))

    # RuntimeError: SymIntArrayRef expected to contain only concrete integers
    @expectedFailureDynamic
    def test_lazy_module4(self):
        m = LazyMLP()
        x = torch.rand([10, 10])
        cnt = torch._dynamo.testing.CompileCounter()
        opt_m = torch.compile(m, backend=cnt, fullgraph=True)
        # first iteration
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))
        # input shape changed and second iteration
        x = torch.rand([20, 20])
        try:
            opt_m(x)
        except RuntimeError:
            self.assertIn("must have same reduction dim", traceback.format_exc())

    # RuntimeError: SymIntArrayRef expected to contain only concrete integers
    @expectedFailureDynamic
    def test_lazy_module5(self):
        # Test lazy module works well with list/tuple input
        m = LazyModuleWithListInput()
        x = [torch.rand([5, 5])] * 3 + [None]
        opt_m = torch.compile(m, backend="eager", fullgraph=True)
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))

    # RuntimeError: SymIntArrayRef expected to contain only concrete integers
    @expectedFailureDynamic
    def test_lazy_module6(self):
        # Test new lazy submodule in lazy module's initialize_parameters
        m = LazyModuleWithLazySubmodule()
        x = [torch.rand([5, 5])] * 3
        opt_m = torch.compile(m, backend="eager", fullgraph=True)
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))

    # RuntimeError: SymIntArrayRef expected to contain only concrete integers
    @expectedFailureDynamic
    def test_lazy_module7(self):
        # Test lazy module works well with namedtuple/dict input
        m = LazyModuleWithNamedTupleInput()
        x = MyInput(
            x={"a": [torch.rand([5, 5])] * 3, "b": torch.rand([5, 5])},
            y=torch.rand([5, 5]),
        )
        opt_m = torch.compile(backend="eager", fullgraph=True)(m)
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_lazy_module_no_cls_to_become(self):
        # make sure super() works in the case where cls_to_become is None
        m = LazyChildModuleNoClsToBecome()
        x = torch.rand(2, 2)
        opt_m = torch.compile(m, backend="eager", fullgraph=True)
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_lazy_module_kwargs(self):
        m = LazyModuleKwArgs()
        x = [torch.rand([5, 5])] * 3
        y = [torch.rand([5, 5])] * 2
        opt_m = torch.compile(backend="eager", fullgraph=True)(m)
        exp_res = m(x, y)
        self.assertTrue(torch.allclose(exp_res, opt_m(x, y)))

    # RuntimeError: SymIntArrayRef expected to contain only concrete integers
    @expectedFailureDynamic
    def test_lazy_module_speculation_log_divergence(self):
        class ModWithOneLazyLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = torch.nn.LazyLinear(8)

            def forward(self, x):
                return self.layer(x)

        # This allows us to restart tracing without clearing speculation log
        def id_and_fail_inlining(x):
            torch._dynamo.graph_break()
            return x

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def test(mod, x):
            res = mod(x)
            # Speculation log must not diverge in the 2nd round of tracing,
            # after we've initialized the `LazyLinear` into a `Linear` in the
            # 1st round.
            res2 = id_and_fail_inlining(res)
            return res

        mod = ModWithOneLazyLinear()
        x = torch.ones(10, 3)

        # Make sure we don't get recompilation across multiple runs
        actual_res = test(mod, x)
        expect_res = mod(x)
        self.assertTrue(torch.allclose(expect_res, actual_res))
        actual_res = test(mod, x)
        expect_res = mod(x)
        self.assertTrue(torch.allclose(expect_res, actual_res))
        self.assertEqual(cnt.frame_count, 1)

    def test_call_fn_with_non_const_inputs_safe(self):
        class ModuleSpecialFwd(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=20, kernel_size=(5, 5)
                )

            def _conv_forward(self, x):
                return self.conv._conv_forward(x, self.conv.weight, self.conv.bias)

            def forward(self, x):
                return self._conv_forward(x)

        mod = ModuleSpecialFwd()
        rx = torch.randn([3, 10, 10])
        real = mod(rx)
        graph, _ = torch._dynamo.export(mod)(rx)
        self.assertTrue(torch._dynamo.testing.same(real, graph(rx)))

    def test_conv_call_forward_directly(self):
        m = ConvCallForwardDirectly()
        x = torch.rand([4, 3, 9, 9])
        ref = m(x)
        opt_m = torch.compile(backend="eager", fullgraph=True)(m)
        res = opt_m(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_conv_transpose_call_forward_directly(self):
        m = ConvTransposeCallForwardDirectly()
        x = torch.rand([4, 4, 4, 4])
        ref = m(x)
        opt_m = torch.compile(backend="eager", fullgraph=True)(m)
        res = opt_m(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_conv_call_super_forward_directly(self):
        x = torch.randn(4, 4)
        m = ConvCallSuperForwardDirectly(4, 4, 4)
        ref = m(x)
        opt_m = torch.compile(backend="eager", fullgraph=True)(m)
        res = opt_m(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_conv_transpose_call_super_forward_directly(self):
        x = torch.randn(4, 4, 4)
        m = ConvTransposeCallSuperForwardDirectly(4, 4, 4)
        ref = m(x)
        opt_m = torch.compile(backend="eager", fullgraph=True)(m)
        res = opt_m(x)
        self.assertTrue(torch.allclose(ref, res))

    @torch._dynamo.config.patch("allow_unspec_int_on_nn_module", True)
    def test_nn_module_unspec_int_attr(self):
        for module_class in [ModuleWithIntAttr, UnspecModuleWithIntAttr]:
            mod = module_class()
            cnt = torch._dynamo.testing.CompileCounter()
            opt_mod = torch.compile(backend=cnt)(copy.deepcopy(mod))
            x = torch.rand(3, 4)

            # Compiling `self.step` as static
            ref1 = mod(x)
            res1 = opt_mod(x)
            self.assertTrue(torch.allclose(ref1, res1))
            self.assertEqual(cnt.frame_count, 1)

            mod.step += 1
            opt_mod.step += 1

            # Second time: compiling `self.step` as dynamic
            ref2 = mod(x)
            res2 = opt_mod(x)
            self.assertTrue(torch.allclose(ref2, res2))
            self.assertEqual(cnt.frame_count, ifdynstaticdefault(2, 1))

            mod.step += 1
            opt_mod.step += 1

            # Third time: no re-compilation!
            ref3 = mod(x)
            res3 = opt_mod(x)
            self.assertTrue(torch.allclose(ref3, res3))
            self.assertEqual(cnt.frame_count, ifdynstaticdefault(2, 1))


class NNModuleTestsDevice(torch._dynamo.test_case.TestCase):
    @expectedFailureDynamic
    @skipIfHpu
    def test_lazy_module3(self, device):
        m = LazyMLP()
        x = torch.rand([10, 10])
        cnt = torch._dynamo.testing.CompileCounter()
        opt_m = torch._dynamo.optimize(cnt, nopython=True)(m)
        # first iteration
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))
        # move to device and second iteration
        m = m.to(device)
        x = x.to(device)
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))
        self.assertEqual(cnt.frame_count, 2)


class MockModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(10, 10)
        self.buf0 = torch.nn.Buffer(torch.randn(10, 10))

    def forward(self, x):
        return self.relu(self.linear(x) + self.buf0)


class OptimizedModuleTest(torch._dynamo.test_case.TestCase):
    def test_nn_module(self):
        mod = MockModule()
        cnt = torch._dynamo.testing.CompileCounter()
        opt_mod = torch.compile(mod, backend=cnt)
        self.assertIsInstance(opt_mod, torch._dynamo.OptimizedModule)

        x = torch.randn(10, 10)
        self.assertTrue(torch._dynamo.testing.same(mod(x), opt_mod(x)))
        self.assertEqual(cnt.frame_count, 1)

    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_attr_precedence(self):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = 3

            def forward(self, x, c=4):
                return x * c

            def linear(self, x):
                return x

            def b(self, x):
                raise RuntimeError("Should not be called")

        class MyMod(Mod):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(11, 11)
                self.a = 2
                self.b = 2
                self.scale = 1

            def scale(self, x):
                # Should not be called because it is shadowed by the instance
                # attribute
                raise RuntimeError("Should not be called")

            def forward(self, x, c=None):
                return self.linear(x) * self.a * self.b * self.scale

        mod = MyMod()
        x = torch.ones(3, 3)
        ref = mod(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_mod = torch.compile(mod, backend=cnts)
        opt_mod(torch.ones(3, 3))
        res = opt_mod(torch.ones(3, 3))

        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(ref, res)

    def test_to(self):
        mod = MockModule()
        cnt = torch._dynamo.testing.CompileCounter()
        opt_mod = torch.compile(mod, backend=cnt)
        x = torch.randn(10, 10)
        self.assertTrue(torch._dynamo.testing.same(mod(x), opt_mod(x)))
        self.assertEqual(cnt.frame_count, 1)

        # Ensure that there is no recompilation
        opt_mod(x)
        self.assertEqual(cnt.frame_count, 1)

        opt_mod = opt_mod.to(device="cpu").to(dtype=torch.float64)
        self.assertIsInstance(opt_mod, torch._dynamo.OptimizedModule)
        x = torch.randn(10, 10).to(dtype=torch.float64)
        opt_mod(x)
        # Ensure that there is a recompilation
        self.assertEqual(cnt.frame_count, 2)

        # Ensure that there is no recompilation
        opt_mod(x)
        self.assertEqual(cnt.frame_count, 2)

        torch._dynamo.reset()
        opt_mod(x)
        self.assertEqual(cnt.frame_count, 3)

    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_param_order(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param1 = torch.nn.Parameter(torch.ones([1]))
                self.param2 = torch.nn.Parameter(torch.ones([2]))

            def forward(self, x):
                return x

        mod = MyModule()
        coeffs = [2, 3]

        def fn(x):
            for idx, p in enumerate(mod.parameters()):
                x += p.sum() * coeffs[idx]

            for idx, p in enumerate(mod.named_parameters()):
                x += p[1].sum() * coeffs[idx]

            return x

        ref = fn(torch.ones(1))
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(torch.ones(1))

        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)

        mod._parameters["param1"] = mod._parameters.pop("param1")
        ref = fn(torch.ones(1))
        res = opt_fn(torch.ones(1))

        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)

    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_buffer_order(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.b1 = torch.nn.Buffer(torch.ones([1]))
                self.b2 = torch.nn.Buffer(torch.ones([2]))

            def forward(self, x):
                return x

        mod = MyModule()
        coeffs = [2, 3]

        def fn(x):
            for idx, p in enumerate(mod.buffers()):
                x += p.sum() * coeffs[idx]

            for idx, p in enumerate(mod.named_buffers()):
                x += p[1].sum() * coeffs[idx]

            return x

        ref = fn(torch.ones(1))
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(torch.ones(1))

        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)

        mod._buffers["b1"] = mod._buffers.pop("b1")
        ref = fn(torch.ones(1))
        res = opt_fn(torch.ones(1))

        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)

    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_module_order(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = torch.nn.Linear(3, 3)
                self.linear2 = torch.nn.Linear(10, 10)

            def forward(self, x):
                return x

        mod = MyModule()
        coeffs = [2, 3, 4]

        coeffs_for_mod = {mod: 10, mod.linear1: 20, mod.linear2: 30}

        # Check order of _modules
        def fn(x):
            for idx, p in enumerate(mod.modules()):
                # Something silly to force depedency on the order
                x += coeffs_for_mod[p] * coeffs[idx]
            for idx, p in enumerate(mod.named_modules()):
                x += coeffs_for_mod[p[1]] * coeffs[idx]
            for idx, p in enumerate(mod.children()):
                x += coeffs_for_mod[p] * coeffs[idx]
            for idx, p in enumerate(mod.named_children()):
                x += coeffs_for_mod[p[1]] * coeffs[idx]
            return x

        ref = fn(torch.ones(1))
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(torch.ones(1))

        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)

        mod._modules["linear1"] = mod._modules.pop("linear1")
        ref = fn(torch.ones(1))
        res = opt_fn(torch.ones(1))

        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)

    def test_attr(self):
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.buf0 = torch.nn.Buffer(torch.randn(10, 10))

            def forward(self, x):
                return self.r(torch.sin(x)) + self.buf0

        mod = MockModule()
        opt_mod = torch.compile(mod, backend="eager")

        # Check parameters and buffers
        for p1, p2 in zip(mod.parameters(), opt_mod.parameters()):
            self.assertTrue(id(p1) == id(p2))
        for b1, b2 in zip(mod.buffers(), opt_mod.buffers()):
            self.assertTrue(id(b1) == id(b2))

        def get_parameter_dtype(mod: torch.nn.Module):
            parameters_and_buffers = itertools.chain(mod.parameters(), mod.buffers())
            return next(parameters_and_buffers).dtype

        opt_mod = torch.compile(get_parameter_dtype, backend="eager")
        out_dtype = opt_mod(mod)
        self.assertEqual(out_dtype, torch.float32)

    def test_dir(self):
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.buf0 = torch.nn.Buffer(torch.nn.Buffer(torch.randn(10, 10)))
                self.register_parameter(
                    name="param0", param=torch.nn.Parameter(torch.randn(10, 10))
                )

            def forward(self, x):
                return self.r(torch.sin(x)) + self.buf0

        mod = MockModule()
        mod_keys = dir(mod)
        opt_mod = torch.compile(mod, backend="eager")
        opt_mod_keys = dir(opt_mod)

        # Check user-defined attributes, parameters and buffers
        self.assertIn("linear", opt_mod_keys)
        self.assertIn("buf0", opt_mod_keys)
        self.assertIn("param0", opt_mod_keys)

        # Check all attributes, parameters and buffers
        self.assertTrue(len(set(mod_keys).difference(opt_mod_keys)) == 0)

    def test_no_recompile_on_nn_guarded_modules(self):
        size = (10, 10)
        recompile_limit = 1
        num_submodules = 4
        cnts = torch._dynamo.testing.CompileCounterWithBackend("eager")

        class SubModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(*size)

            def forward(self, x):
                a = torch.sin(torch.cos(x))
                return self.linear(a)

        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mods = [SubModule() for _ in range(num_submodules)]
                self.mods = [torch.compile(mod, backend=cnts) for mod in self.mods]

            def forward(self, x):
                for mod in self.mods:
                    x = mod(x)
                return x

        mod = MockModule()
        # Each submod is compiled separately and has a different nn module
        # guard. Ensure that recompilation logic is handle correctly.
        with unittest.mock.patch(
            "torch._dynamo.config.error_on_recompile", True
        ), unittest.mock.patch(
            "torch._dynamo.config.recompile_limit",
            recompile_limit,
        ):
            x = torch.randn(*size, requires_grad=True)
            mod(x)
            if torch._dynamo.config.inline_inbuilt_nn_modules:
                self.assertEqual(cnts.frame_count, 1)
            else:
                self.assertEqual(cnts.frame_count, num_submodules)

    @patch.object(torch._dynamo.config, "accumulated_recompile_limit", 2)
    @patch.object(torch._dynamo.config, "inline_inbuilt_nn_modules", False)
    def test_recompile_limit_on_freed_module(self):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.lin(x)

        def fn(x, mod):
            return mod(x)

        cnts = torch._dynamo.testing.CompileCounterWithBackend("eager")
        opt_mod = torch.compile(fn, backend=cnts)
        for i in range(8):
            mod = Mod()
            opt_mod(torch.randn(5, 5), mod)

        # fn compiles twice
        self.assertEqual(cnts.frame_count, 2)

    @patch.object(torch._dynamo.config, "inline_inbuilt_nn_modules", True)
    def test_inline_inbuilt_nn_modules(self):
        size = (10, 10)
        recompile_limit = 1
        num_submodules = 4
        cnts = torch._dynamo.testing.CompileCounterWithBackend("eager")

        class SubModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(*size)

            def forward(self, x):
                a = torch.sin(torch.cos(x))
                return self.linear(a)

        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mods = [SubModule() for _ in range(num_submodules)]
                self.mods = [torch.compile(mod, backend=cnts) for mod in self.mods]

            def forward(self, x):
                for mod in self.mods:
                    x = mod(x)
                return x

        mod = MockModule()
        # Each submod is compiled separately and has a different nn module
        # guard. Ensure that recompilation logic is handle correctly.
        with unittest.mock.patch(
            "torch._dynamo.config.error_on_recompile", True
        ), unittest.mock.patch(
            "torch._dynamo.config.recompile_limit",
            recompile_limit,
        ):
            x = torch.randn(*size, requires_grad=True)
            mod(x)
            self.assertEqual(cnts.frame_count, 1)

    def test_recompile_limit_on_guarded_nn_modules(self):
        recompile_limit = 2
        num_submodules = 4
        cnts = torch._dynamo.testing.CompileCounterWithBackend("eager")

        class SubModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                a = torch.sin(torch.cos(x))
                return self.relu(a)

        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mods = [SubModule() for _ in range(num_submodules)]
                self.mods = [torch.compile(mod, backend=cnts) for mod in self.mods]

            def forward(self, x):
                for mod in self.mods:
                    x = mod(x)
                return x

        mod = MockModule()
        # For the third iteration, we would reach the cache size limit, and
        # therefore the total number of expected frame count is 2 *
        # num_submodules.
        with unittest.mock.patch(
            "torch._dynamo.config.recompile_limit",
            recompile_limit,
        ):
            for size in [
                (4,),
                (4, 4),
                (4, 4, 4),
            ]:
                x = torch.randn(size)
                mod(x)
        if torch._dynamo.config.inline_inbuilt_nn_modules:
            self.assertEqual(cnts.frame_count, 2)
        else:
            self.assertEqual(cnts.frame_count, 2 * num_submodules)

    def test_recursion(self):
        mod = MockModule()
        cnt = torch._dynamo.testing.CompileCounter()
        opt_mod = torch.compile(mod, backend=cnt)

        for _ in range(5):
            opt_mod = torch.compile(opt_mod, backend=cnt)
        opt_mod(torch.randn(10, 10))
        self.assertEqual(cnt.frame_count, 1)

    def test_composition(self):
        class InnerModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(torch.sin(x))

        opt_inner_mod = InnerModule()

        class OuterModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mod = opt_inner_mod

            def forward(self, x):
                return self.mod(torch.cos(x))

        outer_mod = OuterModule()
        cnt = torch._dynamo.testing.CompileCounter()
        opt_outer_mod = torch.compile(outer_mod, backend=cnt)

        x = torch.randn(4)
        self.assertIsInstance(opt_outer_mod, torch._dynamo.OptimizedModule)
        self.assertTrue(torch._dynamo.testing.same(outer_mod(x), opt_outer_mod(x)))
        self.assertEqual(cnt.frame_count, 1)

    def test_composition_with_opt_mod(self):
        class InnerModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(torch.sin(x))

        inner_mod = InnerModule()
        cnt = torch._dynamo.testing.CompileCounter()
        opt_inner_mod = torch.compile(inner_mod, backend=cnt)

        class OuterModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mod = opt_inner_mod

            def forward(self, x):
                return self.mod(torch.cos(x))

        outer_mod = OuterModule()
        opt_outer_mod = torch.compile(outer_mod, backend=cnt)

        x = torch.randn(4)
        self.assertIsInstance(opt_outer_mod, torch._dynamo.OptimizedModule)
        self.assertTrue(torch._dynamo.testing.same(outer_mod(x), opt_outer_mod(x)))
        # There will be a graph break for the inner mod being OptimizedModule
        self.assertEqual(cnt.frame_count, 2)

    def test_module_patch(self):
        mod = ModulePatch1()
        mod.forward = types.MethodType(ModulePatch2.forward, mod)

        def fn(x):
            return mod(x)

        self.assertTrue(
            torch.allclose(
                torch.compile(fn, backend="eager", fullgraph=True)(torch.ones(10)),
                torch.zeros(1),
            )
        )

    @patch.object(torch._dynamo.config, "skip_nnmodule_hook_guards", False)
    def test_hooks_outer(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return 2 * x + 1

        m = TestModule()

        def forward_hook(
            module: torch.nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor
        ) -> torch.Tensor:
            return 2 * output + 1

        handle = m.register_forward_hook(forward_hook)
        inp = torch.tensor(1.0, requires_grad=True)

        failure_reason = None

        def guard_fail_fn(failure):
            nonlocal failure_reason
            failure_reason = failure[0]

        compiled_m = torch._dynamo.optimize(
            guard_fail_fn=guard_fail_fn, backend="eager"
        )(m)

        self.assertEqual(compiled_m(inp), m(inp))
        self.assertEqual(compiled_m(inp).item(), 7)
        self.assertTrue(failure_reason is None)

        # what if we remove our hook? we should recompile?
        handle.remove()
        self.assertEqual(compiled_m(inp), m(inp))
        self.assertEqual(compiled_m(inp).item(), 3)
        # self.assertTrue(failure_reason == "hook")

        """
        Summary:
          - removing a hook doesn't fail a guard, because we weren't compiling the hook
            (at least into the same graph) as forward in the first place! We do correctly
            omit calling the removed hook, but since this hook is a post forward hook,
            the 'RETURN' from forward is breaking the graph.

            Why is 'forward' the entrypoint to an InstructionTranslator, after I changed
            the eval_frame entrypoint to Module.__call__?
        """

    @patch.object(torch._dynamo.config, "skip_nnmodule_hook_guards", False)
    def test_hooks_inner(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return 2 * x + 1

        m = TestModule()

        def forward_hook(
            module: torch.nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor
        ) -> torch.Tensor:
            return 2 * output + 1

        handle = m.register_forward_hook(forward_hook)

        def outer_func(tensor):
            x = tensor * 2 + 1
            y = m(x)
            return y

        inp = torch.tensor(1.0, requires_grad=True)

        failure_reason = None

        def guard_fail_fn(failure):
            nonlocal failure_reason
            failure_reason = failure[0]

        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        compiled_func = torch._dynamo.optimize(
            guard_fail_fn=guard_fail_fn,
            backend=cc,
        )(outer_func)

        self.assertEqual(compiled_func(inp), outer_func(inp))
        self.assertEqual(compiled_func(inp).item(), 15)

        # We are compiling 1 big graph for all 3 functions including the hook.
        self.assertEqual(cc.frame_count, 1)
        self.assertEqual(cc.op_count, 6)

        # If we remove the hook, we should recompile
        handle.remove()
        self.assertEqual(compiled_func(inp), outer_func(inp))
        self.assertEqual(compiled_func(inp).item(), 7)
        self.assertTrue("forward_hooks" in failure_reason)
        self.assertEqual(cc.frame_count, 1 + 1)
        self.assertEqual(cc.op_count, 6 + 4)

        # what if instead of removing, we alter our hook?
        torch._dynamo.reset()
        m = TestModule()
        handle = m.register_forward_hook(forward_hook)
        failure_reason = None
        self.assertEqual(compiled_func(inp), outer_func(inp))
        self.assertEqual(compiled_func(inp).item(), 15)

        def new_forward_hook(
            module: torch.nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor
        ) -> torch.Tensor:
            return 2 * output + 2

        m._forward_hooks[handle.id] = new_forward_hook
        self.assertEqual(compiled_func(inp), outer_func(inp))
        self.assertEqual(compiled_func(inp).item(), 16)

    @patch.object(torch._dynamo.config, "guard_nn_modules", False)
    @patch.object(torch._dynamo.config, "skip_nnmodule_hook_guards", True)
    @patch.object(torch._dynamo.config, "inline_inbuilt_nn_modules", False)
    def test_hooks_skip_guards(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return 2 * x + 1

        m = TestModule()

        def forward_hook(
            module: torch.nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor
        ) -> torch.Tensor:
            return 2 * output + 1

        handle = m.register_forward_hook(forward_hook)

        def outer_func(tensor):
            x = tensor * 2 + 1
            y = m(x)
            return y

        inp = torch.tensor(1.0, requires_grad=True)

        failure_reason = None

        def guard_fail_fn(failure):
            nonlocal failure_reason
            failure_reason = failure[0]

        cc = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        compiled_func = torch._dynamo.optimize(
            guard_fail_fn=guard_fail_fn,
            backend=cc,
        )(outer_func)

        m = TestModule()
        handle = m.register_forward_hook(forward_hook)
        failure_reason = None
        self.assertEqual(compiled_func(inp), outer_func(inp))
        self.assertEqual(compiled_func(inp).item(), 15)
        self.assertEqual(cc.frame_count, 1)
        self.assertEqual(cc.op_count, 6)

        # if we remove the hook, dynamo shouldn't notice
        handle.remove()
        self.assertNotEqual(compiled_func(inp), outer_func(inp))
        self.assertEqual(compiled_func(inp).item(), 15)
        self.assertEqual(cc.frame_count, 1)

    def _forward_hook_test_helper(self, model):
        forward_handles = {}
        compiled_activations = {}
        eager_activations = {}
        activations = None

        def save_activations(name, mod, inp, out):
            activations[name] = inp

        for name, module in model.named_modules():
            forward_handles[name] = module.register_forward_hook(
                partial(save_activations, name)
            )

        compiled_model = torch.compile(model, backend="aot_eager")

        activations = compiled_activations
        for i in range(2):
            # second iteration is key, hooks would have fired during aot trace
            # on first iter
            compiled_activations.clear()
            x = torch.randn((20, 10))
            pred = compiled_model(x)
            loss = pred.sum()
            loss.backward()

        activations = eager_activations
        for i in range(2):
            # second iteration is key, hooks would have fired during aot trace
            # on first iter
            eager_activations.clear()
            x = torch.randn((20, 10))
            pred = model(x)
            loss = pred.sum()
            loss.backward()

        print(f"Recorded Layers: {compiled_activations.keys()}\n\n")
        print(f"Expected Layers: {eager_activations.keys()}")

        self.assertTrue(compiled_activations.keys() == eager_activations.keys())
        self.assertTrue(activations.keys() == forward_handles.keys())

    def test_hooks_allowed_modules(self):
        # this test shouldn't care whether hook guards are enabled or not
        class ToyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = torch.nn.Sequential(
                    *[torch.nn.Linear(10, 10000), torch.nn.ReLU()]
                    + [torch.nn.Linear(10000, 5), torch.nn.ReLU()]
                )

            def forward(self, x):
                return self.net(x)

        model = ToyModel()
        self._forward_hook_test_helper(model)

    def test_hooks_allowed_modules_compiles(self):
        class ToyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = torch.nn.Sequential(
                    *[torch.nn.Linear(10, 10000), torch.nn.ReLU()]
                    + [torch.nn.Linear(10000, 5), torch.nn.ReLU()]
                )

            def forward(self, x):
                return self.net(x)

        model = ToyModel()
        activations = []

        def save_activations(mod, inp, out):
            activations.append(inp)

        for name, module in model.named_modules():
            module.register_forward_hook(save_activations)

        cnt = torch._dynamo.testing.CompileCounter()
        model = torch.compile(model, backend=cnt, fullgraph=True)
        for i in range(2):
            # second iteration is key, hooks would have fired during aot trace
            # on first iter
            activations.clear()
            x = torch.randn((20, 10))
            pred = model(x)
            loss = pred.sum()
            loss.backward()
        self.assertEqual(len(activations), 6)
        self.assertEqual(cnt.frame_count, 1)

    def test_hooks_allowed_modules_compiles_self_contained(self):
        class ToyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = torch.nn.Sequential(
                    *[torch.nn.Linear(10, 10000), torch.nn.ReLU()]
                    + [torch.nn.Linear(10000, 5), torch.nn.ReLU()]
                )

            def forward(self, x):
                return self.net(x) * self.net(x)

        model = ToyModel()
        forward_handles = {}

        def output_modifying_hook(mod, inp, out):
            return 2 * out + 1

        for name, module in model.named_modules():
            forward_handles[name] = module.register_forward_hook(output_modifying_hook)

        cnt = torch._dynamo.testing.CompileCounter()

        x = torch.randn((20, 10))
        pred_eager = model(x)
        loss_eager = pred_eager.sum()
        eager_loss_bwd = loss_eager.backward()

        model = torch.compile(model, backend=cnt, fullgraph=True)
        pred = model(x)

        loss = pred.sum()
        loss_bwd = loss.backward()

        self.assertEqual(eager_loss_bwd, loss_bwd)
        self.assertEqual(cnt.frame_count, 2)

        # Ndim change, recompile
        pred = model(torch.randn([10, 10, 10]))
        self.assertEqual(cnt.frame_count, 4)

        # Stable
        pred = model(torch.randn([10, 10, 10]))
        self.assertEqual(cnt.frame_count, 4)

    def test_dunder_call_explicitly(self):
        # hooks should be triggered if explicit calling `__call__`
        class ToyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10000)

            def forward(self, x):
                return self.linear.__call__(x)

        model = ToyModel()
        self._forward_hook_test_helper(model)

    def test_backward_hooks(self):
        # this test shouldn't care whether hook guards are enabled or not

        class CustomLinear(torch.nn.Module):
            # not an 'allowed module', so should not graph-break
            def __init__(self, a, b):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(a, b))

            def forward(self, x):
                return torch.mm(x, self.weight)

        class ToyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = torch.nn.Sequential(
                    *[CustomLinear(10, 10)]
                    + [CustomLinear(10, 10000)]
                    + [CustomLinear(10000, 5)]
                )

            def forward(self, x):
                return self.net(x)

        model = ToyModel()
        backward_hook_handles = {}
        pre_backward_hook_handles = {}

        grad_sizes = {}

        def backward_hook(name, mod, grad_inp, grad_out):
            grad_sizes[name] = (
                (gi.shape for gi in grad_inp),
                (go.shape for go in grad_out),
            )
            return None

        pre_grad_sizes = {}

        def backward_pre_hook(name, mod, grad_out):
            pre_grad_sizes[name] = (go.shape for go in grad_out)
            return None

        for name, module in model.named_modules():
            backward_hook_handles[name] = module.register_full_backward_hook(
                partial(backward_hook, name)
            )

            pre_backward_hook_handles[name] = module.register_full_backward_pre_hook(
                partial(backward_pre_hook, name)
            )

        model = torch.compile(model, backend="aot_eager")

        for i in range(2):
            # second iteration is key, hooks would have fired during aot trace
            # on first iter
            x = torch.randn((20, 10))
            pred = model(x)
            loss = pred.sum()
            loss.backward()

        self.assertTrue(grad_sizes.keys() == backward_hook_handles.keys())
        self.assertTrue(pre_grad_sizes.keys() == pre_backward_hook_handles.keys())

    def test_udo_instance_method_as_hook(self):
        class CustomClass:
            def __init__(self, module):
                self.module = module
                self.handle = self.module.register_forward_pre_hook(
                    self.func1, prepend=True, with_kwargs=True
                )

            def func1(self, module, args, kwargs):
                return (args[0] + 1,), kwargs

            def __call__(self, x):
                return self.module(x)

        class ToyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return x * x

        model = ToyModel()
        x = torch.zeros((3, 4))
        obj = CustomClass(model)
        out = torch.compile(obj, fullgraph=True)(x)
        self.assertEqual(out, (x + 1) * (x + 1))

    def test_module_dict_iter_name(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.activations = torch.nn.ModuleDict(
                    [["lrelu", torch.nn.LeakyReLU()], ["prelu", torch.nn.PReLU()]]
                )

            def forward(self, x):
                for activation_name in self.activations:
                    x = self.activations[activation_name](x)
                return x

        cnt = torch._dynamo.testing.CompileCounter()
        # Eager
        eager_res = MyModule()(torch.ones(10, 10))

        # Compile
        optim_res = torch.compile(MyModule(), backend=cnt)(torch.ones(10, 10))
        self.assertEqual(eager_res, optim_res)
        self.assertEqual(cnt.frame_count, 1)

    def test_module_dict_iter_keys(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.activations = torch.nn.ModuleDict(
                    [["lrelu", torch.nn.LeakyReLU()], ["prelu", torch.nn.PReLU()]]
                )

            def forward(self, x):
                for activation_name in self.activations.keys():
                    x = self.activations[activation_name](x)
                return x

        cnt = torch._dynamo.testing.CompileCounter()
        # Eager
        eager_res = MyModule()(torch.ones(10, 10))

        # Compile
        optim_res = torch.compile(MyModule(), backend=cnt)(torch.ones(10, 10))
        self.assertEqual(eager_res, optim_res)
        self.assertEqual(cnt.frame_count, 1)

    def test_module_setattr(self):
        models = torch.nn.Sequential(torch.nn.Linear(3, 3))
        models[0].abc = False

        def run():
            models[0].abc = True
            x = torch.randn(1, 3)
            return models(x)

        run = torch.compile(run, fullgraph=True)
        run()
        self.assertTrue(models[0].abc)

    @torch._dynamo.config.patch(inline_inbuilt_nn_modules=False)
    def test_assign_does_not_exist(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                self.text_encoding = x + 1
                return self.text_encoding

        mod = MyModule()
        out = torch.compile(mod, fullgraph=True)(torch.randn(10))
        assert mod.text_encoding is out

    def test_module_dict_iter_values(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.activations = torch.nn.ModuleDict(
                    [["lrelu", torch.nn.LeakyReLU()], ["prelu", torch.nn.PReLU()]]
                )

            def forward(self, x):
                for activation in self.activations.values():
                    x = activation(x)
                return x

        cnt = torch._dynamo.testing.CompileCounter()
        # Eager
        eager_res = MyModule()(torch.ones(10, 10))

        # Compile
        optim_res = torch.compile(MyModule(), backend=cnt)(torch.ones(10, 10))
        self.assertEqual(eager_res, optim_res)
        self.assertEqual(cnt.frame_count, 1)

    def test_unspecialized_seq(self):
        models = torch.nn.Sequential(torch.nn.Linear(3, 3))

        def fn(x):
            models[0].training = False
            return models(x)

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.randn(1, 3)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_no_op_assignment(self):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buffer = torch.rand([4])

            def forward(self, x):
                # should be a no-op, but causes dynamo to lose the static input
                x = x + 1
                self.buffer = self.buffer.to(x)
                return self.buffer + x

        compiles_without_buffers = 0

        def debug_compile(gm, *args, **kwargs):
            nonlocal compiles_without_buffers
            compiles_without_buffers += len(list(gm.buffers())) == 0
            return gm

        @torch.compile(backend=debug_compile)
        def foo(mod, x):
            return mod(x)

        mod = Mod()
        foo(mod, torch.rand([4]))
        if torch._dynamo.config.inline_inbuilt_nn_modules:
            self.assertEqual(compiles_without_buffers, 1)
        else:
            self.assertEqual(compiles_without_buffers, 0)

        foo(mod, torch.rand([4], dtype=torch.half))
        if torch._dynamo.config.inline_inbuilt_nn_modules:
            self.assertEqual(compiles_without_buffers, 2)
        else:
            self.assertEqual(compiles_without_buffers, 1)

        class Mod2(Mod):
            def __setattr__(self, name, value):
                return super().__setattr__(name, value)

        foo(Mod2(), torch.rand([4]))
        # causes two compilations, bc unimplemented custom setattr
        self.assertTrue(compiles_without_buffers >= 2)

    def test_unspec_non_inlinable_module(self):
        mod = UnspecNonInlinableModule()
        opt_fn = torch.compile(mod, backend="eager")
        x = torch.randn(100)
        actual = opt_fn(x)
        expected = mod(x)
        self.assertEqual(actual, expected)

    @torch._dynamo.config.patch("inline_inbuilt_nn_modules", True)
    def test_mark_static_previously_seen_tensor(self):
        # This test verifies that dynamo will mark
        # the buffers/params of a module as static
        # even if this param was previously seen
        # (ex. as a different input)
        num_compiles = 0

        def debug_compiler(gm, _):
            nonlocal num_compiles
            num_compiles += 1

            input_nodes = [
                n for n in gm.graph.nodes if n.op == "placeholder" and n.name == "l_b_"
            ]

            self.assertGreater(len(input_nodes), 0)
            for input_node in input_nodes:
                self.assertEqual(
                    input_node.meta["tensor_dict"]["_dynamo_static_input_type"],
                    "unguarded",
                )

            return gm

        class TestModule(torch.nn.Module):
            def __init__(self, buf) -> None:
                super().__init__()
                # Changing this one to nn.Buffer fails because `nn.Buffer` does a .detach()
                # so the value in self.tx.output.side_effects will no longer evaluate to True
                self.register_buffer("buf", buf)

            def forward(self, x):
                return self.buf * x

        @torch.compile(backend=debug_compiler)
        def fn(x, b, mod):
            z = b + 1
            return z * mod(x)

        buf = torch.ones(2, 2)
        inp = torch.ones(2)
        mod = TestModule(buf)
        fn(inp, buf, mod)
        self.assertEqual(num_compiles, 1)

    @torch._dynamo.config.patch("inline_inbuilt_nn_modules", True)
    def test_mark_static_nn_module_tensor(self):
        # This test verifies that dynamo will mark
        # the nn module tensor attributes as static
        num_compiles = 0

        def debug_compiler(gm, _):
            nonlocal num_compiles
            num_compiles += 1

            input_nodes = [
                n
                for n in gm.graph.nodes
                if n.op == "placeholder" and n.name == "l_mod_buf"
            ]

            self.assertGreater(len(input_nodes), 0)
            for input_node in input_nodes:
                self.assertEqual(
                    input_node.meta["tensor_dict"]["_dynamo_static_input_type"],
                    "unguarded",
                )

            return gm

        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buf = torch.ones(2, 2)

            def forward(self, x):
                return self.buf * x

        mod = TestModule()

        @torch.compile(backend=debug_compiler)
        def fn(x):
            return x * mod(x)

        inp = torch.ones(2)
        fn(inp)
        self.assertEqual(num_compiles, 1)

    @torch._dynamo.config.patch("inline_inbuilt_nn_modules", True)
    @torch._inductor.config.patch("freezing", True)
    @torch.no_grad()
    def test_mark_static_with_freezing(self):
        # This test verifies that dynamo will
        # add buffers/params as attributes of the
        # graph w/ guards if freezing is enabled
        num_compiles = 0

        def debug_compiler(gm, _):
            nonlocal num_compiles
            num_compiles += 1

            input_nodes = [
                n for n in gm.graph.nodes if n.op == "placeholder" and n.name == "l_b_"
            ]
            self.assertEqual(len(input_nodes), 0)
            self.assertEqual(len(list(gm.buffers())), 1)
            return gm

        class TestModule(torch.nn.Module):
            def __init__(self, buf) -> None:
                super().__init__()
                self.buf = torch.nn.Buffer(buf)

            def forward(self, x):
                return self.buf * x

        @torch.compile(backend=debug_compiler)
        def fn(x, mod):
            return mod(x)

        buf = torch.ones(2, 2)
        inp = torch.ones(2)
        mod = TestModule(buf)
        fn(inp, mod)
        self.assertEqual(num_compiles, 1)
        mod.buf = torch.rand_like(buf)
        fn(inp, mod)
        self.assertEqual(num_compiles, 2)

    @patch.object(torch._dynamo.config, "guard_nn_modules", True)
    def test_guard_on_torch_nn_modules(self):
        # https://github.com/pytorch/pytorch/issues/110048

        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.multiplier = 10

            def forward(self, x):
                return self.linear(x) * self.multiplier

        mod = MockModule()

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def generate(x, c):
            return mod(x) + c

        for _ in range(0, 10):
            generate(torch.randn(10, 10), 0)
            generate(torch.randn(10, 10), 1)
        self.assertEqual(cnt.frame_count, 2)

        # Ensure that modification in user module causes recompile
        mod.multiplier = 11
        generate(torch.randn(10, 10), 0)
        self.assertEqual(cnt.frame_count, 3)

    def test_setattr_on_compiled_module(self):
        # https://github.com/pytorch/pytorch/issues/114844

        class ReplayMutation(torch.nn.Module):
            def __init__(self, inp_size, out_size, inner_size):
                super().__init__()
                self.Linear1 = torch.nn.Linear(inp_size, inner_size)
                self.Linear2 = torch.nn.Linear(inner_size, out_size)
                self.x = None

            def forward(self, inp):
                res = self.Linear1(inp)
                self.x = res
                return self.Linear2(res)

        N, D_in, H, inner = 2, 2, 2, 4
        model = ReplayMutation(D_in, H, inner)
        model2 = copy.deepcopy(model)
        input = torch.ones(N, D_in)

        # Keep some intermediate value in model.x
        model.x = torch.tensor([[100, 100, 100, 100], [200, 200, 200, 200]])
        model(input)

        compiled_model = torch.compile(model2, backend="eager")
        compiled_model.x = torch.tensor([[100, 100, 100, 100], [200, 200, 200, 200]])
        compiled_model(input)

        self.assertEqual(model.x, compiled_model.x)

    def test_delattr_on_compiled_module(self):
        class Mod(torch.nn.Module):
            def forward(self, x):
                return x + 1

        model = Mod()
        compiled_model = torch.compile(model)
        compiled_model.foo = 42
        del compiled_model.foo

        self.assertFalse(hasattr(model, "foo"))
        self.assertFalse(hasattr(compiled_model, "foo"))

    def test_globals_change_in_other_file(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # Let `update_global` get invoked in a nested frame, to make sure
            # Dynamo is properly modelling globals across frames and files.
            test_functions.call(update_global)
            a = test_functions.update_global(x)
            # Ensure that the updated global values are read
            return x * a * (_variable + _variable1 + test_functions._variable)

        res = fn(torch.ones(10))
        self.assertEqual(_variable, 1)
        self.assertEqual(_variable1, 1)
        # Ensure that the reconstructed bytecode updates the global value in the
        # other file.
        self.assertEqual(test_functions._variable, 1)
        self.assertEqual(res, 3 * torch.ones(10))

    @unittest.skipIf(
        "inductor" not in torch._dynamo.list_backends(),
        "inductor backend is not available",
    )
    def test_save_and_load_inductor(self):
        mod = MockModule()
        opt_mod = torch.compile(mod, backend="inductor")
        inp = torch.randn(10, 10)
        opt_mod(inp)

        with tempfile.TemporaryDirectory() as tmpdirname:
            torch.save(opt_mod, os.path.join(tmpdirname, "model.pt"))
            # weights_only=False as this is a legacy use case that loads a module
            loaded_model = torch.load(
                os.path.join(tmpdirname, "model.pt"), weights_only=False
            )
        loaded_model(inp)
        self.assertTrue(same_two_models(loaded_model, mod, [inp]))
        self.assertTrue(same_two_models(loaded_model, opt_mod, [inp]))

        torch._dynamo.reset()  # force recompiles
        torch._inductor.metrics.generated_kernel_count = 0
        loaded_model(inp)
        self.assertGreater(torch._inductor.metrics.generated_kernel_count, 0)

    def test_save_and_load_all_backends(self):
        mod = MockModule()
        inp = torch.randn(10, 10)
        for backend in torch._dynamo.list_backends():
            try:
                opt_mod = torch.compile(mod, backend=backend)
                with tempfile.TemporaryDirectory() as tmpdirname:
                    torch.save(opt_mod, os.path.join(tmpdirname, "model.pt"))
                    # weights_only=False as this is a legacy use case that loads a module
                    loaded_model = torch.load(
                        os.path.join(tmpdirname, "model.pt"), weights_only=False
                    )
                torch._dynamo.reset()  # force recompiles
                torch._inductor.metrics.generated_kernel_count = 0
                opt_mod(inp)
                opt_success = torch._inductor.metrics.generated_kernel_count == 0
                torch._dynamo.reset()  # force recompiles
                torch._inductor.metrics.generated_kernel_count = 0
                loaded_model(inp)
                loaded_success = torch._inductor.metrics.generated_kernel_count == 0
                self.assertEqual(opt_success, loaded_success)
            except torch._dynamo.exc.BackendCompilerFailed:
                pass

    def test_monkeypatching_forward(self):
        class FakeModule(torch.nn.Module):
            def forward(self, x):
                return torch.sin(x)

        class MyModule(torch.nn.Module):
            def __init__(self, x):
                super().__init__()

            def forward(self, x):
                return torch.cos(x)

        def helper():
            torch._dynamo.reset()
            mod = MyModule(3)

            def fn(x):
                return mod(x)

            cnt = torch._dynamo.testing.CompileCounter()
            opt_fn = torch.compile(fn, backend=cnt)
            x = torch.randn(10)

            opt_fn(x)
            opt_fn(x)
            self.assertEqual(cnt.frame_count, 1)

            # Monkeypatch forward
            mod.forward = types.MethodType(FakeModule.forward, mod)
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)
            self.assertEqual(cnt.frame_count, 2)

        helper()
        with torch._dynamo.config.patch(inline_inbuilt_nn_modules=True):
            helper()

    def test_user_defined_nn_module_dynamic(self):
        class Conv2d(torch.nn.Conv2d):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def forward(self, x):
                x = torch.nn.functional.conv2d(
                    x,
                    self.weight,
                    self.bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )
                return x

        cnts = torch._dynamo.testing.CompileCounter()
        mod1 = Conv2d(64, 64, kernel_size=(2, 2), stride=(1, 1))
        mod2 = Conv2d(64, 64, kernel_size=(2, 2), stride=(2, 2))
        mod3 = Conv2d(64, 64, kernel_size=(2, 2), stride=(3, 3))

        opt_mod1 = torch.compile(mod1, backend=cnts, fullgraph=True)
        opt_mod2 = torch.compile(mod2, backend=cnts, fullgraph=True)
        opt_mod3 = torch.compile(mod3, backend=cnts, fullgraph=True)

        x = torch.randn(1, 64, 64, 64)
        opt_mod1(x)
        opt_mod2(x)
        opt_mod3(x)

        # Must be 3 compilations. If not marked static there would be 2, because strides would be converted to symints.
        self.assertEqual(cnts.frame_count, 3)

    @patch.object(torch._dynamo.config, "inline_inbuilt_nn_modules", True)
    def test_overridden_call(self):
        class OverRiddenCallModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def __call__(self, x):
                # Overrides the __call__ method of torch.nn.Module
                return 5 * self.forward(x)

            def forward(self, x):
                return x * 3

        m = OverRiddenCallModule()

        def fn(x):
            return m(x)

        x = torch.ones(4)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    @torch._dynamo.config.patch("skip_tensor_guards_with_matching_dict_tags", False)
    @torch._dynamo.config.patch("inline_inbuilt_nn_modules", True)
    def test_param_requires_grad(self):
        def adjust_model(model):
            to_freeze = model.num_iter % 2 == 0
            if to_freeze:
                for param in model.layer2.parameters():
                    param.requires_grad = False
            else:
                for param in model.layer2.parameters():
                    param.requires_grad = True

        class MyModule(torch.nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()

                self.layer1 = torch.nn.Linear(hidden_size, hidden_size)
                self.layer2 = torch.nn.Linear(hidden_size, hidden_size)

                self.num_iter = 0

            def forward(self, x):
                x = self.layer2(x + self.layer1.bias)

                self.num_iter += 1
                return x

        input_size = 1024
        hidden_size = 1024
        output_size = 1
        num_samples = 2048
        features = torch.randn(num_samples, input_size)

        model = MyModule(input_size, hidden_size, output_size)

        cnt = torch._dynamo.testing.CompileCounter()
        opt_model = torch.compile(model, backend=cnt, fullgraph=True)

        for _ in range(3):
            model.zero_grad(True)
            adjust_model(model)
            res = opt_model(features)
            res.sum().backward()

        # Check that we have recompiled twice, which leads to 3 frames
        self.assertEqual(cnt.frame_count, 3)

    def test_branch_on_nn_module_custom_len(self):
        class Cache(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.key_cache = []
                self.len_invoked = 0

            def __len__(self):
                self.len_invoked += 1
                return len(self.key_cache)

        @torch.compile(fullgraph=True, backend="eager")
        def f(x):
            cache = Cache()
            if cache:
                return x + 1, cache
            return x + 2, cache

        x = torch.ones(1)
        res, cache = f(x)
        self.assertEqual(res, x + 2)
        # Make sure Dynamo actually traced the method.
        self.assertEqual(cache.len_invoked, 1)

    def test_branch_on_nn_module_custom_bool(self):
        class Cache(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.key_cache = [0]
                self.bool_invoked = 0

            def __bool__(self):
                self.bool_invoked += 1
                return len(self.key_cache)

        @torch.compile(fullgraph=True, backend="eager")
        def f(x):
            cache = Cache()
            if cache:
                return x + 1, cache
            return x + 2, cache

        x = torch.ones(1)
        res, cache = f(x)
        self.assertEqual(res, x + 1)
        # Make sure Dynamo actually traced the method.
        self.assertEqual(cache.bool_invoked, 1)

    def test_patch_module(self):
        def set_attrs_from_orig_model(cls_instance, mod, *func_names):
            cls_instance.__dict__.update(mod.__dict__)
            if func_names is not None:
                for func in func_names:
                    setattr(cls_instance, func, getattr(mod, func))

        class PatchedMyModule(torch.nn.Module):
            def __init__(self, mod):
                super().__init__()
                set_attrs_from_orig_model(self, mod, "resolve_input")

            def forward(self, x):
                x = self.resolve_input(x)
                return x

        class MyModule(torch.nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.linear = torch.nn.Linear(
                    in_features=input_dim, out_features=output_dim
                )

            def resolve_input(self, x):
                x = self.linear(x)
                return x

            def forward(self, x):
                x = self.linear(x)
                return x

        module = MyModule(input_dim=1, output_dim=1)
        patched_module = PatchedMyModule(module)
        compiled_module = torch.compile(patched_module, backend="eager", fullgraph=True)

        input_tensor = torch.tensor([1.0], dtype=torch.float)
        ref = module(input_tensor)
        res = compiled_module(input_tensor)
        self.assertEqual(ref, res)


devices = ["cuda", "hpu"]
instantiate_device_type_tests(NNModuleTestsDevice, globals(), only_for=devices)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
