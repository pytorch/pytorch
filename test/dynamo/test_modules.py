# Owner(s): ["module: dynamo"]

import collections
import itertools
import traceback
import types
import unittest
from copy import deepcopy
from functools import partial
from typing import Tuple
from unittest.mock import patch

import torch

import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.eval_frame import unsupported
from torch._dynamo.mutation_guard import GenerationTracker
from torch._dynamo.testing import same
from torch.nn import functional as F
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import Parameter, UninitializedParameter

try:
    from . import test_functions
except ImportError:
    import test_functions


class BasicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        return F.relu(self.linear1(x)) * self.scale


class FnMember(torch.nn.Module):
    def __init__(self):
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
    def __init__(self):
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x * self.scale


class IsTrainingCheck(torch.nn.Module):
    def __init__(self):
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
    def __init__(self):
        super().__init__()
        self.train(False)


class ModuleMethodCall(torch.nn.Module):
    def __init__(self):
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
    def __init__(self):
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
    def __init__(self):
        super().__init__()
        self.layer1 = BasicModule()
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        x = self.layer1(x) * self.scale
        return unsupported(x, x)


class UnsupportedModuleCall(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = UnsupportedModule()

    def forward(self, x):
        return 1 + self.mod(x * 1.5)


class ModuleWithStaticForward(torch.nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class ModuleCallModuleWithStaticForward(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = ModuleWithStaticForward()

    def forward(self, x):
        return self.mod(x)


class ModuleStaticMethodCall(torch.nn.Module):
    def __init__(self):
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
    def __init__(self):
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
    def __init__(self):
        super().__init__()
        self.scale = torch.randn(1, 10)

    @property
    def scale_alias(self):
        return self.scale

    def forward(self, x):
        return x * self.scale_alias


class ConstLoop(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.count = 3

    def forward(self, x):
        for i in range(self.count):
            x = torch.sigmoid(self.linear1(x))
        return x


class ViaModuleCall(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)

    def forward(self, x):
        return test_functions.constant3(torch.sigmoid(self.linear1(x)), x)


class IsNoneLayer(torch.nn.Module):
    def __init__(self):
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
    def __init__(self):
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
    def __init__(self):
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
    def __init__(self):
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
    def __init__(self):
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


class CustomGetItemModuleDict(torch.nn.Module):
    def __init__(self):
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
    def __init__(self):
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
    def __init__(self):
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
    def __init__(self):
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
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)

    def forward(self, x, offset=1):
        x = F.relu(self.layer1(x)) + offset
        return x


class Seq(torch.nn.Module):
    def __init__(self):
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
    def __init__(self):
        self.val = 0.5
        self.count = 3


class CfgModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = Cfg()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        for i in range(self.cfg.count):
            x = self.layer(x + self.cfg.val)
        return x


class StringMember(torch.nn.Module):
    def __init__(self):
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
            self.add_module("denselayer%d" % (i + 1), _Block())

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNetBlocks(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = _DenseBlock()

    def forward(self, x):
        return self.layers(x)


class MaterializedModule(torch.nn.Module):
    """Once the below lazy module is initialized with its first input,
    it is transformed into this module."""

    param: Parameter

    def __init__(self):
        super().__init__()
        self.register_parameter("param", None)

    def forward(self, x):
        return x


class LazyModule(LazyModuleMixin, MaterializedModule):
    param: UninitializedParameter
    cls_to_become = MaterializedModule

    def __init__(self):
        super().__init__()
        self.param = UninitializedParameter()

    def initialize_parameters(self, x):
        # force graph break to ensure this was not inlined
        torch._dynamo.graph_break()
        self.param.materialize(x.shape)


class LazyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.LazyLinear(10)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.LazyLinear(1)
        self.relu2 = torch.nn.ReLU()

    def forward(self, input):
        x = self.relu1(self.fc1(input))
        y = self.relu2(self.fc2(x))
        return y


class LazyLayerWithListInput(LazyModuleMixin, torch.nn.Module):
    def __init__(self):
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
    def __init__(self):
        super().__init__()
        self.layer = LazyLayerWithListInput()

    def forward(self, input):
        return self.layer(input[:-1])


class LazyModuleWithLazySubmodule(LazyModuleMixin, torch.nn.Module):
    def __init__(self):
        super().__init__()

    def initialize_parameters(self, input):
        with torch.no_grad():
            self.layer = LazyLayerWithListInput()

    def forward(self, x):
        return self.layer(x)


class LazyParentModule(LazyModuleMixin, torch.nn.Module):
    def __init__(self):
        super().__init__()

    def impl(self, x):
        return x.cos() + self._val


class LazyChildModuleNoClsToBecome(LazyParentModule):
    def __init__(self):
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
    def __init__(self):
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
    def __init__(self):
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
            self.add_module("denselayer%d" % (i + 1), _Block())

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
            self.add_module("denselayer%d" % (i + 1), _Block())

    def forward(self, init_features):
        features = [init_features]
        for k in self.keys():
            new_features = self[k](features)
            features.append(new_features)
        return torch.cat(features, 1)


class CallForwardDirectly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.layer1.forward(x)
        x = self.layer2.forward(x)
        return x


class ConvCallForwardDirectly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)

    def forward(self, x):
        return self.layer.forward(x)


class ConvTransposeCallForwardDirectly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.ConvTranspose2d(4, 4, 4)

    def forward(self, x):
        return self.layer.forward(x)


class ModuleNameString(torch.nn.Module):
    def __init__(self):
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
    def linear(self, x):
        return x * 2.0


class ModuleAttributePrecedence(ModuleAttributePrecedenceBase):
    def __init__(self):
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
    def __init__(self):
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
    def __init__(self):
        super().__init__()
        for i in range(2):
            self.add_module("l@yer-%d" % (i + 1), BasicModule())

    def forward(self, x):
        for _, layer in self.items():
            x = layer(x)
        return x


class SequentialWithDuplicatedModule(torch.nn.Module):
    # Sequential module(self.layer) contains three duplicated ReLU module.
    def __init__(self):
        super(SequentialWithDuplicatedModule, self).__init__()
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
    def __init__(self):
        super(SequentialWithDuplicatedModule2, self).__init__()
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


class ModulePatch1(torch.nn.Module):
    pass


class ModulePatch2(torch.nn.Module):
    def forward(self, x):
        return x - 1


def make_test(fn, expected_ops=None):
    def test_fn(self):
        return torch._dynamo.testing.standard_test(
            self, fn=fn, nargs=1, expected_ops=expected_ops
        )

    fn.eval()
    return test_fn


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
    test_modulelist = make_test(CustomGetItemModuleList())
    test_moduledict = make_test(ModuleDict())
    test_moduledict = make_test(CustomGetItemModuleDict())
    test_super1 = make_test(SuperModule())
    test_super2 = make_test(SuperModule2())
    test_super_class_method = make_test(SuperChildCallsClassMethod())
    test_children = make_test(Children())
    test_named_children = make_test(NamedChildren())
    test_densenet = make_test(DenseNetBlocks())
    test_parameters1 = make_test(ParametersModule1())
    test_parameters2 = make_test(ParametersModule2())
    test_parameters3 = make_test(ParametersModule3(), expected_ops=5)
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

    def test_module_forward_has_graph_break(self):
        m = ModuleForwardHasGraphBreak()
        x = torch.rand([10, 10])
        ref = m(x)
        opt_m = torch._dynamo.optimize("eager")(m)
        res = opt_m(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_unsupportedmethod(self):
        m = UnsupportedMethodCall()
        i = torch.randn(10)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_m = torch._dynamo.optimize(cnt)(m)
        r = opt_m(i)
        self.assertTrue(torch._dynamo.testing.same(r, m(i)))
        self.assertEqual(cnt.op_count, 5)

    def test_unsupportedmodule(self):
        m = UnsupportedModuleCall()
        i = torch.randn(10)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_m = torch._dynamo.optimize(cnt)(m)
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
        self.assertEqual(cnt.frame_count, 3)

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

        class TensorProxy(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return super().__torch_function__(func, types, args, kwargs)

        torch._dynamo.config.traceable_tensor_subclasses.add(TensorProxy)

        try:
            x = torch.randn(1).as_subclass(TensorProxy)
            cnt = torch._dynamo.testing.CompileCounter()
            out1 = foo(x)
            opt_foo = torch._dynamo.optimize(cnt, nopython=True)(foo)
            out2 = opt_foo(x)

            self.assertEqual(cnt.op_count, 4)
            self.assertTrue(torch._dynamo.testing.same(out1, out2))

        finally:
            torch._dynamo.config.traceable_tensor_subclasses.remove(TensorProxy)

    def test_torch_function_with_closure(self):
        def run():
            counter = 0

            def foo(x):
                # function call, twice to test wrapping
                x = F.sigmoid(x)
                x = F.sigmoid(x)
                # method call, twice to test wrapping
                x = x.sigmoid()
                x = x.sigmoid()
                return x

            class TensorProxy(torch.Tensor):
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    nonlocal counter
                    # for now, only support reads from closure cells
                    # TODO(future PR): support writes as well
                    counter + 1
                    return super().__torch_function__(func, types, args, kwargs)

            torch._dynamo.config.traceable_tensor_subclasses.add(TensorProxy)

            try:
                x = torch.randn(1).as_subclass(TensorProxy)
                x = torch.randn(1)
                cnt = torch._dynamo.testing.CompileCounter()
                out1 = foo(x)
                opt_foo = torch._dynamo.optimize(cnt, nopython=True)(foo)
                out2 = opt_foo(x)

                self.assertEqual(cnt.op_count, 4)
                self.assertTrue(torch._dynamo.testing.same(out1, out2))
            finally:
                torch._dynamo.config.traceable_tensor_subclasses.remove(TensorProxy)

        run()

    @patch.object(torch._dynamo.config, "raise_on_ctx_manager_usage", False)
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

        module_dict = torch.nn.ModuleDict({"cat": torch.nn.Conv2d(1, 1, 1)})
        pre = m(data)
        cnt.clear()

        with torch._dynamo.optimize(cnt, nopython=False):
            opt_pre = m(data)
            m = M(module_dict)
            data = torch.randn(1)
            out1 = m(data)

        out_post = m(data)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)
        self.assertTrue(torch._dynamo.testing.same(pre, opt_pre))
        self.assertTrue(torch._dynamo.testing.same(out1, out_post))

    def test_lazy_module1(self):
        input_shape = (16, 3, 6, 7, 8)

        cnt = torch._dynamo.testing.CompileCounter()
        module = LazyModule()

        def test_static_module():
            input = torch.ones(*input_shape)
            module(input)

        # test no graph break
        opt_test_static_module = torch._dynamo.optimize(cnt, nopython=True)(
            test_static_module
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

        opt_test_unspecialized = torch._dynamo.optimize(cnt)(test_unspecialized)
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
        opt_test_torch_static = torch._dynamo.optimize(cnt, nopython=True)(
            test_torch_static
        )
        opt_test_torch_static()
        out = opt_test_torch_static()

        self.assertTrue(same(out, module(torch.ones(*input_shape))))

        self.assertTrue(
            isinstance(module, torch.nn.modules.batchnorm.BatchNorm3d),
            "Module should be transformed to an instance of BatchNorm3d.",
        )
        self.assertEqual(cnt.frame_count, 1, "No guards should have triggered.")

    def test_lazy_module2(self):
        # Test FX graph 'call_module' works well if argument is lazy module
        m = LazyMLP()
        x = torch.rand([10, 10])
        opt_m = torch._dynamo.optimize("eager", nopython=True)(m)
        # We should run compile mode firstly, otherwise the module
        # would be initialized when running eager mode.
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_lazy_module3(self):
        m = LazyMLP()
        x = torch.rand([10, 10])
        cnt = torch._dynamo.testing.CompileCounter()
        opt_m = torch._dynamo.optimize(cnt, nopython=True)(m)
        # first iteration
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))
        # move to cuda and second iteration
        m = m.to("cuda")
        x = x.to("cuda")
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))
        self.assertEqual(cnt.frame_count, 2)

    def test_lazy_module4(self):
        m = LazyMLP()
        x = torch.rand([10, 10])
        cnt = torch._dynamo.testing.CompileCounter()
        opt_m = torch._dynamo.optimize(cnt, nopython=True)(m)
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

    def test_lazy_module5(self):
        # Test lazy module works well with list/tuple input
        m = LazyModuleWithListInput()
        x = [torch.rand([5, 5])] * 3 + [None]
        opt_m = torch._dynamo.optimize("eager", nopython=True)(m)
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_lazy_module6(self):
        # Test new lazy submodule in lazy module's initialize_parameters
        m = LazyModuleWithLazySubmodule()
        x = [torch.rand([5, 5])] * 3
        opt_m = torch._dynamo.optimize("eager", nopython=True)(m)
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_lazy_module_no_cls_to_become(self):
        # make sure super() works in the case where cls_to_become is None
        m = LazyChildModuleNoClsToBecome()
        x = torch.rand(2, 2)
        opt_m = torch._dynamo.optimize("eager", nopython=True)(m)
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_call_fn_with_non_const_inputs_safe(self):
        class ModuleSpecialFwd(torch.nn.Module):
            def __init__(self):
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
        graph, _ = torch._dynamo.export(mod, rx)
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


class MockModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(10, 10)
        self.register_buffer("buf0", torch.randn(10, 10))

    def forward(self, x):
        return self.relu(self.linear(x) + self.buf0)


class OptimizedModuleTest(torch._dynamo.test_case.TestCase):
    def test_nn_module(self):
        mod = MockModule()
        cnt = torch._dynamo.testing.CompileCounter()
        opt_mod = torch._dynamo.optimize(cnt)(mod)
        self.assertIsInstance(opt_mod, torch._dynamo.OptimizedModule)

        x = torch.randn(10, 10)
        self.assertTrue(torch._dynamo.testing.same(mod(x), opt_mod(x)))
        self.assertEqual(cnt.frame_count, 1)

    def test_to(self):
        mod = MockModule()
        cnt = torch._dynamo.testing.CompileCounter()
        opt_mod = torch._dynamo.optimize(cnt)(mod)
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

    def test_attr(self):
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.register_buffer("buf0", torch.randn(10, 10))

            def forward(self, x):
                return self.r(torch.sin(x)) + self.buf0

        mod = MockModule()
        opt_mod = torch._dynamo.optimize("eager")(mod)

        # Check parameteres and buffers
        for p1, p2 in zip(mod.parameters(), opt_mod.parameters()):
            self.assertTrue(id(p1) == id(p2))
        for b1, b2 in zip(mod.buffers(), opt_mod.buffers()):
            self.assertTrue(id(b1) == id(b2))

        def get_parameter_dtype(mod: torch.nn.Module):
            parameters_and_buffers = itertools.chain(mod.parameters(), mod.buffers())
            return next(parameters_and_buffers).dtype

        opt_mod = torch._dynamo.optimize("eager")(get_parameter_dtype)
        out_dtype = opt_mod(mod)
        self.assertEqual(out_dtype, torch.float32)

    def test_dir(self):
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.register_buffer("buf0", torch.randn(10, 10))
                self.register_parameter(
                    name="param0", param=torch.nn.Parameter(torch.randn(10, 10))
                )

            def forward(self, x):
                return self.r(torch.sin(x)) + self.buf0

        mod = MockModule()
        mod_keys = dir(mod)
        opt_mod = torch._dynamo.optimize("eager")(mod)
        opt_mod_keys = dir(opt_mod)

        # Check user-defined attributes, parameters and buffers
        self.assertIn("linear", opt_mod_keys)
        self.assertIn("buf0", opt_mod_keys)
        self.assertIn("param0", opt_mod_keys)

        # Check all attributes, parameters and buffers
        self.assertTrue(len(set(mod_keys).difference(opt_mod_keys)) == 0)

    def test_recursion(self):
        mod = MockModule()
        cnt = torch._dynamo.testing.CompileCounter()
        opt_mod = torch._dynamo.optimize(cnt)(mod)

        for _ in range(5):
            opt_mod = torch._dynamo.optimize(cnt)(opt_mod)
        opt_mod(torch.randn(10, 10))
        self.assertEqual(cnt.frame_count, 1)

    def test_composition(self):
        class InnerModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(torch.sin(x))

        opt_inner_mod = InnerModule()

        class OuterModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mod = opt_inner_mod

            def forward(self, x):
                return self.mod(torch.cos(x))

        outer_mod = OuterModule()
        cnt = torch._dynamo.testing.CompileCounter()
        opt_outer_mod = torch._dynamo.optimize(cnt)(outer_mod)

        x = torch.randn(4)
        self.assertIsInstance(opt_outer_mod, torch._dynamo.OptimizedModule)
        self.assertTrue(torch._dynamo.testing.same(outer_mod(x), opt_outer_mod(x)))
        self.assertEqual(cnt.frame_count, 1)

    def test_composition_with_opt_mod(self):
        class InnerModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(torch.sin(x))

        inner_mod = InnerModule()
        cnt = torch._dynamo.testing.CompileCounter()
        opt_inner_mod = torch._dynamo.optimize(cnt)(inner_mod)

        class OuterModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mod = opt_inner_mod

            def forward(self, x):
                return self.mod(torch.cos(x))

        outer_mod = OuterModule()
        opt_outer_mod = torch._dynamo.optimize(cnt)(outer_mod)

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
                torch._dynamo.optimize("eager", nopython=True)(fn)(torch.ones(10)),
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
            module: torch.nn.Module, inputs: Tuple[torch.Tensor], output: torch.Tensor
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
          - removing a hook doesn't fail a guard, becuase we weren't compiling the hook
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
            module: torch.nn.Module, inputs: Tuple[torch.Tensor], output: torch.Tensor
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
        self.assertTrue("forward_hooks.keys" in failure_reason)
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
            module: torch.nn.Module, inputs: Tuple[torch.Tensor], output: torch.Tensor
        ) -> torch.Tensor:
            return 2 * output + 2

        m._forward_hooks[handle.id] = new_forward_hook
        self.assertEqual(compiled_func(inp), outer_func(inp))
        self.assertEqual(compiled_func(inp).item(), 16)
        self.assertTrue("___check_obj_id(L['m']._forward_hooks" in failure_reason)

    @patch.object(torch._dynamo.config, "skip_nnmodule_hook_guards", True)
    def test_hooks_skip_guards(self):
        class TestModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return 2 * x + 1

        m = TestModule()

        def forward_hook(
            module: torch.nn.Module, inputs: Tuple[torch.Tensor], output: torch.Tensor
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
        activations = dict()

        def save_activations(name, mod, inp, out):
            activations[name] = inp

        for name, module in model.named_modules():
            forward_handles[name] = module.register_forward_hook(
                partial(save_activations, name)
            )

        model = torch.compile(model, backend="aot_eager")

        for i in range(2):
            # second iteration is key, hooks would have fired during aot trace
            # on first iter
            activations.clear()
            x = torch.randn((20, 10))
            pred = model(x)
            loss = pred.sum()
            loss.backward()

        print(f"Recorded Layers: {activations.keys()}\n\n")
        print(f"Expected Layers: {forward_handles.keys()}")
        self.assertTrue(activations.keys() == forward_handles.keys())

    def test_hooks_allowed_modules(self):
        # this test shouldn't care whether hook guards are enabled or not
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Sequential(
                    *[torch.nn.Linear(10, 10000), torch.nn.ReLU()]
                    + [torch.nn.Linear(10000, 5), torch.nn.ReLU()]
                )

            def forward(self, x):
                return self.net(x)

        model = ToyModel()
        self._forward_hook_test_helper(model)

    def test_dunder_call_explicitly(self):
        # hooks should be triggered if explicit calling `__call__`
        class ToyModel(torch.nn.Module):
            def __init__(self):
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
            def __init__(self):
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

    def test_module_dict_iter_name(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
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
        optim_res = torch._dynamo.optimize(cnt)(MyModule())(torch.ones(10, 10))
        self.assertEqual(eager_res, optim_res)
        self.assertEqual(cnt.frame_count, 1)

    def test_module_dict_iter_keys(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
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
        optim_res = torch._dynamo.optimize(cnt)(MyModule())(torch.ones(10, 10))
        self.assertEqual(eager_res, optim_res)
        self.assertEqual(cnt.frame_count, 1)

    def test_module_dict_iter_values(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
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
        optim_res = torch._dynamo.optimize(cnt)(MyModule())(torch.ones(10, 10))
        self.assertEqual(eager_res, optim_res)
        self.assertEqual(cnt.frame_count, 1)

    def test_unspecialized_seq(self):
        models = torch.nn.Sequential(torch.nn.Linear(3, 3))

        def fn(x):
            models[0].training = False
            return models(x)

        opt_fn = torch._dynamo.optimize("eager")(fn)
        x = torch.randn(1, 3)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
