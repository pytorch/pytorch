# Owner(s): ["module: onnx"]

"""Test consistency between the output values of fx.GraphModule produced by FX frontend
and torch.nn.Modules.

Note [Add new module test cases]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To add a new module test case, add a new entry to `MODULE_INFOS` below. The entry should
be a `ModuleInfo` object. The `ModuleInfo` object should have the following fields:

    1. `module_cls`: the class of the nn.Module to be tested.
    2. `module_inputs_func`: a function that returns a list of `ModuleInput`s.
        `ModuleInput` is a class that contains the constructor arguments and
        the forward arguments for the module. Both are represented by `FunctionInput`.
        `FunctionInput` is a class that contains the positional arguments and
        the keyword arguments for the function.

This is adopted from 'torch/testing/_internal/common_modules.py'. More details
and example usages can be found there.

Note [Add new FxFrontend]
~~~~~~~~~~~~~~~~~~~~~~~~~

To add a new FxFrontend, add a new entry to `FX_FRONTEND_INFOS` below. The entry should
be a `FrontendInfo` object. The `FrontendInfo` object should have the following fields:

    1. `fx_frontend`: the FxFrontend to be tested.
    2. `skips`: a mapping from module_info name to reason for skipping the test.
        `ModuleInfo.formatted_name` is used as the model name to skip tests.


"""

from __future__ import annotations

import functools
import sys
from typing import List, Mapping, Optional, Union

import pytest

import torch
from torch import nn
from torch.onnx._internal.fx import frontend
from torch.testing._internal import common_device_type, common_modules, common_utils
from torch.utils import _pytree as pytree

_FrontendType = Union[frontend.FxFrontend, frontend.FxFrontendUnpackKwargs]


class FrontendInfo:
    """A container for FxFrontend and its skips for test cases."""

    __slots__ = ["fx_frontend", "skips"]

    fx_frontend: _FrontendType

    # Mapping from module_info name to reason for skipping the test.
    # NOTE: `ModuleInfo.formatted_name` is used as the model name to skip tests.
    skips: Mapping[str, str]

    def __init__(
        self,
        fx_frontend: _FrontendType,
        skips: Optional[Mapping[str, str]] = None,
    ):
        self.fx_frontend = fx_frontend
        self.skips = skips or {}


FX_FRONTEND_INFOS: List[FrontendInfo] = []


# MODIFY THIS SECTION to add a FxFrontend or modify skips ######################
# See Note [Add new FxFrontend] for more details.

FX_FRONTEND_INFOS.append(
    FrontendInfo(
        frontend.FxFrontendUnpackKwargs(frontend.AOTAutogradFrontend(dynamic=True)),
        skips={
            "MLPModel": "aotautograd alters graph inputs/outputs when there are parameters."
        },
    )
)

FX_FRONTEND_INFOS.append(
    FrontendInfo(
        frontend.DynamoExport(tracing_mode="symbolic", aten_graph=True),
        skips={},
    )
)

FX_FRONTEND_INFOS.append(
    FrontendInfo(
        frontend.DynamoExport(tracing_mode="real", aten_graph=True),
        skips={},
    )
)

FX_FRONTEND_INFOS.append(
    FrontendInfo(
        frontend.FxFrontendUnpackKwargs(frontend.DynamoOptimize(dynamic=True)),
        skips={},
    )
)

FX_FRONTEND_INFOS.append(
    FrontendInfo(
        frontend.FxFrontendUnpackKwargs(frontend.DynamoOptimize(dynamic=False)),
        skips={},
    )
)

FX_FRONTEND_INFOS.append(
    FrontendInfo(
        frontend.FxFrontendUnpackKwargs(frontend.MakeFx(tracing_mode="real")),
        skips={},
    )
)


# END OF SECTION TO MODIFY to add a FxFrontend or modify skips #################

# MODIFY THIS SECTION to add or modify nn.Module test cases ####################
# See Note [Add new module test cases] for more details.
# TODO: Organize test models into a separate file to be re-used by other tests.
class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(8, 8, bias=True)
        self.fc1 = nn.Linear(8, 4, bias=True)
        self.fc2 = nn.Linear(4, 2, bias=True)
        self.fc3 = nn.Linear(2, 2, bias=True)

    def forward(self, tensor_x: torch.Tensor):
        tensor_x = self.fc0(tensor_x)
        tensor_x = torch.sigmoid(tensor_x)
        tensor_x = self.fc1(tensor_x)
        tensor_x = torch.sigmoid(tensor_x)
        tensor_x = self.fc2(tensor_x)
        tensor_x = torch.sigmoid(tensor_x)
        output = self.fc3(tensor_x)
        return output


def module_inputs_mlp_model(
    module_info, device, dtype, requires_grad, training, **kwargs
):
    make_input = functools.partial(
        common_modules.make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    module_inputs = [
        common_modules.ModuleInput(
            constructor_input=common_modules.FunctionInput(),
            forward_input=common_modules.FunctionInput(make_input((97, 8))),
        ),
    ]
    return module_inputs


class SingleAddModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor_x: torch.Tensor):
        output = tensor_x + 1
        return output


def module_inputs_single_add(
    module_info, device, dtype, requires_grad, training, **kwargs
):
    make_input = functools.partial(
        common_modules.make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    module_inputs = [
        common_modules.ModuleInput(
            constructor_input=common_modules.FunctionInput(),
            forward_input=common_modules.FunctionInput(make_input((97, 8))),
        ),
    ]
    return module_inputs


# Utilize `ModuleInfo` to define the test cases.
# See Note [Add new module test cases] for more details.
MODULES_DB = [
    common_modules.ModuleInfo(MLPModel, module_inputs_func=module_inputs_mlp_model),
    common_modules.ModuleInfo(
        SingleAddModel, module_inputs_func=module_inputs_single_add
    ),
]

# END OF SECTION TO MODIFY to add or modify nn.Module test cases ###############


class _TestFxFrontendConsistency(common_utils.TestCase):
    frontend_info: FrontendInfo

    @common_modules.modules(
        MODULES_DB, train_eval_mode=common_modules.TrainEvalMode.eval_only
    )
    def test_output_match(
        self,
        device: str,
        dtype: torch.dtype,
        module_info: common_modules.ModuleInfo,
        training: bool,
    ):
        """Base test method for testing each module_info with fx frontend."""
        # device is provided by instantiate_device_type_tests, but we only want to run in cpu.
        assert device == "cpu"

        if skip_reason := self.frontend_info.skips.get(module_info.formatted_name):
            pytest.skip(skip_reason)

        fx_frontend = self.frontend_info.fx_frontend

        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(
            module_info,
            device=device,
            dtype=dtype,
            requires_grad=False,
            training=training,
        )

        for (i, module_input) in enumerate(module_inputs):
            with self.subTest(
                module_name=module_info.formatted_name,
                sample_num=i,
                fx_frontend=fx_frontend.name,
            ):
                init_args, init_kwargs = (
                    module_input.constructor_input.args,
                    module_input.constructor_input.kwargs,
                )
                model = module_cls(*init_args, **init_kwargs)
                model.to(device).to(dtype)
                model.train(training)

                args, kwargs = (
                    module_input.forward_input.args,
                    module_input.forward_input.kwargs,
                )

                if isinstance(fx_frontend, frontend.FxFrontendUnpackKwargs):
                    graph_module, new_args = fx_frontend.trace(model, *args, **kwargs)
                    fx_outputs = graph_module(*new_args)
                else:
                    graph_module = fx_frontend.trace(model, *args, **kwargs)
                    fx_outputs = graph_module(*args, **kwargs)

                outputs = model(*args, **kwargs)

                self.assertEqual(
                    pytree.tree_flatten(outputs)[0],
                    pytree.tree_flatten(fx_outputs)[0],
                )


# 'Parameterize' the test class to run for each fx frontend.
# Adopted from `parameterized` package.
# The reason not using `parameterized` package is that it doesn't support
# applying additional `common_device_type.instantiate_device_type_tests`.
for frontend_info in FX_FRONTEND_INFOS:
    test_class_module = sys.modules[_TestFxFrontendConsistency.__module__].__dict__
    new_test_class_name = f"TestFxFrontendConsistency_{frontend_info.fx_frontend.name}"
    test_class_module[new_test_class_name] = type(
        new_test_class_name,
        (_TestFxFrontendConsistency,),
        dict(_TestFxFrontendConsistency.__dict__, frontend_info=frontend_info),
    )

    # Adds 'instantiated' device-specific test cases to the given scope.
    common_device_type.instantiate_device_type_tests(
        test_class_module[new_test_class_name], globals(), only_for="cpu"
    )

# Remove all test methods from base class to avoid them being picked up and run by
# the test runner.
for method_name in list(_TestFxFrontendConsistency.__dict__):
    if method_name.startswith("test_"):
        delattr(_TestFxFrontendConsistency, method_name)

if __name__ == "__main__":
    common_utils.run_tests()
