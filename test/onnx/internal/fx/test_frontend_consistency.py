# Owner(s): ["module: onnx"]

"""Test consistency between the output values of fx.GraphModule produced by FX frontend
and torch.nn.Modules.
"""

from __future__ import annotations

import functools
from typing import Sequence, Union

import torch
from torch import nn
from torch.onnx._internal.fx import frontend
from torch.testing._internal import common_device_type, common_modules, common_utils
from torch.utils import _pytree as pytree

_FrontendType = Union[frontend.FxFrontend, frontend.FxFrontendUnpackKwargs]
FX_FRONTENDS: Sequence[_FrontendType] = [
    # TODO(bowbao): skip aotautograd, because it alters graph inputs/outputs.
    # frontend.FxFrontendUnpackKwargs(frontend.AOTAutogradFrontend(dynamic=True)),
    frontend.DynamoExport(tracing_mode="symbolic", aten_graph=True),
    frontend.DynamoExport(tracing_mode="real", aten_graph=True),
    frontend.FxFrontendUnpackKwargs(frontend.DynamoOptimize(dynamic=True)),
    frontend.FxFrontendUnpackKwargs(frontend.MakeFx(tracing_mode="real")),
    frontend.FxFrontendUnpackKwargs(frontend.DynamoOptimize(dynamic=False)),
]

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


MODULES_DB = [
    common_modules.ModuleInfo(MLPModel, module_inputs_func=module_inputs_mlp_model)
]


class TestFxFrontendConsistency(common_utils.TestCase):
    @classmethod
    def create_test_base(cls, fx_frontend: _FrontendType):
        """Returns the base test method for the given fx frontend."""

        @common_modules.modules(
            MODULES_DB, train_eval_mode=common_modules.TrainEvalMode.eval_only
        )
        def _output_match_base(
            self: common_utils.TestCase,
            device: str,
            dtype: torch.dtype,
            module_info: common_modules.ModuleInfo,
            training: common_modules.TrainEvalMode,
        ):
            """Base test method for testing each fx frontend, used by instantiate_device_type_tests."""
            # device is provided by instantiate_device_type_tests, but we only want to run in cpu.
            assert device == "cpu"

            module_cls = module_info.module_cls
            module_inputs = module_info.module_inputs_func(
                module_info,
                device=device,
                dtype=dtype,
                requires_grad=False,
                training=False,
            )

            for (i, module_input) in enumerate(module_inputs):
                with self.subTest(
                    module_name=module_cls.__name__,
                    sample_num=i,
                    fx_frontend=fx_frontend.name,
                ):
                    init_args, init_kwargs = (
                        module_input.constructor_input.args,
                        module_input.constructor_input.kwargs,
                    )
                    model = module_cls(*init_args, **init_kwargs)
                    model.to(device).to(dtype)
                    model.eval()

                    args, kwargs = (
                        module_input.forward_input.args,
                        module_input.forward_input.kwargs,
                    )

                    if isinstance(fx_frontend, frontend.FxFrontendUnpackKwargs):
                        graph_module, new_args = fx_frontend(model, *args, **kwargs)
                        fx_outputs = graph_module(*new_args)
                    else:
                        graph_module = fx_frontend(model, *args, **kwargs)
                        fx_outputs = graph_module(*args, **kwargs)

                    outputs = model(*args, **kwargs)

                    self.assertEqual(
                        pytree.tree_flatten(outputs)[0],
                        pytree.tree_flatten(fx_outputs)[0],
                    )

        test_name = f"test_output_match_frontend_{fx_frontend.name}"
        _output_match_base.__name__ = test_name
        return _output_match_base

    @classmethod
    def parameterize_frontends(cls, frontends: Sequence[_FrontendType]):
        for fx_frontend in frontends:
            # Generate a test method for each frontend.
            base_method = cls.create_test_base(fx_frontend)
            # Important to rename the test method so that DecorateInfo can find it
            test_name = base_method.__name__

            # TODO(bowbao): Potentially re-use 'add_decorate_info' to have finer-grained
            # control over which modules to skip or expect to fail, based on frontend.

            setattr(cls, test_name, base_method)


TestFxFrontendConsistency.parameterize_frontends(FX_FRONTENDS)
common_device_type.instantiate_device_type_tests(
    TestFxFrontendConsistency, globals(), only_for="cpu"
)

if __name__ == "__main__":
    common_utils.run_tests()
