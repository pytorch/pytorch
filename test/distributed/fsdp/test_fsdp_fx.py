# Owner(s): ["oncall: distributed"]

from typing import Any

import torch
from torch.distributed.fsdp._symbolic_trace import _init_execution_info, _patch_tracer
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
)


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight1 = torch.nn.Parameter(torch.randn(6, 6))
        self.weight2 = torch.nn.Parameter(torch.randn(6, 6))
        self.weight_unused = torch.nn.Parameter(torch.randn(2, 2))
        self.layer0 = torch.nn.Linear(6, 6)
        self.layer1 = torch.nn.Linear(6, 6, bias=False)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(6, 3, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 6, bias=False),
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x: Any, run_all_layers: bool):
        z = self.relu(self.layer0(x))
        z = self.relu(self.layer2(z))
        z = z @ self.weight1
        if run_all_layers:
            z = self.relu(self.layer1(z))
            z = z @ self.weight2
            # used to test the case where a module is called more than once
            z = self.relu(self.layer0(x))
        return z


class TestSymbolicTracing(FSDPTest):
    def test_symbolic_tracing_outputs(self):
        """
        test ``execution_info.module_forward_order`` and ``execution_info.module_to_execution_infos``
        after running ``tracer.trace()`` inside ``_patch_tracer``.
        """
        model = Model()
        tracer = torch.fx.Tracer()
        execution_info = _init_execution_info(model)
        original_call_module = tracer.call_module
        original_create_proxy = tracer.create_proxy
        with _patch_tracer(
            tracer=tracer, root_module=model, execution_info=execution_info
        ):
            concrete_args = {"run_all_layers": True}
            tracer.trace(model, concrete_args)
        # the member functions of tracer should not be changed
        self.assertEqual(original_call_module, tracer.call_module)
        self.assertEqual(original_create_proxy, tracer.create_proxy)
        # test tracer.module_forward_order
        correct_module_forward_order = [
            model,
            model.layer0,
            model.relu,
            model.layer2,
            model.layer2[0],
            model.layer2[1],
            model.layer2[2],
            model.relu,
            model.layer1,
            model.relu,
            model.layer0,
            model.relu,
        ]
        self.assertEqual(
            execution_info.module_forward_order, correct_module_forward_order
        )
        # test execution_info.module_to_execution_infos
        self.assertEqual(
            execution_info.module_to_execution_infos[model],
            [
                (model.layer0, list(model.layer0.named_parameters())),
                (model.layer2, list(model.layer2.named_parameters())),
                (model, [("weight1", model.weight1)]),
                (model.layer1, list(model.layer1.named_parameters())),
                (model, [("weight2", model.weight2)]),
                (model.layer0, list(model.layer0.named_parameters())),
            ],
        )
        self.assertEqual(
            execution_info.module_to_execution_infos[model.layer0],
            [(model.layer0, list(model.layer0.named_parameters()))],
        )
        self.assertEqual(
            execution_info.module_to_execution_infos[model.layer1],
            [(model.layer1, list(model.layer1.named_parameters()))],
        )
        self.assertEqual(
            execution_info.module_to_execution_infos[model.layer2],
            [
                (model.layer2[0], list(model.layer2[0].named_parameters())),
                (model.layer2[2], list(model.layer2[2].named_parameters())),
            ],
        )
        self.assertEqual(execution_info.module_to_execution_infos[model.relu], [])
        # test tracer.param_exec_order
        correct_param_order = [
            model.layer0.weight,
            model.layer0.bias,
            model.layer2[0].weight,
            model.layer2[2].weight,
            model.weight1,
            model.layer1.weight,
            model.weight2,
        ]
        self.assertEqual(execution_info.param_exec_order, correct_param_order)


instantiate_parametrized_tests(TestSymbolicTracing)

if __name__ == "__main__":
    run_tests()
