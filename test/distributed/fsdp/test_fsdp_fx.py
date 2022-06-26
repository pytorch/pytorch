# Owner(s): ["oncall: distributed"]

import torch
from torch.testing._internal.common_fsdp import FSDPTest
from torch.distributed.fsdp.symbolic_trace import _patch_tracer
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
)


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight1 = torch.nn.Parameter(torch.randn(6, 6))
        self.weight2 = torch.nn.Parameter(torch.randn(6, 6))
        self.layer0 = torch.nn.Linear(6, 6)
        self.layer1 = torch.nn.Linear(6, 6, bias=False)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(6, 3, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 6, bias=False),
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        z = self.relu(self.layer0(x))
        z = self.relu(self.layer2(z))
        z = z @ self.weight1
        z = self.relu(self.layer1(z))
        z = z @ self.weight2
        # used to test the case where a module is called more than once
        z = self.relu(self.layer0(x))
        return z


class TestSymbolicTracing(FSDPTest):
    def test_symbolic_tracing_outputs(self):
        model = Model()
        tracer = torch.fx.Tracer()
        execution_info = _patch_tracer(tracer, model)
        tracer.trace(model)
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
            model.relu
        ]
        self.assertEqual(
            execution_info.module_forward_order,
            correct_module_forward_order
        )
        # test execution_info.module_execution_info_dict
        self.assertEqual(
            execution_info.module_execution_info_dict[model],
            [
                (model.layer0, list(model.layer0.named_parameters())),
                (model.layer2, list(model.layer2.named_parameters())),
                (model, [("weight1", model.weight1)]),
                (model.layer1, list(model.layer1.named_parameters())),
                (model, [("weight2", model.weight2)]),
                (model.layer0, list(model.layer0.named_parameters())),
            ]
        )
        self.assertEqual(
            execution_info.module_execution_info_dict[model.layer0],
            [(model.layer0, list(model.layer0.named_parameters()))],
        )
        self.assertEqual(
            execution_info.module_execution_info_dict[model.layer1],
            [(model.layer1, list(model.layer1.named_parameters()))],
        )
        self.assertEqual(
            execution_info.module_execution_info_dict[model.layer2],
            [
                (model.layer2[0], list(model.layer2[0].named_parameters())),
                (model.layer2[2], list(model.layer2[2].named_parameters())),
            ]
        )
        self.assertEqual(execution_info.module_execution_info_dict[model.relu], [])
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
