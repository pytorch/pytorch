# Owner(s): ["oncall: distributed"]
from collections import defaultdict
from typing import Dict

import torch
from torch.distributed._tensor.experimental.tp_transform import (
    tensor_parallel_transformation,
)
from torch.distributed.tensor.parallel.style import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
)
from torch.testing._internal.common_distributed import (
    run_with_both_funcol_impls_with_arg,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class MLPListModule(torch.nn.Module):
    """
    A dummy model with list of MLPs.
    """

    def __init__(self, num_mlps=3, bias=True):
        super().__init__()
        self.mlps = torch.nn.ModuleList()
        for _ in range(num_mlps):
            self.mlps.append(
                torch.nn.Sequential(
                    torch.nn.Linear(6, 18),
                    torch.nn.ReLU(),
                    torch.nn.Linear(18, 6, bias=bias),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.chunk(x, 2, dim=1)[0]
        for mlp in self.mlps:
            x = mlp(x)
        return x + torch.ones_like(x)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 5)
        self.bn = torch.nn.BatchNorm1d(5)

    def forward(self, x):
        return self.bn(self.fc(x))


@instantiate_parametrized_tests
class TensorParallelTest(DTensorTestBase):
    def setUp(self) -> None:
        super().setUp()

    def assert_has_c10d_ops(
        self, gm: torch.fx.GraphModule, expected_ops_count: Dict[str, int]
    ) -> None:
        actual_ops_count: Dict[str, int] = defaultdict(int)
        for node in gm.graph.nodes:
            if node.op == "call_function":
                if "c10d_functional" in str(node.target):
                    actual_ops_count[str(node.target)] += 1
        self.assertDictEqual(expected_ops_count, actual_ops_count)

    @with_comms
    @run_with_both_funcol_impls_with_arg
    def test_tp_transform_with_uncovered_op(self, use_native_funcol):
        model = DummyModel().to(device=self.device_type)
        inputs = (torch.randn(7, 3, requires_grad=False).to(device=self.device_type),)
        with torch.no_grad():
            res = model(*inputs)
        exported_program = torch.export.export(
            model,
            inputs,
        )
        tp_exported_program = tensor_parallel_transformation(
            exported_program,
            self.rank,
            self.world_size,
            self.device_type,
            {"fc": ColwiseParallel},
        )
        tp_model = tp_exported_program.module()
        with torch.no_grad():
            tp_res = tp_model(*inputs)
        self.assertEqual(res, tp_res)
        # Expect all_gather to be inserted to distributed sharded fc resutls
        if use_native_funcol:
            self.assert_has_c10d_ops(
                tp_exported_program.graph_module,
                {
                    "_c10d_functional.all_gather_into_tensor.default": 1,
                    "_c10d_functional.wait_tensor.default": 1,
                },
            )
        else:
            self.assert_has_c10d_ops(
                tp_exported_program.graph_module,
                {
                    "c10d_functional.all_gather_into_tensor.default": 1,
                    "c10d_functional.wait_tensor.default": 1,
                },
            )

    @with_comms
    @run_with_both_funcol_impls_with_arg
    def test_tp_transform_e2e(self, use_native_funcol):
        torch.manual_seed(0)
        model = MLPListModule(2).to(device=self.device_type)
        inputs = (torch.randn((10, 12)).to(device=self.device_type),)
        parallel_strategies: Dict[str, ParallelStyle] = {
            "mlps.0.0": ColwiseParallel,
            "mlps.0.2": RowwiseParallel,
            "mlps.1.0": ColwiseParallel,
            "mlps.1.2": RowwiseParallel,
        }

        with torch.inference_mode():
            res = model(*inputs)
        exported_program = torch.export.export(
            model,
            inputs,
        )
        tp_exported_program = tensor_parallel_transformation(
            exported_program,
            self.rank,
            self.world_size,
            self.device_type,
            parallel_strategies,
        )
        tp_model = tp_exported_program.module()
        with torch.inference_mode():
            tp_res = tp_model(*inputs)
        self.assertEqual(res, tp_res)
        # Expect all_reduce to be inserted at the end of each MLP
        if use_native_funcol:
            self.assert_has_c10d_ops(
                tp_exported_program.graph_module,
                {
                    "_c10d_functional.all_reduce.default": 2,
                    "_c10d_functional.wait_tensor.default": 2,
                },
            )
        else:
            self.assert_has_c10d_ops(
                tp_exported_program.graph_module,
                {
                    "c10d_functional.all_reduce.default": 2,
                    "c10d_functional.wait_tensor.default": 2,
                },
            )

    @with_comms
    @run_with_both_funcol_impls_with_arg
    def test_tp_transform_no_bias(self, use_native_funcol):
        torch.manual_seed(0)
        model = MLPListModule(1, bias=False).to(device=self.device_type)
        inputs = (torch.randn((10, 12)).to(device=self.device_type),)
        parallel_strategies: Dict[str, ParallelStyle] = {
            "mlps.0.0": ColwiseParallel,
            "mlps.0.2": RowwiseParallel,
        }

        with torch.inference_mode():
            res = model(*inputs)
        exported_program = torch.export.export(
            model,
            inputs,
        )
        tp_exported_program = tensor_parallel_transformation(
            exported_program,
            self.rank,
            self.world_size,
            self.device_type,
            parallel_strategies,
        )
        tp_model = tp_exported_program.module()
        with torch.inference_mode():
            tp_res = tp_model(*inputs)
        self.assertEqual(res, tp_res)
        if use_native_funcol:
            self.assert_has_c10d_ops(
                tp_exported_program.graph_module,
                {
                    "_c10d_functional.all_reduce.default": 1,
                    "_c10d_functional.wait_tensor.default": 1,
                },
            )
        else:
            self.assert_has_c10d_ops(
                tp_exported_program.graph_module,
                {
                    "c10d_functional.all_reduce.default": 1,
                    "c10d_functional.wait_tensor.default": 1,
                },
            )


if __name__ == "__main__":
    run_tests()
