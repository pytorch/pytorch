# Owner(s): ["oncall: distributed"]
from collections import OrderedDict

import torch
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate
from torch.distributed.tensor.parallel._utils import _create_1d_device_mesh
from torch.distributed.tensor.parallel.api import (
    _parallelize_linear,
    _parallelize_mlp,
    parallelize_module,
)
from torch.distributed.tensor.parallel.style import (
    ColwiseParallel,
    make_input_replicate_1d,
    make_output_replicate_1d,
    PairwiseParallel,
    ParallelStyle,
    RowwiseParallel,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class MLPModule(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        torch.manual_seed(5)
        self.net1 = torch.nn.Linear(10, 16, device=device)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(16, 12, device=device)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


class TensorParallelAPITests(DTensorTestBase):
    @property
    def world_size(self):
        gpu_num = torch.cuda.device_count()
        return gpu_num if gpu_num % 2 == 0 and gpu_num > 4 else 4

    @with_comms
    def test_create_1d_device_mesh(self):
        dim_one_size = 2
        mesh_shape = (
            torch.arange(self.world_size)
            .reshape(
                self.world_size // dim_one_size,
                dim_one_size,
            )
            .to(torch.int)
        )
        mesh = DeviceMesh(self.device_type, mesh_shape)
        # When 1D dim is 1.
        one_dimention_mesh_shape = mesh_shape[self.rank // dim_one_size, :]
        pg = mesh.get_dim_groups()[1]
        new_mesh = _create_1d_device_mesh(mesh, 1)
        expected_mesh = one_dimention_mesh_shape

        self.assertEqual(new_mesh.mesh, expected_mesh)
        self.assertEqual(new_mesh.device_type, self.device_type)
        self.assertEqual(new_mesh.get_dim_groups(), [pg])
        # When 1D dim is 0.
        one_dimention_mesh_shape = mesh_shape[:, self.rank % dim_one_size]
        pg = mesh.get_dim_groups()[0]
        new_mesh = _create_1d_device_mesh(mesh, 0)
        expected_mesh = one_dimention_mesh_shape
        self.assertEqual(new_mesh.mesh, expected_mesh)
        self.assertEqual(new_mesh.device_type, self.device_type)
        self.assertEqual(new_mesh.get_dim_groups(), [pg])

    @with_comms
    def test_create_1d_device_mesh_error(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        with self.assertRaisesRegex(
            AssertionError,
            "Expect tp_mesh_dim within range \\[-1, 1\\), but found 3.",
        ):
            _create_1d_device_mesh(mesh, 3)

    def _compare_params(
        self,
        local_module,
        dist_module,
        rank0_only,
        skip_rowwise_bias=False,
        compare_grad=False,
    ):
        replicate = [Replicate()]
        for name, param in local_module.named_parameters():
            dist_param = dist_module.get_parameter(name)
            param = param.grad if compare_grad else param
            dist_param = dist_param.grad if compare_grad else dist_param
            if (
                (not rank0_only)
                or (self.rank == 0)
                or (
                    name not in ["net2.bias"]
                    and not skip_rowwise_bias
                    or name not in ["bias", "net2.bias"]
                )
            ):
                self.assertEqual(
                    param,
                    dist_param.redistribute(
                        device_mesh=dist_param.device_mesh, placements=replicate
                    ).to_local(),
                    f"{name} not equal between dist and non-dist",
                )

    def _compare_module(
        self, local_module, dist_module, inp_size, rank0_only=True, rowwise=False
    ):
        LR = 0.25  # the learning rate we use for testing
        local_optim = torch.optim.SGD(local_module.parameters(), lr=LR)
        dist_optim = torch.optim.SGD(dist_module.parameters(), lr=LR)
        torch.manual_seed(0)
        inp = torch.rand(*inp_size, device=self.device_type)
        self._compare_params(local_module, dist_module, rank0_only)

        # check forward correctness
        local_output = local_module(inp)
        inp = inp.chunk(self.world_size, dim=-1)[self.rank] if rowwise else inp
        dist_output = dist_module(inp)
        dist_output = (
            dist_output.redistribute(dist_output.device_mesh, [Replicate()]).to_local()
            if isinstance(dist_output, DTensor)
            else dist_output
        )
        self.assertEqual(local_output, dist_output)

        local_output.sum().backward()
        dist_output.sum().backward()

        # check backward and ensure gradients are same
        self._compare_params(local_module, dist_module, rank0_only, rowwise, True)

        local_optim.step()
        dist_optim.step()
        self._compare_params(local_module, dist_module, rank0_only, rowwise)

    @with_comms
    def test_parallelize_mlp(self):
        inp_size = [12, 10]
        model = MLPModule(self.device_type)
        model_tp = MLPModule(self.device_type)

        # Ensure model are initialized the same way.
        self.assertEqual(model.net1.weight, model_tp.net1.weight)
        self.assertEqual(model.net1.bias, model_tp.net1.bias)
        self.assertEqual(model.net2.weight, model_tp.net2.weight)
        self.assertEqual(model.net2.bias, model_tp.net2.bias)

        # Parallelize module.
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        model_tp = _parallelize_mlp(model_tp, device_mesh, PairwiseParallel())
        self._compare_module(model, model_tp, inp_size)

    @with_comms
    def test_parallelize_mlp_with_module_api(self):
        inp_size = [12, 10]
        model = MLPModule(self.device_type)
        model_tp = MLPModule(self.device_type)

        # Ensure model are initialized the same way.
        self.assertEqual(model.net1.weight, model_tp.net1.weight)
        self.assertEqual(model.net1.bias, model_tp.net1.bias)
        self.assertEqual(model.net2.weight, model_tp.net2.weight)
        self.assertEqual(model.net2.bias, model_tp.net2.bias)

        # Parallelize module.
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        model_tp = parallelize_module(
            model_tp,
            device_mesh,
            {
                "net1": ColwiseParallel(
                    make_input_replicate_1d, make_output_replicate_1d
                ),
                "net2": ColwiseParallel(
                    make_input_replicate_1d, make_output_replicate_1d
                ),
            },
        )
        self._compare_module(model, model_tp, inp_size, rank0_only=False)

    @with_comms
    def test_parallelize_mlp_with_module_api_nested(self):
        inp_size = [12, 10]
        model = torch.nn.Sequential(
            OrderedDict([("dummy_encoder", MLPModule(self.device_type))])
        )
        model_tp = torch.nn.Sequential(
            OrderedDict([("dummy_encoder", MLPModule(self.device_type))])
        )

        # Ensure model are initialized the same way.
        self.assertEqual(
            model.dummy_encoder.net1.weight, model_tp.dummy_encoder.net1.weight
        )
        self.assertEqual(
            model.dummy_encoder.net1.bias, model_tp.dummy_encoder.net1.bias
        )
        self.assertEqual(
            model.dummy_encoder.net2.weight, model_tp.dummy_encoder.net2.weight
        )
        self.assertEqual(
            model.dummy_encoder.net2.bias, model_tp.dummy_encoder.net2.bias
        )

        # Parallelize module.
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        model_tp = parallelize_module(
            model_tp,
            device_mesh,
            {
                "dummy_encoder.net1": ColwiseParallel(
                    make_input_replicate_1d, make_output_replicate_1d
                ),
                "dummy_encoder.net2": ColwiseParallel(
                    make_input_replicate_1d, make_output_replicate_1d
                ),
            },
        )
        self._compare_module(model, model_tp, inp_size, rank0_only=False)

    @with_comms
    def test_parallelize_mlp_error(self):
        class DummyParallel(ParallelStyle):
            def __init__(self) -> None:
                super().__init__(make_input_replicate_1d, make_output_replicate_1d)

        model_tp = MLPModule(self.device_type)
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        with self.assertRaisesRegex(
            NotImplementedError,
            "Only support PairwiseParallel for MLP parallelization.",
        ):
            _parallelize_mlp(model_tp, device_mesh, DummyParallel())

        with self.assertRaisesRegex(
            RuntimeError, "More than one nn.Linear needed for a MLP."
        ):
            _parallelize_mlp(torch.nn.Linear(10, 5), device_mesh, PairwiseParallel())

    @with_comms
    def test_linear_row_wise_parallel(self):
        # test RowwiseParallel
        inp_size = [9, 16]
        rowwise = RowwiseParallel()

        torch.manual_seed(5)
        model = torch.nn.Linear(16, 10, device=self.device_type)
        torch.manual_seed(5)
        model_tp = torch.nn.Linear(16, 10, device=self.device_type)

        # parallelize model_tp
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        model_tp = _parallelize_linear(model_tp, device_mesh, rowwise)

        # let each rank generate unique local input
        torch.manual_seed(self.rank)
        self._compare_module(model, model_tp, inp_size, rowwise=True)

    @with_comms
    def test_linear_col_wise_parallel(self):
        # test ColwiseParallel
        inp_size = [8, 10]
        colwise = ColwiseParallel(make_input_replicate_1d, make_output_replicate_1d)

        torch.manual_seed(5)
        model = torch.nn.Linear(10, 16, device=self.device_type)
        torch.manual_seed(5)
        model_tp = torch.nn.Linear(10, 16, device=self.device_type)

        # parallelize model_tp
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        model_tp = _parallelize_linear(model_tp, device_mesh, colwise)

        self._compare_module(model, model_tp, inp_size)


if __name__ == "__main__":
    run_tests()
