# Owner(s): ["oncall: distributed"]
from collections import OrderedDict
from copy import deepcopy

import torch
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate, Shard
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.parallel.api import parallelize_module
from torch.distributed.tensor.parallel.style import (
    ColwiseParallel,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    PrepareModuleOutput,
    RowwiseParallel,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorTestBase,
    map_local_tensor_for_rank,
    MLPModule,
    MLPStacked,
    with_comms,
)


class DummyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x


class TensorParallelAPITests(DTensorTestBase):
    @property
    def world_size(self):
        gpu_num = torch.accelerator.device_count()
        return gpu_num if gpu_num % 2 == 0 and gpu_num > 4 else 4

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
                    name != "net2.bias"
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
        inp = map_local_tensor_for_rank(
            inp,
            self.rank,
            lambda inp, rank: inp.chunk(self.world_size, dim=-1)[rank]
            if rowwise
            else inp,
        )
        # inp = inp.chunk(self.world_size, dim=-1)[self.rank] if rowwise else inp
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
    def test_parallelize_mlp_with_module_api(self):
        inp_size = [12, 10]
        model = MLPModule(self.device_type)
        model_tp = deepcopy(model)

        # Parallelize module.
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        model_tp = parallelize_module(
            model_tp,
            device_mesh,
            {
                "net1": ColwiseParallel(output_layouts=Replicate()),
                "net2": ColwiseParallel(output_layouts=Replicate()),
            },
        )
        self._compare_module(model, model_tp, inp_size, rank0_only=False)

    @with_comms
    def test_parallelize_mlp_with_module_api_nested(self):
        inp_size = [12, 10]
        model = torch.nn.Sequential(
            OrderedDict([("dummy_encoder", MLPModule(self.device_type))])
        )
        model_tp = deepcopy(model)

        # Parallelize module.
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        model_tp = parallelize_module(
            model_tp,
            device_mesh,
            {
                "dummy_encoder.net1": ColwiseParallel(output_layouts=Replicate()),
                "dummy_encoder.net2": ColwiseParallel(output_layouts=Replicate()),
            },
        )
        self._compare_module(model, model_tp, inp_size, rank0_only=False)

    @with_comms
    def test_linear_row_wise_parallel(self):
        # test RowwiseParallel
        inp_size = [9, 16]
        rowwise = RowwiseParallel()

        torch.manual_seed(5)
        model = torch.nn.Linear(16, 10, device=self.device_type)
        model_tp = deepcopy(model)

        # parallelize model_tp
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        model_tp = parallelize_module(model_tp, device_mesh, rowwise)

        # let each rank generate unique local input
        torch.manual_seed(self.rank)
        self._compare_module(model, model_tp, inp_size, rowwise=True)

    @with_comms
    def test_linear_col_wise_parallel(self):
        # test ColwiseParallel
        inp_size = [8, 10]
        colwise = ColwiseParallel(output_layouts=Replicate())

        torch.manual_seed(5)
        model = torch.nn.Linear(10, 16, device=self.device_type)
        model_tp = deepcopy(model)

        # parallelize model_tp
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        model_tp = parallelize_module(model_tp, device_mesh, colwise)

        self._compare_module(model, model_tp, inp_size)

    @with_comms
    def test_prepare_module_input(self):
        module = DummyModule()
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        parallelize_module(
            module,
            device_mesh,
            PrepareModuleInput(
                input_layouts=Shard(0), desired_input_layouts=Replicate()
            ),
        )
        inp = torch.rand(5, 7, device=self.device_type)
        output = module(inp).redistribute(device_mesh, [Shard(0)]).to_local()
        self.assertEqual(inp, output)

    @with_comms
    def test_prepare_module_output(self):
        module = DummyModule()
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        parallelize_module(
            module,
            device_mesh,
            PrepareModuleOutput(
                output_layouts=Replicate(), desired_output_layouts=Shard(0)
            ),
        )
        torch.manual_seed(15)
        inp = torch.rand(16, 7, device=self.device_type)
        dtensor = DTensor.from_local(inp, device_mesh, [Replicate()], run_check=False)
        output = module(dtensor)
        inp = dtensor.redistribute(device_mesh, [Shard(0)]).to_local()
        self.assertEqual(inp, output)

    @with_comms
    def test_prepare_module_input_output(self):
        module = DummyModule()
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        parallelize_module(
            module,
            device_mesh,
            PrepareModuleInputOutput(
                input_layouts=Shard(0),
                desired_input_layouts=Replicate(),
                output_layouts=Replicate(),
                desired_output_layouts=Shard(1),
            ),
        )
        inp = torch.rand(5, 7, device=self.device_type)
        output = module(inp)
        inp = (
            DTensor.from_local(inp, device_mesh, [Shard(0)], run_check=False)
            .redistribute(device_mesh, [Shard(1)])
            .to_local()
        )
        self.assertEqual(inp, output)

    @with_comms
    def test_parallelize_module_with_star(self):
        inp_size = [12, 10]
        model = MLPModule(self.device_type)
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        model_tp = deepcopy(model)
        model_tp = parallelize_module(
            model_tp,
            device_mesh,
            {
                "net*": ColwiseParallel(output_layouts=Replicate()),
            },
        )
        self._compare_module(model, model_tp, inp_size, rank0_only=False)

    @with_comms
    def test_parallelize_module_src_data_rank(self):
        # set seed different for each rank
        torch.manual_seed(self.rank)
        model = MLPModule(self.device_type)
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        comm_mode = CommDebugMode()

        # test src_data_rank == 1
        with comm_mode:
            model_tp = deepcopy(model)
            model_tp = parallelize_module(
                model_tp,
                device_mesh,
                {
                    "net*": ColwiseParallel(output_layouts=Replicate()),
                },
                src_data_rank=1,
            )

        self.assertTrue(comm_mode.get_total_counts() > 0)
        tp_full_params = [param.full_tensor() for param in model_tp.parameters()]
        if self.rank == 1:
            orig_model_params = list(model.parameters())
            for idx, param in enumerate(tp_full_params):
                self.assertEqual(param, orig_model_params[idx])

        # test src_data_rank == None
        model_tp_no_comm = deepcopy(model)
        with comm_mode:
            parallelize_module(
                model_tp_no_comm,
                device_mesh,
                {
                    "net1": ColwiseParallel(),
                    "net2": RowwiseParallel(),
                },
                src_data_rank=None,
            )
            self.assertEqual(comm_mode.get_total_counts(), 0)

    @with_comms
    def test_parallelize_module_with_question(self):
        inp_size = [12, 10]
        model = MLPModule(self.device_type)
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        model_tp = deepcopy(model)
        model_tp = parallelize_module(
            model_tp,
            device_mesh,
            {
                "net?": ColwiseParallel(output_layouts=Replicate()),
            },
        )
        self._compare_module(model, model_tp, inp_size, rank0_only=False)

    @with_comms
    def test_parallelize_module_with_digit(self):
        inp_size = [12, 10]
        model = MLPModule(self.device_type)
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        model_tp = deepcopy(model)
        model_tp = parallelize_module(
            model_tp,
            device_mesh,
            {
                "net[1-2]": ColwiseParallel(output_layouts=Replicate()),
            },
        )
        self._compare_module(model, model_tp, inp_size, rank0_only=False)

    @with_comms
    def test_parallelize_module_multi_wildcard(self):
        inp_size = [12, 10]
        model = MLPStacked(self.device_type, n_layers=2)
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        model_tp = deepcopy(model)
        model_tp = parallelize_module(
            model_tp,
            device_mesh,
            {
                "layers.*.net[1]": ColwiseParallel(),
                "layers.*.net[2]": RowwiseParallel(),
            },
        )
        self._compare_module(model, model_tp, inp_size, rank0_only=False)

    @with_comms
    def test_parallelize_module_with_root_module(self):
        inp_size = [16, 10]
        model = MLPModule(self.device_type)
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        model_tp = deepcopy(model)
        model_tp = parallelize_module(
            model_tp,
            device_mesh,
            {
                "": PrepareModuleInputOutput(
                    input_layouts=Replicate(),
                    desired_input_layouts=Shard(0),
                    output_layouts=Shard(0),
                    desired_output_layouts=Replicate(),
                ),
                "net1": ColwiseParallel(input_layouts=Shard(0)),
                "net2": RowwiseParallel(output_layouts=Shard(0)),
            },
        )
        self._compare_module(model, model_tp, inp_size, rank0_only=False)

    @with_comms
    def test_parallelize_module_with_no_match(self):
        inp_size = [16, 10]
        model = MLPModule(self.device_type)
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        model_tp = deepcopy(model)
        with self.assertWarns(UserWarning):
            model_tp = parallelize_module(
                model_tp,
                device_mesh,
                {
                    "net0.hello.world": ColwiseParallel(),
                    "net1": ColwiseParallel(),
                    "net2": RowwiseParallel(),
                    "net3": ColwiseParallel(),
                },
            )
        self._compare_module(model, model_tp, inp_size, rank0_only=False)

    @with_comms
    def test_under_devicemesh_context(self):
        # test ColwiseParallel
        inp_size = [8, 10]
        colwise = ColwiseParallel(output_layouts=Replicate())

        torch.manual_seed(5)
        model = torch.nn.Linear(10, 16, device=self.device_type)
        model_tp = deepcopy(model)

        # Call parallelize_module under DeviceMesh context.
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        with device_mesh:
            model_tp = parallelize_module(model_tp, parallelize_plan=colwise)

        self._compare_module(model, model_tp, inp_size)

    @with_comms
    def test_empty_plan(self):
        torch.manual_seed(5)
        model = torch.nn.Linear(10, 16, device=self.device_type)

        # Call parallelize_module with empty plan.
        # Goal is not to crash.
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        with self.assertWarns(UserWarning):
            parallelize_module(model, device_mesh)


TensorParallelAPITestsWithLocalTensor = create_local_tensor_test_class(
    TensorParallelAPITests,
    skipped_tests=[
        # Uses mesh_scatter that has local rank dependent logic
        "test_parallelize_module_src_data_rank",
    ],
)


if __name__ == "__main__":
    run_tests()
