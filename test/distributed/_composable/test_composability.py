# Owner(s): ["oncall: distributed"]

import copy
import functools
from typing import List, Type
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed._composable import checkpoint, replicate
from torch.distributed._composable.fsdp import CPUOffloadPolicy, fully_shard
from torch.distributed._tensor import DTensor, init_device_mesh
from torch.distributed._tensor.debug.comm_mode import CommDebugMode
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
)
from torch.distributed.device_mesh import DeviceMesh
import io
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn as nn

import torch.nn.functional as F
from torch.distributed._tensor import (
    DeviceMesh,
    DTensor,
    DTensor as DT,
    init_device_mesh,
    Replicate,
    Shard,
)
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._common_utils import (
    _get_module_fsdp_state,
    clean_tensor_name,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.distributed.tensor.parallel.ddp import _pre_dp_module_transform
from torch.distributed.tensor.parallel.fsdp import DTensorExtensions
from torch.distributed.tensor.parallel.input_reshard import input_reshard

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    check_sharded_parity,
    FSDPTest,
    MLP,
    MLPStack,
)

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfRocm,
)

from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    MLPModule,
    ModelArgs,
    Transformer,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir
import copy
import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp.fully_shard import (
    fully_shard,
    MixedPrecisionPolicy,
)
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    PipelineScheduleSingle,
    Schedule1F1B,
    ScheduleGPipe,
)
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_distributed import (
    MultiProcContinousTest,
    requires_nccl,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skip_but_pass_in_sandcastle_if,
)


# Tensor-Parallel degree
TP_DEGREE = 2
LR = 3e-5

c10d_ops = torch.ops.c10d
funcol = torch.ops.c10d_functional

def init_model(device_type, model_parallel_size=TP_DEGREE):
    torch.manual_seed(0)
    model = MLPModule(device_type)
    torch.manual_seed(0)
    twod_model = MLPModule(device_type)
    model = DDP(model)

    # 2-D mesh is [dp, tp]
    world_size = dist.get_world_size()
    twod_mesh = DeviceMesh(
        device_type=device_type,
        mesh=torch.arange(0, world_size).view(-1, model_parallel_size),
    )
    mesh_2d = init_device_mesh(
        device_type,
        (world_size // model_parallel_size, model_parallel_size),
        mesh_dim_names=("dp", "tp"),
    )

    dp_pg = mesh_2d.get_group(mesh_dim=0)

    parallelize_plan = {
        "net1": ColwiseParallel(),
        "net2": RowwiseParallel(),
    }
    twod_model = parallelize_module(twod_model, mesh_2d["tp"], parallelize_plan)
    _pre_dp_module_transform(twod_model)
    # TODO: Add tests when using gradient_as_bucket_view and static_graph for DDP.
    twod_model = DDP(twod_model, process_group=dp_pg)
    return model, twod_model, dp_pg

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(5, 8)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(8, 4)
        self.net3 = nn.Linear(4, 12)

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        return x

    def get_input(self):
        return torch.rand(4, 5, device="cuda")

class SimpleModelUneven(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Linear(5, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 15)
        self.net3 = nn.Linear(15, 30)
        self.net4 = nn.Linear(30, 5)

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        x = self.net4(x)
        return x

    def get_input(self):
        return torch.rand(4, 5, device="cuda")

# MLP Layer
class MLPLayerModule(torch.nn.Module):
    def __init__(self, d_hid: int):
        super().__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        return x


class TestFullyShard2DTraining(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    def init_global_mesh(self) -> DeviceMesh:
        # Prefer to test with >=4 GPUs, but for 2 GPUs, use 2-way TP
        dp_size = 2 if self.world_size > 2 else 1
        return init_device_mesh(
            "cuda", (dp_size, self.world_size // dp_size), mesh_dim_names=("dp", "tp")
        )

    @skip_if_lt_x_gpu(2)
    @skipIfRocm
    def test_train_parity_2d_mlp(self):
        global_mesh = self.init_global_mesh()
        self.run_subtests(
            {
                "reshard_after_forward": [False, True],
                "use_activation_checkpointing": [False, True],
                "mlp_dim": [3, 16, 17],
            },
            functools.partial(self._test_train_parity_2d_mlp, global_mesh),
        )

    def _test_train_parity_2d_mlp(
        self,
        global_mesh: DeviceMesh,
        reshard_after_forward: bool,
        use_activation_checkpointing: bool,
        mlp_dim: int,
    ):
        dp_mesh, tp_mesh = global_mesh["dp"], global_mesh["tp"]
        dp_pg = dp_mesh.get_group()  # used for `replicate()`

        torch.manual_seed(42)
        model = MLPStack(mlp_dim)
        ref_model = copy.deepcopy(model).cuda()
        replicate(ref_model, device_ids=[self.rank], process_group=dp_pg)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2, foreach=False)
        model.parallelize(
            tp_mesh,
            dp_mesh,
            use_activation_checkpointing,
            reshard_after_forward=reshard_after_forward,
        )
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=False)

        torch.manual_seed(42 + dp_pg.rank() + 1)
        device = torch.device("cuda")
        for iter_idx in range(10):
            inp = torch.randn((8, mlp_dim), device=device)
            losses: List[torch.Tensor] = []
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                losses.append(_model(inp).sum())
                losses[-1].backward()
                _optim.step()
            self.assertEqual(losses[0], losses[1])

    @skip_if_lt_x_gpu(2)
    @skipIfRocm
    def test_tp_with_fsdp_offloading(self):
        global_mesh = init_device_mesh(
            "cuda", (1, self.world_size), mesh_dim_names=("dp", "tp")
        )
        dp_mesh, tp_mesh = global_mesh["dp"], global_mesh["tp"]
        torch.manual_seed(42)
        mlp_dim = 16
        model = MLPStack(mlp_dim)
        ref_model = copy.deepcopy(model).cuda()
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2, foreach=False)
        # Parallelize with N-way TP and 1-way FSDP
        model.parallelize(
            tp_mesh,
            dp_mesh,
            use_activation_checkpointing=False,
            reshard_after_forward=True,
            offload_policy=CPUOffloadPolicy(),
        )
        for param in model.parameters():
            self.assertEqual(param.device.type, "cpu")
        num_mlps = sum(isinstance(module, MLP) for module in model.modules())
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=False)

        # NOTE: We still see the FSDP all-gather/reduce-scatter c10d ops
        # called, but they will just be no-ops without issuing any kernels.
        # We prefer to keep the no-op check at the c10d level, not in FSDP.
        inp = torch.randn((4, mlp_dim), device="cuda")  # same on all ranks
        for iter_idx in range(10):
            ref_optim.zero_grad()
            optim.zero_grad()

            with CommDebugMode() as fwd_comm_mode:
                loss = model(inp).sum()

            fwd_comm_counts = fwd_comm_mode.get_comm_counts()
            self.assertEqual(len(fwd_comm_counts), 2)
            self.assertEqual(fwd_comm_counts[funcol.all_reduce], num_mlps)
            self.assertEqual(fwd_comm_counts[c10d_ops._allgather_base_], num_mlps)
            ref_loss = ref_model(inp).sum()
            self.assertEqual(loss, ref_loss)

            with CommDebugMode() as bwd_comm_mode:
                loss.backward()
            bwd_comm_counts = bwd_comm_mode.get_comm_counts()
            self.assertEqual(len(bwd_comm_counts), 3)
            # First MLP's input gradient does not need to be all-reduced
            self.assertEqual(bwd_comm_counts[funcol.all_reduce], num_mlps - 1)
            self.assertEqual(bwd_comm_counts[c10d_ops._allgather_base_], num_mlps)
            self.assertEqual(bwd_comm_counts[c10d_ops._reduce_scatter_base_], num_mlps)
            ref_loss.backward()

            optim.step()
            ref_optim.step()

    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_train_parity_2d_transformer_checkpoint_resume(self):
        """
        Tests train parity of a 2D transformer without checkpointing against a
        2D transformer with a checkpoint save/load.
        """
        self.run_subtests(
            {
                "use_seq_parallel": [False, True],
                # If reusing, then load into the same model/optimizer instance
                # else construct new ones (requiring eager optim state init)
                "reuse_model_optim": [False, True],
                "optimizer_class": [torch.optim.Adam, torch.optim.AdamW],
                # TODO: need to update `parallelize` before including foreach=True for testing
                "foreach": [False],
            },
            self._test_train_parity_2d_transformer_checkpoint_resume,
        )

    def _test_train_parity_2d_transformer_checkpoint_resume(
        self,
        use_seq_parallel: bool,
        reuse_model_optim: bool,
        optimizer_class: Type[torch.optim.Optimizer],
        foreach: bool,
    ):
        def train_step(
            _model: nn.Module, _optim: torch.optim.Optimizer, _inp: torch.Tensor
        ) -> torch.Tensor:
            loss = _model(_inp).sum()
            loss.backward()
            _optim.step()
            _optim.zero_grad()
            return loss

        def parallelize(_model: Transformer, mesh: DeviceMesh, use_seq_parallel: bool):
            _model = Transformer.parallelize(_model, mesh["tp"], use_seq_parallel)
            for layer in _model.layers:
                fully_shard(layer, mesh=mesh["dp"])
            fully_shard(_model, mesh=mesh["dp"])
            return _model

        global_mesh = self.init_global_mesh()
        # Baseline: run two iterations without checkpointing
        seed = 42
        torch.manual_seed(seed)
        model_args = ModelArgs(dropout_p=0.0)
        model_no_cp = parallelize(
            Transformer(model_args), global_mesh, use_seq_parallel
        )
        optim_no_cp = optimizer_class(
            model_no_cp.parameters(), lr=1e-2, foreach=foreach
        )

        torch.manual_seed(42 + global_mesh["dp"].get_local_rank() + 1)
        inp = torch.randint(0, model_args.vocab_size, (3, 16), device="cuda")
        loss_no_cp1 = train_step(model_no_cp, optim_no_cp, inp)
        loss_no_cp2 = train_step(model_no_cp, optim_no_cp, inp)

        # Test: run one iteration, save checkpoint, zero states or init new
        # model/optimizer, load checkpoint, and run another iteration
        torch.manual_seed(seed)
        model_cp = parallelize(Transformer(model_args), global_mesh, use_seq_parallel)
        optim_cp = optimizer_class(model_cp.parameters(), lr=1e-2, foreach=foreach)

        loss_cp1 = train_step(model_cp, optim_cp, inp)
        self.assertEqual(loss_no_cp1, loss_cp1)

        sharded_sd = {
            "model": get_model_state_dict(model_cp),
            # Use `get_optimizer_state_dict` to handle eager optim state init
            # when constructing a new optimizer instance
            "optim": get_optimizer_state_dict(model_cp, optim_cp),
        }
        dcp.save(
            state_dict=sharded_sd,
            storage_writer=dcp.FileSystemWriter(self.temp_dir),
        )
        if reuse_model_optim:
            with torch.no_grad():
                for param in model_cp.parameters():
                    param.zero_()
                optim_sd = optim_cp.state_dict()
                for param_states in optim_sd["state"].values():
                    for state_value in param_states.values():
                        if torch.is_tensor(state_value):
                            state_value.zero_()
        else:
            torch.manual_seed(seed + 1)  # different seed
            model_cp = parallelize(
                Transformer(model_args), global_mesh, use_seq_parallel
            )
            optim_cp = optimizer_class(model_cp.parameters(), lr=1e-2, foreach=foreach)
        self.assertNotEqual(loss_no_cp2, train_step(model_cp, optim_cp, inp))

        sharded_sd = {
            "model": get_model_state_dict(model_cp),
            "optim": get_optimizer_state_dict(model_cp, optim_cp),
        }
        dcp.load(
            state_dict=sharded_sd,
            storage_reader=dcp.FileSystemReader(self.temp_dir),
        )
        self.assertGreater(len(optim_cp.state_dict()["state"]), 0)

        loss_cp2 = train_step(model_cp, optim_cp, inp)
        self.assertEqual(loss_no_cp2, loss_cp2)

class TestFullyShardHSDPTraining(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    def test_train_parity_hsdp(self):
        shard_size = 2 if self.world_size > 2 else 1
        replicate_size = self.world_size // shard_size
        global_mesh = init_device_mesh(
            "cuda", (replicate_size, shard_size), mesh_dim_names=("replicate", "shard")
        )
        self.run_subtests(
            {
                "reshard_after_forward": [False, True],
                "use_activation_checkpointing": [False, True],
                "mlp_dim": [3, 16, 17],
                "sync_gradients_at_last_batch": [True, False],
            },
            functools.partial(self._test_train_parity_hsdp, global_mesh),
        )

    def _test_train_parity_hsdp(
        self,
        global_mesh: DeviceMesh,
        reshard_after_forward: bool,
        use_activation_checkpointing: bool,
        mlp_dim: int,
        sync_gradients_at_last_batch: bool,
    ):
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.LayerNorm(mlp_dim, bias=False),
            MLP(mlp_dim, dim_multiplier=3),
            MLP(mlp_dim),
            MLP(mlp_dim, dim_multiplier=3),
        )
        ref_model = copy.deepcopy(model).cuda()
        replicate(ref_model, device_ids=[self.rank])
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        for mlp in model:
            if use_activation_checkpointing:
                checkpoint(mlp)
            fully_shard(
                mlp, mesh=global_mesh, reshard_after_forward=reshard_after_forward
            )
        fully_shard(
            model, mesh=global_mesh, reshard_after_forward=reshard_after_forward
        )
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        check_sharded_parity(self, ref_model, model)
        torch.manual_seed(42 + self.rank + 1)
        device = torch.device("cuda")
        num_microbatches = 3
        for iter_idx in range(5):
            for microbatch_idx in range(num_microbatches):
                is_last_microbatch = microbatch_idx == num_microbatches - 1
                if sync_gradients_at_last_batch:
                    model.set_requires_gradient_sync(is_last_microbatch)
                inp = torch.randn((8, mlp_dim), device=device)
                losses: List[torch.Tensor] = []
                for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                    losses.append(_model(inp).sum())
                    losses[-1].backward()
                self.assertEqual(losses[0], losses[1])
            check_sharded_parity(self, ref_model, model)
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.step()
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            check_sharded_parity(self, ref_model, model)


class Test2dParallelIntegration(DTensorTestBase):
    def _check_module(self, m1, m2, check_grad=False):
        named_parameters = dict(m1.named_parameters())
        for name, param_m2 in m2.named_parameters():
            if name not in named_parameters:
                print(name, named_parameters.keys())
            self.assertTrue(name in named_parameters)
            param_m1 = named_parameters[name]
            if check_grad:
                param_m2 = param_m2.grad
                param_m1 = param_m1.grad
            if isinstance(param_m2, DTensor):
                replicate = [Replicate()]
                param_m2 = param_m2.redistribute(
                    device_mesh=param_m2.device_mesh, placements=replicate
                ).to_local()
            self.assertEqual(param_m2, param_m1)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_ddp_integration_functionality(self) -> None:
        model, twod_model, dp_pg = init_model(self.device_type)
        optim = torch.optim.Adam(model.parameters(), lr=LR)
        twod_optim = torch.optim.Adam(twod_model.parameters(), lr=LR)

        # Create Input
        input_seed = dist.get_rank(dp_pg)
        torch.manual_seed(input_seed + 1)
        input = torch.rand(4, 10, device=self.device_type)

        output = model(input)
        twod_output = twod_model(input)
        self.assertEqual(output, twod_output)

        output.sum().backward()
        twod_output.sum().backward()
        self._check_module(model, twod_model, check_grad=True)

        optim.step()
        twod_optim.step()
        self._check_module(model, twod_model)

        torch.manual_seed(input_seed + 1004)
        input = torch.rand(16, 10, device=self.device_type)

        output = model(input)
        twod_output = twod_model(input)
        self.assertEqual(output, twod_output)

        # TODO: Add save/load of 2D verification.

# TODO: add additional tests for multi_param_group, optim_in_backward,
# and fsdp_nested.
class TestNew2dParallelTraining(DTensorTestBase):
    def _compare_params(self, m1, m2):
        with FSDP.summon_full_params(m1):
            with FSDP.summon_full_params(m2):
                for n_p1, n_p2 in zip(m1.named_parameters(), m2.named_parameters()):
                    p1 = n_p1[1]
                    p2 = n_p2[1]
                    if n_p1[0] != n_p2[0]:
                        self.assertTrue(n_p1[0] in n_p2[0])
                    name = n_p1[0]
                    if name == "net2.bias" and self.rank != 0:
                        continue
                    if type(p2) is DT:
                        p2 = p2.redistribute(p2.device_mesh, [Replicate()]).to_local()
                    self.assertTrue(torch.allclose(p1, p2), f"{p1} vs {p2}")

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_raise_invalid_tp_composition(self):
        with self.assertRaisesRegex(
            RuntimeError, r"Found TP device_mesh on the \d dimension of its parent mesh"
        ):
            mesh_2d = init_device_mesh(
                self.device_type, (2, self.world_size // 2), mesh_dim_names=("tp", "dp")
            )
            parallelize_plan = {
                "net1": ColwiseParallel(),
                "net2": RowwiseParallel(),
            }
            model_2d = parallelize_module(
                SimpleModel().cuda(), mesh_2d["tp"], parallelize_plan
            )

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_fsdp_state_enable_extension(self):
        mesh_2d = init_device_mesh(
            self.device_type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
        )
        model = FSDP(
            SimpleModel().cuda(),
            device_mesh=mesh_2d["dp"],
        )
        fsdp_state = _get_module_fsdp_state(model)
        self.assertTrue(isinstance(fsdp_state._fsdp_extension, DTensorExtensions))

    def _test_2d_e2e_training(
        self,
        use_orig_params=False,
        recompute_activation=False,
    ) -> None:
        torch.manual_seed(0)
        model = SimpleModel().cuda(self.rank)
        model = FSDP(model, use_orig_params=use_orig_params)
        optim = torch.optim.Adam(model.parameters(), lr=0.01)

        torch.manual_seed(0)
        mesh_2d = init_device_mesh(
            self.device_type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
        )
        tp_mesh = mesh_2d["tp"]
        dp_mesh = mesh_2d["dp"]
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }
        model_2d = parallelize_module(SimpleModel().cuda(), tp_mesh, parallelize_plan)
        model_2d = FSDP(
            model_2d,
            device_mesh=dp_mesh,
            use_orig_params=use_orig_params,
        )
        optim_2d = torch.optim.Adam(model_2d.parameters(), lr=0.01)

        if recompute_activation:
            model_2d = input_reshard(model_2d, mesh_2d["tp"], 0)

        # Check named parameters are returning the same name at least.
        param_names_2d = [
            clean_tensor_name(name) for name, _ in model_2d.named_parameters()
        ]
        for name, _ in model.named_parameters():
            name = clean_tensor_name(name)
            if name not in param_names_2d:
                print(name, param_names_2d)
            self.assertTrue(name in param_names_2d)
        self._compare_params(model, model_2d)

        # TODO: add additional tests for multi_param_group and optim_in_backward.

        for i in range(5):
            # Ensure all input across TP ranks are same.
            # TODO: add a get_group_rank() to DeviceMesh.
            torch.manual_seed(i + dist.get_rank(dp_mesh.get_group(mesh_dim=0)))
            input = torch.rand(4, 5).cuda(self.rank)
            output = model(input)
            output_2d = model_2d(input)
            self.assertEqual(output, output_2d)
            output.sum().backward()
            output_2d.sum().backward()
            optim.step()
            optim_2d.step()
            self.assertEqual(model(input), model_2d(input))

        # Ensure all params are still the same after optimizer update.
        self._compare_params(model, model_2d)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_e2e_training_default(self):
        self._test_2d_e2e_training()

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_e2e_training_use_orig_params(self):
        self._test_2d_e2e_training(use_orig_params=True)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_e2e_training_not_use_orig_params(self):
        # TODO: need to revisit input_reshard API about why it failed multi-gpu tests.
        # self._test_2d_e2e_training(recompute_activation=True)
        self._test_2d_e2e_training(recompute_activation=False)


# TODO: update all state dict unit tests to use distributed.checkpoint.state_dict,
# and consolidate all the state_dict test in test.distributed.checkpoint.
class TestNew2dParallelStateDict(DTensorTestBase):
    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_fsdp_2d_extension(self):
        """
        Test whether _fsdp_extension from FSDPstate has been set correctly.
        """
        mesh_2d = init_device_mesh(
            self.device_type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
        )
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
            "net3": ColwiseParallel(),
        }
        model_2d = parallelize_module(
            SimpleModel().cuda(),
            mesh_2d["tp"],
            parallelize_plan=parallelize_plan,
        )
        model_2d = FSDP(model_2d, device_mesh=mesh_2d["dp"], use_orig_params=True)
        model_2d_fsdp_state = _get_module_fsdp_state(model_2d)
        self.assertTrue(
            isinstance(model_2d_fsdp_state._fsdp_extension, DTensorExtensions)
        )

        mesh_1d = init_device_mesh("cuda", (self.world_size,))
        model_1d = FSDP(SimpleModel().cuda(), device_mesh=mesh_1d, use_orig_params=True)
        model_1d_fsdp_state = _get_module_fsdp_state(model_1d)
        self.assertEqual(model_1d_fsdp_state._fsdp_extension, None)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @parametrize("is_even_sharded_model", [True, False])
    def test_2d_state_dict(self, is_even_sharded_model):
        simple_model = SimpleModel if is_even_sharded_model else SimpleModelUneven

        # Create a model without wrapper
        torch.manual_seed(0)
        no_wrap_model = simple_model().cuda(self.rank)
        no_wrap_state_dict = no_wrap_model.state_dict()

        # Create a model and sharded it with 2D FSDP + TP
        torch.manual_seed(0)
        mesh_2d = init_device_mesh(
            self.device_type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
        )
        tp_mesh = mesh_2d["tp"]
        dp_mesh = mesh_2d["dp"]
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }
        model_2d = parallelize_module(simple_model().cuda(), tp_mesh, parallelize_plan)
        model_2d = FSDP(model_2d, device_mesh=dp_mesh, use_orig_params=True)

        FSDP.set_state_dict_type(
            model_2d,
            StateDictType.SHARDED_STATE_DICT,
        )
        state_dict_2d = model_2d.state_dict()

        for no_wrap_items, two_d_items in zip(
            no_wrap_state_dict.items(), state_dict_2d.items()
        ):
            no_wrap_k, no_wrap_v = no_wrap_items
            two_d_k, two_d_v = two_d_items

            self.assertEqual(no_wrap_k, two_d_k)

            # check if all value in 2D state_dict are DTensor
            self.assertTrue(isinstance(two_d_v, DT))
            self.assertEqual(len(two_d_v.placements), 2)
            # the outer dimension is the FSDP dimension and the placement is always Shard(0)
            self.assertEqual(two_d_v.placements[0], Shard(0))
            self.assertEqual(two_d_v.device_mesh, mesh_2d)

            # check if the parameter value is the same between 2D model and the model without wrapper
            all_gather_two_d_v = two_d_v.redistribute(
                mesh_2d, (Replicate(), Replicate())
            )
            self.assertEqual(
                torch.allclose(no_wrap_v, all_gather_two_d_v.to_local()), True
            )

    @with_comms
    @skip_if_lt_x_gpu(4)
    @parametrize("is_even_sharded_model", [True, False])
    def test_2d_load_state_dict(self, is_even_sharded_model):
        simple_model = SimpleModel if is_even_sharded_model else SimpleModelUneven

        torch.manual_seed(0)
        mesh_2d = init_device_mesh(
            self.device_type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
        )
        tp_mesh = mesh_2d["tp"]
        dp_mesh = mesh_2d["dp"]
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }
        model_2d = parallelize_module(simple_model().cuda(), tp_mesh, parallelize_plan)
        model_2d = FSDP(model_2d, device_mesh=dp_mesh, use_orig_params=True)
        optim_2d = torch.optim.Adam(model_2d.parameters(), lr=0.01)

        FSDP.set_state_dict_type(
            model_2d,
            StateDictType.SHARDED_STATE_DICT,
        )
        checkpoint = io.BytesIO()
        torch.save(model_2d.state_dict(), checkpoint)
        # Deepcopy to save current state_dict to compare with the state_dict loaded back below.
        ref_state_dict = deepcopy(model_2d.state_dict())

        # Update the parameters so model.state_dict() will be different from ref_dtensor_sd.
        model_2d(model_2d.get_input().cuda(self.rank)).sum().backward()
        optim_2d.step()

        # Load ref_state_dict back.
        checkpoint.seek(0)
        load_ref_state_dict = torch.load(checkpoint)
        model_2d.load_state_dict(load_ref_state_dict)
        new_state_dict = model_2d.state_dict()

        # Check whether new_state_dict is the same as ref_state_dict.
        for (k1, v1), (k2, v2) in zip(ref_state_dict.items(), new_state_dict.items()):
            # check whether fqn are the same
            self.assertEqual(k1, k2)

            self.assertEqual(type(v1), DT)
            self.assertEqual(type(v2), DT)
            # check whether DTensor are the same
            # TODO: 2D DTensor comparison is not supported at the time, so we are comparing the spec and the local tensor for now.
            # TODO: Update it to compare the two DTensors once 2D DTensor comparison is supported.
            self.assertEqual(v1.to_local(), v2.to_local())
            self.assertEqual(v1.device_mesh, v2.device_mesh)
            self.assertEqual(v1.placements, v2.placements)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @parametrize("is_even_sharded_model", [True, False])
    def test_2d_optim_state_dict(self, is_even_sharded_model):
        simple_model = SimpleModel if is_even_sharded_model else SimpleModelUneven

        # Create a model without wrapper
        torch.manual_seed(0)
        no_wrap_model = simple_model().cuda(self.rank)
        no_wrap_state_dict = no_wrap_model.state_dict()
        no_wrap_optim = torch.optim.Adam(no_wrap_model.parameters(), lr=0.01)
        no_wrap_model(no_wrap_model.get_input().cuda(self.rank)).sum().backward()
        no_wrap_optim.step()
        no_wrap_osd = get_optimizer_state_dict(no_wrap_model, optimizers=no_wrap_optim)

        # Create a model and sharded it with 2D FSDP + TP
        torch.manual_seed(0)
        mesh_2d = init_device_mesh(
            self.device_type, (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
        )
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }
        model_2d = parallelize_module(
            simple_model().cuda(), mesh_2d["tp"], parallelize_plan
        )
        model_2d = FSDP(model_2d, device_mesh=mesh_2d["dp"], use_orig_params=True)
        FSDP.set_state_dict_type(
            model_2d,
            StateDictType.SHARDED_STATE_DICT,
        )
        optim_2d = torch.optim.Adam(model_2d.parameters(), lr=0.01)
        model_2d(model_2d.get_input().cuda(self.rank)).sum().backward()
        optim_2d.step()
        optim_2d_osd = get_optimizer_state_dict(model_2d, optimizers=optim_2d)
        ref_optim_2d_osd = deepcopy(optim_2d_osd)

        no_wrap_osd_states = no_wrap_osd["state"]
        optim_2d_osd_states = optim_2d_osd["state"]

        self.assertEqual(len(no_wrap_osd_states), len(optim_2d_osd_states))
        self.assertEqual(no_wrap_osd_states.keys(), optim_2d_osd_states.keys())
        for fqn, states in no_wrap_osd_states.items():
            dist_states = optim_2d_osd_states.get(fqn)

            for state_name, state in states.items():
                dist_state = dist_states.get(state_name)
                # If a state  is DTensor, we all gather it in both DP and TP dimension to
                # compare with no_wrap state.
                if isinstance(dist_state, DT):
                    dist_state = (
                        dist_state.cuda()
                        .redistribute(placements=(Replicate(), Replicate()))
                        .to_local()
                    )
                self.assertTrue(isinstance(dist_state, torch.Tensor))
                self.assertTrue(torch.allclose(state, dist_state))

        # Update the parameters 2d optim states will be different from ref_optim_state_dict.
        model_2d(model_2d.get_input().cuda(self.rank)).sum().backward()
        optim_2d.step()

        set_optimizer_state_dict(
            model_2d, optimizers=optim_2d, optim_state_dict=ref_optim_2d_osd
        )
        new_optim_2d_osd = get_optimizer_state_dict(model_2d, optimizers=optim_2d)

        ref_optim_2d_osd_states = ref_optim_2d_osd["state"]
        new_optim_2d_osd_states = optim_2d_osd["state"]

        # Compare the new optim state dict after load with the reference one
        self.assertEqual(len(ref_optim_2d_osd_states), len(new_optim_2d_osd_states))
        self.assertEqual(ref_optim_2d_osd_states.keys(), new_optim_2d_osd_states.keys())
        for fqn, states in ref_optim_2d_osd_states.items():
            new_states = new_optim_2d_osd_states.get(fqn)

            for state_name, state in states.items():
                new_state = new_states.get(state_name)

                if isinstance(new_state, DT):
                    self.assertEqual(new_state.placements, state.placements)
                    self.assertEqual(new_state.device_mesh, state.device_mesh)
                    self.assertTrue(
                        torch.allclose(new_state.to_local(), state.to_local())
                    )
                else:
                    self.assertEqual(new_state, state)


class ComposabilityTest(MultiProcContinousTest):
    @classmethod
    def backend_str(cls) -> str:
        # Testing with NCCL backend
        return "nccl"

    @classmethod
    def setUpClass(cls):
        """
        Class-scope test fixture. Run once for entire test class, before any test starts.
        Set up the device.
        """
        super().setUpClass()
        dev_id = cls.rank % torch.cuda.device_count()
        cls.device = torch.device(f"cuda:{dev_id}")
        # TODO: investigate why this is needed to prevent multiple NCCL ranks from hitting the same device
        torch.cuda.set_device(cls.device)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "Test requires 4+ GPUs")
    @parametrize("dp_type", ["DDP", "FSDP"])
    @parametrize("ScheduleClass", [ScheduleGPipe, Schedule1F1B])
    def test_manual_with_data_parallel(self, dp_type, ScheduleClass):
        device_mesh = init_device_mesh(
            "cuda", mesh_shape=(2, 2), mesh_dim_names=("dp", "pp")
        )
        pp_group = device_mesh["pp"].get_group()
        dp_mesh = device_mesh["dp"]

        # create "entire model"
        total_layers = 8
        dim = 10
        full_model = nn.ModuleList([MLPLayerModule(dim) for _ in range(total_layers)])
        ref_model = nn.Sequential(*copy.deepcopy(full_model))
        ref_model.to(self.device)

        # Prepare inputs
        num_microbatches = 8
        inputs = [
            torch.rand((num_microbatches, dim), device=self.device)
            for _ in range(dp_mesh.size())
        ]
        input = inputs[dp_mesh.get_local_rank()]
        input_mb = [[input[i].reshape((1, dim))] for i in range(num_microbatches)]

        # dummy loss needed just to force backwards to run in schedule step
        def loss_fn(y, target):
            return y.sum()

        # Get stage module i from the entire model
        def get_stage_module(stage_idx, num_stages):
            # divide the model (8 layers) by the number of stages
            layers_per_stage = total_layers // num_stages
            assert layers_per_stage * num_stages == total_layers
            # return offset so validation code can match partial layer back to orig model
            offset = stage_idx * layers_per_stage
            partial_model = nn.Sequential(
                *full_model[offset : (stage_idx + 1) * layers_per_stage]
            )
            partial_model.to(self.device)
            return partial_model, offset

        # Apply DP to stage module
        def apply_dp(partial_model, dp_type):
            if dp_type == "FSDP":
                # apply FSDP
                mp_policy = MixedPrecisionPolicy(
                    # TODO(whc) need to fix PP + FSDP-mixed-precision
                    # tracer for PP assumes f32 and is caught off guard when runtime FSDP interacts using bf16 inputs
                    # param_dtype=torch.bfloat16, reduce_dtype=torch.float32
                    param_dtype=torch.float32,
                    reduce_dtype=torch.float32,
                )
                fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
                for layer in partial_model.children():
                    fully_shard(
                        layer,
                        **fsdp_config,
                        reshard_after_forward=False,
                    )
                dp_model = fully_shard(partial_model, **fsdp_config)
            elif dp_type == "DDP":
                dp_model = DDP(partial_model, process_group=dp_mesh.get_group())
            else:
                raise RuntimeError(f"unsupported dp type {dp_type}")
            return dp_model

        # Create pipeline stage
        def build_stage(stage_idx, num_stages):
            partial_model, offset = get_stage_module(stage_idx, num_stages)
            dp_model = apply_dp(partial_model, dp_type)
            stage = PipelineStage(
                dp_model,
                stage_idx,
                num_stages,
                self.device,
                group=pp_group,
                input_args=input_mb[0],
            )
            return stage, offset

        # Attach to a schedule
        if issubclass(ScheduleClass, PipelineScheduleSingle):
            pipeline_stage, offset = build_stage(pp_group.rank(), pp_group.size())
            partial_models = [pipeline_stage.submod]
            offsets = [offset]
            pipeline_schedule = ScheduleClass(
                pipeline_stage,
                n_microbatches=num_microbatches,
                loss_fn=loss_fn,
            )
        else:
            n_virtual = 2
            num_stages = pp_group.size() * n_virtual
            stages = []
            offsets = []
            for i in range(n_virtual):
                stage, offset = build_stage(pp_group.rank() + n_virtual * i, num_stages)
                stages.append(stage)
                offsets.append(offset)
                partial_models = [pipeline_stage.submod for pipeline_stage in stages]
            pipeline_schedule = ScheduleClass(
                stages,
                n_microbatches=num_microbatches,
                loss_fn=loss_fn,
            )

        # Run
        pipeline_schedule._step_microbatches(arg_mbs=input_mb, target_mbs=input_mb)

        # Ref model runs on 2 different inputs, accumulating grads across them.
        # this ensures that we detect if the FSDP reduce becomes a no-op.
        # (in fsdp case, we use one of these inputs on each DP rank)
        (ref_model(inputs[0]).sum()).backward()
        (ref_model(inputs[1]).sum()).backward()

        # simulate the built-in averaging done by FSDP
        for p in ref_model.parameters():
            p.grad /= dp_mesh.size()

        # Validate that whichever weights we have locally match that part of our local/full ref model
        # (we force FSDP's grads to be all-gathered (.full_tensor) to make it simpler)
        ref_parameters = dict(ref_model.named_parameters())
        if dp_type == "FSDP":
            for partial_model, offset in zip(partial_models, offsets):
                for name, p in partial_model.named_parameters():
                    parts = name.split(".")
                    parts[0] = str(int(parts[0]) + offset)
                    name = ".".join(parts)
                    ref_p = ref_parameters[name]
                    self.assertTrue(isinstance(p.grad, DTensor))
                    self.assertEqual(ref_p.grad, p.grad.full_tensor())
        elif dp_type == "DDP":
            for partial_model, offset in zip(partial_models, offsets):
                for name, p in partial_model.named_parameters():
                    parts = name.split(".")[1:]  # remove the "module." prefix
                    parts[0] = str(int(parts[0]) + offset)
                    name = ".".join(parts)
                    ref_p = ref_parameters[name]
                    self.assertEqual(ref_p.grad, p.grad)

instantiate_parametrized_tests(TestNew2dParallelStateDict)
if __name__ == "__main__":
    run_tests()
    # Check if GPU and NCCL are available
    
    if not (
        dist.is_available()
        and dist.is_nccl_available()
        and torch.cuda.device_count() >= 4
    ):
        print(
            "Composability test requires at least 4 GPUs, but not enough found, skipping",
            file=sys.stderr,
        )
        sys.exit(0)

    rank = int(os.getenv("RANK", -1))
    world_size = int(os.getenv("WORLD_SIZE", 4))
    print("ranks: ", rank, world_size)
    if rank != -1:
        # Launched with torchrun or other multi-proc launchers. Directly run the test.
        ComposabilityTest.run_rank(rank, world_size)
    else:
        # Launched as a single process. Spawn subprocess to run the tests.
        # Also need a rendezvous file for `init_process_group` purpose.
        rdvz_file = tempfile.NamedTemporaryFile(delete=False).name
        torch.multiprocessing.spawn(
            ComposabilityTest.run_rank,
            nprocs=world_size,
            args=(world_size, rdvz_file),
        )
    
