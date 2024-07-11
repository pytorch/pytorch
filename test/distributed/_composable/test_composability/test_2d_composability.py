# Owner(s): ["oncall: distributed"]

import contextlib
import copy
import functools
import itertools
import unittest
from typing import Iterable, List, Tuple, Union
import io
from copy import deepcopy
from typing import List, Type

import torch
import torch.distributed as dist
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn

import torch.nn.functional as F
from torch.distributed._composable import replicate
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    fully_shard,
    OffloadPolicy,
)
from torch.distributed._tensor import DTensor, init_device_mesh
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.distributed._composable.fsdp import CPUOffloadPolicy
from torch.distributed._composable.fsdp.fully_shard import fully_shard
from torch.distributed._tensor import (
    DeviceMesh,
    DTensor,
    DTensor as DT,
    init_device_mesh,
    Replicate,
    Shard,
)
from torch.distributed._tensor.debug.comm_mode import CommDebugMode
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
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
    FSDPTestMultiThread,
    MLP,
    patch_all_gather,
    patch_reduce_scatter,
    test_compiled_fsdp,
)
from torch.testing._internal.common_utils import (
    get_cycles_per_ms,
    run_tests,
    wrapSwapTensorsTest,
)
from torch.testing._internal.common_fsdp import FSDPTest, MLP, MLPStack

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


def init_model(device_type, model_parallel_size=2):
    torch.manual_seed(0)
    model = MLPModule(device_type)
    torch.manual_seed(0)
    twod_model = MLPModule(device_type)
    model = DDP(model)

    # 2-D mesh is [dp, tp]
    world_size = dist.get_world_size()
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


class TestFullyShard2DTraining(FSDPTest):
    global c10d_ops
    global funcol
    c10d_ops = torch.ops.c10d
    funcol = torch.ops.c10d_functional

    @property
    def world_size(self) -> int:
        return 2

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_root_move_forward_input_to_device(self):
        device = torch.device("cuda", 0)

        class ParamlessModule(nn.Module):
            def forward(self, x: torch.Tensor, ys: Tuple[torch.Tensor, ...]):
                # Check that FSDP moved the inputs to GPU, including recursing
                # into the tuple data structure
                assert x.device == device, f"Expects {device} but got {x.device}"
                assert (
                    ys[0].device == device
                ), f"Expects {device} but got {ys[0].device}"
                assert (
                    ys[1].device == device
                ), f"Expects {device} but got {ys[1].device}"
                y = ys[0] + ys[1]
                return x + y + 1

        model = ParamlessModule()
        fully_shard(model)
        x = torch.randn((3,))
        ys = (torch.randn((3,)), torch.randn((3,)))
        self.assertEqual(x.device, torch.device("cpu"))
        self.assertEqual(ys[0].device, torch.device("cpu"))
        self.assertEqual(ys[1].device, torch.device("cpu"))
        model(x, ys)


class TestFullyShardRegisteredParams(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 4

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_param_registration_after_forward(self):
        """Tests the parameter registration after forward."""
        device = torch.device("cuda", 0)
        # Single FSDP group
        for reshard_after_forward in (True, False, 2):
            torch.manual_seed(42)
            model = MLP(3, device)
            # Since seed is per process, not per thread, we broadcast to ensure
            # the same parameters across ranks
            for param in model.parameters():
                dist.broadcast(param, src=0)
            ref_model = copy.deepcopy(model)
            fully_shard(model, reshard_after_forward=reshard_after_forward)  # root only
            inp = torch.randn((2, 3), device="cuda")
            self._assert_dtensor_params(model.parameters())
            self._assert_same_params(model.parameters(), ref_model.parameters())
            model(inp)  # root does not reshard after forward
            self._assert_tensor_params(model.parameters())
            self._assert_same_params(model.parameters(), ref_model.parameters())
            model.reshard()  # however, we can manually reshard
            self._assert_dtensor_params(model.parameters())
            self._assert_same_params(model.parameters(), ref_model.parameters())

        # Multiple FSDP groups
        for reshard_after_forward in (True, False, 2):
            torch.manual_seed(42)
            model = nn.Sequential(MLP(3, device), MLP(3, device))
            for param in model.parameters():
                dist.broadcast(param, src=0)
            ref_model = copy.deepcopy(model)
            fully_shard(model[0].in_proj, reshard_after_forward=reshard_after_forward)
            fully_shard(model[0].out_proj, reshard_after_forward=reshard_after_forward)
            fully_shard(model, reshard_after_forward=reshard_after_forward)

            self._assert_dtensor_params(model.parameters())
            self._assert_same_params(model.parameters(), ref_model.parameters())
            model(inp)
            non_root_params = list(model[0].in_proj.parameters()) + list(
                model[0].out_proj.parameters()
            )
            root_params = list(set(model.parameters()) - set(non_root_params))
            if reshard_after_forward is False:
                self._assert_tensor_params(non_root_params)
            else:
                self._assert_dtensor_params(non_root_params)
            self._assert_tensor_params(root_params)
            self._assert_same_params(model.parameters(), ref_model.parameters())
            for module in model.modules():
                if isinstance(module, FSDPModule):
                    module.reshard()  # however, we can manually reshard
            self._assert_dtensor_params(model.parameters())
            self._assert_same_params(model.parameters(), ref_model.parameters())

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_param_registration_after_backward(self):
        """Tests the parameter registration after backward."""
        device = torch.device("cuda", 0)
        # Single FSDP group
        for reshard_after_forward in (True, False, 2):
            model = MLP(8, device)
            fully_shard(model, reshard_after_forward=reshard_after_forward)  # root only
            inp = torch.randn((2, 8), device="cuda")
            self._assert_dtensor_params(model.parameters())
            model(inp).sum().backward()
            self._assert_dtensor_params(model.parameters())

        # Multiple FSDP groups
        for reshard_after_forward in (True, False, 2):
            model = MLP(8, device)
            fully_shard(model.in_proj, reshard_after_forward=reshard_after_forward)
            fully_shard(model.out_proj, reshard_after_forward=reshard_after_forward)
            fully_shard(model, reshard_after_forward=reshard_after_forward)
            self._assert_dtensor_params(model.parameters())
            model(inp).sum().backward()
            self._assert_dtensor_params(model.parameters())

    def _assert_tensor_params(self, params: Iterable[nn.Parameter]):
        self.assertGreater(len(list(params)), 0)
        for param in params:
            self.assertNotIsInstance(param, DTensor)
            self.assertIsInstance(param, torch.Tensor)

    def _assert_dtensor_params(self, params: Iterable[nn.Parameter]):
        self.assertGreater(len(list(params)), 0)
        for param in params:
            self.assertIsInstance(param, DTensor)

    def _assert_same_params(
        self, params: Iterable[nn.Parameter], ref_params: Iterable[nn.Parameter]
    ):
        params, ref_params = list(params), list(ref_params)
        self.assertEqual(len(params), len(ref_params))
        for param, ref_param in zip(params, ref_params):
            if isinstance(param, DTensor):
                param = param.full_tensor()
            self.assertEqual(param.shape, ref_param.shape)
            self.assertEqual(param, ref_param)


class TestFullyShardCastAfterInit(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 2

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    @wrapSwapTensorsTest(True)
    def test_to_float64_after_init(self):
        """Tests that the user can cast the module to float64 after init."""
        # NOTE: Test fp64 instead of a lower precision dtype like bf16 for
        # better numerics. The important part is changing the dtype.
        torch.manual_seed(42)
        mlp_dim, device, dtype = 4, torch.device("cuda"), torch.float64
        model = MLP(mlp_dim, device=device)
        for param in model.parameters():
            dist.broadcast(param, src=0)
        ref_model = copy.deepcopy(model).to(dtype)
        replicate(ref_model)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        for module in (model.in_proj, model.out_proj, model):
            fully_shard(module)
        model.to(dtype)
        for param in model.parameters():
            self.assertEqual(param.dtype, dtype)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        check_sharded_parity(self, ref_model, model)
        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((2, mlp_dim), device="cuda", dtype=dtype)
        for iter_idx in range(10):
            losses: List[torch.Tensor] = []
            for _model in (ref_model, model):
                losses.append(_model(inp).sum())
                losses[-1].backward()
            self.assertEqual(losses[0], losses[1])
            check_sharded_parity(self, ref_model, model)
            for _optim in (ref_optim, optim):
                _optim.step()
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))


class TestFullyShard1DTrainingCore(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(8, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    def test_train_parity_single_group(self):
        """Tests train parity with DDP for a single FSDP group."""
        self.run_subtests(
            {
                "lin_shapes": [[(16, 15), (15, 8)], [(7, 15), (15, 3)]],
            },
            self._test_train_parity_single_group,
        )

    def _test_train_parity_single_group(self, lin_shapes: List[Tuple[int, int]]):
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(*lin_shapes[0]), nn.ReLU(), nn.Linear(*lin_shapes[1])
        )
        ref_model = copy.deepcopy(model).cuda()
        replicate(ref_model, device_ids=[self.rank])
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        fully_shard(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        torch.manual_seed(42 + self.rank + 1)
        inp = (torch.randn((4, lin_shapes[0][0]), device="cuda"),)
        for iter_idx in range(10):
            losses: List[torch.Tensor] = []
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                losses.append(_model(*inp).sum())
                losses[-1].backward()
                _optim.step()
            self.assertEqual(losses[0], losses[1])

    @skip_if_lt_x_gpu(2)
    @test_compiled_fsdp(compile_compute_on_module=Transformer)
    def test_train_parity_multi_group(self):
        """
        Tests train parity against DDP when using multiple parameter groups for
        communication (for communication and computation overlap plus memory
        reduction).
        """
        self.run_subtests(
            {
                "reshard_after_forward": [True, False, 2],
                "device_type": ["cuda"],
                "offload_policy": [OffloadPolicy()],
                "delay_after_forward": [False, True],
                "delay_before_all_gather": [False, True],
                "delay_before_reduce_scatter": [False, True],
                "delay_before_optim": [False, True],
            },
            self._test_train_parity_multi_group,
        )

    @skip_if_lt_x_gpu(2)
    def test_train_parity_multi_group_cpu_offload_eager(self):
        """
        Tests train parity against DDP when using multiple parameter groups for
        communication and CPU offloading.
        """
        self.run_subtests(
            {
                "reshard_after_forward": [True],  # save CI time
                "offload_policy": [
                    CPUOffloadPolicy(pin_memory=True),
                    CPUOffloadPolicy(pin_memory=False),
                ],
                "device_type": ["cuda"],
                "delay_after_forward": [False, True],
                "delay_before_all_gather": [False, True],
                "delay_before_reduce_scatter": [False, True],
                "delay_before_optim": [False, True],
            },
            self._test_train_parity_multi_group,
        )

    def _test_train_parity_multi_group(
        self,
        reshard_after_forward: Union[bool, int],
        offload_policy: OffloadPolicy,
        device_type: str,
        delay_after_forward: bool,
        delay_before_all_gather: bool,
        delay_before_reduce_scatter: bool,
        delay_before_optim: bool,
    ):
        # Only test individual delays or all four delays to save test time
        if (
            delay_after_forward
            + delay_before_all_gather
            + delay_before_reduce_scatter
            + delay_before_optim
            in (2, 3)
        ):
            return
        assert device_type in ("cuda", "cpu"), f"{device_type}"
        torch.manual_seed(42)
        vocab_size = 1024
        model_args = ModelArgs(
            n_layers=3,
            n_heads=4,
            vocab_size=vocab_size,
            max_seq_len=64,
            dropout_p=0,
        )
        model = Transformer(model_args)
        ref_model = copy.deepcopy(model)
        if device_type == "cuda":
            replicate(ref_model.cuda(), device_ids=[self.rank])
        else:
            gloo_pg = dist.new_group(backend="gloo")
            replicate(ref_model, process_group=gloo_pg)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        mesh = init_device_mesh(device_type, (self.world_size,))
        fully_shard_fn = functools.partial(
            fully_shard,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
            offload_policy=offload_policy,
        )
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard_fn(module)
        fully_shard_fn(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        delay_in_ms = 100
        orig_all_gather = dist.all_gather_into_tensor
        orig_reduce_scatter = dist.reduce_scatter_tensor

        def delayed_all_gather(*args, **kwargs):
            torch.cuda._sleep(int(delay_in_ms * get_cycles_per_ms()))
            return orig_all_gather(*args, **kwargs)

        def delayed_reduce_scatter(*args, **kwargs):
            torch.cuda._sleep(int(delay_in_ms * get_cycles_per_ms()))
            return orig_reduce_scatter(*args, **kwargs)

        torch.manual_seed(42 + self.rank + 1)
        patch_all_gather_ctx = (
            patch_all_gather(delayed_all_gather)
            if delay_before_all_gather
            else contextlib.nullcontext()
        )
        patch_reduce_scatter_ctx = (
            patch_reduce_scatter(delayed_reduce_scatter)
            if delay_before_reduce_scatter
            else contextlib.nullcontext()
        )
        with patch_all_gather_ctx, patch_reduce_scatter_ctx:
            for iter_idx in range(10):
                inp = torch.randint(0, vocab_size, (3, 64), device=device_type)
                losses: List[torch.Tensor] = []
                for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                    _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                    losses.append(_model(inp).sum())
                    if _model is model and delay_after_forward:
                        torch.cuda._sleep(int(delay_in_ms * get_cycles_per_ms()))
                    losses[-1].backward()
                    if _model is model and delay_before_optim:
                        torch.cuda._sleep(int(delay_in_ms * get_cycles_per_ms()))
                    _optim.step()
                self.assertEqual(losses[0], losses[1])

    @skip_if_lt_x_gpu(2)
    def test_non_root_forward_backward(self):
        """
        Tests running forward/backward through the root and then through a
        non-root. The non-root needs to synchronize streams/queue the callback.
        """
        torch.manual_seed(42)
        lin_dim = 32
        model = nn.Sequential(*[MLP(lin_dim, torch.device("cpu")) for _ in range(3)])
        ref_model = copy.deepcopy(model).cuda()
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        for mlp in model:
            fully_shard(mlp)
        fully_shard(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        torch.manual_seed(42 + self.rank)
        inp = torch.randn((8, lin_dim), device=torch.device("cuda"))

        ref_root_loss = ref_model(inp).sum()
        ref_root_loss.backward()
        for param in ref_model.parameters():
            dist.all_reduce(param.grad)
            param.grad.detach().div_(self.world_size)
        ref_optim.step()
        ref_optim.zero_grad()
        ref_nonroot_loss = ref_model[0](inp).sum()
        ref_nonroot_loss.backward()
        for param in ref_model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
                param.grad.detach().div_(self.world_size)
        ref_optim.step()

        root_loss = model(inp).sum()
        root_loss.backward()
        torch.cuda._sleep(int(100 * get_cycles_per_ms()))
        optim.step()
        optim.zero_grad()
        nonroot_loss = model[0](inp).sum()
        nonroot_loss.backward()
        optim.step()

        self.assertEqual(ref_root_loss, root_loss)
        self.assertEqual(ref_nonroot_loss, nonroot_loss)
        self.assertEqual(ref_model(inp).sum(), model(inp).sum())

    @skip_if_lt_x_gpu(2)
    def test_multi_forward_module(self):
        """
        Tests parity with DDP when running a module that participates multiple
        times in forward.
        """
        self.run_subtests(
            {"reshard_after_forward": [True, False, 2]},
            self._test_multi_forward_module,
        )

    def _test_multi_forward_module(self, reshard_after_forward: Union[bool, int]):
        class MultiForwardModule(nn.Module):
            def __init__(self, device: torch.device):
                super().__init__()
                self.inner = nn.Linear(4, 4, device=device)
                self.outer = nn.Linear(4, 5, device=device)

            def forward(self, x):
                i = self.inner(x)
                j = self.inner(x)
                return self.outer(i + j)

        torch.manual_seed(42)
        model = MultiForwardModule(device="cuda")
        ref_model = copy.deepcopy(model)
        replicate(ref_model, device_ids=[self.rank])
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        fully_shard(model.inner)
        fully_shard(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        torch.manual_seed(42 + self.rank)
        inp = torch.randn((32, 4), device="cuda")
        for iter_idx in range(10):
            losses: List[torch.Tensor] = []
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                losses.append(_model(inp).sum())
                losses[-1].backward()
                _optim.step()
            self.assertEqual(losses[0], losses[1])

    @skip_if_lt_x_gpu(2)
    def test_explicit_prefetching(self):
        torch.manual_seed(42)
        model_args = ModelArgs(n_layers=8, dropout_p=0.0)
        model = Transformer(model_args)
        ref_model = replicate(copy.deepcopy(model).cuda())
        ref_optim = torch.optim.AdamW(ref_model.parameters(), lr=1e-2)
        for layer in itertools.chain(model.layers, [model]):
            fully_shard(layer)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

        num_to_forward_prefetch = num_to_backward_prefetch = 2
        for i, layer in enumerate(model.layers):
            if i >= len(model.layers) - num_to_forward_prefetch:
                break
            layers_to_prefetch = [
                model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
            ]
            layer.set_modules_to_forward_prefetch(layers_to_prefetch)
        for i, layer in enumerate(model.layers):
            if i < num_to_backward_prefetch:
                continue
            layers_to_prefetch = [
                model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
            ]
            layer.set_modules_to_backward_prefetch(layers_to_prefetch)

        torch.manual_seed(42 + self.rank)
        inp = torch.randint(0, model_args.vocab_size, (2, 8), device="cuda")
        for iter_idx in range(10):
            losses: List[torch.Tensor] = []
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad()
                losses.append(_model(inp).sum())
                losses[-1].backward()
                _optim.step()
            self.assertEqual(losses[0], losses[1])

    @skip_if_lt_x_gpu(2)
    def test_post_optim_event(self):
        torch.manual_seed(42)
        model_args = ModelArgs(dropout_p=0.0)
        model = Transformer(model_args)
        ref_model = replicate(copy.deepcopy(model).cuda())
        ref_optim = torch.optim.AdamW(ref_model.parameters(), lr=1e-2)
        for layer in itertools.chain(model.layers, [model]):
            fully_shard(layer)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

        def step_post_hook(
            fsdp_module: FSDPModule, opt: torch.optim.Optimizer, args, kwargs
        ) -> None:
            post_optim_event = torch.cuda.current_stream().record_event()
            fsdp_module.set_post_optim_event(post_optim_event)

        optim.register_step_post_hook(functools.partial(step_post_hook, model))

        torch.manual_seed(42 + self.rank)
        inp = torch.randint(0, model_args.vocab_size, (2, 8), device="cuda")
        # Track all losses and check for equality at the end to avoid a CPU
        # sync point after each iteration
        ref_losses: List[torch.Tensor] = []
        losses: List[torch.Tensor] = []
        for iter_idx in range(10):
            ref_optim.zero_grad()
            ref_losses.append(ref_model(inp).sum())
            ref_losses[-1].backward()
            ref_optim.step()
        for iter_idx in range(10):
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


class Test2dParallelIntegration(DTensorTestBase):
    global LR
    LR = 3e-5

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
            parallelize_module(SimpleModel().cuda(), mesh_2d["tp"], parallelize_plan)

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


instantiate_parametrized_tests(TestNew2dParallelStateDict)

if __name__ == "__main__":
    run_tests()
