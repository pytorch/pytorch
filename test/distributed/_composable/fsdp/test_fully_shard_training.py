# Owner(s): ["oncall: distributed"]

import contextlib
import copy
import functools
import itertools
import unittest
from collections import defaultdict
from typing import Iterable, List, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import checkpoint, replicate
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    fully_shard,
    OffloadPolicy,
    register_fsdp_forward_method,
)
from torch.distributed._tensor import DTensor, init_device_mesh
from torch.distributed._tensor.debug.comm_mode import CommDebugMode
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    _CHECKPOINT_PREFIX,
    apply_activation_checkpointing,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    check_sharded_parity,
    FSDPTest,
    FSDPTestMultiThread,
    MLP,
    MLPStack,
    patch_all_gather,
    patch_reduce_scatter,
    test_compiled_fsdp,
)
from torch.testing._internal.common_utils import (
    get_cycles_per_ms,
    run_tests,
    wrapSwapTensorsTest,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
    TransformerBlock,
)


c10d_ops = torch.ops.c10d
funcol = torch.ops.c10d_functional


class TestFullyShardForwardInputs(FSDPTestMultiThread):
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
            losses.append(model(inp).sum())
            losses[-1].backward()
            optim.step()
            # Sleep after the optimizer step to allow CPU to run ahead into the
            # next iteration's forward, exercising the post-optim stream sync
            torch.cuda._sleep(int(25 * get_cycles_per_ms()))
        for ref_loss, loss in zip(ref_losses, losses):
            self.assertEqual(ref_loss, loss)


class TestFullyShard1DTrainingCompose(FSDPTest):
    @property
    def world_size(self) -> int:
        # Since these tests run with a larger transformer model, they may see
        # some numeric drift with >2 GPUs
        return min(torch.cuda.device_count(), 2)

    @skip_if_lt_x_gpu(2)
    @test_compiled_fsdp(compile_compute_on_module=Transformer)
    def test_train_parity_with_activation_checkpointing(self):
        """
        Tests train parity against DDP when composing with activation
        checkpointing.
        """
        self.run_subtests(
            {
                "reshard_after_forward": [True, False],
                "checkpoint_impl": ["composable", "utils", "wrapper"],
                "module_grouping": ["block", "mem_eff", "mem_eff_weight_tied"],
            },
            self._test_train_parity_with_activation_checkpointing,
        )

    def _test_train_parity_with_activation_checkpointing(
        self,
        reshard_after_forward: Union[bool, int],
        checkpoint_impl: str,
        module_grouping: str,
    ):
        assert checkpoint_impl in ("composable", "utils", "wrapper")
        testing_compile = fully_shard != torch.distributed._composable.fsdp.fully_shard
        if testing_compile and checkpoint_impl == "composable":
            return
        torch.manual_seed(42)
        vocab_size = 1024
        with torch.device(torch.device("cuda")):
            model_args = ModelArgs(
                n_layers=3,
                n_heads=4,
                vocab_size=vocab_size,
                max_seq_len=64,
                dropout_p=0,
                checkpoint_activations=(checkpoint_impl == "utils"),
                # For the mem-efficient module grouping, we separate the
                # embeddings from the output projection, which does not support
                # weight tying
                weight_tying=module_grouping != "mem_eff",
            )
            model = Transformer(model_args)
        ref_model = replicate(copy.deepcopy(model), device_ids=[self.rank])
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)

        # Apply activation checkpointing
        prefixes_to_ignore = ()
        if checkpoint_impl == "wrapper":
            prefixes_to_ignore = (_CHECKPOINT_PREFIX,)
            apply_activation_checkpointing(
                model, check_fn=lambda m: isinstance(m, TransformerBlock)
            )
        elif checkpoint_impl == "composable":
            for module in model.modules():
                if isinstance(module, TransformerBlock):
                    checkpoint(module)

        # Apply FSDP
        fsdp_kwargs = {"reshard_after_forward": reshard_after_forward}
        if module_grouping == "mem_eff":
            assert model_args.n_layers == 3
            fully_shard(model.layers[0], **fsdp_kwargs)
            fully_shard([model.layers[1], model.layers[2]], **fsdp_kwargs)
            fully_shard([model.tok_embeddings, model.pos_embeddings], **fsdp_kwargs)
            fully_shard([model.norm, model.output], **fsdp_kwargs)
        elif module_grouping == "mem_eff_weight_tied":
            fully_shard([model.tok_embeddings, model.output], **fsdp_kwargs)
            for layer in model.layers:
                fully_shard(layer, **fsdp_kwargs)
        elif module_grouping == "block":
            for layer in model.layers:
                fully_shard(layer, **fsdp_kwargs)
        else:
            raise NotImplementedError(f"Unknown module grouping: {module_grouping}")
        fully_shard(model, **fsdp_kwargs)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        torch.manual_seed(42 + self.rank)
        # Reuse the same input across iterations to avoid loss explosion from
        # trying to learn from random inputs
        inp = torch.randint(0, vocab_size, (3, 64), device="cuda")
        check_sharded_parity(
            self, ref_model, model, prefixes_to_ignore=prefixes_to_ignore
        )
        for iter_idx in range(10):
            losses: List[torch.Tensor] = []
            for _model in (ref_model, model):
                torch.manual_seed(iter_idx + 1)  # for dropout determinism
                losses.append(_model(inp).sum())
                losses[-1].backward()
            if not testing_compile:
                check_sharded_parity(
                    self, ref_model, model, prefixes_to_ignore=prefixes_to_ignore
                )
            self.assertEqual(losses[0], losses[1])
            for _optim in (ref_optim, optim):
                _optim.step()
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            if not testing_compile:
                check_sharded_parity(
                    self, ref_model, model, prefixes_to_ignore=prefixes_to_ignore
                )


class TestFullyShardSharedParams(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    def test_train_parity_with_shared_params(self):
        self.run_subtests(
            {
                "reshard_after_forward": [False, True],
                "use_activation_checkpointing": [False, True],
            },
            self._test_train_shared_params,
        )

    def _test_train_shared_params(
        self,
        reshard_after_forward: bool,
        use_activation_checkpointing: bool,
    ):
        torch.manual_seed(42)
        model_args = ModelArgs(n_layers=3, dropout_p=0.0, weight_tying=True)
        model = Transformer(model_args)
        ref_model = copy.deepcopy(model).cuda()
        replicate(ref_model, device_ids=[self.rank])
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                if use_activation_checkpointing:
                    checkpoint(module)
                fully_shard(module, reshard_after_forward=reshard_after_forward)
        fully_shard(model, reshard_after_forward=reshard_after_forward)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        torch.manual_seed(42 + self.rank + 1)
        for iter_idx in range(10):
            inp = torch.randint(0, model_args.vocab_size, (2, 16), device="cuda")
            losses: List[torch.Tensor] = []
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                losses.append(_model(inp).sum())
                losses[-1].backward()
                _optim.step()
            self.assertEqual(losses[0], losses[1])


class TestFullyShardGradientAccumulation(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    def test_gradient_accumulation(self):
        """
        Tests gradient accumulation with/without gradient reduction and
        with/without resharding after backward.
        """
        meshes = [init_device_mesh("cuda", (self.world_size,))]  # always test FSDP
        if self.world_size == 4:  # test HSDP too if enough GPUs
            shard_size, replicate_size = 2, 2
            meshes.append(init_device_mesh("cuda", (replicate_size, shard_size)))
        self.run_subtests(
            {
                "mesh": meshes,
                "reshard_after_forward": [True, False, 2],
                # "all": disable reduce-scatter for all modules
                # "root_only": disable reduce-scatter for root's linear only
                # "some_mlps": disable reduce-scatter for some MLPs
                "mode": ["all", "root_only", "some_mlps"],
                "reshard_after_backward": [False, True],
                "offload_policy": [OffloadPolicy(), CPUOffloadPolicy()],
                # For HSDP only:
                # `True`: reduce-scatter only (no all-reduce) each microbatch
                # until the last microbatch
                # `False`: neither reduce-scatter nor all-reduce each
                # microbatch until the last microbatch
                "reduce_scatter_only": [False, True],
            },
            self._test_gradient_accumulation,
        )

    def _test_gradient_accumulation(
        self,
        mesh: DeviceMesh,
        reshard_after_forward: Union[bool, int],
        mode: str,
        reshard_after_backward: bool,
        offload_policy: OffloadPolicy,
        reduce_scatter_only: bool,  # for HSDP
    ):
        if (
            (
                not reshard_after_backward
                and (reshard_after_forward is not False or mode == "some_mlps")
            )
            or (
                isinstance(offload_policy, CPUOffloadPolicy)
                and reshard_after_forward is not True
            )
            or (mesh.ndim != 2 and reduce_scatter_only)
        ):
            return  # skip since not common or applicable

        torch.manual_seed(42)
        batch_size, lin_dim, num_mlps, num_microbatches = (2, 32, 3, 3)
        if mode == "some_mlps":
            num_mlps_to_disable_reduce_scatter = 2
        modules = [nn.Linear(lin_dim, lin_dim)]
        modules.extend(MLP(lin_dim) for _ in range(num_mlps))
        model = nn.Sequential(*modules)
        ref_model = copy.deepcopy(model).cuda()
        fully_shard_fn = functools.partial(
            fully_shard,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
            offload_policy=offload_policy,
        )
        for mlp in model[1:]:
            fully_shard_fn(mlp)
        fully_shard_fn(model)  # root gets the 1st linear
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        def set_grad_sync_flag(
            module: nn.Module, is_last_microbatch: bool, recurse: bool = True
        ):
            if reduce_scatter_only:
                module.set_requires_all_reduce(is_last_microbatch, recurse=recurse)
            else:
                module.set_requires_gradient_sync(is_last_microbatch, recurse=recurse)

        def set_backward_flags(_model: nn.Module, is_last_microbatch: bool):
            if mode == "all":
                set_grad_sync_flag(_model, is_last_microbatch)
                if not reshard_after_backward:
                    _model.set_reshard_after_backward(is_last_microbatch)
            elif mode == "some_mlps":
                for mlp in model[1 : 1 + num_mlps_to_disable_reduce_scatter]:
                    set_grad_sync_flag(mlp, is_last_microbatch)
                    if not reshard_after_backward:
                        mlp.set_reshard_after_backward(is_last_microbatch)
            elif mode == "root_only":
                set_grad_sync_flag(model, is_last_microbatch, recurse=False)
                if not reshard_after_backward:
                    model.set_reshard_after_backward(is_last_microbatch, recurse=False)

        torch.manual_seed(42 + self.rank + 1)
        for iter_idx in range(5):
            comm_count_list = []

            for microbatch_idx in range(num_microbatches):
                is_last_microbatch = microbatch_idx == num_microbatches - 1
                set_backward_flags(model, is_last_microbatch)
                inp = torch.randn(batch_size, lin_dim, device="cuda")
                losses: List[torch.Tensor] = []
                for _model in (ref_model, model):
                    with CommDebugMode() as comm_mode:
                        losses.append(_model(inp).sum())
                        losses[-1].backward()
                    comm_count_list.append(comm_mode.get_comm_counts())
                self.assertEqual(losses[0], losses[1])

            comm_counts = defaultdict(int)
            for comm_count_dict in comm_count_list:
                for collective, count in comm_count_dict.items():
                    comm_counts[collective] += count

            all_gather_count = comm_counts[c10d_ops._allgather_base_]
            reduce_scatter_count = comm_counts[c10d_ops._reduce_scatter_base_]
            all_reduce_count = comm_counts[c10d_ops.allreduce_]

            # Expect one reduce-scatter per MLP plus one for the root's linear
            # on the last microbatch
            expected_reduce_scatter_count = num_mlps + 1
            if mode == "some_mlps":
                # Expect additional reduce-scatters for non-disabled MLPs and
                # the root's linear
                expected_reduce_scatter_count += (
                    num_mlps - num_mlps_to_disable_reduce_scatter + 1
                ) * (num_microbatches - 1)
            elif mode == "root_only":
                # Expect additional reduce-scatters for all MLPs
                expected_reduce_scatter_count += (num_mlps) * (num_microbatches - 1)
            expected_all_reduce_count = (
                expected_reduce_scatter_count if mesh.ndim == 2 else 0
            )
            if reduce_scatter_only:
                # Specially for HSDP if only reduce-scattering but not
                # all-reducing until the last microbatch, expect one
                # reduce-scatter per MLP plus for the root per microbatch
                expected_reduce_scatter_count = (num_mlps + 1) * num_microbatches
            self.assertEqual(reduce_scatter_count, expected_reduce_scatter_count)
            self.assertEqual(all_reduce_count, expected_all_reduce_count)

            # Expect one all-gather per MLP plus one for the root's linear in
            # the first microbatch's forward
            expected_all_gather_count = num_mlps + 1
            if reshard_after_forward is not False:  # `True` or `2`
                # Add the number of MLPs without the +1 for the backward
                # all-gathers since the root does not reshard after forward
                expected_all_gather_count += num_mlps
                # Multiply by the number of microbatches since these
                # all-gathers run every microbatch
                expected_all_gather_count *= num_microbatches
            elif reshard_after_backward:  # `reshard_after_forward=False`
                expected_all_gather_count *= num_microbatches
            elif mode == "all":  # `reshard_after_forward/backward=False`
                # Only reshard parameters after the last microbatch's backward,
                # so there should not be any more all-gathers
                pass
            elif mode == "root_only":  # `reshard_after_forward/backward=False`
                # The MLPs should still contribute all-gathers in each
                # microbatch forward
                expected_all_gather_count += num_mlps * (num_microbatches - 1)
            self.assertEqual(all_gather_count, expected_all_gather_count)

            for param in ref_model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            check_sharded_parity(self, ref_model, model)
            for _optim in (optim, ref_optim):
                _optim.step()
                # When `set_to_none=False`, we are exercising mixing
                # gradient accumulation with and without communication
                _optim.zero_grad(set_to_none=(iter_idx % 2))

    @skip_if_lt_x_gpu(2)
    def test_1f1b_microbatching(self):
        self.run_subtests(
            {
                "use_explicit_unshard": [False, True],
                "reshard_after_backward": [False, True],
            },
            self._test_1f1b_microbatching,
        )

    def _test_1f1b_microbatching(
        self, use_explicit_unshard: bool, reshard_after_backward: bool
    ):
        torch.manual_seed(42)
        model_args = ModelArgs(dropout_p=0.0)
        model = Transformer(model_args)
        ref_model = copy.deepcopy(model).cuda()
        ref_optim = torch.optim.AdamW(ref_model.parameters(), lr=1e-2)
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard(module, reshard_after_forward=False)
        fully_shard(model, reshard_after_forward=False)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

        num_microbatches = 3
        local_batch_size = 2
        torch.manual_seed(42 + self.rank + 1)
        inps = [
            torch.randint(
                0, model_args.vocab_size, (local_batch_size, 16), device="cuda"
            )
            for _ in range(num_microbatches)
        ]

        # Before pipelining, we may prefer to issue all all-gathers ahead of
        # time to increase overlap opportunity at no difference in parameter
        # memory usage since we do not reshard after forward
        if use_explicit_unshard:
            for module in model.modules():
                if isinstance(module, FSDPModule):
                    module.unshard(async_op=True)

        # Emulate the 1f1b pipeline schedule and only reduce gradients on the
        # last microbatch
        losses: List[torch.Tensor] = []
        ref_losses: List[torch.Tensor] = []
        for inp_idx, inp in enumerate(inps):
            is_last_microbatch = inp_idx == num_microbatches - 1
            model.set_requires_gradient_sync(is_last_microbatch)
            model.set_is_last_backward(is_last_microbatch)
            if not reshard_after_backward:
                model.set_reshard_after_backward(is_last_microbatch)
            losses.append(model(inp).sum())
            losses[-1].backward()
            ref_losses.append(ref_model(inp).sum())
            ref_losses[-1].backward()
        for param in ref_model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        for loss, ref_loss in zip(losses, ref_losses):
            self.assertEqual(loss, ref_loss)
        optim.step()
        ref_optim.step()
        check_sharded_parity(self, ref_model, model)


class TestFullyShardNDTraining(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(8, torch.cuda.device_count())

    def init_global_mesh(self) -> DeviceMesh:
        # Prefer to test with >=8 GPUs, but for 2 GPUs, use 2-way TP
        dp_size = 2 if self.world_size > 2 else 1
        pp_size = 2 if self.world_size > 4 else 1
        return init_device_mesh(
            "cuda",
            (pp_size, dp_size, self.world_size // (dp_size * pp_size)),
            mesh_dim_names=("pp", "dp", "tp"),
        )

    @skip_if_lt_x_gpu(4)
    def test_2d_mlp_with_nd_mesh(self):
        global_mesh = self.init_global_mesh()
        self.run_subtests(
            {
                "reshard_after_forward": [False, True],
                "use_activation_checkpointing": [False, True],
                "mlp_dim": [3, 16, 17],
                "foreach": [False],
            },
            functools.partial(self._test_2d_mlp_with_nd_mesh, global_mesh),
        )

    def _test_2d_mlp_with_nd_mesh(
        self,
        global_mesh: DeviceMesh,
        reshard_after_forward: bool,
        use_activation_checkpointing: bool,
        mlp_dim: int,
        foreach: bool,
    ):
        global_mesh = self.init_global_mesh()
        pp_mesh, dp_mesh, tp_mesh = (
            global_mesh["pp"],
            global_mesh["dp"],
            global_mesh["tp"],
        )
        dp_pg = dp_mesh.get_group()  # used for `replicate()`

        torch.manual_seed(42)
        model = MLPStack(mlp_dim)
        ref_model = copy.deepcopy(model).cuda()
        replicate(ref_model, device_ids=[self.rank], process_group=dp_pg)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2, foreach=foreach)
        model.parallelize(
            tp_mesh,
            dp_mesh,
            use_activation_checkpointing,
            reshard_after_forward=reshard_after_forward,
        )
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=foreach)

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

        for n, p in model.named_parameters():
            self.assertIsInstance(p, DTensor)
            self.assertEqual(p.device_mesh.ndim, 2)
            self.assertEqual(len(p.placements), 2)
            self.assertEqual(p.device_mesh.mesh_dim_names, ("dp", "tp"))


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


class TestFullyShardCustomForwardMethod(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 2)

    @skip_if_lt_x_gpu(2)
    def test_register_fsdp_forward_method(self):
        """Based on https://github.com/pytorch/pytorch/issues/109385"""

        class VisionTransformer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.patch_proj = nn.Conv2d(3, 1024, kernel_size=14, stride=14)

            def forward_features(self, imgs: torch.Tensor) -> torch.Tensor:
                return self.patch_proj(imgs).flatten(2).transpose(1, 2)

            def forward(self, imgs: torch.Tensor) -> torch.Tensor:
                return self.forward_features(imgs).sum(dim=1)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.vit, self.projector = VisionTransformer(), nn.Linear(1024, 256)

            def forward(self, imgs: torch.Tensor) -> torch.Tensor:
                # Run `vit.forward_features`, which is not `forward`!
                patch_embeddings = self.vit.forward_features(imgs)
                return self.projector(patch_embeddings)

        torch.manual_seed(42)
        model = Model()
        ref_model = copy.deepcopy(model).cuda()
        fully_shard(model.vit)
        fully_shard(model.projector)
        fully_shard(model)
        register_fsdp_forward_method(model.vit, "forward_features")

        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn(4, 3, 224, 224, device="cuda")
        ref_loss = ref_model(inp).sum()
        loss = model(inp).sum()
        self.assertEqual(ref_loss, loss)
        ref_loss.backward()
        loss.backward()
        for param in ref_model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        check_sharded_parity(self, ref_model, model)


if __name__ == "__main__":
    run_tests()
