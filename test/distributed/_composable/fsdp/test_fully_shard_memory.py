# Owner(s): ["oncall: distributed"]

import contextlib
import functools
import gc
import math
import types
import unittest
import weakref
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from torch.distributed.tensor import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import (
    get_cycles_per_ms,
    run_tests,
    TEST_CUDA,
    TEST_HPU,
    TestCase,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
    TransformerBlock,
)


device_type = torch.device(get_devtype())


class TestFullyShardMemory(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(2, torch.get_device_module(device_type).device_count())

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(TEST_HPU, " 'empty_cache' is not supported on hpu")
    def test_fully_shard_training_memory(self):
        self.run_subtests(
            {
                "reshard_after_forward": [True, False],
                "use_cpu_offload": [True, False],
                "run_optim_in_backward": [True, False],
            },
            self._test_fully_shard_training_memory,
        )

    def _test_fully_shard_training_memory(
        self,
        reshard_after_forward: bool,
        use_cpu_offload: bool,
        run_optim_in_backward: bool,
    ):
        if (
            # CPU offloading is typically for memory savings, so we expect
            # users to want to reshard after forward
            (not reshard_after_forward and use_cpu_offload)
            # Optimizer in backward frees sharded gradient GPU memory early for
            # memory savings, so we expect users to want to reshard after
            # forward; plus, it has no real effect with CPU offloading
            or (
                run_optim_in_backward and (not reshard_after_forward or use_cpu_offload)
            )
        ):
            return  # skip since not a common use case
        if self.world_size != 2:
            raise AssertionError(
                f"Requires world size of 2 since some values are hard coded: {self.world_size}"
            )
        torch.manual_seed(42)
        # Pre-run a linear forward (gemm and bias) and backward (gemm) to
        # allocate the cuBLAS workspaces before measuring the memory usage
        # since the workspace size can differ between hardwares
        lin = torch.nn.Linear(768, 768, device=device_type)
        # NOTE: before https://github.com/pytorch/pytorch/pull/163955,
        # the input shape was (1, 768), so that the forward gemm used
        # cublaslt, and the backward used cublas.
        # With the aforementioned PR, and with shape (1, 768),
        # the cublas path is used both in forward and in backward,
        # altering peak memory usage not accounting for cublaslt.
        # Here we change the input shape to (2, 768), and that swaps
        # the cublas/cublaslt selection in the forward/backward,
        # but that does not affect the peak memory usage stored in `base_mem_mb`.
        # Reasons for the flip:
        # before PR: no Lt in addmm when mat2 has nrows/ncols <= 1,
        # after PR: no Lt in addmm when either mat1 or mat2 have nrows/ncols <= 1,
        # since the input preparation can swap matrices based on output
        # row-/col-majorness.
        inp = torch.randn(2, 768, device=device_type)
        lin(inp).sum().backward()
        torch.get_device_module(device_type).empty_cache()
        base_mem_mb = self._get_peak_active_memory_mb()
        vocab_size = 32
        model_args = ModelArgs(
            vocab_size=vocab_size, n_layers=3, dim=768, n_heads=12, weight_tying=False
        )
        model = Transformer(model_args)
        model_unsharded_numel = sum(p.numel() for p in model.parameters())
        model_sharded_numel = (model_unsharded_numel + 1) // 2
        max_unsharded_numel = sum(
            p.numel() for p in model.layers[0].parameters()
        )  # i.e. block unsharded numel
        non_block_numel = round(
            sum(p.numel() for p in model.tok_embeddings.parameters())
            + sum(p.numel() for p in model.pos_embeddings.parameters())
            + sum(p.numel() for p in model.norm.parameters())
            + sum(p.numel() for p in model.output.parameters())
        )
        fully_shard_fn = functools.partial(
            fully_shard,
            reshard_after_forward=reshard_after_forward,
            offload_policy=CPUOffloadPolicy() if use_cpu_offload else OffloadPolicy(),
        )
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard_fn(module)
        fully_shard_fn(model)
        # Do not use foreach since intermediates increase peak memory
        optim_kwargs = {"lr": 1e-2, "foreach": False}
        if run_optim_in_backward:
            self._register_optim_in_backward(model, **optim_kwargs)
        else:
            optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=False)

        # Init: Each module is moved to GPU before sharding parameters
        peak_mem_mb = self._get_peak_active_memory_mb()
        curr_mem_mb = self._get_curr_active_memory_mb()
        # Allow for some buffer for the peak memory since original parameters
        # are not freed until a `fully_shard` call returns
        buffer_mb = 4
        if use_cpu_offload:
            # Parameters are offloaded after sharding
            init_mem_mb = (1.5 * max_unsharded_numel) * 4 / 1e6
        else:
            init_mem_mb = (model_sharded_numel + max_unsharded_numel) * 4 / 1e6
        self.assertLessEqual(peak_mem_mb - base_mem_mb, init_mem_mb + buffer_mb)
        self.assertLessEqual(curr_mem_mb - base_mem_mb, init_mem_mb)

        # Use a small input to minimize activation memory usage
        inp = torch.randint(0, vocab_size, (1, 4), device=device_type.type)

        # Forward:
        loss = model(inp)
        mem_mb = self._get_peak_active_memory_mb()
        # Allow for some buffer for fragmentation/activations (where this
        # number is kept much smaller than the actual memory usage, which is on
        # the order of 100-200+ MB)
        buffer_mb = 16
        # The default workspace for hipblaslt is larger than for cublas/cublaslt
        # which requires a slight increase to this buffer value.
        buffer_mb = 16 if torch.version.cuda else 18
        if reshard_after_forward:
            # 3x max unsharded block parameters (current all-gather + copy-out
            # and next all-gather), non-block parameters, and other
            expected_mem_mb = (
                3 * max_unsharded_numel + non_block_numel
            ) * 4 / 1e6 + buffer_mb
            if not use_cpu_offload:
                # Sharded parameters
                expected_mem_mb += model_sharded_numel * 4 / 1e6
        else:
            if use_cpu_offload:
                raise AssertionError("Expected use_cpu_offload to be False")
            # Sharded parameters, unsharded parameters, 1x max unsharded block parameters
            # (copy-out) and other (peak at end of forward)
            expected_mem_mb = (
                model_sharded_numel + model_unsharded_numel + max_unsharded_numel
            ) * 4 / 1e6 + buffer_mb
        self.assertLessEqual(mem_mb - base_mem_mb, expected_mem_mb)

        # Backward:
        loss.sum().backward()
        mem_mb = self._get_peak_active_memory_mb()
        if reshard_after_forward:
            # 2x max unsharded block parameters (all-gather + copy-out), 2x max
            # unsharded block gradients (gradients, reduce-scatter input),
            # non-block parameters, and other
            # NOTE: Reduce-scatter output is counted as part of the 1x sharded
            # gradients below since the gradients view into the output
            expected_mem_mb = (
                4 * max_unsharded_numel + non_block_numel
            ) * 4 / 1e6 + buffer_mb
            if not use_cpu_offload:
                if run_optim_in_backward:
                    # 1x sharded parameters
                    expected_mem_mb += model_sharded_numel * 4 / 1e-6
                    # 1x sharded block gradients
                    expected_mem_mb += max_unsharded_numel // self.world_size * 4 / 1e-6
                else:
                    # 2x sharded parameters/gradients
                    expected_mem_mb += 2 * model_sharded_numel * 4 / 1e6
        else:
            if use_cpu_offload:
                raise AssertionError("Expected use_cpu_offload to be False")
            # Sharded parameters, unsharded parameters, 1.5x max unsharded
            # block parameters (reduce-scatter input/output), and other (peak
            # at beginning of backward)
            expected_mem_mb = (
                model_sharded_numel + model_unsharded_numel + 1.5 * max_unsharded_numel
            ) * 4 / 1e6 + buffer_mb
        self.assertLessEqual(mem_mb - base_mem_mb, expected_mem_mb)
        del loss
        torch.get_device_module(device_type).reset_peak_memory_stats()

        # Optimizer step: unsharded parameters/gradients freed
        if not run_optim_in_backward:
            optim.step()
        mem_mb = self._get_peak_active_memory_mb()
        expected_mem_mb = buffer_mb
        if not use_cpu_offload:
            # 1x sharded parameters, 2x sharded optimizer states
            expected_mem_mb += (3 * model_sharded_numel) * 4 / 1e6
            if not run_optim_in_backward:
                # 1x sharded gradients
                expected_mem_mb += model_sharded_numel * 4 / 1e6
        self.assertLessEqual(mem_mb - base_mem_mb, expected_mem_mb)

        # Zero grad: sharded gradients freed
        if not run_optim_in_backward:
            optim.zero_grad()
        torch.get_device_module(
            device_type
        ).reset_peak_memory_stats()  # reset after freeing
        mem_mb = self._get_peak_active_memory_mb()
        expected_mem_mb = 0
        if not use_cpu_offload:
            # 1x sharded parameters
            expected_mem_mb += model_sharded_numel * 4 / 1e6 + buffer_mb
            # 2x sharded optimizer states
            expected_mem_mb += (2 * model_sharded_numel) * 4 / 1e6 + buffer_mb
        self.assertLessEqual(mem_mb - base_mem_mb, expected_mem_mb)

    @skip_if_lt_x_gpu(2)
    def test_fully_shard_training_memory_no_gc(self):
        """Memory should not grow across training steps when GC is disabled.

        Regression test: reference cycles in FSDP's autograd integration can
        hold GPU tensors alive when Python's cyclic GC is disabled. This
        catches leaks like the one introduced in #173415.
        """
        torch.manual_seed(42)
        gc.disable()
        try:
            vocab_size = 1024
            model_args = ModelArgs(
                vocab_size=vocab_size,
                n_layers=6,
                dim=2048,
                n_heads=16,
                weight_tying=False,
            )
            model = Transformer(model_args)
            for module in model.modules():
                if isinstance(module, TransformerBlock):
                    fully_shard(module.attention)
                    fully_shard(module.feed_forward)
                    fully_shard(module)
            fully_shard(model)
            optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=False)
            inp = torch.randint(0, vocab_size, (2, 16), device=device_type.type)

            # Warm-up step to stabilize memory (cuBLAS workspaces, etc.)
            loss = model(inp)
            loss.sum().backward()
            optim.step()
            optim.zero_grad()
            gc.collect()  # one-time collection after warm-up
            torch.get_device_module(device_type).synchronize()
            mem_after_warmup = self._get_curr_active_memory_mb()

            num_steps = 10
            for _ in range(num_steps):
                loss = model(inp)
                loss.sum().backward()
                optim.step()
                optim.zero_grad()

            torch.get_device_module(device_type).synchronize()
            mem_after_steps = self._get_curr_active_memory_mb()
            # Allow a small buffer (2 MB) for non-determinism, but no
            # per-step growth should occur.
            self.assertLessEqual(
                mem_after_steps - mem_after_warmup,
                2,
                lambda msg: f"{msg}\nMemory grew by {mem_after_steps - mem_after_warmup} MB over "
                f"{num_steps} steps with gc disabled, indicating a reference "
                f"cycle leak in FSDP's autograd graph",
            )
        finally:
            gc.enable()

    @skip_if_lt_x_gpu(2)
    def test_fully_shard_del_memory(self):
        base_mem_mb = self._get_peak_active_memory_mb()
        vocab_size = 32
        model_args = ModelArgs(
            vocab_size=vocab_size, n_layers=3, dim=768, n_heads=12, weight_tying=False
        )
        model = Transformer(model_args)
        # Initializing the model on CPU should not change the GPU memory usage
        post_model_init_mem_mb = self._get_peak_active_memory_mb()
        self.assertEqual(base_mem_mb, post_model_init_mem_mb)

        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard(module)
        fully_shard(model)
        unsharded_numel = sum(p.numel() for p in model.parameters())
        sharded_numel = unsharded_numel // self.world_size
        buffer_mb = 4
        mem_mb = self._get_curr_active_memory_mb()
        expected_mb = sharded_numel * 4 / 1e6 + buffer_mb
        self.assertLessEqual(mem_mb - base_mem_mb, expected_mb)

        # Deleting the model should free all of the FSDP-managed GPU memory
        del model
        # Manually call garbage collection since there are ref cycles in FSDP
        gc.collect()
        mem_mb = self._get_curr_active_memory_mb()
        self.assertEqual(mem_mb, base_mem_mb)

    @skip_if_lt_x_gpu(2)
    def test_unsharded_grads_freed_after_copy_in(self):
        from torch.distributed.fsdp._fully_shard import _fsdp_collectives

        model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 32)).to(device_type)
        fully_shard(model)
        copy_in = _fsdp_collectives.foreach_reduce_scatter_copy_in
        cast_and_view = _fsdp_collectives._cast_and_view_sharded_grads
        grad_refs: list[weakref.ref] = []
        num_alive_grads: list[int] = []

        def recording_copy_in(grads, *args):
            grad_refs[:] = [weakref.ref(grad) for grad in grads]
            copy_in(grads, *args)

        def counting_cast_and_view(*args):
            num_alive_grads.append(sum(ref() is not None for ref in grad_refs))
            return cast_and_view(*args)

        with (
            mock.patch.object(
                _fsdp_collectives, "foreach_reduce_scatter_copy_in", recording_copy_in
            ),
            mock.patch.object(
                _fsdp_collectives,
                "_cast_and_view_sharded_grads",
                counting_cast_and_view,
            ),
        ):
            model(torch.randn(4, 16, device=device_type)).sum().backward()
        # Clearing foreach_reduce's list after the copy-in must free the
        # unsharded gradients before the reduce allocates its outputs
        self.assertEqual(num_alive_grads, [0])

    def _get_peak_active_memory_mb(self) -> int:
        mem_stats = torch.get_device_module(device_type).memory_stats()
        # HPU uses different memory stat keys.
        if not TEST_HPU:
            return round(mem_stats["active_bytes.all.peak"] / 1e6)
        return round(mem_stats["MaxInUse"] / 1e6)

    def _get_curr_active_memory_mb(self) -> int:
        mem_stats = torch.get_device_module(device_type).memory_stats()
        # HPU uses different memory stat keys.
        if not TEST_HPU:
            return round(mem_stats["active_bytes.all.current"] / 1e6)
        return round(mem_stats["InUse"] / 1e6)

    def _register_optim_in_backward(
        self, model: torch.nn.Module, **optim_kwargs
    ) -> None:
        param_to_optim = {}
        for param in model.parameters():
            param_to_optim[param] = torch.optim.AdamW([param], **optim_kwargs)

        def optim_hook(param: torch.nn.Parameter) -> None:
            param_to_optim[param].step()
            param_to_optim[param].zero_grad()

        for param in model.parameters():
            param.register_post_accumulate_grad_hook(optim_hook)


class TestFullyShardHSDPSyncCorrectness(FSDPTest):
    """Sync-correctness guards for HSDP+AR buffer lifetime.

    These tests exercise cases where the AR output buffer could be reclaimed
    too early by the caching allocator — specifically when `reduce_dtype !=
    orig_dtype`, so the dtype cast at `_fsdp_collectives.py:_to_dtype_if_needed`
    materializes a new tensor and makes the original AR buffer refless inside
    `foreach_reduce`.
    """

    @property
    def world_size(self) -> int:
        return min(4, torch.get_device_module(device_type).device_count())

    # This test is CUDA-specific because it relies on torch.cuda._sleep.
    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(not TEST_CUDA, "HSDP sync correctness test is CUDA-only")
    def test_ar_buffer_lifetime_mixed_dtype(self):
        """Regression guard for PR #140044 (`[FSDP2] Fix CUDA sync for bf16
        HSDP AR, fp32 params`).

        Mechanism: under HSDP+AR when `reduce_dtype != orig_dtype`, the cast
        at `_to_dtype_if_needed(reduce_output, orig_dtype)` materializes a
        new tensor and leaves the original (bf16 / fp16) AR output buffer
        reachable only through `all_reduce_input` inside `foreach_reduce`.
        PR #140044's `_all_reduce_state` holds that ref across layers — which
        keeps the buffer off the caching allocator's free list. Without it,
        the allocator deterministically reuses the same physical block for
        every layer's reduce-scatter output; under slow AR, each layer's RS
        overwrites the shared block before the previous layer's AR has
        finished reading it, so all layers' sharded grads collapse to the
        last-executed layer's value.

        The test surfaces the race by injecting `torch.cuda._sleep` before
        `dist.all_reduce`, keeping the AR kernel active for ~500 ms on the
        AR stream. Sharded grads are compared against a fast-AR reference.

        Parametrized over dtype-mismatch configs so the guard covers the
        whole surface area of the bug (not just PR #140044's canonical
        bf16+fp32 setup).

        Passes on current code (PR #140044's `_all_reduce_state` / PR #179443's
        `comm_ctx.all_reduce_state`). Fails if cross-layer ref tracking is
        removed. See design doc in
        `agent_space/fsdp2_early_release_ar_buffer.md` and PR #180900.
        """
        self.run_subtests(
            {
                "mp_dtype": [torch.bfloat16, torch.float16],
                "use_custom_all_reduce": [False, True],
            },
            self._test_ar_buffer_lifetime_mixed_dtype,
        )

    def _test_ar_buffer_lifetime_mixed_dtype(
        self, mp_dtype: torch.dtype, use_custom_all_reduce: bool
    ):
        torch.manual_seed(0)
        mesh = init_device_mesh(
            device_type.type,
            (2, 2),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        # reduce_dtype = mp_dtype, orig_dtype = fp32 → cast fires.
        mp = MixedPrecisionPolicy(param_dtype=mp_dtype, reduce_dtype=mp_dtype)

        dim = 512
        model = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
        ).to(device_type, dtype=torch.float32)
        for layer in model:
            if isinstance(layer, nn.Linear):
                fully_shard(layer, mesh=mesh, mp_policy=mp)
        fully_shard(model, mesh=mesh, mp_policy=mp)

        torch.manual_seed(42)
        inp = torch.randn(4, dim, device=device_type.type, dtype=torch.float32)

        def run_one(slow_ar: bool):
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None

            orig_all_reduce = dist.all_reduce

            def slow_all_reduce(*args, **kwargs):
                torch.get_device_module(device_type)._sleep(
                    int(500 * get_cycles_per_ms())
                )
                return orig_all_reduce(*args, **kwargs)

            if use_custom_all_reduce:
                custom_comm = mock.MagicMock(
                    side_effect=lambda tensor, group, op: slow_all_reduce(
                        tensor, group=group, op=op
                    )
                )
                for layer in model:
                    if isinstance(layer, nn.Linear):
                        layer.set_custom_all_reduce(custom_comm if slow_ar else None)

            patch_ctx = (
                mock.patch.object(dist, "all_reduce", slow_all_reduce)
                if slow_ar and not use_custom_all_reduce
                else contextlib.nullcontext()
            )
            with patch_ctx:
                out = model(inp)
                loss = out.sum()
                loss.backward()
            if use_custom_all_reduce and slow_ar:
                self.assertEqual(custom_comm.call_count, 3)

            return {
                name: (
                    p.grad.to_local().float().sum().item()
                    if hasattr(p.grad, "to_local")
                    else p.grad.float().sum().item()
                )
                for name, p in model.named_parameters()
                if p.grad is not None
            }

        ref = run_one(slow_ar=False)
        # Repeat the slow-AR run to flush out non-deterministic allocator
        # scheduling: the race depends on the allocator picking the freed
        # block for the next layer's RS, which may not fire on every call.
        for _ in range(3):
            slow = run_one(slow_ar=True)
            for name, ref_sum in ref.items():
                slow_sum = slow[name]
                # When the reduce-dtype RS-output block is shared across
                # layers (no `_all_reduce_state` ref holding it alive), slow
                # AR lets later-layer RSes overwrite the block before
                # earlier layers' AR completes → every layer's grad
                # collapses to the last-executed layer's value. Reference
                # values across the three layers differ by orders of
                # magnitude; any collapse shows up as a large mismatch here.
                self.assertFalse(
                    math.isnan(slow_sum),
                    lambda msg: f"{msg}\nNaN in slow-AR grad for {name}",
                )
                self.assertAlmostEqual(
                    slow_sum,
                    ref_sum,
                    delta=max(abs(ref_sum) * 1e-3, 1e-3),
                    msg=(
                        lambda msg: f"{msg}\nHSDP AR buffer lifetime regression ({mp_dtype}): "
                        f"grad sum for {name} differs under slow AR. "
                        f"reference={ref_sum:.4f}, slow={slow_sum:.4f}. "
                        f"All layers collapsing to the same value indicates "
                        f"the RS-output buffer is being reused across layers "
                        f"(see PR #140044, PR #180900)."
                    ),
                )

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(not TEST_CUDA, "HSDP sync correctness test is CUDA-only")
    def test_partial_repack_after_slow_all_reduce(self):
        # No input requires grad, so both post-backwards run in the final
        # callback. layer's finalize_backward releases its fp32 all-reduce
        # buffer (its bf16 sharded grads are casts), then aux repacks its
        # pending partial because aux.second first gets a gradient. The repack
        # must not reuse that buffer while the slowed all-reduce still uses it.
        class TwoLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = nn.Linear(32, 32, bias=False)
                self.second = nn.Linear(32, 32, bias=False)

            def forward(self, inp, use_second=True):
                output = self.first(inp)
                return output + self.second(inp) if use_second else output

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = TwoLinear()
                self.aux = TwoLinear()

            def forward(self, inp, use_layer):
                output = self.aux(inp, use_second=use_layer)
                return output + self.layer(inp) if use_layer else output

        model = Model().to(device_type, torch.bfloat16)
        mesh = init_device_mesh(
            device_type.type,
            (2, self.world_size // 2),
            mesh_dim_names=("replicate", "shard"),
        )
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
        for module in (model.layer, model.aux, model):
            fully_shard(module, mesh=mesh, mp_policy=mp_policy)
        orig_all_reduce = dist.all_reduce

        def slow_all_reduce(*args, **kwargs):
            torch.cuda._sleep(int(200 * get_cycles_per_ms()))
            return orig_all_reduce(*args, **kwargs)

        with mock.patch.object(dist, "all_reduce", slow_all_reduce):
            for use_layer, value in ((False, 2.0), (True, 3.0)):
                model.set_requires_all_reduce(use_layer)
                inp = torch.full((1, 32), value, device=device_type).bfloat16()
                model(inp, use_layer).sum().backward()
        for module, expected in ((model.layer, (3, 3)), (model.aux, (5, 3))):
            for param, value in zip(module.parameters(), expected):
                actual = param.grad.full_tensor()
                self.assertEqual(actual, torch.full_like(actual, value))


class TestFullyShardAllReduceHookSyncCorrectness(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(2, torch.get_device_module(device_type).device_count())

    # This test is CUDA-specific because it relies on torch.cuda._sleep.
    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(not TEST_CUDA, "all-reduce hook sync test is CUDA-only")
    def test_all_reduce_hook_buffer_lifetime_mixed_dtype(self):
        # Without native HSDP, the all-reduce hook and the cast to orig_dtype
        # read the RS output on the hook stream. If nothing holds it past the
        # cast, the next RS (for the first linear) can reuse its block before
        # the slow hook finishes, and the last linear's grad reads that data.
        torch.manual_seed(0)
        dim = 512
        model = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
        ).to(device_type)
        # bf16 reduce with fp32 params, so the cast drops the last RS output ref
        mp = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
        )
        linears = [layer for layer in model if isinstance(layer, nn.Linear)]
        for linear in linears:
            fully_shard(linear, mp_policy=mp)
        fully_shard(model, mp_policy=mp)
        torch.manual_seed(42 + self.rank)
        inp = torch.randn(4, dim, device=device_type)

        model(inp).sum().backward()
        ref_grads = [param.grad.to_local().clone() for param in model.parameters()]
        model.zero_grad()

        sleep_cycles = int(200 * get_cycles_per_ms())

        def slow_hook(output: torch.Tensor) -> None:
            torch.get_device_module(device_type)._sleep(sleep_cycles)

        for linear in linears:
            linear.set_all_reduce_hook(slow_hook)
        model(inp).sum().backward()
        for param, ref_grad in zip(model.parameters(), ref_grads):
            self.assertEqual(param.grad.to_local(), ref_grad)


class TestFullyShardReduceOutputCurrentStreamUnit(TestCase):
    """Stream-allocation and event-guarded retention contracts without GPU."""

    def _make_fake_streams(self):
        events: list[tuple] = []

        class FakeEvent:
            def __init__(self, stream_name: str, seq: int) -> None:
                self.stream_name = stream_name
                self.seq = seq

        class FakeStream:
            def __init__(self, name: str) -> None:
                self.name = name
                self._seq = 0

            def wait_stream(self, other: "FakeStream") -> None:
                events.append(("wait_stream", self.name, other.name))

            def wait_event(self, event: FakeEvent) -> None:
                events.append(("wait_event", self.name, event.stream_name, event.seq))

            def record_event(self) -> FakeEvent:
                self._seq += 1
                ev = FakeEvent(self.name, self._seq)
                events.append(("record_event", self.name, ev.seq))
                return ev

        current = FakeStream("current")
        rs_stream = FakeStream("rs")
        ar_stream = FakeStream("ar")
        state = {"current": current}

        @contextlib.contextmanager
        def use_stream(stream: FakeStream):
            prev = state["current"]
            state["current"] = stream
            try:
                yield
            finally:
                state["current"] = prev

        handle = types.SimpleNamespace(
            current_stream=lambda: state["current"],
            stream=use_stream,
        )
        return handle, current, rs_stream, ar_stream, events

    def _make_fake_fsdp_param(
        self,
        shape: tuple[int, ...],
        sharded_grad_dtype: torch.dtype,
        *,
        padded_shape: tuple[int, ...] | None = None,
        needs_nonzero_dim_padding: bool = False,
    ):
        padded = torch.Size(padded_shape or shape)
        sharded_size = torch.Size(shape)
        stride = torch.empty(sharded_size).stride()
        sharded_param = types.SimpleNamespace(
            grad=None, _post_accumulate_grad_hooks=None
        )
        return types.SimpleNamespace(
            fsdp_placement=types.SimpleNamespace(dim=0),
            padded_sharded_param_size=padded,
            sharded_size=sharded_size,
            contiguous_sharded_stride=stride,
            sharded_grad_dtype=sharded_grad_dtype,
            needs_nonzero_dim_padding=needs_nonzero_dim_padding,
            offload_to_cpu=False,
            pin_memory=False,
            sharded_param=sharded_param,
            to_sharded_dtensor=lambda grad: grad,
        )

    def test_foreach_reduce_allocates_in_current_stream_and_retains_refs(self):
        from torch.distributed.fsdp._fully_shard import _fsdp_collectives

        cases = [
            # (reduce_output_in_current_stream, sharded_dtype, has_ar, ar_grads, has_hook, nonzero_pad)
            (False, torch.float32, False, False, False, False),
            (True, torch.float32, False, False, False, False),
            (True, torch.float32, False, False, False, True),
            (True, torch.bfloat16, False, False, False, False),
            (True, torch.bfloat16, True, True, False, False),
            (True, torch.bfloat16, True, False, False, False),
            (True, torch.bfloat16, False, False, True, False),
        ]
        for (
            in_current,
            sharded_dtype,
            has_ar,
            ar_grads,
            has_hook,
            nonzero_pad,
        ) in cases:
            handle, current, rs_stream, ar_stream, events = self._make_fake_streams()
            alloc_streams: list[str] = []
            comm_streams: list[str] = []
            orig_empty = torch.empty

            def tracking_empty(*args, **kwargs):
                alloc_streams.append(handle.current_stream().name)
                return orig_empty(*args, **kwargs)

            class FakeRSComm:
                def allocate(self, size, *, dtype, device):
                    alloc_streams.append(handle.current_stream().name)
                    return orig_empty(*size, dtype=dtype, device="cpu")

                def __call__(self, *, output_tensor, input_tensor, group, op):
                    comm_streams.append(handle.current_stream().name)
                    output_tensor.copy_(input_tensor[: output_tensor.numel()])

            class FakeARComm:
                def __call__(self, *, tensor, group, op):
                    comm_streams.append(handle.current_stream().name)

            fake_dev = types.SimpleNamespace(type="fake_acc")
            fsdp_param = self._make_fake_fsdp_param(
                (2, 2),
                sharded_dtype,
                padded_shape=(2, 3) if nonzero_pad else (2, 2),
                needs_nonzero_dim_padding=nonzero_pad,
            )
            unsharded_grad = torch.ones(
                (4, 3 if nonzero_pad else 2), dtype=torch.float32
            )

            def fake_copy_in(grads, *, shard_dims, world_size):
                def _fill(buf):
                    buf.fill_(1.0)

                return _fill

            hook = (lambda _t: None) if has_hook else None
            with (
                mock.patch.object(
                    _fsdp_collectives, "_get_device_handle", return_value=handle
                ),
                mock.patch.object(
                    _fsdp_collectives,
                    "record_grad_output_stream",
                    side_effect=AssertionError(
                        "record_grad_output_stream should not run"
                    )
                    if in_current and (not has_ar or ar_grads)
                    else None,
                ),
                mock.patch.object(torch, "empty", side_effect=tracking_empty),
                mock.patch.object(
                    torch.Tensor,
                    "record_stream",
                    side_effect=AssertionError("record_stream must not be called"),
                ),
            ):
                (
                    rs_input,
                    rs_event,
                    post_stream,
                    post_event,
                    ar_input,
                    ar_event,
                    partial_out,
                ) = _fsdp_collectives.foreach_reduce(
                    [fsdp_param],
                    [unsharded_grad],
                    types.SimpleNamespace(size=lambda: 2),
                    rs_stream,
                    FakeRSComm(),
                    torch.float32,
                    fake_dev,
                    1.0,
                    types.SimpleNamespace(size=lambda: 2) if has_ar else None,
                    ar_stream,
                    ar_grads,
                    None,
                    hook,
                    prepare_reduce_scatter_inputs=fake_copy_in,
                    all_reduce_comm=FakeARComm(),
                    reduce_output_in_current_stream=in_current,
                )

            self.assertEqual(comm_streams[0], "rs")
            if not in_current:
                # Default behavior: input on current, output on rs
                self.assertEqual(alloc_streams[:2], ["current", "rs"])
            elif has_ar and not ar_grads:
                # HSDP partial step stays internal to rs stream
                self.assertEqual(alloc_streams[:2], ["current", "rs"])
                self.assertIsNotNone(partial_out)
            else:
                # Every allocation (input, output, and any cast/unpad) lives on current stream
                self.assertTrue(
                    all(stream == "current" for stream in alloc_streams),
                    f"Expected all allocations on current stream, got {alloc_streams}",
                )
                if has_ar or has_hook:
                    self.assertEqual(post_stream, ar_stream)
                    self.assertEqual(rs_event.stream_name, "rs")
                    self.assertIs(ar_event, post_event)
                    if sharded_dtype != torch.float32:
                        self.assertIsInstance(ar_input, list)
                        self.assertEqual(len(ar_input), 2)
                    if has_hook and sharded_dtype != torch.float32:
                        self.assertIn(("wait_stream", "ar", "current"), events)
                else:
                    self.assertEqual(post_stream, rs_stream)
                    self.assertIs(rs_event, post_event)
                    self.assertIsInstance(rs_input, list)
                    expected_len = (
                        3 if (sharded_dtype != torch.float32 or nonzero_pad) else 2
                    )
                    self.assertEqual(len(rs_input), expected_len)

    def test_reduce_single_rank_allocates_cast_in_current_stream(self):
        from torch.distributed.fsdp._fully_shard import _fsdp_collectives

        handle, current, rs_stream, _, _ = self._make_fake_streams()
        alloc_streams: list[str] = []
        orig_empty = torch.empty

        def tracking_empty(*args, **kwargs):
            alloc_streams.append(handle.current_stream().name)
            return orig_empty(*args, **kwargs)

        fsdp_param = self._make_fake_fsdp_param((2, 2), torch.bfloat16)
        grad = torch.ones((2, 2), dtype=torch.float32)
        with (
            mock.patch.object(
                _fsdp_collectives, "_get_device_handle", return_value=handle
            ),
            mock.patch.object(torch, "empty", side_effect=tracking_empty),
        ):
            keepalive, rs_event, _, post_event, _, _, _ = (
                _fsdp_collectives._reduce_single_rank(
                    [fsdp_param],
                    [grad],
                    rs_stream,
                    torch.float32,
                    types.SimpleNamespace(type="fake_acc"),
                    [],
                    reduce_output_in_current_stream=True,
                )
            )
        self.assertEqual(alloc_streams, ["current"])
        self.assertIs(rs_event, post_event)
        self.assertEqual(len(keepalive), 2)
        self.assertEqual(fsdp_param.sharded_param.grad.dtype, torch.bfloat16)

    def test_release_all_reduce_state_waits_on_allocation_stream(self):
        from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
            AllReduceState,
            FSDPParamGroup,
        )

        handle, current, rs_stream, ar_stream, events = self._make_fake_streams()
        ev = ar_stream.record_event()
        group = object.__new__(FSDPParamGroup)
        group.device_handle = handle
        group._all_reduce_state = AllReduceState(
            torch.empty(1), ev, allocation_stream=rs_stream
        )
        group._release_all_reduce_state()
        self.assertIsNone(group._all_reduce_state)
        self.assertIn(("wait_event", "rs", "ar", ev.seq), events)
        self.assertIn(("wait_event", "current", "ar", ev.seq), events)


class TestFullyShardReduceOutputCurrentStream(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(2, torch.get_device_module(device_type).device_count())

    @skip_if_lt_x_gpu(2)
    def test_set_reduce_output_in_current_stream_recurse(self):
        model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8)).to(device_type)
        # Wrap only model[0] and root model so both hold non-empty param groups.
        fully_shard(model[0])
        fully_shard(model)

        for mod in (model, model[0]):
            groups = mod._get_fsdp_state()._fsdp_param_groups
            self.assertGreater(len(groups), 0)
            self.assertTrue(all(not g.reduce_output_in_current_stream for g in groups))

        model.set_reduce_output_in_current_stream(True, recurse=False)
        self.assertTrue(
            all(
                g.reduce_output_in_current_stream
                for g in model._get_fsdp_state()._fsdp_param_groups
            )
        )
        self.assertTrue(
            all(
                not g.reduce_output_in_current_stream
                for g in model[0]._get_fsdp_state()._fsdp_param_groups
            )
        )

        model.set_reduce_output_in_current_stream(True)
        for mod in (model, model[0]):
            self.assertTrue(
                all(
                    g.reduce_output_in_current_stream
                    for g in mod._get_fsdp_state()._fsdp_param_groups
                )
            )

    @skip_if_lt_x_gpu(2)
    def test_reduce_output_in_current_stream_training_parity(self):
        self.run_subtests(
            {
                "mixed_precision": [False, True],
                "grad_accum_steps": [1, 2],
                "use_cpu_offload": [False, True],
                "run_optim_in_backward": [False, True],
            },
            self._test_reduce_output_in_current_stream_training_parity,
        )

    def _test_reduce_output_in_current_stream_training_parity(
        self,
        mixed_precision: bool,
        grad_accum_steps: int,
        use_cpu_offload: bool,
        run_optim_in_backward: bool,
    ):
        if use_cpu_offload and run_optim_in_backward:
            return
        from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
            DefaultReduceScatter,
        )

        def run_mode(enable_current_stream: bool):
            torch.manual_seed(42)
            model = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
            ).to(device_type)
            for param in model.parameters():
                dist.broadcast(param.detach(), src=0)
            mp_policy = (
                MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
                )
                if mixed_precision
                else MixedPrecisionPolicy()
            )
            offload = CPUOffloadPolicy() if use_cpu_offload else OffloadPolicy()
            for mod in (model[0], model[2], model[4], model):
                fully_shard(mod, mp_policy=mp_policy, offload_policy=offload)
            if enable_current_stream:
                model.set_reduce_output_in_current_stream(True)

            if run_optim_in_backward:
                param_optims = {
                    p: torch.optim.AdamW([p], lr=1e-2, foreach=False)
                    for p in model.parameters()
                }

                def _hook(p):
                    param_optims[p].step()
                    param_optims[p].zero_grad()

                for p in model.parameters():
                    p.register_post_accumulate_grad_hook(_hook)
            else:
                optim = torch.optim.AdamW(model.parameters(), lr=1e-2, foreach=False)

            dev_mod = torch.get_device_module(device_type)
            alloc_streams: list[torch.Stream] = []
            comm_streams: list[torch.Stream] = []
            orig_alloc = DefaultReduceScatter.allocate
            orig_call = DefaultReduceScatter.__call__

            def spy_alloc(self_comm, size, *, dtype, device):
                alloc_streams.append(dev_mod.current_stream())
                return orig_alloc(self_comm, size, dtype=dtype, device=device)

            def spy_call(self_comm, *args, **kwargs):
                comm_streams.append(dev_mod.current_stream())
                return orig_call(self_comm, *args, **kwargs)

            compute_stream = dev_mod.current_stream()
            losses = []
            with (
                mock.patch.object(DefaultReduceScatter, "allocate", spy_alloc),
                mock.patch.object(DefaultReduceScatter, "__call__", spy_call),
            ):
                for step in range(2):
                    for micro in range(grad_accum_steps):
                        torch.manual_seed(100 + step * 10 + micro + self.rank)
                        inp = torch.randn(4, 64, device=device_type)
                        loss = model(inp).square().sum()
                        loss.backward()
                        losses.append(loss.item())
                    if not run_optim_in_backward:
                        optim.step()
                        optim.zero_grad()

            self.assertGreater(len(alloc_streams), 0)
            self.assertTrue(all(s != compute_stream for s in comm_streams))
            if enable_current_stream:
                self.assertTrue(all(s == compute_stream for s in alloc_streams))
            else:
                self.assertTrue(any(s != compute_stream for s in alloc_streams))

            params = [p.to_local().detach().cpu() for p in model.parameters()]
            return losses, params

        ref_losses, ref_params = run_mode(False)
        cur_losses, cur_params = run_mode(True)
        for ref_l, cur_l in zip(ref_losses, cur_losses):
            self.assertAlmostEqual(ref_l, cur_l, places=4)
        for ref_p, cur_p in zip(ref_params, cur_params):
            self.assertEqual(cur_p, ref_p, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    run_tests()
