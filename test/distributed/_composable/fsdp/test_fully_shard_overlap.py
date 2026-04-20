# Owner(s): ["oncall: distributed"]

import contextlib
import copy
import functools
import unittest
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    FSDPMeshInfo,
    ShardPlacementResult,
)
from torch.distributed.tensor import init_device_mesh, Shard
from torch.distributed.tensor.experimental import implicit_replication
from torch.testing._internal.common_distributed import (
    skip_if_lt_x_gpu,
    skip_if_rocm_arch_multiprocess,
)
from torch.testing._internal.common_fsdp import (
    FSDPTest,
    get_devtype,
    patch_all_gather,
    patch_all_reduce,
    patch_reduce_scatter,
)
from torch.testing._internal.common_utils import (
    get_cycles_per_ms,
    MI200_ARCH,
    run_tests,
    TEST_HPU,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)


device_type = torch.device(get_devtype())
device_module = torch.get_device_module(device_type)


def _time_fn(fn: Callable):
    start_event = device_module.Event(enable_timing=True)
    end_event = device_module.Event(enable_timing=True)
    dist.barrier()
    device_module.synchronize()
    start_event.record()
    fn()
    end_event.record()
    device_module.synchronize()
    return start_event.elapsed_time(end_event)


class TestFullyShardOverlap(FSDPTest):
    """
    NOTE: Testing stream overlap in PyTorch CI is tricky.

    One approach is to use CUDA sleeps to emulate kernels in each stream;
    however, ``torch.cuda._sleep`` requires inputs in units of cycles. The
    ``get_cycles_per_ms`` function to convert from ms to cycles is computed
    once and cached thereafter, which means that if there is variation later,
    the cached value may not be accurate. This leads to flakiness in CI.

    To address this, we relax the tests as simple sanity checks that the
    overlapped times are less than a non-overlapped baseline, but we do not
    test that the overlapped time is less than a precisely calculated time.
    """

    @property
    def world_size(self) -> int:
        return min(2, torch.get_device_module(device_type).device_count())

    @skip_if_rocm_arch_multiprocess(MI200_ARCH)
    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_fully_shard_training_overlap(self):
        torch.manual_seed(42)

        # Use non-trivial comm. time but still shorter than compute time
        dim, num_linears, compute_sleep_ms, comm_sleep_ms = (4, 3, 25, 10)
        model = nn.Sequential(
            *[LinearWithSleep(dim, compute_sleep_ms) for _ in range(num_linears)]
        )
        ref_model = copy.deepcopy(model).to(device_type)
        for lin in model:
            if len(list(lin.parameters())) != 1:
                raise AssertionError("Expects only one weight")
            fully_shard(lin, reshard_after_forward=True)
        fully_shard(model, reshard_after_forward=True)

        orig_all_gather_into_tensor = dist.all_gather_into_tensor
        orig_reduce_scatter_tensor = dist.reduce_scatter_tensor
        comm_stream = torch.get_device_module(device_type).Stream()

        def delay_collective():
            # Share a stream so that all-gather and reduce-scatter block each
            # other like in `ProcessGroupNCCL`
            comm_stream.wait_stream(
                torch.get_device_module(device_type).current_stream()
            )
            with torch.get_device_module(device_type).stream(comm_stream):
                torch.get_device_module(device_type)._sleep(
                    int(comm_sleep_ms * get_cycles_per_ms())
                )
            torch.get_device_module(device_type).current_stream().wait_stream(
                comm_stream
            )

        def delayed_all_gather(*args, **kwargs):
            delay_collective()
            return orig_all_gather_into_tensor(*args, **kwargs)

        def delayed_reduce_scatter(*args, **kwargs):
            delay_collective()
            return orig_reduce_scatter_tensor(*args, **kwargs)

        inp = torch.randn((2, dim), device=device_type.type)
        loss = model(inp).sum()  # warmup CUDA and allocator
        loss.backward()

        def ref_fwd():
            with patch_all_gather(delayed_all_gather):
                # Run dummy all-gathers per weight (which is one FSDP group)
                for lin in ref_model:
                    dummy_ag_output = torch.empty_like(lin.weight)
                    dummy_ag_input = torch.chunk(dummy_ag_output, self.world_size)[
                        self.rank
                    ]
                    dist.all_gather_into_tensor(dummy_ag_output, dummy_ag_input)
                return ref_model(inp)

        def fwd():
            with patch_all_gather(delayed_all_gather):
                model(inp)

        ref_fwd_time = _time_fn(ref_fwd)
        fwd_time = _time_fn(fwd)
        # Forward: only 1st all-gather is exposed
        # NOTE: Do not enforce the expected forward time due to flakiness in CI
        # expected_fwd_time = comm_sleep_ms + num_linears * compute_sleep_ms + buffer_ms
        self.assertLessEqual(fwd_time, ref_fwd_time)

        def ref_fwd_bwd():
            with patch_all_gather(delayed_all_gather):
                # Run dummy all-gathers per weight (which is one FSDP group)
                for lin in ref_model:
                    dummy_ag_output = torch.empty_like(lin.weight)
                    dummy_ag_input = torch.chunk(dummy_ag_output, self.world_size)[
                        self.rank
                    ]
                    dist.all_gather_into_tensor(dummy_ag_output, dummy_ag_input)
                loss = ref_model(inp).sum()
                # Run dummy all-gathers per weight again since we are
                # resharding after forward
                for lin in ref_model:
                    dummy_ag_output = torch.empty_like(lin.weight)
                    dummy_ag_input = torch.chunk(dummy_ag_output, self.world_size)[
                        self.rank
                    ]
                    dist.all_gather_into_tensor(dummy_ag_output, dummy_ag_input)
                loss.backward()
                # Run dummy reduce-scatters per weight
                for lin in ref_model:
                    dummy_rs_input = torch.empty_like(lin.weight)
                    dummy_rs_output = torch.chunk(dummy_rs_input, self.world_size)[
                        self.rank
                    ]
                    dist.reduce_scatter_tensor(dummy_rs_output, dummy_rs_input)

        def fwd_bwd():
            with (
                patch_all_gather(delayed_all_gather),
                patch_reduce_scatter(delayed_reduce_scatter),
            ):
                loss = model(inp).sum()
                loss.backward()

        ref_fwd_bwd_time = _time_fn(ref_fwd_bwd)
        fwd_bwd_time = _time_fn(fwd_bwd)
        # Backward: only 1st all-gather and last reduce-scatter are exposed;
        # double the backward compute since computing two gradients per layer
        # NOTE: Do not enforce the expected forward-backward time due to
        # flakiness in CI
        # expected_bwd_time = (
        #     comm_sleep_ms * 2 + num_linears * 2 * compute_sleep_ms + buffer_ms * 2
        # )
        self.assertLessEqual(fwd_bwd_time, ref_fwd_bwd_time)

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_fully_shard_post_optim_event_overlap(self):
        torch.manual_seed(42)

        # Use non-trivial comm. time but still shorter than compute time
        dim, compute_sleep_ms, comm_sleep_ms = (4, 25, 10)
        # Define the model to have a high-compute linear followed by a
        # low-compute linear, where only the low-compute linear uses FSDP
        model = nn.Sequential(
            LinearWithSleep(dim, compute_sleep_ms), nn.Linear(dim, dim)
        ).to(device_type)
        fully_shard(model[1], reshard_after_forward=False)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

        orig_all_gather_into_tensor = dist.all_gather_into_tensor

        def delayed_all_gather(*args, **kwargs):
            torch.get_device_module(device_type)._sleep(
                int(comm_sleep_ms * get_cycles_per_ms())
            )
            return orig_all_gather_into_tensor(*args, **kwargs)

        inp = torch.randn((2, dim), device=device_type)

        def run_train_steps(num_iters: int, use_post_optim_event: bool):
            for _ in range(num_iters):
                optim.zero_grad()
                with patch_all_gather(delayed_all_gather):
                    loss = model(inp).sum()
                loss.backward()
                with implicit_replication():
                    optim.step()
                if use_post_optim_event:
                    post_optim_event = (
                        torch.get_device_module(device_type)
                        .current_stream()
                        .record_event()
                    )
                    model[1].set_post_optim_event(post_optim_event)

        run_train_steps(1, False)  # warmup CUDA and allocator
        num_iters = 5
        baseline_time = _time_fn(functools.partial(run_train_steps, num_iters, False))
        test_time = _time_fn(functools.partial(run_train_steps, num_iters, True))

        buffer_ms = 4  # CPU delays and copies
        # Baseline: FSDP all-gather is exposed since the FSDP module waits for
        # the current stream and hence the high-compute linear
        self.assertLessEqual(
            baseline_time,
            num_iters * (3 * compute_sleep_ms + comm_sleep_ms + buffer_ms),
        )
        # Test: FSDP all-gather is overlapped with the high-compute linear
        # since the FSDP module only waits for the post-optim event (except on
        # the 1st iteration when no event has been recorded)
        expected_test_time = (
            num_iters * (3 * compute_sleep_ms + buffer_ms) + comm_sleep_ms
        )
        self.assertLessEqual(test_time, expected_test_time)
        # Since `get_cycles_per_ms` uses lru cache, there may be some variance
        # between the initially determined cycles vs. the current cycles per
        # ms, so we relax the baseline check to just that it is greater than
        # the test time rather than the expected test time
        self.assertGreater(baseline_time, test_time)

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(TEST_HPU, "pin_memory requires GPU")
    def test_offload_post_accumulate_grad_hook_sees_reduced_grad(self):
        """A param with a registered ``post_accumulate_grad_hook`` under
        ``CPUOffloadPolicy`` must have its D2H copy forced synchronous —
        otherwise the hook (e.g. optimizer-in-backward, see
        ``torch/_tensor.py``) reads stale pinned memory while the async
        memcpy is still in flight. ``foreach_reduce`` extends its
        ``non_blocking`` gate to disable async offload when a hook is
        registered. Reproducibility was ~2/3 runs with diffs up to 2.0
        before the fix.
        """
        from torch.distributed.fsdp import CPUOffloadPolicy

        torch.manual_seed(42)
        dim = 4
        model = nn.Sequential(
            *[nn.Linear(dim, dim, device=device_type) for _ in range(3)]
        )
        ref = copy.deepcopy(model).to(device_type)
        off = copy.deepcopy(model).to(device_type)
        for lin in ref:
            fully_shard(lin)
        fully_shard(ref)
        for lin in off:
            fully_shard(lin, offload_policy=CPUOffloadPolicy())
        fully_shard(off, offload_policy=CPUOffloadPolicy())

        hook_observed: dict[int, torch.Tensor] = {}

        def _hook(p: torch.Tensor) -> None:
            g = p.grad
            local = g.to_local() if hasattr(g, "to_local") else g
            hook_observed[id(p)] = local.detach().cpu().clone()

        for p in off.parameters():
            p.register_post_accumulate_grad_hook(_hook)

        inp = torch.randn((2, dim), device=device_type)
        ref(inp).sum().backward()
        off(inp).sum().backward()

        for ref_p, off_p in zip(ref.parameters(), off.parameters()):
            if ref_p.grad is None or id(off_p) not in hook_observed:
                continue
            ref_local = (
                ref_p.grad.to_local() if hasattr(ref_p.grad, "to_local") else ref_p.grad
            )
            torch.testing.assert_close(
                hook_observed[id(off_p)],
                ref_local.cpu(),
                atol=1e-5,
                rtol=1e-4,
                msg=lambda msg: (
                    f"post_accumulate_grad_hook observed stale grad: {msg}. "
                    f"foreach_reduce should force non_blocking=False for the "
                    f"D2H copy when a hook is registered."
                ),
            )


class Matmul(torch.autograd.Function):
    # Use CUDA sleeps to emulate compute time
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, sleep_ms: int):
        ctx.save_for_backward(input, weight)
        ctx.sleep_ms = sleep_ms
        torch.get_device_module(device_type)._sleep(int(sleep_ms * get_cycles_per_ms()))
        return input @ weight

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (input, weight) = ctx.saved_tensors
        torch.get_device_module(device_type)._sleep(
            int(2 * ctx.sleep_ms * get_cycles_per_ms())
        )
        grad_input = grad_output @ weight.T
        grad_weight = input.T @ grad_output
        return grad_input, grad_weight, None


class LinearWithSleep(nn.Module):
    def __init__(self, dim: int, sleep_ms: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((dim, dim)))
        self.sleep_ms = sleep_ms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.relu(Matmul.apply(x, self.weight, self.sleep_ms))


class TestFullyShardPerParamMeshOverlap(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.get_device_module(device_type).device_count())

    @staticmethod
    @contextlib.contextmanager
    def _delayed_foreach_reduce(sleep_ms: int):
        """Inject a CUDA sleep after each reduce-scatter / all-reduce.

        Each process group gets its own delay stream so that delays for
        different groups run concurrently (just like real NCCL ops on
        separate communicators).  The event returned to FSDP is replaced
        with a delayed event on this per-group stream.

        This amplifies the difference between per-group and shared RS
        state:
        - Per-group RS state: the compute stream waits only on its own
          group's delayed event, so dp and efsdp delays overlap.
        - Shared RS state: the compute stream waits on *all* groups'
          delayed events, serializing the dp and efsdp delays.
        """
        import torch.distributed.fsdp._fully_shard._fsdp_param_group as _pg_mod

        orig = _pg_mod.foreach_reduce
        # One delay stream per process group so that sleeps for different
        # groups (e.g. dp vs efsdp reduce-scatter) execute in parallel,
        # mirroring how real NCCL ops on separate communicators overlap.
        delay_streams: dict[dist.ProcessGroup, torch.cuda.Stream] = {}

        def wrapped(
            fsdp_params,
            unsharded_grads,
            reduce_scatter_group,
            reduce_scatter_stream,
            *args,
            **kwargs,
        ):
            result = orig(
                fsdp_params,
                unsharded_grads,
                reduce_scatter_group,
                reduce_scatter_stream,
                *args,
                **kwargs,
            )
            rs_input, rs_event, *rest = result
            if reduce_scatter_group not in delay_streams:
                delay_streams[reduce_scatter_group] = device_module.Stream()
            ds = delay_streams[reduce_scatter_group]
            ds.wait_event(rs_event)
            with device_module.stream(ds):
                device_module._sleep(int(sleep_ms * get_cycles_per_ms()))
                delayed_event = ds.record_event()
            return (rs_input, delayed_event, *rest)

        dist.barrier()
        _pg_mod.foreach_reduce = wrapped
        try:
            yield
        finally:
            dist.barrier()
            _pg_mod.foreach_reduce = orig

    @skip_if_rocm_arch_multiprocess(MI200_ARCH)
    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_fully_shard_per_param_mesh_training_overlap(self):
        self._test_per_param_mesh_overlap(simulate_no_grad_input=False)

    @skip_if_rocm_arch_multiprocess(MI200_ARCH)
    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_fully_shard_per_param_mesh_no_grad_input_overlap(self):
        self._test_per_param_mesh_overlap(simulate_no_grad_input=True)

    def _test_per_param_mesh_overlap(self, simulate_no_grad_input: bool):
        """Verify per-param-group reduce-scatter state avoids cross-group
        serialization in per-param-mesh FSDP with expert parallelism.

        When ``simulate_no_grad_input`` is True, block inputs are detached so
        that autograd's RegisterPostBackwardFunction nodes are never inserted,
        exercising the safety-net path in _root_post_backward_final_callback.
        """
        import types

        from torch.distributed._composable.replicate_with_fsdp import replicate

        ep_degree = 2
        efsdp_size = self.world_size // ep_degree
        comm_sleep_ms = 500
        world_mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("world",),
        )
        dp_mesh = world_mesh._unflatten(0, (self.world_size,), ("fsdp",))["fsdp"]
        sparse_mesh = dp_mesh._unflatten(0, (efsdp_size, ep_degree), ("efsdp", "ep"))
        ep_mesh = sparse_mesh["ep"]
        dp_mesh_info = FSDPMeshInfo(mesh=dp_mesh, shard_mesh_dim=0)
        efsdp_mesh_info = FSDPMeshInfo(mesh=sparse_mesh["efsdp"], shard_mesh_dim=0)
        model_args = ModelArgs(
            n_layers=20,
            vocab_size=1024,
            max_seq_len=64,
            dim=1280,
            n_heads=16,
            dropout_p=0.0,
            num_experts=2,
        )
        inp = torch.randint(
            0,
            model_args.vocab_size,
            (2, model_args.max_seq_len),
            device=device_type.type,
        )

        def _build_fsdp_model():
            torch.manual_seed(42)
            model = Transformer(model_args)
            Transformer.parallelize(
                model, tp_mesh=None, use_seq_parallel=False, ep_mesh=ep_mesh
            )
            for block in model.layers:
                expert_params = set(block.expert_layer.experts.parameters())

                def _shard_placement_fn(param, _expert_params=expert_params):
                    if param in _expert_params:
                        return ShardPlacementResult(
                            placement=Shard(0), mesh_info=efsdp_mesh_info
                        )
                    return ShardPlacementResult(
                        placement=Shard(0), mesh_info=dp_mesh_info
                    )

                fully_shard(block, mesh=dp_mesh, shard_placement_fn=_shard_placement_fn)
            fully_shard(
                [model.tok_embeddings, model.norm, model.output],
                mesh=dp_mesh,
                reshard_after_forward=True,
            )
            fully_shard(model, mesh=dp_mesh, reshard_after_forward=True)
            blocks = model.layers
            for i in range(len(blocks) - 1):
                blocks[i].set_modules_to_forward_prefetch([blocks[i + 1]])
            for i in range(1, len(blocks)):
                blocks[i].set_modules_to_backward_prefetch([blocks[i - 1]])
            return model

        def _build_replicate_model():
            torch.manual_seed(42)
            model = Transformer(model_args)
            Transformer.parallelize(
                model, tp_mesh=None, use_seq_parallel=False, ep_mesh=ep_mesh
            )
            for block in model.layers:
                expert_params = set(block.expert_layer.experts.parameters())
                replicate(block, mesh=dp_mesh, ignored_params=expert_params)
            replicate([model.tok_embeddings, model.norm, model.output], mesh=dp_mesh)
            replicate(model, mesh=dp_mesh)
            return model

        # Both models measured under _delayed_foreach_reduce so the delay
        # applies uniformly to foreach_reduce (all-reduce for replicate,
        # reduce-scatter for FSDP).
        with self._delayed_foreach_reduce(comm_sleep_ms):
            # --- replicate + EP (baseline) ---
            rep_model = _build_replicate_model()

            def rep_fwd_bwd():
                rep_model(inp).sum().backward()  # noqa: F821

            for _ in range(5):
                rep_fwd_bwd()
                rep_model.zero_grad(set_to_none=True)
            rep_time = _time_fn(rep_fwd_bwd)
            rep_model.zero_grad(set_to_none=True)
            del rep_model
            # --- FSDP + EP ---
            fsdp_model = _build_fsdp_model()

            if simulate_no_grad_input:
                # Detach inputs to each block so that autograd's
                # RegisterPostBackwardFunction nodes are never inserted,
                # forcing the safety-net path in
                # _root_post_backward_final_callback.
                def _no_grad_input_forward(self, tokens):
                    h = self.tok_embeddings(tokens)
                    h = h + self.pos_embeddings(
                        torch.arange(tokens.size(1), device=tokens.device)
                    )
                    for layer in self.layers:
                        h = h + layer(h.detach())
                    h = self.norm(h)
                    return self.output(h).float()

                fsdp_model.forward = types.MethodType(
                    _no_grad_input_forward, fsdp_model
                )

            def fsdp_fwd_bwd():
                fsdp_model(inp).sum().backward()  # noqa: F821

            for _ in range(5):
                fsdp_fwd_bwd()
                fsdp_model.zero_grad(set_to_none=True)
            fsdp_time = _time_fn(fsdp_fwd_bwd)
            fsdp_model.zero_grad(set_to_none=True)
        # replicate: 1 all-reduce/block → N+1 serialized waits.
        # FSDP: 2 reduce-scatters/block (dp + efsdp).
        #   Per-group RS state: dp reduce-scatter overlaps with efsdp's
        #     wait → ~N+1 waits ≈ replicate (ratio ≈ 1.0).
        #   Shared RS state: both reduce-scatters serialize → ~2N+1
        #     waits → ratio ≈ (2N+1)/(N+1) → 1.95 for N=20.
        self.assertLess(
            fsdp_time / rep_time,
            1.5,
            f"FSDP/replicate ratio {fsdp_time / rep_time:.2f} >= 1.5; "
            f"per-group RS state may not be preventing cross-group stalls",
        )


class TestFullyShardHSDPStreamBehavior(FSDPTest):
    """HSDP CUDA stream ordering tests using ``_sleep`` injection:

    1. Default compute waits on AG (forward can't proceed before params ready).
    2. AG and RS pipeline through the same dp_shard NCCL communicator (they
       serialize on a shared comm stream, not overlap).
    3. AR waits on RS within a layer's post-reduce.
    4. ARs across layers pipeline on ``comm_ctx.all_reduce_stream``.
    5. Default compute does not wait on RS/AR during backward — only the
       end-of-backward ``finalize_backward`` synchronizes.

    Like ``TestFullyShardOverlap``, these assertions are qualitative (loose
    bounds) to tolerate ``get_cycles_per_ms`` drift in CI.
    """

    @property
    def world_size(self) -> int:
        return min(4, torch.get_device_module(device_type).device_count())

    def _build_hsdp_model(
        self, n_layers: int, dim: int, compute_sleep_ms: int
    ) -> tuple[nn.Module, torch.Tensor]:
        torch.manual_seed(42)
        mesh = init_device_mesh(
            device_type.type,
            (2, 2),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        model = nn.Sequential(
            *[LinearWithSleep(dim, compute_sleep_ms) for _ in range(n_layers)]
        )
        for lin in model:
            fully_shard(lin, mesh=mesh, reshard_after_forward=True)
        fully_shard(model, mesh=mesh, reshard_after_forward=True)
        inp = torch.randn((2, dim), device=device_type.type)
        # Warmup (CUDA, allocator, NCCL)
        model(inp).sum().backward()
        return model, inp

    @staticmethod
    def _delay_on_shared_stream(comm_stream: torch.cuda.Stream, sleep_ms: int) -> None:
        """Sleep on ``comm_stream`` and force the current stream to wait.

        Mirrors ``delay_collective`` in ``test_fully_shard_training_overlap``:
        ties the collective's observable latency to ``sleep_ms``.
        """
        comm_stream.wait_stream(device_module.current_stream())
        with device_module.stream(comm_stream):
            device_module._sleep(int(sleep_ms * get_cycles_per_ms()))
        device_module.current_stream().wait_stream(comm_stream)

    def _make_model_pair(
        self,
        *,
        mesh_shape: tuple[int, ...] = (2, 2),
        n_layers: int = 2,
        dim: int = 4,
        mp: MixedPrecisionPolicy | None = None,
        reshard_after_forward: bool = True,
    ) -> tuple[nn.Module, nn.Module, torch.Tensor]:
        """Build a (ref, off, inp) triple where ``off`` has
        ``CPUOffloadPolicy`` and ``ref`` does not, with otherwise
        identical init and topology on the given mesh shape. 1D ``(N,)``
        shapes yield plain FSDP; 2D ``(R, S)`` shapes yield HSDP. Returns
        an input tensor in the param dtype.
        """
        torch.manual_seed(42)
        mesh_dim_names = (
            ("fsdp",) if len(mesh_shape) == 1 else ("dp_replicate", "dp_shard")
        )
        mesh = init_device_mesh(
            device_type.type, mesh_shape, mesh_dim_names=mesh_dim_names
        )
        param_dtype = mp.param_dtype if mp is not None else torch.float32
        torch.manual_seed(0)
        base = nn.Sequential(
            *[nn.Linear(dim, dim, device=device_type) for _ in range(n_layers)]
        )
        ref = copy.deepcopy(base).to(device_type).to(param_dtype)
        off = copy.deepcopy(base).to(device_type).to(param_dtype)
        shard_kwargs = {
            "mesh": mesh,
            "reshard_after_forward": reshard_after_forward,
        }
        if mp is not None:
            shard_kwargs["mp_policy"] = mp
        for lin in ref:
            fully_shard(lin, **shard_kwargs)
        fully_shard(ref, **shard_kwargs)
        off_kwargs = {**shard_kwargs, "offload_policy": CPUOffloadPolicy()}
        for lin in off:
            fully_shard(lin, **off_kwargs)
        fully_shard(off, **off_kwargs)
        inp = torch.randn((2, dim), device=device_type.type, dtype=param_dtype)
        return ref, off, inp

    def _assert_grad_parity(
        self,
        ref_model: nn.Module,
        off_model: nn.Module,
        *,
        atol: float = 1e-3,
        rtol: float = 1e-2,
        msg_tag: str = "",
    ) -> None:
        """Assert ``.grad`` parity across a ref+off pair, handling DTensor
        and CPU vs GPU tensor placement transparently.
        """
        for ref_p, off_p in zip(ref_model.parameters(), off_model.parameters()):
            if ref_p.grad is None:
                continue
            ref_local = (
                ref_p.grad.to_local() if hasattr(ref_p.grad, "to_local") else ref_p.grad
            )
            off_local = (
                off_p.grad.to_local() if hasattr(off_p.grad, "to_local") else off_p.grad
            )
            torch.testing.assert_close(
                off_local.cpu(),
                ref_local.cpu(),
                atol=atol,
                rtol=rtol,
                msg=lambda m, _tag=msg_tag: f"{_tag}grad mismatch: {m}",
            )

    def _assert_comm_ctx_drained(
        self,
        model: nn.Module,
        *,
        mp_cast: bool = True,
        grad_offload: bool = True,
        all_gather: bool = True,
        reduce_scatter: bool = True,
        msg_tag: str = "",
    ) -> None:
        """Walk every FSDP state in ``model`` and assert the selected
        cross-layer ``comm_ctx`` slots are drained.
        """
        for module in model.modules():
            if getattr(module, "_get_fsdp_state", None) is None:
                continue
            for pg in module._get_fsdp_state()._fsdp_param_groups:
                ctx = pg.comm_ctx
                if all_gather:
                    self.assertIsNone(
                        ctx.all_gather_state, f"{msg_tag}all_gather_state leaked"
                    )
                if reduce_scatter:
                    self.assertEqual(
                        len(ctx.reduce_scatter_states),
                        0,
                        f"{msg_tag}reduce_scatter_states leaked "
                        f"({len(ctx.reduce_scatter_states)})",
                    )
                if mp_cast:
                    self.assertIsNone(
                        ctx.all_reduce_state,
                        f"{msg_tag}all_reduce_state leaked",
                    )
                if grad_offload:
                    self.assertIsNone(
                        ctx.grad_offload_state,
                        f"{msg_tag}grad_offload_state leaked",
                    )

    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_all_reduce_waits_on_reduce_scatter(self):
        """(3) Within a layer's post-reduce, AR cannot start before RS
        completes. Injecting RS delay increases backward time by at least
        one RS delay even when AR is untouched."""
        dim, n_layers, compute_sleep_ms, rs_sleep_ms = 4, 2, 20, 25
        model, inp = self._build_hsdp_model(n_layers, dim, compute_sleep_ms)

        orig_rs = dist.reduce_scatter_tensor
        rs_stream = device_module.Stream()

        def delayed_rs(*args, **kwargs):
            self._delay_on_shared_stream(rs_stream, rs_sleep_ms)
            return orig_rs(*args, **kwargs)

        def baseline():
            model(inp).sum().backward()

        def test():
            with patch_reduce_scatter(delayed_rs):
                model(inp).sum().backward()

        baseline_time = _time_fn(baseline)
        test_time = _time_fn(test)
        # AR must wait on RS ⇒ the last layer's RS+AR tail after backward
        # compute is at least rs_sleep_ms longer.
        added = test_time - baseline_time
        buffer_ms = 8
        self.assertGreater(
            added,
            rs_sleep_ms - buffer_ms,
            f"added={added:.1f}ms < one rs_sleep={rs_sleep_ms}ms; AR may "
            f"not be waiting on RS",
        )

    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_all_reduces_pipelined_across_layers(self):
        """(4) All ARs share ``comm_ctx.all_reduce_stream`` and the
        dp_replicate NCCL comm ⇒ they pipeline serially. Use cheap compute
        so ARs can't hide in backward compute; then the added time must
        grow with ``n_layers * ar_sleep_ms`` (pipelined), not stay at one
        AR latency (which is what parallel execution would yield)."""
        # compute << ar_sleep so compute cannot mask the AR delays.
        dim, n_layers, compute_sleep_ms, ar_sleep_ms = 4, 3, 3, 40
        model, inp = self._build_hsdp_model(n_layers, dim, compute_sleep_ms)

        orig_ar = dist.all_reduce
        ar_stream = device_module.Stream()

        def delayed_ar(*args, **kwargs):
            self._delay_on_shared_stream(ar_stream, ar_sleep_ms)
            return orig_ar(*args, **kwargs)

        def baseline():
            model(inp).sum().backward()

        def test():
            with patch_all_reduce(delayed_ar):
                model(inp).sum().backward()

        baseline_time = _time_fn(baseline)
        test_time = _time_fn(test)
        # Serialized: added ≈ n_layers * ar_sleep (all exposed since compute
        # can't overlap). Parallel (broken): added ≈ ar_sleep (last only).
        added = test_time - baseline_time
        serial_bound = n_layers * ar_sleep_ms
        buffer_ms = 15
        self.assertGreater(
            added,
            serial_bound * 0.7 - buffer_ms,
            f"added={added:.1f}ms ≪ serial bound {serial_bound}ms; ARs may "
            f"not be pipelined on all_reduce_stream",
        )

    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_set_requires_all_reduce_skips_ar_keeps_rs(self):
        """(7) HSDP grad accumulation: ``set_requires_all_reduce(False)``
        disables AR but keeps RS. Inject AR delay — no effect on bwd time
        (AR doesn't run). Inject RS delay — still adds to bwd time (RS
        always runs). Exercises the ``all_reduce_grads=False`` branch in
        ``foreach_reduce``."""
        dim, n_layers, compute_sleep_ms, delay_ms = 4, 2, 5, 40
        model, inp = self._build_hsdp_model(n_layers, dim, compute_sleep_ms)
        model.set_requires_all_reduce(False)

        orig_rs = dist.reduce_scatter_tensor
        orig_ar = dist.all_reduce
        delay_stream = device_module.Stream()

        def delayed_rs(*args, **kwargs):
            self._delay_on_shared_stream(delay_stream, delay_ms)
            return orig_rs(*args, **kwargs)

        def delayed_ar(*args, **kwargs):
            self._delay_on_shared_stream(delay_stream, delay_ms)
            return orig_ar(*args, **kwargs)

        def baseline():
            model(inp).sum().backward()

        def test_ar_only():
            # AR delay should not run because set_requires_all_reduce(False).
            with patch_all_reduce(delayed_ar):
                model(inp).sum().backward()

        def test_rs_only():
            with patch_reduce_scatter(delayed_rs):
                model(inp).sum().backward()

        baseline_time = _time_fn(baseline)
        ar_only_time = _time_fn(test_ar_only)
        rs_only_time = _time_fn(test_rs_only)

        buffer_ms = 8
        # AR must NOT execute ⇒ delay doesn't propagate.
        self.assertLess(
            ar_only_time - baseline_time,
            delay_ms,
            f"AR delay {delay_ms}ms added "
            f"{ar_only_time - baseline_time:.1f}ms with "
            f"set_requires_all_reduce(False); AR should not run",
        )
        # RS must still execute ⇒ delay propagates (at least 1 RS exposed).
        self.assertGreater(
            rs_only_time - baseline_time,
            delay_ms - buffer_ms,
            f"RS delay added only {rs_only_time - baseline_time:.1f}ms; "
            f"RS should still run under set_requires_all_reduce(False)",
        )

    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_post_reduce_cast_gates_next_layer_post_reduce(self):
        """(8) HSDP + MP (bf16 params, fp32 reduce): the post-reduce dtype
        cast runs on ``post_reduce_stream`` and records
        ``all_reduce_event`` *after* the cast (not after the AR). That
        event is carried to the next layer via ``prev_all_reduce_event`` so
        the next layer's AR stream waits for the prev layer's cast to
        finish — otherwise the prev fp32 buffer could be freed while the
        cast is still reading it.

        Inject a sleep inside ``_to_dtype_if_needed`` (runs on
        post_reduce_stream). Casts serialize across layers via
        ``prev_all_reduce_event`` ⇒ added time grows with ``n_layers``.
        """
        torch.manual_seed(42)
        dim, n_layers, compute_sleep_ms, cast_sleep_ms = 4, 3, 5, 30
        mesh = init_device_mesh(
            device_type.type,
            (2, 2),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        mp = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
        model = nn.Sequential(
            *[LinearWithSleep(dim, compute_sleep_ms) for _ in range(n_layers)]
        ).to(device_type, dtype=torch.bfloat16)
        for lin in model:
            fully_shard(lin, mesh=mesh, mp_policy=mp, reshard_after_forward=True)
        fully_shard(model, mesh=mesh, mp_policy=mp, reshard_after_forward=True)
        inp = torch.randn((2, dim), device=device_type.type, dtype=torch.bfloat16)
        model(inp).sum().backward()  # warmup

        import torch.distributed.fsdp._fully_shard._fsdp_collectives as _coll

        orig_cast = _coll._to_dtype_if_needed

        def delayed_cast(tensor, dtype):
            if dtype is not None and tensor.dtype != dtype:
                device_module._sleep(int(cast_sleep_ms * get_cycles_per_ms()))
            return orig_cast(tensor, dtype)

        def baseline():
            model(inp).sum().backward()

        def test():
            dist.barrier()
            _coll._to_dtype_if_needed = delayed_cast
            try:
                model(inp).sum().backward()
            finally:
                dist.barrier()
                _coll._to_dtype_if_needed = orig_cast

        baseline_time = _time_fn(baseline)
        test_time = _time_fn(test)
        added = test_time - baseline_time
        # Serial via prev_all_reduce_event: added ≈ n_layers * cast_sleep.
        # If the event were just AR's (bug): casts overlap, added ≈ 1x.
        # Require strictly greater than 1 cast_sleep to catch that bug.
        buffer_ms = 8
        self.assertGreater(
            added,
            (n_layers - 1) * cast_sleep_ms * 0.7 - buffer_ms,
            f"added={added:.1f}ms ≪ serialized bound "
            f"{(n_layers - 1) * cast_sleep_ms:.0f}ms; post-reduce cast "
            f"may not be gating next layer via prev_all_reduce_event",
        )

    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_finalize_backward_syncs_default_stream_with_comm(self):
        """(9) After ``.backward()`` returns, the default stream must be
        gated on all pending RS/AR events (``finalize_backward`` chains
        the waits). ``_time_fn`` records its end event on the default
        stream; if the default stream is correctly gated, the end event's
        GPU timestamp can only advance after comm completes, so the
        measured elapsed time must include the injected delay."""
        dim, n_layers, compute_sleep_ms, delay_ms = 4, 2, 5, 50
        model, inp = self._build_hsdp_model(n_layers, dim, compute_sleep_ms)

        orig_rs = dist.reduce_scatter_tensor
        orig_ar = dist.all_reduce
        rs_stream = device_module.Stream()
        ar_stream = device_module.Stream()

        def delayed_rs(*args, **kwargs):
            self._delay_on_shared_stream(rs_stream, delay_ms)
            return orig_rs(*args, **kwargs)

        def delayed_ar(*args, **kwargs):
            self._delay_on_shared_stream(ar_stream, delay_ms)
            return orig_ar(*args, **kwargs)

        def baseline():
            model(inp).sum().backward()

        def test():
            with (
                patch_reduce_scatter(delayed_rs),
                patch_all_reduce(delayed_ar),
            ):
                model(inp).sum().backward()

        baseline_time = _time_fn(baseline)
        test_time = _time_fn(test)
        # End event records on default stream. If finalize_backward chained
        # RS/AR waits onto default stream, the end event can't record until
        # those waits complete. Added time must exceed one delay (at least
        # one comm event is captured by the default stream's dependency).
        added = test_time - baseline_time
        buffer_ms = 8
        self.assertGreater(
            added,
            delay_ms - buffer_ms,
            f"added={added:.1f}ms < delay {delay_ms}ms; default stream "
            f"may not be gated on pending RS/AR comm after finalize",
        )

    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_ar_and_rs_overlap_across_layers(self):
        """(10) HSDP uses two NCCL comms — dp_replicate (AR) and dp_shard
        (RS/AG). AR on layer K can therefore overlap with RS on layer K-1:
        different streams, different communicators. Inject delays in both
        on separate streams; total added time must be strictly below a
        fully-serialized ``2 * n_layers * delay_ms`` bound."""
        dim, n_layers, compute_sleep_ms, delay_ms = 4, 4, 3, 20
        model, inp = self._build_hsdp_model(n_layers, dim, compute_sleep_ms)

        orig_rs = dist.reduce_scatter_tensor
        orig_ar = dist.all_reduce
        rs_stream = device_module.Stream()
        ar_stream = device_module.Stream()

        def delayed_rs(*args, **kwargs):
            self._delay_on_shared_stream(rs_stream, delay_ms)
            return orig_rs(*args, **kwargs)

        def delayed_ar(*args, **kwargs):
            self._delay_on_shared_stream(ar_stream, delay_ms)
            return orig_ar(*args, **kwargs)

        def baseline():
            model(inp).sum().backward()

        def test():
            with (
                patch_reduce_scatter(delayed_rs),
                patch_all_reduce(delayed_ar),
            ):
                model(inp).sum().backward()

        baseline_time = _time_fn(baseline)
        test_time = _time_fn(test)
        added = test_time - baseline_time
        # Fully-serialized (shared comm): 2*N*delay. Pipelined (two comms,
        # AR_K overlaps with RS_{K-1} across layers): ≤ (N+1)*delay.
        fully_serial = 2 * n_layers * delay_ms
        buffer_ms = 8
        self.assertLess(
            added,
            fully_serial - buffer_ms,
            f"added={added:.1f}ms ≥ fully-serial bound {fully_serial}ms; "
            f"AR and RS may be serializing instead of overlapping across "
            f"dp_replicate / dp_shard comms",
        )

    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_partial_reduce_output_accumulation_across_microbatches(self):
        """(11) HSDP grad accumulation:
        ``set_requires_all_reduce(False)`` for K-1 microbatches causes
        ``partial_reduce_output += reduce_output`` to serialize on the
        reduce-scatter stream (line 656 of ``_fsdp_collectives.py``). The
        final microbatch with AR enabled triggers AR over the accumulated
        value. Correctness proof: accumulated gradient == gradient from a
        single-step reference with the concatenated input batch."""
        torch.manual_seed(42)
        dim, n_layers = 4, 2
        n_microbatches = 4
        mesh = init_device_mesh(
            device_type.type,
            (2, 2),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        # Build the test model (accumulation via set_requires_all_reduce).
        torch.manual_seed(0)
        model = nn.Sequential(
            *[nn.Linear(dim, dim, device=device_type) for _ in range(n_layers)]
        )
        ref_model = copy.deepcopy(model).to(device_type)
        for lin in model:
            fully_shard(lin, mesh=mesh, reshard_after_forward=True)
        fully_shard(model, mesh=mesh, reshard_after_forward=True)
        for lin in ref_model:
            fully_shard(lin, mesh=mesh, reshard_after_forward=True)
        fully_shard(ref_model, mesh=mesh, reshard_after_forward=True)

        inps = [
            torch.randn((2, dim), device=device_type.type)
            for _ in range(n_microbatches)
        ]

        # Reference: one big batch = concat of microbatches, one backward.
        ref_model(torch.cat(inps, dim=0)).sum().backward()

        # Test: accumulate K-1 microbatches with AR disabled, final with
        # AR enabled. Small RS delay stresses the cross-microbatch ordering
        # of the += into partial_reduce_output.
        orig_rs = dist.reduce_scatter_tensor
        rs_stream = device_module.Stream()

        def delayed_rs(*args, **kwargs):
            self._delay_on_shared_stream(rs_stream, 5)
            return orig_rs(*args, **kwargs)

        with patch_reduce_scatter(delayed_rs):
            for i, inp in enumerate(inps):
                model.set_requires_all_reduce(i == n_microbatches - 1)
                model(inp).sum().backward()

        for ref_p, test_p in zip(ref_model.parameters(), model.parameters()):
            if ref_p.grad is None:
                continue
            torch.testing.assert_close(
                test_p.grad,
                ref_p.grad,
                atol=1e-4,
                rtol=1e-3,
                msg=lambda msg: (
                    f"accumulated grad mismatch: {msg}. "
                    f"partial_reduce_output may not be correctly "
                    f"serializing the += across microbatches."
                ),
            )

    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_cpu_offload_grad_copy_synchronized(self):
        """(12) ``CPUOffloadPolicy`` copies grads to CPU on a dedicated
        stream and ``grad_offload_event.synchronize()`` in
        ``finalize_backward`` blocks the CPU until the copy completes. If
        that sync were missing, reading ``param.grad`` on CPU after
        ``.backward()`` returns could observe partial / stale data.

        Covers both 1D FSDP and HSDP 2×2. The HSDP case additionally
        guards the ``reduce_output`` lifetime: for HSDP the D2H memcpy
        runs on ``all_reduce_stream`` while ``reduce_output`` was
        allocated on ``reduce_scatter_stream``. Without keep-alive
        management, a later RS-stream allocation can reuse the source
        block before the memcpy drains, corrupting the CPU grad.
        """
        self.run_subtests(
            {
                "mesh_shape": [(self.world_size,), (2, 2)],
                "n_models": [1, 2],
            },
            self._test_cpu_offload_grad_copy_synchronized,
        )

    def _test_cpu_offload_grad_copy_synchronized(
        self, mesh_shape: tuple[int, ...], n_models: int
    ):
        torch.manual_seed(42)
        dim, n_layers = 4, 2
        mesh_dim_names = (
            ("fsdp",) if len(mesh_shape) == 1 else ("dp_replicate", "dp_shard")
        )
        mesh = init_device_mesh(
            device_type.type, mesh_shape, mesh_dim_names=mesh_dim_names
        )
        torch.manual_seed(0)

        def _make_model(offload: bool) -> nn.Module:
            torch.manual_seed(0)
            model = nn.Sequential(
                *[nn.Linear(dim, dim, device=device_type) for _ in range(n_layers)]
            )
            kwargs = {"mesh": mesh}
            if offload:
                kwargs["offload_policy"] = CPUOffloadPolicy()
            for lin in model:
                fully_shard(lin, **kwargs)
            fully_shard(model, **kwargs)
            return model

        ref_model = _make_model(offload=False)
        # Building n_models offload models back-to-back stresses the
        # cross-stream reuse race on HSDP: the first model's reduce_output
        # can be reused by the n-th model's RS-stream allocation.
        off_models = [_make_model(offload=True) for _ in range(n_models)]

        inp = torch.randn((2, dim), device=device_type.type)
        ref_model(inp).sum().backward()
        for m in off_models:
            m(inp).sum().backward()

        # Read CPU grads immediately after backward; correctness depends
        # on the full offload sync chain (grad_offload_event sync and
        # reduce_output keep-alive on HSDP).
        tag = f"[mesh_shape={mesh_shape}, n_models={n_models}] "
        for off_model in off_models:
            self._assert_grad_parity(
                ref_model, off_model, atol=1e-5, rtol=1e-4, msg_tag=tag
            )

    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_multi_iteration_no_comm_state_leak(self):
        """(13) Across iterations, every cross-layer comm_ctx slot must be
        drained — ``reduce_scatter_states``, ``all_gather_state``,
        ``all_reduce_state``, ``grad_offload_state``. Regression guard for
        iteration-boundary leaks like #179128. Uses HSDP + bf16 mp + CPU
        offload so all four slots are exercised."""
        mp = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
        _, off, inp = self._make_model_pair(n_layers=3, mp=mp)
        # Warmup (lazy init)
        off(inp).sum().backward()
        off.zero_grad(set_to_none=True)
        for iter_idx in range(3):
            off(inp).sum().backward()
            off.zero_grad(set_to_none=True)
            self._assert_comm_ctx_drained(off, msg_tag=f"iter {iter_idx}: ")

    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_exception_in_backward_does_not_leak_comm_state(self):
        """(14) An exception raised mid-backward must not leave
        ``comm_ctx.all_reduce_state`` or ``comm_ctx.grad_offload_state``
        pointing at a dangling GPU buffer. Both fields are single shared
        slots on comm_ctx; a missed cleanup would persist across
        iterations. Uses HSDP + bf16 mp + CPU offload so both slots are
        populated before the failure point."""
        mp = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
        _, off, inp = self._make_model_pair(n_layers=3, mp=mp)
        off(inp).sum().backward()  # warmup
        off.zero_grad(set_to_none=True)

        orig_ar = dist.all_reduce
        call_count = [0]

        def raising_ar(*args, **kwargs):
            call_count[0] += 1
            # Fail on the 2nd AR (i.e., mid-backward, after at least one
            # comm_ctx slot has been populated).
            if call_count[0] == 2:
                raise RuntimeError("simulated mid-backward AR failure")
            return orig_ar(*args, **kwargs)

        with self.assertRaises(RuntimeError):
            with patch_all_reduce(raising_ar):
                off(inp).sum().backward()

        # After the exception, both comm_ctx slots must be clean so the
        # next iteration starts fresh.
        self._assert_comm_ctx_drained(
            off, msg_tag="post-exception: ", all_gather=False, reduce_scatter=False
        )

    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_hsdp_per_param_mesh_offload_slot_collision(self):
        """(23) Per-param-mesh on HSDP with ≥2 groups both using offload.
        The ``grad_offload_state`` slot lives on ``comm_ctx`` shared across
        all param groups of a single root. Two concurrent groups writing
        to the same slot must not clobber each other — verified by
        comparing grads against a non-offloaded HSDP reference."""
        from torch.distributed.fsdp._fully_shard._fsdp_common import (
            HSDPMeshInfo,
            ShardPlacementResult,
        )
        from torch.distributed.tensor import Shard

        torch.manual_seed(42)
        dim = 8
        global_mesh = init_device_mesh(
            device_type.type,
            (2, 2),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        # Two HSDPMeshInfo variants that swap shard / replicate dims on
        # the same 2x2 mesh ⇒ distinct process groups, so params using
        # each variant land in distinct FSDPParamGroups.
        hsdp_a = HSDPMeshInfo(mesh=global_mesh, shard_mesh_dim=1, replicate_mesh_dim=0)
        hsdp_b = HSDPMeshInfo(mesh=global_mesh, shard_mesh_dim=0, replicate_mesh_dim=1)

        torch.manual_seed(0)

        class TwoLinears(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(dim, dim, device=device_type)
                self.b = nn.Linear(dim, dim, device=device_type)

            def forward(self, x):
                return self.b(self.a(x))

        ref_model = TwoLinears()
        off_model = copy.deepcopy(ref_model)
        ref_params_a = set(ref_model.a.parameters())
        off_params_a = set(off_model.a.parameters())

        def _ref_placement_fn(p, _a=ref_params_a):
            return ShardPlacementResult(
                placement=Shard(0),
                mesh_info=hsdp_a if p in _a else hsdp_b,
            )

        def _off_placement_fn(p, _a=off_params_a):
            return ShardPlacementResult(
                placement=Shard(0),
                mesh_info=hsdp_a if p in _a else hsdp_b,
            )

        fully_shard(ref_model, mesh=global_mesh, shard_placement_fn=_ref_placement_fn)
        fully_shard(
            off_model,
            mesh=global_mesh,
            shard_placement_fn=_off_placement_fn,
            offload_policy=CPUOffloadPolicy(),
        )

        # Sanity: off_model root state has ≥2 param groups (per-param-mesh).
        self.assertGreaterEqual(
            len(off_model._get_fsdp_state()._fsdp_param_groups),
            2,
            "per-param-mesh did not produce multiple FSDPParamGroups",
        )

        inp = torch.randn((2, dim), device=device_type.type)
        ref_model(inp).sum().backward()
        off_model(inp).sum().backward()

        for ref_p, off_p in zip(ref_model.parameters(), off_model.parameters()):
            if ref_p.grad is None:
                continue
            ref_local = (
                ref_p.grad.to_local() if hasattr(ref_p.grad, "to_local") else ref_p.grad
            )
            off_local = (
                off_p.grad.to_local() if hasattr(off_p.grad, "to_local") else off_p.grad
            )
            torch.testing.assert_close(
                off_local,
                ref_local.cpu(),
                atol=1e-5,
                rtol=1e-4,
                msg=lambda msg: (
                    f"HSDP per-param-mesh + offload grad mismatch: {msg}. "
                    f"Multiple groups may be colliding on the single "
                    f"comm_ctx.grad_offload_state slot."
                ),
            )
        # Both groups drained after finalize.
        for pg in off_model._get_fsdp_state()._fsdp_param_groups:
            self.assertIsNone(pg.comm_ctx.grad_offload_state)

    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_hsdp_offload_reshard_after_forward_false(self):
        """(24) HSDP + offload + ``reshard_after_forward=False``. Params
        stay gathered after forward so backward uses them directly — no
        re-AG. The ``grad_offload_state`` drain mechanism must still work
        even though the AG timing differs from the `=True` default."""
        mp = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
        ref, off, inp = self._make_model_pair(
            n_layers=3, mp=mp, reshard_after_forward=False
        )
        ref(inp).sum().backward()
        off(inp).sum().backward()
        self._assert_grad_parity(ref, off)
        self._assert_comm_ctx_drained(off)

    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_hsdp_post_optim_event_overlap(self):
        """(26) HSDP parity for ``set_post_optim_event`` — the existing
        ``test_fully_shard_post_optim_event_overlap`` only covers 1D FSDP.
        On HSDP the root module waits AG streams on the user event
        instead of the current stream. Verify the optim event gates both
        AG streams correctly on an HSDP mesh."""
        torch.manual_seed(42)
        dim, compute_sleep_ms, comm_sleep_ms = 4, 25, 10
        mesh = init_device_mesh(
            device_type.type,
            (2, 2),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        model = nn.Sequential(
            LinearWithSleep(dim, compute_sleep_ms), nn.Linear(dim, dim)
        ).to(device_type)
        fully_shard(model[1], mesh=mesh, reshard_after_forward=False)
        optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

        orig_ag = dist.all_gather_into_tensor

        def delayed_ag(*args, **kwargs):
            device_module._sleep(int(comm_sleep_ms * get_cycles_per_ms()))
            return orig_ag(*args, **kwargs)

        inp = torch.randn((2, dim), device=device_type.type)

        def run_train_steps(num_iters: int, use_post_optim_event: bool):
            for _ in range(num_iters):
                optim.zero_grad()
                with patch_all_gather(delayed_ag):
                    model(inp).sum().backward()
                with implicit_replication():
                    optim.step()
                if use_post_optim_event:
                    post_optim_event = device_module.current_stream().record_event()
                    model[1].set_post_optim_event(post_optim_event)

        run_train_steps(1, False)  # warmup
        num_iters = 5
        baseline_time = _time_fn(functools.partial(run_train_steps, num_iters, False))
        test_time = _time_fn(functools.partial(run_train_steps, num_iters, True))
        # With set_post_optim_event, AG prefetch can overlap with the
        # high-compute preceding op ⇒ test should be faster than baseline.
        self.assertGreater(baseline_time, test_time)

    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_hsdp_explicit_prefetch_correctness(self):
        """(27) HSDP + explicit ``set_modules_to_forward_prefetch`` and
        ``set_modules_to_backward_prefetch`` control AG stream ordering
        across layers. Verify grads remain correct under explicit
        prefetch (the stream-ordering fixes this test exercises are
        orthogonal to the offload fix, but share the AG-stream machinery)."""
        torch.manual_seed(42)
        dim, n_layers = 4, 3
        mesh = init_device_mesh(
            device_type.type,
            (2, 2),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        torch.manual_seed(0)
        base = nn.Sequential(
            *[nn.Linear(dim, dim, device=device_type) for _ in range(n_layers)]
        )
        ref_model = copy.deepcopy(base).to(device_type)
        test_model = copy.deepcopy(base).to(device_type)
        for lin in ref_model:
            fully_shard(lin, mesh=mesh)
        fully_shard(ref_model, mesh=mesh)
        layers = list(test_model)
        for lin in layers:
            fully_shard(lin, mesh=mesh)
        fully_shard(test_model, mesh=mesh)
        # Explicit forward prefetch: layer i prefetches layer i+1
        for i in range(len(layers) - 1):
            layers[i].set_modules_to_forward_prefetch([layers[i + 1]])
        # Explicit backward prefetch: layer i prefetches layer i-1
        for i in range(1, len(layers)):
            layers[i].set_modules_to_backward_prefetch([layers[i - 1]])

        inp = torch.randn((2, dim), device=device_type.type)
        ref_model(inp).sum().backward()
        test_model(inp).sum().backward()
        self._assert_grad_parity(
            ref_model,
            test_model,
            atol=1e-5,
            rtol=1e-4,
            msg_tag="explicit fwd/bwd prefetch: ",
        )

    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_hsdp_offload_set_is_last_backward_defers_finalize(self):
        """(19) ``set_is_last_backward(False)`` tells FSDP to skip
        ``finalize_backward`` on that backward. For HSDP + offload, the
        ``grad_offload_state`` slot must survive to the next backward and
        be drained via the prev-slot mechanism when the subsequent
        foreach_reduce runs. Verify across two iterations (False → True):
        grads match a non-offloaded reference and both comm_ctx slots are
        None after the final iteration.
        """
        mp = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
        ref, off, inp1 = self._make_model_pair(n_layers=2, mp=mp)
        inp2 = torch.randn(inp1.shape, device=inp1.device, dtype=inp1.dtype)
        off.set_is_last_backward(False)
        ref(inp1).sum().backward()
        off(inp1).sum().backward()
        off.set_is_last_backward(True)
        ref(inp2).sum().backward()
        off(inp2).sum().backward()
        self._assert_grad_parity(ref, off, msg_tag="deferred finalize: ")
        self._assert_comm_ctx_drained(off, msg_tag="post deferred finalize: ")

    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_fsdp_offload_custom_all_reduce_hook_stream(self):
        """(20) 1D FSDP + ``set_all_reduce_hook(hook, stream=custom_stream)``
        + ``CPUOffloadPolicy``. With a custom hook stream, ``all_reduce_stream
        = self._all_reduce_hook_stream`` — a user-provided stream. The
        stream-scoped ``del prev_grad_offload_state`` routes the block to
        that user stream's free pool. Verify grads match a non-offloaded
        reference."""
        torch.manual_seed(42)
        dim, n_layers = 4, 2
        mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        torch.manual_seed(0)
        base = nn.Sequential(
            *[nn.Linear(dim, dim, device=device_type) for _ in range(n_layers)]
        )
        ref_model = copy.deepcopy(base).to(device_type)
        off_model = copy.deepcopy(base).to(device_type)
        for lin in ref_model:
            fully_shard(lin, mesh=mesh)
        fully_shard(ref_model, mesh=mesh)
        for lin in off_model:
            fully_shard(lin, mesh=mesh, offload_policy=CPUOffloadPolicy())
        fully_shard(off_model, mesh=mesh, offload_policy=CPUOffloadPolicy())

        # Attach a trivial hook on a custom stream. The hook itself is a
        # no-op; we only care that the custom stream is used for the
        # stream-scoped del of prev_grad_offload_state.
        custom_stream = device_module.Stream()

        def noop_hook(reduce_output: torch.Tensor) -> None:
            pass

        for lin in off_model:
            lin.set_all_reduce_hook(noop_hook, stream=custom_stream)

        inp = torch.randn((2, dim), device=device_type.type)
        ref_model(inp).sum().backward()
        off_model(inp).sum().backward()

        for ref_p, off_p in zip(ref_model.parameters(), off_model.parameters()):
            if ref_p.grad is None:
                continue
            torch.testing.assert_close(
                off_p.grad,
                ref_p.grad.cpu(),
                atol=1e-5,
                rtol=1e-4,
                msg=lambda msg: (
                    f"grad mismatch with custom hook stream: {msg}. The "
                    f"grad_offload keep-alive may not be routing correctly "
                    f"to the user-supplied stream's free pool."
                ),
            )

    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_hsdp_offload_async_op_unshard(self):
        """(21) HSDP + ``CPUOffloadPolicy`` + ``_set_unshard_async_op(True)``.
        Async unshard routes AG copy-in through the default stream rather
        than the dedicated copy-in stream. Verify this config still
        produces correct grads under offload."""
        mp = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
        ref, off, inp = self._make_model_pair(n_layers=2, mp=mp)
        off._set_unshard_async_op(True)
        ref(inp).sum().backward()
        off(inp).sum().backward()
        self._assert_grad_parity(ref, off, msg_tag="async_op=True + offload: ")

    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_three_fsdp_roots_sharing_mesh(self):
        """(22) Three independent FSDP roots on the same mesh. Each root
        has its own comm_ctx, but they share the underlying
        ProcessGroups. Serial backward must not cross-contaminate
        comm_ctx state — a pre-condition for the single-slot invariants
        of ``all_reduce_state`` and ``grad_offload_state``.
        Parameterized over 1D FSDP and HSDP 2×2: the invariant is mesh-
        agnostic, and the HSDP case additionally exercises the
        single-slot fields that only exist there."""
        self.run_subtests(
            {"mesh_shape": [(self.world_size,), (2, 2)]},
            self._test_three_fsdp_roots_sharing_mesh,
        )

    def _test_three_fsdp_roots_sharing_mesh(self, mesh_shape: tuple[int, ...]):
        mp = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
        _, ref, inp = self._make_model_pair(mesh_shape=mesh_shape, mp=mp)
        models = [
            self._make_model_pair(mesh_shape=mesh_shape, mp=mp)[1] for _ in range(3)
        ]
        ref(inp).sum().backward()
        for m in models:
            m(inp).sum().backward()
        tag = f"mesh_shape={mesh_shape} "
        for i, m in enumerate(models):
            self._assert_grad_parity(ref, m, msg_tag=f"{tag}root {i} vs ref: ")
            self._assert_comm_ctx_drained(m, msg_tag=f"{tag}root {i}: ")

    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_hsdp_mp_offload_combined_lifetime(self):
        """(15) HSDP + ``MixedPrecisionPolicy(bf16, fp32)`` + ``CPUOffloadPolicy``
        is the only config where both ``all_reduce_state`` and
        ``grad_offload_state`` comm_ctx slots are populated simultaneously.
        Both drain paths use ``del prev_<state> inside
        with stream(all_reduce_stream):`` — a subtle ordering bug in either
        would only surface here. Verify (a) grads match a non-offloaded
        HSDP+MP reference, (b) both comm_ctx slots are None after
        finalize_backward.
        """
        mp = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
        ref, off, inp = self._make_model_pair(n_layers=3, mp=mp)
        ref(inp).sum().backward()
        off(inp).sum().backward()
        self._assert_grad_parity(ref, off, msg_tag="HSDP+MP+offload combined: ")
        self._assert_comm_ctx_drained(off, msg_tag="combined lifetime: ")

    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(TEST_HPU, "Sleep is not supported on HPU")
    def test_hsdp_training_overlap(self):
        """HSDP 2×2 complement to ``test_fully_shard_training_overlap``
        (which covers the same properties on 1D FSDP). Verifies for
        ``fully_shard(..., reshard_after_forward=True)`` fwd+bwd:

          - Compute on the default stream waits on AG (at least the first
            AG of each phase is exposed ⇒ added time ≥ 2*comm_sleep_ms).
          - AG and RS serialize on the shared dp_shard NCCL comm (both
            delays, stacked on one comm_stream, are individually visible).
          - Backward AG prefetch + RS overlap with compute: added time is
            strictly below a fully-sequential comm bound.
        """
        torch.manual_seed(42)
        dim, n_layers, compute_sleep_ms, comm_sleep_ms = 4, 3, 25, 10
        mesh = init_device_mesh(
            device_type.type,
            (2, 2),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        model = nn.Sequential(
            *[LinearWithSleep(dim, compute_sleep_ms) for _ in range(n_layers)]
        )
        for lin in model:
            fully_shard(lin, mesh=mesh, reshard_after_forward=True)
        fully_shard(model, mesh=mesh, reshard_after_forward=True)
        inp = torch.randn((2, dim), device=device_type.type)
        model(inp).sum().backward()  # warmup

        orig_ag = dist.all_gather_into_tensor
        orig_rs = dist.reduce_scatter_tensor
        # Shared comm_stream ⇒ AG and RS delays serialize on it, matching
        # ProcessGroupNCCL's behavior on a shared dp_shard communicator.
        shared_stream = device_module.Stream()

        def delayed_ag(*args, **kwargs):
            self._delay_on_shared_stream(shared_stream, comm_sleep_ms)
            return orig_ag(*args, **kwargs)

        def delayed_rs(*args, **kwargs):
            self._delay_on_shared_stream(shared_stream, comm_sleep_ms)
            return orig_rs(*args, **kwargs)

        def baseline():
            model(inp).sum().backward()

        def test():
            with (
                patch_all_gather(delayed_ag),
                patch_reduce_scatter(delayed_rs),
            ):
                model(inp).sum().backward()

        baseline_time = _time_fn(baseline)
        test_time = _time_fn(test)
        added = test_time - baseline_time
        buffer_ms = 8

        # (a) Both AG and RS delays exposed: at minimum the first fwd AG
        # and the last bwd RS are not overlapped ⇒ added ≥ 2*comm_sleep.
        self.assertGreater(
            added,
            2 * comm_sleep_ms - buffer_ms,
            f"added={added:.1f}ms < 2*{comm_sleep_ms}ms; AG and RS delays "
            f"are not both visible",
        )
        # (b) Overlap prevents fully-sequential comm exposure. With
        # reshard_after_forward=True, a no-overlap path would expose all
        # N fwd all-gathers + N bwd-prefetch all-gathers + N reduce-scatters
        # = 3N * comm_sleep_ms.
        fully_serial = 3 * n_layers * comm_sleep_ms
        self.assertLess(
            added,
            fully_serial - buffer_ms,
            f"added={added:.1f}ms ≥ fully-serial bound {fully_serial}ms; "
            f"AG/RS are not overlapping with compute",
        )


if __name__ == "__main__":
    run_tests()
