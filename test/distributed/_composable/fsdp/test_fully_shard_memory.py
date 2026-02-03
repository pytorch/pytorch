# Owner(s): ["oncall: distributed"]

import functools
import gc
import unittest

import torch
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_CUDA,
    TEST_HPU,
    TEST_XPU,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    MoEArgs,
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
        assert self.world_size == 2, (
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
            assert not use_cpu_offload
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
            assert not use_cpu_offload
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

    def _get_peak_active_memory_mb(self) -> int:
        mem_stats = torch.get_device_module(device_type).memory_stats()

        if TEST_CUDA or TEST_XPU:
            return round(mem_stats["active_bytes.all.peak"] / 1e6)
        if TEST_HPU:
            return round(mem_stats["MaxInUse"] / 1e6)

    def _get_curr_active_memory_mb(self) -> int:
        mem_stats = torch.get_device_module(device_type).memory_stats()
        if TEST_CUDA or TEST_XPU:
            return round(mem_stats["active_bytes.all.current"] / 1e6)
        if TEST_HPU:
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


class TestFullyShardMoEMemory(FSDPTest):
    """
    Test memory behavior with MoE + Expert Parallel + FSDP + Activation Checkpointing.

    This test catches a memory leak regression where memory would grow by ~2.3GB
    per training step when using:
    - MoE with Expert Parallel (all-to-all dispatch/combine)
    - FSDP2 with nested EFSDP for experts
    - Activation checkpointing

    The leak was caused by _apply_to_tensors creating container copies that
    prevented proper garbage collection. The fix uses tree_flatten/tree_unflatten
    with special handling for unregistered dataclasses.

    This test requires 4 GPUs to properly test Expert Parallel (EP=2) with
    FSDP (world_size=4) and EFSDP (size=2 within EP group).
    """

    @property
    def world_size(self) -> int:
        return min(4, torch.get_device_module(device_type).device_count())

    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(TEST_HPU, "MoE test requires CUDA")
    def test_moe_expert_parallel_memory_leak(self):
        """
        Test that MoE + Expert Parallel + FSDP + activation checkpointing
        does not leak memory over multiple training steps.

        Configuration:
            - 4 GPUs total
            - FSDP mesh: 4-way (all GPUs)
            - EP mesh: 2-way (experts partitioned across 2 GPUs)
            - EFSDP mesh: 2-way (FSDP within EP group)
            - Activation checkpointing enabled

        The memory leak manifested as ~2.3GB growth per step, causing OOM
        after ~20 steps on 80GB GPUs. With the fix, memory should be stable.
        """
        assert self.world_size == 4, f"Requires 4 GPUs, got {self.world_size}"

        # Build device meshes: FSDP=4, EP=2, EFSDP=2
        ep_degree = 2
        fsdp_mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )
        efsdp_size = self.world_size // ep_degree
        sparse_mesh = init_device_mesh(
            device_type.type,
            (efsdp_size, ep_degree),
            mesh_dim_names=("efsdp", "ep"),
        )
        ep_mesh = sparse_mesh["ep"]
        efsdp_mesh = sparse_mesh["efsdp"]

        # Use smaller model dimensions for CI but preserve the structure
        # that triggers the memory leak (MoE + EP + activation checkpointing)
        moe_args = MoEArgs(
            num_experts=8,  # Smaller than prod (64) but still tests EP
            top_k=2,
            hidden_dim=256,
        )
        model_args = ModelArgs(
            n_layers=2,
            vocab_size=1024,
            max_seq_len=128,
            dim=256,
            n_heads=4,
            dropout_p=0.0,
            use_attn_mask=True,
            weight_tying=False,
            checkpoint_activations=True,  # Required to trigger the leak
            moe_args=moe_args,
        )

        torch.manual_seed(42)

        # Create model with MoE and Expert Parallel
        model = Transformer(model_args, ep_mesh=ep_mesh)

        # Apply FSDP
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        for layer in model.layers:
            if hasattr(layer.feed_forward, "experts"):
                fully_shard(
                    layer.feed_forward.experts, mesh=efsdp_mesh, mp_policy=mp_policy
                )
            fully_shard(layer, mesh=fsdp_mesh, mp_policy=mp_policy)

        fully_shard(model.tok_embeddings, mesh=fsdp_mesh, mp_policy=mp_policy)
        fully_shard(model.pos_embeddings, mesh=fsdp_mesh, mp_policy=mp_policy)
        fully_shard([model.norm, model.output], mesh=fsdp_mesh, mp_policy=mp_policy)
        fully_shard(model, mesh=fsdp_mesh, mp_policy=mp_policy)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)

        batch_size = 1
        seq_len = model_args.max_seq_len

        # Warmup steps to stabilize memory (optimizer states, CUDA caches, etc.)
        num_warmup_steps = 3
        for _ in range(num_warmup_steps):
            tokens = torch.randint(
                0, model_args.vocab_size, (batch_size, seq_len), device=device_type.type
            )
            labels = torch.randint(
                0, model_args.vocab_size, (batch_size, seq_len), device=device_type.type
            )
            logits = model(tokens)
            loss = F.cross_entropy(
                logits.view(-1, model_args.vocab_size), labels.view(-1)
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Force garbage collection and reset memory stats
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Record baseline memory after warmup
        baseline_memory = torch.cuda.memory_allocated()

        # Run test steps and track memory
        num_test_steps = 5
        memory_readings = [baseline_memory]

        for step in range(num_test_steps):
            torch.manual_seed(100 + step)
            tokens = torch.randint(
                0, model_args.vocab_size, (batch_size, seq_len), device=device_type.type
            )
            labels = torch.randint(
                0, model_args.vocab_size, (batch_size, seq_len), device=device_type.type
            )

            logits = model(tokens)
            loss = F.cross_entropy(
                logits.view(-1, model_args.vocab_size), labels.view(-1)
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            gc.collect()
            torch.cuda.synchronize()
            current_memory = torch.cuda.memory_allocated()
            memory_readings.append(current_memory)

        # Check for memory growth
        # Before the fix: memory grew by ~2.3GB per step (with large model)
        # With CI-sized model, growth would be proportionally smaller but still detectable
        # Allow 50MB threshold to catch significant leaks while tolerating noise
        max_memory_growth_bytes = 50 * 1024 * 1024  # 50 MB threshold
        final_memory = memory_readings[-1]
        memory_growth = final_memory - baseline_memory

        self.assertLessEqual(
            memory_growth,
            max_memory_growth_bytes,
            f"Memory leak detected in MoE + Expert Parallel + FSDP! "
            f"Memory grew by {memory_growth / (1024 * 1024):.2f} MB "
            f"over {num_test_steps} steps (threshold: {max_memory_growth_bytes / (1024 * 1024):.2f} MB). "
            f"Memory readings (MB): {[m / (1024 * 1024) for m in memory_readings]}. "
            f"This regression was fixed in the FSDP _apply_to_tensors -> tree_flatten change.",
        )


if __name__ == "__main__":
    run_tests()
