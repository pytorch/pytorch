# Owner(s): ["oncall: distributed"]

import functools
import gc
import unittest

import torch
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, OffloadPolicy
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


if __name__ == "__main__":
    run_tests()
