# Owner(s): ["oncall: distributed"]

import functools

import torch

from _test_fully_shard_common import Block, GPT, GPTConfig
from torch.distributed._composable.fsdp import fully_shard, OffloadPolicy

from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests


class TestFullyShardMemory(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(2, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    def test_fully_shard_training_memory(self):
        self.run_subtests(
            {"reshard_after_forward": [True, False], "use_cpu_offload": [True, False]},
            self._test_fully_shard_training_memory,
        )

    def _test_fully_shard_training_memory(
        self, reshard_after_forward: bool, use_cpu_offload: bool
    ):
        if not reshard_after_forward and use_cpu_offload:
            # Skip this since it is not a common use case
            return
        assert (
            self.world_size == 2
        ), f"Requires world size of 2 since some values are hard coded: {self.world_size}"
        torch.manual_seed(42)
        torch.cuda.empty_cache()
        base_mem_mb = self._get_peak_active_memory_mb()
        vocab_size = 32
        model = GPT(GPTConfig(vocab_size=vocab_size, n_layer=3))
        model_unsharded_numel = sum(p.numel() for p in model.parameters())
        model_sharded_numel = (model_unsharded_numel + 1) // 2
        max_unsharded_numel = sum(
            p.numel() for p in model.transformer.h[0].parameters()
        )  # i.e. block unsharded numel
        non_block_numel = round(
            sum(p.numel() for p in model.transformer.wte.parameters())
            + sum(p.numel() for p in model.transformer.wpe.parameters())
            + sum(p.numel() for p in model.transformer.ln_f.parameters())
        )
        fully_shard_fn = functools.partial(
            fully_shard,
            reshard_after_forward=reshard_after_forward,
            offload_policy=OffloadPolicy("cpu" if use_cpu_offload else None),
        )
        for module in model.modules():
            if isinstance(module, Block):
                fully_shard_fn(module)
        fully_shard_fn(model)
        # Do not use foreach since intermediates increase peak memory
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=False)

        mem_mb = self._get_peak_active_memory_mb()
        # Init: Each module is moved to GPU before sharding parameters
        if use_cpu_offload:
            # Parameters are offloaded after sharding
            self.assertLessEqual(
                mem_mb - base_mem_mb,
                (1.5 * max_unsharded_numel) * 4 / 1e6,
            )
        else:
            self.assertLessEqual(
                mem_mb - base_mem_mb,
                (model_sharded_numel + max_unsharded_numel) * 4 / 1e6,
            )

        # Use a small input to minimize activation memory usage
        src = torch.randint(0, vocab_size, (1, 4), device="cuda")
        tgt = torch.randint(0, vocab_size, (1, 4), device="cuda")
        inp = (src, tgt)

        # Forward:
        loss = model(*inp)
        mem_mb = self._get_peak_active_memory_mb()
        buffer_mb = 32  # 8.1 MiB cuBLAS workspaces, fragmentation, activations
        if reshard_after_forward:
            # 2x max unsharded block parameters (all-gather + copy-out),
            # non-block parameters, and other
            expected_mem_mb = (
                2 * max_unsharded_numel + non_block_numel
            ) * 4 / 1e6 + buffer_mb
            if not use_cpu_offload:
                # Sharded parameters
                expected_mem_mb += model_sharded_numel * 4 / 1e6
        else:
            # Sharded parameters, unsharded parameters, 1x max unsharded block
            # parameters (copy-out) and other (peak at end of forward)
            expected_mem_mb = (
                model_sharded_numel + model_unsharded_numel + max_unsharded_numel
            ) * 4 / 1e6 + buffer_mb
        self.assertLessEqual(mem_mb - base_mem_mb, expected_mem_mb)

        # Backward:
        loss.backward()
        mem_mb = self._get_peak_active_memory_mb()
        if reshard_after_forward:
            # 1x max unsharded block parameters (all-gather), 2.5x max
            # unsharded block gradients (gradients, reduce-scatter input,
            # reduce-scatter output), non-block parameters, and other
            expected_mem_mb = (
                3.5 * max_unsharded_numel + non_block_numel
            ) * 4 / 1e6 + buffer_mb
            if not use_cpu_offload:
                # 2x sharded parameters/gradients
                expected_mem_mb += 2 * model_sharded_numel * 4 / 1e6
        else:
            # Sharded parameters, unsharded parameters, 1.5x max unsharded
            # block parameters (reduce-scatter input/output), and other (peak
            # at beginning of backward)
            expected_mem_mb = (
                model_sharded_numel + model_unsharded_numel + 1.5 * max_unsharded_numel
            ) * 4 / 1e6 + buffer_mb
        self.assertLessEqual(mem_mb - base_mem_mb, expected_mem_mb)
        del loss
        torch.cuda.reset_peak_memory_stats()

        # Optimizer step: unsharded parameters/gradients freed
        optim.step()
        mem_mb = self._get_peak_active_memory_mb()
        expected_mem_mb = buffer_mb
        if not use_cpu_offload:
            # 1x sharded parameters, 1x sharded gradients, 2x sharded optimizer
            # states
            expected_mem_mb += (4 * model_sharded_numel) * 4 / 1e6
        self.assertLessEqual(mem_mb - base_mem_mb, expected_mem_mb)

        # Zero grad: sharded gradients freed
        optim.zero_grad()
        torch.cuda.reset_peak_memory_stats()  # reset after freeing
        mem_mb = self._get_peak_active_memory_mb()
        # 1x sharded parameters
        expected_mem_mb = model_sharded_numel * 4 / 1e6 + buffer_mb
        if not use_cpu_offload:
            # 2x sharded optimizer states
            expected_mem_mb += (2 * model_sharded_numel) * 4 / 1e6 + buffer_mb
        self.assertLessEqual(mem_mb - base_mem_mb, expected_mem_mb)

    def _get_peak_active_memory_mb(self) -> int:
        mem_stats = torch.cuda.memory_stats()
        return round(mem_stats["active_bytes.all.peak"] / 1e6)


if __name__ == "__main__":
    run_tests()
