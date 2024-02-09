# Owner(s): ["oncall: distributed"]

import functools
from typing import Union

import torch
import torch.distributed as dist

from torch.distributed._composable.fsdp import fully_shard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    FSDPTest,
    patch_all_gather,
    patch_reduce_scatter,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
    TransformerBlock,
)


class TestFullyShardCommunication(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    def test_fully_shard_communication_count(self):
        """
        Tests that FSDP issues the expected number of all-gathers and
        reduce-scatters during forward and backward.
        """
        self.run_subtests(
            {"reshard_after_forward": [True, False, 2]},
            self._test_communication_count,
        )

    def _test_communication_count(
        self,
        reshard_after_forward: Union[bool, int],
    ):
        torch.manual_seed(42)
        model_args = ModelArgs()
        model = Transformer(model_args)
        fully_shard_fn = functools.partial(
            fully_shard, reshard_after_forward=reshard_after_forward
        )
        num_blocks = 0
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard_fn(module)
                num_blocks += 1
        fully_shard_fn(model)
        # We construct `num_blocks` plus 1 FSDP states/communication groups

        orig_all_gather = dist.all_gather_into_tensor
        orig_reduce_scatter = dist.reduce_scatter_tensor
        reduce_scatter_count = all_gather_count = 0

        def all_gather_with_count(*args, **kwargs):
            nonlocal all_gather_count
            all_gather_count += 1
            return orig_all_gather(*args, **kwargs)

        def reduce_scatter_with_count(*args, **kwargs):
            nonlocal reduce_scatter_count
            reduce_scatter_count += 1
            return orig_reduce_scatter(*args, **kwargs)

        torch.manual_seed(42 + self.rank)
        inp = torch.randint(0, model_args.vocab_size, (2, 16), device="cuda")
        with patch_all_gather(all_gather_with_count), patch_reduce_scatter(
            reduce_scatter_with_count
        ):
            loss = model(inp)
        self.assertEqual(all_gather_count, num_blocks + 1)
        self.assertEqual(reduce_scatter_count, 0)
        all_gather_count = reduce_scatter_count = 0
        with patch_all_gather(all_gather_with_count), patch_reduce_scatter(
            reduce_scatter_with_count
        ):
            loss.sum().backward()
        if reshard_after_forward is False:
            self.assertEqual(
                all_gather_count,
                0,
                f"Expects 0 but got {all_gather_count} for reshard_after_forward={reshard_after_forward}",
            )
        else:
            # The root always does not reshard after forward
            self.assertEqual(all_gather_count, num_blocks)
        self.assertEqual(reduce_scatter_count, num_blocks + 1)


if __name__ == "__main__":
    run_tests()
