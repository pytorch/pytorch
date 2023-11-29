# Owner(s): ["oncall: distributed"]

import copy
import functools

from typing import List, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from _test_fully_shard_common import (
    check_sharded_grad_parity,
    MLP,
    patch_reduce_scatter,
)
from torch.distributed._composable.fsdp import fully_shard, OffloadPolicy
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests


class TestFullyShardGradientAccumulation(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    def test_requires_gradient_sync_(self):
        """
        Tests the ``set_requires_gradient_sync`` API to exercise gradient
        accumulation without reduce-scattering. This test includes mixing with
        gradient accumulation *with* reduce-scattering.
        """
        self.run_subtests(
            {
                "reshard_after_forward": [True, False, 2],
                "offload_policy": [OffloadPolicy(), OffloadPolicy("cpu")],
                # For `True`, disable reduce-scatter for all MLPs, and for
                # `False`, only disable it for some MLPs
                "recurse": [True, False],
            },
            self._test_requires_gradient_sync_,
        )

    def _test_requires_gradient_sync_(
        self,
        reshard_after_forward: Union[bool, int],
        offload_policy: OffloadPolicy,
        recurse: bool,
    ):
        torch.manual_seed(42)
        local_batch_size, lin_dim, num_mlps, num_microbatches = (2, 32, 3, 3)
        global_batch_size = local_batch_size * self.world_size
        if not recurse:
            num_mlps_to_disable_reduce_scatter = 2
        model = nn.Sequential(
            *[MLP(lin_dim, torch.device("cpu")) for _ in range(num_mlps)]
        )
        ref_model = copy.deepcopy(model).cuda()
        fully_shard_fn = functools.partial(
            fully_shard,
            reshard_after_forward=reshard_after_forward,
            offload_policy=offload_policy,
        )
        for mlp in model:
            fully_shard_fn(mlp)
        fully_shard_fn(model)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        orig_reduce_scatter = dist.reduce_scatter_tensor
        reduce_scatter_count = 0

        def reduce_scatter_with_count(*args, **kwargs):
            nonlocal reduce_scatter_count
            reduce_scatter_count += 1
            return orig_reduce_scatter(*args, **kwargs)

        torch.manual_seed(1)  # same on all ranks
        for iter_idx in range(5):
            with patch_reduce_scatter(reduce_scatter_with_count):
                for microbatch_idx in range(num_microbatches):
                    is_last_microbatch = microbatch_idx == num_microbatches - 1
                    if recurse:
                        model.set_requires_gradient_sync(is_last_microbatch)
                    else:
                        for mlp in model[:num_mlps_to_disable_reduce_scatter]:
                            mlp.set_requires_gradient_sync(
                                is_last_microbatch, recurse=False
                            )
                    global_inp = torch.rand((global_batch_size, lin_dim), device="cuda")
                    local_inp = global_inp[
                        self.rank
                        * local_batch_size : (self.rank + 1)
                        * local_batch_size
                    ].detach()
                    losses: List[torch.Tensor] = []
                    for _model, _optim, inp in (
                        (ref_model, ref_optim, global_inp),
                        (model, optim, local_inp),
                    ):
                        losses.append(_model(inp).sum())
                        losses[-1].backward()
                    dist.all_reduce(losses[1])  # partial -> replicated
                    self.assertEqual(losses[0], losses[1])
            # Expect one reduce-scatter per MLP on the last microbatch
            expected_reduce_scatter_count = num_mlps
            if not recurse:
                # Expect additional reduce-scatters for non-disabled MLPs
                expected_reduce_scatter_count += (
                    num_mlps - num_mlps_to_disable_reduce_scatter
                ) * (num_microbatches - 1)
            self.assertEqual(reduce_scatter_count, expected_reduce_scatter_count)
            reduce_scatter_count = 0
            for param in ref_model.parameters():
                if param.grad is not None:
                    param.grad.div_(self.world_size)
            check_sharded_grad_parity(self, ref_model, model)
            for _optim in (optim, ref_optim):
                _optim.step()
                # When `set_to_none=False`, we are exercising mixing
                # gradient accumulation with and without communication
                _optim.zero_grad(set_to_none=(iter_idx % 2))


if __name__ == "__main__":
    run_tests()
