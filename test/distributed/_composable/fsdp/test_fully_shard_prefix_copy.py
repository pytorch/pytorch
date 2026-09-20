# Owner(s): ["oncall: distributed"]

import math
import unittest
from unittest.mock import Mock

import torch
import torch.distributed as dist
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    parametrize,
    run_tests,
    TestCase,
)


if dist.is_available() and not IS_WINDOWS:
    from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
        _default_reduce_scatter_input_fn,
        foreach_reduce_scatter_copy_in,
    )
    from torch.distributed.fsdp.experimental import (
        reduce_scatter_input_fn_for_nonzero_dim_shards,
    )
    from torch.distributed.tensor import Shard


@unittest.skipIf(IS_WINDOWS, "FSDP2 is not supported on Windows")
@unittest.skipIf(not dist.is_available(), "distributed not available")
class TestFSDPPrefixCopy(TestCase):
    @parametrize("use_nonzero_dim", [False, True])
    @parametrize("world_size", [1, 4])
    @parametrize("mixed_layout", [False, True])
    @dtypes(torch.bfloat16)
    def test_reduce_scatter_preparation(
        self, device, dtype, use_nonzero_dim, world_size, mixed_layout
    ):
        shapes = [
            (world_size * 3 - 1, 5),
            (2, world_size * 3, 5),
            (2, 3, world_size * 2),
        ]
        grads = [make_tensor(shape, device=device, dtype=dtype) for shape in shapes]
        if mixed_layout:
            grads[2] = grads[2].transpose(0, 1).contiguous().transpose(0, 1)
        shards = []
        for dim, grad in enumerate(grads):
            shape = list(grad.shape)
            shape[dim] = math.ceil(shape[dim] / world_size) * world_size
            padded = grad.new_zeros(shape)
            padded.narrow(dim, 0, grad.size(dim)).copy_(grad)
            shards.append(padded.chunk(world_size, dim))
        expected = torch.cat(
            [shard[rank].flatten() for rank in range(world_size) for shard in shards]
        ).float()
        params = [Mock(fsdp_placement=Shard(dim)) for dim in range(len(grads))]
        prepare = (
            reduce_scatter_input_fn_for_nonzero_dim_shards
            if use_nonzero_dim
            else _default_reduce_scatter_input_fn
        )
        prepared = prepare(params, grads, world_size)
        sizes = prepared.padded_unsharded_sizes
        if not use_nonzero_dim or world_size == 1:
            self.assertIs(prepared.copy_in, foreach_reduce_scatter_copy_in)
        self.assertEqual(len(sizes), len(params))
        self.assertEqual(sum(size.numel() for size in sizes), expected.numel())
        output = torch.empty_like(expected)
        prepared.copy_in(grads, output, world_size)
        self.assertEqual(output, expected, atol=0, rtol=0)


instantiate_device_type_tests(TestFSDPPrefixCopy, globals(), only_for=("cpu", "cuda"))

if __name__ == "__main__":
    run_tests()
