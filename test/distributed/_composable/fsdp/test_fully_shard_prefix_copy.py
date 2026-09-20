# Owner(s): ["oncall: distributed"]

import math
import unittest
from collections import Counter
from unittest.mock import Mock

import torch
import torch.distributed as dist
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    _default_all_gather_output_fn,
    _default_reduce_scatter_input_fn,
    AllGatherResult,
    foreach_reduce_scatter_copy_in,
)
from torch.distributed.fsdp._fully_shard._fsdp_param import ShardedState
from torch.distributed.tensor import Shard
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    deviceCountAtLeast,
    dtypes,
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._python_dispatch import TorchDispatchMode


class _OpCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.counts = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[func] += 1
        return func(*args, **(kwargs or {}))


@unittest.skipIf(IS_WINDOWS, "FSDP2 is not supported on Windows")
@unittest.skipIf(not dist.is_available(), "distributed not available")
class TestPrefixCopy(TestCase):
    @parametrize("nonzero_shards", [False, True])
    @parametrize("world_size", [1, 4])
    @parametrize("mixed_layout", [False, True])
    @dtypes(torch.bfloat16)
    def test_reduce_scatter_preparation(
        self, device, dtype, nonzero_shards, world_size, mixed_layout
    ):
        shapes = [
            (world_size * 3 - 1, 5),
            (2, world_size * 3, 5),
            (2, 3, world_size * 2),
        ]
        grads = [make_tensor(shape, device=device, dtype=dtype) for shape in shapes]
        if mixed_layout:
            grads[2] = grads[2].transpose(0, 1).contiguous().transpose(0, 1)
        shard_dims = list(range(len(grads))) if nonzero_shards else [0] * len(grads)
        shards = []
        for dim, grad in zip(shard_dims, grads):
            shape = list(grad.shape)
            shape[dim] = math.ceil(shape[dim] / world_size) * world_size
            padded = grad.new_zeros(shape)
            padded.narrow(dim, 0, grad.size(dim)).copy_(grad)
            shards.append(padded.chunk(world_size, dim))
        expected = torch.cat(
            [shard[rank].flatten() for rank in range(world_size) for shard in shards]
        ).float()
        params = [Mock(fsdp_placement=Shard(dim)) for dim in shard_dims]
        prepared = _default_reduce_scatter_input_fn(params, grads, world_size)
        sizes = prepared.padded_unsharded_sizes
        use_prefix_copy = nonzero_shards and world_size > 1
        if use_prefix_copy:
            self.assertIsNot(prepared.copy_in, foreach_reduce_scatter_copy_in)
        else:
            self.assertIs(prepared.copy_in, foreach_reduce_scatter_copy_in)
        self.assertEqual(len(sizes), len(params))
        self.assertEqual(sum(size.numel() for size in sizes), expected.numel())
        output = torch.empty_like(expected)
        with _OpCounter() as counter:
            prepared.copy_in(grads, output, world_size)
        self.assertEqual(output, expected, atol=0, rtol=0)
        self.assertEqual(
            counter.counts[torch.ops.fsdp._chunk_cat_with_prefixes_.default],
            int(use_prefix_copy),
        )
        self.assertEqual(
            counter.counts[torch.ops.fsdp.chunk_cat.default], int(not use_prefix_copy)
        )

    @parametrize(
        "layout",
        [
            "shard0",
            "shard1",
            "singleton_prefix",
            "mixed",
            "extension",
            "post_forward",
            "zero_prefix",
            "mixed_fallback",
        ],
    )
    def test_all_gather_default(self, device, layout):
        world_size = 4
        layouts = {
            "mixed": ("shard0", "shard1"),
            "zero_prefix": ("zero_prefix", "shard0"),
            "mixed_fallback": ("shard1", "extension", "post_forward", "zero_prefix"),
        }.get(layout, (layout,))
        params, expected, shards, outputs = [], [], [], []
        for kind in layouts:
            dim = 0 if kind in ("shard0", "post_forward") else 1
            shape = {
                "shard0": (12, 5),
                "singleton_prefix": (1, 12, 5),
                "post_forward": (16, 5),
                "zero_prefix": (0, 12, 5),
            }.get(kind, (2, 12, 5))
            dtype = torch.bfloat16 if layout == "mixed" and dim == 1 else torch.float32
            tensor = make_tensor(shape, device=device, dtype=dtype)
            rank_shards = tensor.chunk(world_size, dim=dim)
            shard_size = list(rank_shards[0].size())
            state = ShardedState.SHARDED
            if kind == "post_forward":
                state = ShardedState.SHARDED_POST_FORWARD
                shard_size[dim] //= 2
            output = tensor.new_empty(tensor.numel())
            params.append(
                Mock(
                    fsdp_placement=Shard(dim),
                    padded_sharded_param_size=torch.Size(shard_size),
                    sharded_state=state,
                    _sharded_local_tensor=(
                        Mock(spec=["fsdp_pre_all_gather"])
                        if kind == "extension"
                        else rank_shards[0]
                    ),
                    _sharded_post_forward_param_data=(
                        rank_shards[0].flatten() if kind == "post_forward" else None
                    ),
                    all_gather_outputs=[output],
                )
            )
            expected.append(tensor)
            shards.append(rank_shards)
            outputs.append(output)

        gather_dtype = torch.uint8 if layout == "mixed" else torch.float32
        packed = torch.cat(
            [
                shard[rank].contiguous().view(gather_dtype).flatten()
                for rank in range(world_size)
                for shard in shards
            ]
        )
        splits = [
            tensor.nbytes // world_size
            if gather_dtype == torch.uint8
            else tensor.numel() // world_size
            for tensor in expected
        ]
        result = AllGatherResult(
            packed,
            None,
            None,
            [[tensor.dtype] for tensor in expected],
            [[tensor.numel() // world_size] for tensor in expected],
            splits,
        )
        versions = [output._version for output in outputs]
        with torch.no_grad(), _OpCounter() as counter:
            _default_all_gather_output_fn(params, result, world_size)

        self.assertEqual(
            [output.view(tensor.shape) for output, tensor in zip(outputs, expected)],
            expected,
            atol=0,
            rtol=0,
        )
        self.assertEqual([output._version for output in outputs], versions)
        use_prefix_copy = "shard1" in layouts or "singleton_prefix" in layouts
        prefix_op = torch.ops.fsdp._split_with_sizes_copy_with_prefixes_.default
        self.assertEqual(counter.counts[prefix_op], int(use_prefix_copy))
        self.assertEqual(
            counter.counts[torch.ops.fsdp.split_with_sizes_copy.default],
            int(not use_prefix_copy),
        )
        fallback_layouts = ("extension", "zero_prefix")
        self.assertEqual(
            counter.counts[torch.ops.aten.cat.out],
            sum(kind in fallback_layouts for kind in layouts),
        )

    @parametrize("num_chunks", [1, 4])
    @parametrize("num_prefixes", [1, 128])
    @parametrize("inference", [False, True])
    @dtypes(torch.bfloat16, torch.float32, torch.int64)
    def test_split_copy(self, device, dtype, num_chunks, num_prefixes, inference):
        with torch.inference_mode(inference):
            shapes = [
                (num_chunks * 3, 5),
                (num_prefixes, num_chunks * 3, 5),
                (2, 3, num_chunks * 2, 5),
            ]
            expected = [make_tensor(s, device=device, dtype=dtype) for s in shapes]
            shards = [torch.chunk(t, num_chunks, dim) for dim, t in enumerate(expected)]
            packed = torch.cat(
                [s[rank].flatten() for rank in range(num_chunks) for s in shards]
            )
            source = torch.cat([packed.new_zeros(3), packed, packed.new_zeros(3)])[3:-3]
            buffers = [t.new_full((t.numel() + 10,), 7) for t in expected]
            outputs = [b[5:-5].view(s) for b, s in zip(buffers, shapes)]
            splits = [t.numel() // num_chunks for t in expected]
            prefixes = [math.prod(s[:dim]) for dim, s in enumerate(shapes)]
            versions = [t._version for t in outputs] if not inference else None

            result = torch.ops.fsdp._split_with_sizes_copy_with_prefixes_(
                outputs, source, splits, prefixes, num_chunks
            )

            self.assertIsNone(result)
            self.assertEqual(outputs, expected, atol=0, rtol=0)
            self.assertEqual(source, packed, atol=0, rtol=0)
            for index, buffer in enumerate(buffers):
                self.assertEqual(buffer[:5], buffer.new_full((5,), 7))
                self.assertEqual(buffer[-5:], buffer.new_full((5,), 7))
                if versions is not None:
                    self.assertGreater(outputs[index]._version, versions[index])

    def test_split_copy_mixed_dtype(self, device):
        num_chunks = 4
        shapes = [(12, 5), (2, 12, 5), (2, 3, 8, 5)]
        dtypes = [torch.bfloat16, torch.float32, torch.int64]
        expected = [
            make_tensor(shape, device=device, dtype=dtype)
            for shape, dtype in zip(shapes, dtypes)
        ]
        shards = [torch.chunk(t, num_chunks, dim) for dim, t in enumerate(expected)]
        packed = torch.cat(
            [
                shard[rank].contiguous().view(torch.uint8).flatten()
                for rank in range(num_chunks)
                for shard in shards
            ]
        )
        source = torch.cat([packed.new_zeros(3), packed, packed.new_zeros(3)])[3:-3]
        outputs = [torch.empty_like(t) for t in expected]
        splits = [t.nbytes // num_chunks for t in expected]

        torch.ops.fsdp._split_with_sizes_copy_with_prefixes_(
            outputs, source, splits, [1, 2, 6], num_chunks
        )

        self.assertEqual(outputs, expected, atol=0, rtol=0)

    @parametrize("all_empty", [False, True])
    def test_split_copy_empty(self, device, all_empty):
        source = torch.arange(12, device=device, dtype=torch.float32)
        if all_empty:
            source = source[:0]
        outputs = [torch.empty(0, device=device), torch.empty_like(source)]
        torch.ops.fsdp._split_with_sizes_copy_with_prefixes_(
            outputs, source, [0, source.numel() // 4], [128, 1], 4
        )
        self.assertEqual(outputs[1], source)
        torch.ops.fsdp._split_with_sizes_copy_with_prefixes_([], source[:0], [], [], 4)

    @parametrize(
        "invalid,match",
        [
            ("chunks", "positive num_chunks"),
            ("split_count", "per output"),
            ("prefix_count", "per output"),
            ("negative_split", "non-negative"),
            ("split_sum", "sum"),
            ("zero_prefix", "positive prefix"),
            ("indivisible_prefix", "divisible"),
            ("input_size", "divisible"),
            ("output_size", "output size"),
            ("dtype", "dtype"),
            ("input_contiguity", "contiguous"),
            ("output_contiguity", "contiguous"),
        ],
    )
    def test_split_copy_invalid(self, device, invalid, match):
        source = torch.zeros(16, device=device)
        outputs = [torch.empty_like(source)]
        splits, prefixes, num_chunks = [8], [2], 2
        if invalid == "chunks":
            num_chunks = 0
        elif invalid == "split_count":
            splits = []
        elif invalid == "prefix_count":
            prefixes = []
        elif invalid == "negative_split":
            splits = [-8]
        elif invalid == "split_sum":
            splits = [10]
        elif invalid == "zero_prefix":
            prefixes = [0]
        elif invalid == "indivisible_prefix":
            prefixes = [3]
        elif invalid == "input_size":
            source = source[:-1]
        elif invalid == "output_size":
            outputs = [outputs[0][:-2]]
        elif invalid == "dtype":
            outputs = [outputs[0].to(torch.float64)]
        elif invalid == "input_contiguity":
            source = torch.zeros(32, device=device)[::2]
        elif invalid == "output_contiguity":
            outputs = [torch.empty(32, device=device)[::2]]
        with self.assertRaisesRegex(RuntimeError, match):
            torch.ops.fsdp._split_with_sizes_copy_with_prefixes_(
                outputs, source, splits, prefixes, num_chunks
            )

    @parametrize("num_chunks", [1, 4])
    @parametrize("num_prefixes", [1, 128])
    @parametrize("noncontiguous", [False, True])
    @parametrize(
        "in_dtype,out_dtype",
        [
            (torch.bfloat16, torch.bfloat16),
            (torch.bfloat16, torch.float32),
            (torch.float32, torch.float32),
            (torch.float32, torch.float64),
        ],
    )
    def test_chunk_cat(
        self, device, num_chunks, num_prefixes, noncontiguous, in_dtype, out_dtype
    ):
        shapes = [
            (num_chunks + 1, 5),
            (num_prefixes, num_chunks * 3, 5),
            (2, 3, num_chunks * 2, 5),
        ]
        tensors = [make_tensor(s, device=device, dtype=in_dtype) for s in shapes]
        if noncontiguous:
            tensors[0] = tensors[0].t().contiguous().t()
        rank_outputs = []
        for rank in range(num_chunks):
            rank_shards = []
            for dim, tensor in enumerate(tensors):
                shard_size = (tensor.size(dim) + num_chunks - 1) // num_chunks
                start = min(rank * shard_size, tensor.size(dim))
                length = min(shard_size, tensor.size(dim) - start)
                shape = list(tensor.shape)
                shape[dim] = shard_size
                shard = torch.zeros(shape, device=device, dtype=out_dtype)
                shard.narrow(dim, 0, length).copy_(tensor.narrow(dim, start, length))
                rank_shards.append(shard.flatten())
            rank_outputs.append(torch.cat(rank_shards))
        expected = torch.stack(rank_outputs)
        buffer = expected.new_full((expected.numel() + 10,), 7)
        output = buffer[5:-5].view(expected.shape)
        version = output._version

        result = torch.ops.fsdp._chunk_cat_with_prefixes_(
            output, tensors, [0, 1, 2], num_chunks
        )

        self.assertIs(result, output)
        self.assertGreater(output._version, version)
        self.assertEqual(output, expected, atol=0, rtol=0)
        self.assertEqual(buffer[:5], buffer.new_full((5,), 7))
        self.assertEqual(buffer[-5:], buffer.new_full((5,), 7))

    @parametrize("num_leading_dims", [1, 2])
    def test_chunk_cat_empty_prefix(self, device, num_leading_dims):
        tensor = make_tensor((2, 8, 3), device=device, dtype=torch.float32)
        if num_leading_dims == 2:
            tensor = tensor.unsqueeze(0)
        expected = torch.stack(
            [shard.flatten() for shard in torch.chunk(tensor, 4, num_leading_dims)]
        )
        output = torch.empty_like(expected)
        torch.ops.fsdp._chunk_cat_with_prefixes_(
            output, [tensor[:0], tensor], [num_leading_dims] * 2, 4
        )
        self.assertEqual(output, expected, atol=0, rtol=0)

    @parametrize(
        "invalid,match",
        [
            ("chunks", "positive num_chunks"),
            ("empty_list", "non-empty"),
            ("empty_tensor", "non-empty"),
            ("dim_count", "per input"),
            ("negative_dim", "non-negative"),
            ("dim_too_large", "input ndim"),
            ("indivisible_shard", "evenly divisible"),
            ("dtype", "same dtype"),
            ("prefix_contiguity", "contiguous"),
            ("output_contiguity", "contiguous"),
        ],
    )
    def test_chunk_cat_invalid(self, device, invalid, match):
        tensors = [torch.zeros(2, 8, device=device)]
        dims, num_chunks = [1], 4
        output = torch.empty(4, 4, device=device)
        if invalid == "chunks":
            num_chunks = 0
        elif invalid == "empty_list":
            tensors, dims = [], []
        elif invalid == "empty_tensor":
            tensors = [tensors[0][:0]]
        elif invalid == "dim_count":
            dims = []
        elif invalid == "negative_dim":
            dims = [-1]
        elif invalid == "dim_too_large":
            dims = [2]
        elif invalid == "indivisible_shard":
            tensors = [torch.zeros(2, 7, device=device)]
        elif invalid == "dtype":
            tensors.append(tensors[0].to(torch.float64))
            dims.append(1)
        elif invalid == "prefix_contiguity":
            tensors = [torch.zeros(8, 2, device=device).t()]
        elif invalid == "output_contiguity":
            output = output.t()
        with self.assertRaisesRegex(RuntimeError, match):
            torch.ops.fsdp._chunk_cat_with_prefixes_(output, tensors, dims, num_chunks)

    @parametrize("operation", ["split", "chunk"])
    @parametrize("view", ["conjugate", "negative"])
    @parametrize("flagged_output", [False, True])
    def test_view_flags(self, device, operation, view, flagged_output):
        tensor = make_tensor((2, 8, 3), device=device, dtype=torch.complex64)
        packed = torch.stack([t.flatten() for t in torch.chunk(tensor, 4, dim=1)])
        view_fn = torch.Tensor.conj if view == "conjugate" else torch._neg_view
        if not flagged_output:
            tensor = view_fn(tensor)
            packed = view_fn(packed)
        expected = tensor if operation == "split" else packed
        output = torch.empty_like(expected)
        if flagged_output:
            output = view_fn(output)
        if operation == "split" and flagged_output:
            with self.assertRaisesRegex(RuntimeError, "mutable TensorLists"):
                torch.ops.fsdp._split_with_sizes_copy_with_prefixes_(
                    [output], packed, [tensor.numel() // 4], [2], 4
                )
            return
        if operation == "split":
            torch.ops.fsdp._split_with_sizes_copy_with_prefixes_(
                [output], packed, [tensor.numel() // 4], [2], 4
            )
        else:
            torch.ops.fsdp._chunk_cat_with_prefixes_(output, [tensor], [1], 4)
        self.assertEqual(output, expected, atol=0, rtol=0)

    @parametrize("operation", ["split", "chunk"])
    def test_functionalize(self, device, operation):
        tensor = make_tensor((2, 8, 3), device=device, dtype=torch.float32)
        packed = torch.stack([t.flatten() for t in torch.chunk(tensor, 4, dim=1)])
        expected = tensor if operation == "split" else packed
        output = torch.empty_like(expected)

        def copy(destination):
            if operation == "split":
                torch.ops.fsdp._split_with_sizes_copy_with_prefixes_(
                    [destination], packed, [tensor.numel() // 4], [2], 4
                )
            else:
                torch.ops.fsdp._chunk_cat_with_prefixes_(destination, [tensor], [1], 4)
            return destination

        result = torch.func.functionalize(copy)(output)
        self.assertEqual(output, expected, atol=0, rtol=0)
        self.assertEqual(result, expected, atol=0, rtol=0)

    @onlyCUDA
    def test_device_mismatch(self, device):
        tensor = torch.zeros(2, 8, device=device)
        output = torch.empty(16)
        with self.assertRaisesRegex(RuntimeError, "same device"):
            torch.ops.fsdp._split_with_sizes_copy_with_prefixes_(
                [output], tensor, [4], [2], 4
            )
        with self.assertRaisesRegex(RuntimeError, "same device"):
            torch.ops.fsdp._chunk_cat_with_prefixes_(output, [tensor], [1], 4)

    @onlyCUDA
    @deviceCountAtLeast(2)
    def test_device_and_stream(self, devices):
        stream = torch.cuda.Stream(device=devices[0])
        with torch.cuda.stream(stream):
            tensor = make_tensor((2, 8, 3), device=devices[0], dtype=torch.float32)
            expected = torch.stack([t.flatten() for t in torch.chunk(tensor, 4, 1)])
            packed = torch.empty_like(expected)
            output = torch.empty_like(tensor)
            with torch.cuda.device(devices[1]):
                torch.ops.fsdp._chunk_cat_with_prefixes_(packed, [tensor], [1], 4)
                torch.ops.fsdp._split_with_sizes_copy_with_prefixes_(
                    [output], packed, [12], [2], 4
                )
                self.assertEqual(
                    torch.cuda.current_device(), torch.device(devices[1]).index
                )
        stream.synchronize()
        self.assertEqual(packed, expected, atol=0, rtol=0)
        self.assertEqual(output, tensor, atol=0, rtol=0)

    @onlyCUDA
    def test_cuda_graph(self, device):
        tensors = [
            make_tensor(shape, device=device, dtype=torch.bfloat16)
            for shape in [(8, 3), (128, 8, 3)]
        ]
        num_chunks = 4
        splits = [t.numel() // num_chunks for t in tensors]
        packed = tensors[0].new_empty((num_chunks, sum(splits)))
        outputs = [torch.empty_like(t) for t in tensors]
        torch.ops.fsdp._chunk_cat_with_prefixes_(packed, tensors, [0, 1], num_chunks)
        torch.ops.fsdp._split_with_sizes_copy_with_prefixes_(
            outputs, packed, splits, [1, 128], num_chunks
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            torch.ops.fsdp._chunk_cat_with_prefixes_(
                packed, tensors, [0, 1], num_chunks
            )
            torch.ops.fsdp._split_with_sizes_copy_with_prefixes_(
                outputs, packed, splits, [1, 128], num_chunks
            )
        for tensor in tensors:
            tensor.add_(1)
        graph.replay()
        shards = [torch.chunk(t, num_chunks, dim) for dim, t in enumerate(tensors)]
        expected = torch.stack(
            [
                torch.cat([shard[rank].flatten() for shard in shards])
                for rank in range(num_chunks)
            ]
        )
        self.assertEqual(packed, expected, atol=0, rtol=0)
        self.assertEqual(outputs, tensors, atol=0, rtol=0)


instantiate_device_type_tests(TestPrefixCopy, globals(), only_for=("cpu", "cuda"))

if __name__ == "__main__":
    run_tests()
