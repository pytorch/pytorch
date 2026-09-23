# Owner(s): ["oncall: distributed"]

import math
import unittest

import torch
import torch.distributed as dist
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    deviceCountAtLeast,
    dtypes,
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    IS_WINDOWS,
    parametrize,
    run_tests,
    TestCase,
)


@unittest.skipIf(IS_WINDOWS, "FSDP2 is not supported on Windows")
@unittest.skipIf(not dist.is_available(), "distributed not available")
class TestPrefixCopy(TestCase):
    @parametrize("num_chunks", [1, 4])
    @parametrize("outer_size", [1, 128])
    @parametrize("inference", [False, True])
    @dtypes(torch.bfloat16, torch.float32, torch.int64)
    def test_split_copy(self, device, dtype, num_chunks, outer_size, inference):
        with torch.inference_mode(inference):
            shapes = [
                (num_chunks * 3, 5),
                (outer_size, num_chunks * 3, 5),
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
            outer_sizes = [math.prod(s[:dim]) for dim, s in enumerate(shapes)]
            versions = [t._version for t in outputs] if not inference else None

            result = torch.ops.fsdp._all_gather_copy_out_(
                outputs, source, splits, outer_sizes=outer_sizes, num_chunks=num_chunks
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

        torch.ops.fsdp._all_gather_copy_out_(
            outputs, source, splits, [1, 2, 6], num_chunks
        )

        self.assertEqual(outputs, expected, atol=0, rtol=0)

    @parametrize("byte_input", [False, True])
    @parametrize("resize", [False, True])
    def test_split_copy_strided_input(self, device, byte_input, resize):
        expected = [
            torch.arange(16, device=device, dtype=torch.float32).view(4, 4),
            torch.arange(16, device=device, dtype=torch.bfloat16).view(2, 8),
        ]
        if not byte_input:
            expected[1] = expected[1].float()
        dtype = torch.uint8 if byte_input else torch.float32
        parts = [
            torch.stack(
                [shard.contiguous().view(dtype).flatten() for shard in t.chunk(2, dim)]
            )
            for dim, t in enumerate(expected)
        ]
        if resize:
            parts = [part[:, :-1] for part in parts]
        packed = torch.cat(parts, dim=1).flatten()
        buffer = packed.new_full((packed.numel(), 2), 7)
        source = buffer[:, 0]
        source.copy_(packed)
        outputs = [torch.empty_like(t).flatten() for t in expected]
        torch.ops.fsdp._all_gather_copy_out_(
            outputs, source, [part.size(1) for part in parts], [1, 2], 2
        )
        for output, part, outer_size in zip(outputs, parts, (1, 2)):
            restored = output.view(dtype).view(outer_size, 2, -1).transpose(0, 1)
            self.assertEqual(restored.flatten()[: part.numel()], part.flatten())
        self.assertEqual(source, packed)
        self.assertEqual(buffer[:, 1], buffer.new_full((packed.numel(),), 7))

    @parametrize("byte_input", [False, True])
    @parametrize("all_empty", [False, True])
    def test_split_copy_cached_output_size(self, device, byte_input, all_empty):
        num_chunks = 2
        dtypes = [torch.float32, torch.bfloat16, torch.int64, torch.float32]
        if not byte_input:
            dtypes = [torch.float32] * len(dtypes)
        buffers = [torch.full((22,), 7, device=device, dtype=dtype) for dtype in dtypes]
        outputs = [buffer[3:-3] for buffer in buffers]
        outer_sizes = [32 if byte_input else 2, 2, 4, 1]
        splits = [32 if byte_input else 8, 3, 1, 0]
        if all_empty:
            splits = [0] * len(splits)
        dtype = torch.uint8 if byte_input else torch.float32
        source = torch.arange(num_chunks * sum(splits), device=device, dtype=dtype)
        parts = source.view(num_chunks, -1).split(splits, dim=1)
        versions = [output._version for output in outputs]
        pointers = [output.data_ptr() for output in outputs]

        torch.ops.fsdp._all_gather_copy_out_(
            outputs, source, splits, outer_sizes, num_chunks
        )

        for output, part, outer_size, buffer, version, pointer in zip(
            outputs, parts, outer_sizes, buffers, versions, pointers
        ):
            packed = output.view(dtype).view(outer_size, num_chunks, -1)
            packed = packed.transpose(0, 1).flatten()
            self.assertEqual(packed[: part.numel()], part.flatten(), atol=0, rtol=0)
            if part.numel() == 0:
                self.assertEqual(output, output.new_full(output.shape, 7))
            self.assertEqual(output.size(), (16,))
            self.assertEqual(output.data_ptr(), pointer)
            self.assertGreater(output._version, version)
            self.assertEqual(buffer[:3], buffer.new_full((3,), 7))
            self.assertEqual(buffer[-3:], buffer.new_full((3,), 7))

    @parametrize("count", [65, 129])
    def test_split_copy_cached_output_batches(self, device, count):
        num_chunks = 8
        inner_sizes = (1, 15, 16, 17, 31, 32, 4097)
        outer_sizes = [2 + i % 3 for i in range(count)]
        rank_sizes = [
            outer_size * inner_sizes[i % len(inner_sizes)]
            for i, outer_size in enumerate(outer_sizes)
        ]
        offsets = [1 + i % 17 for i in range(count)]
        buffers = [
            torch.full(
                (num_chunks * rank_size + offset + 3,),
                7,
                device=device,
                dtype=torch.uint8,
            )
            for rank_size, offset in zip(rank_sizes, offsets)
        ]
        outputs = [buffer[offset:-3] for buffer, offset in zip(buffers, offsets)]
        splits = [rank_size - 1 for rank_size in rank_sizes]
        source = torch.arange(num_chunks * sum(splits), device=device)
        source = source.remainder(127).to(torch.uint8)
        parts = source.view(num_chunks, -1).split(splits, dim=1)

        with DeterministicGuard(True):
            torch.ops.fsdp._all_gather_copy_out_(
                outputs, source, splits, outer_sizes, num_chunks
            )

        for output, part, outer_size, buffer, offset in zip(
            outputs, parts, outer_sizes, buffers, offsets
        ):
            staging = torch.full_like(output, 255)
            staging[: part.numel()] = part.flatten()
            chunks = staging.view(num_chunks, outer_size, -1).unbind(0)
            expected = torch.cat(chunks, dim=1).flatten()
            self.assertEqual(output, expected)
            self.assertEqual(buffer[:offset], buffer.new_full((offset,), 7))
            self.assertEqual(buffer[-3:], buffer.new_full((3,), 7))

    @parametrize("dlpack", [False, True])
    def test_split_copy_cached_output_aliases(self, device, dlpack):
        num_chunks = 8
        size = num_chunks * 4 * 17
        buffer = torch.full((size + 22,), 7, device=device, dtype=torch.uint8)
        offsets = [3, 19]
        outputs = [buffer[offset : offset + size] for offset in offsets]
        if dlpack:
            outputs[1] = torch.from_dlpack(outputs[1])
        outer_sizes, splits = [2, 4], [65, 33]
        source = torch.arange(num_chunks * sum(splits), device=device)
        source = source.remainder(127).to(torch.uint8)
        parts = source.view(num_chunks, -1).split(splits, dim=1)
        expected = buffer.clone()
        for part, outer_size, offset in zip(parts, outer_sizes, offsets):
            staging = torch.full_like(outputs[0], 255)
            staging[: part.numel()] = part.flatten()
            chunks = staging.view(num_chunks, outer_size, -1).unbind(0)
            expected[offset : offset + size] = torch.cat(chunks, dim=1).flatten()

        with DeterministicGuard(True):
            torch.ops.fsdp._all_gather_copy_out_(
                outputs, source, splits, outer_sizes, num_chunks
            )

        self.assertEqual(buffer, expected)

    @parametrize("all_empty", [False, True])
    def test_split_copy_empty(self, device, all_empty):
        source = torch.arange(12, device=device, dtype=torch.float32)
        if all_empty:
            source = source[:0]
        outputs = [torch.empty(0, device=device), torch.empty_like(source)]
        torch.ops.fsdp._all_gather_copy_out_(
            outputs, source, [0, source.numel() // 4], [128, 1], 4
        )
        self.assertEqual(outputs[1], source)
        torch.ops.fsdp._all_gather_copy_out_([], source[:0], [], [], 4)

    @parametrize(
        "invalid,match",
        [
            ("chunks", "positive num_chunks"),
            ("split_count", "per output"),
            ("outer_size_count", "per output"),
            ("negative_split", "non-negative"),
            ("split_sum", "sum"),
            ("zero_outer_size", "positive outer size"),
            ("indivisible_outer_size", "divisible"),
            ("input_size", "divisible"),
            ("output_size", "output size"),
            ("output_outer_size", "output size"),
            ("dtype", "dtype"),
            ("output_contiguity", "contiguous"),
        ],
    )
    def test_split_copy_invalid(self, device, invalid, match):
        source = torch.zeros(16, device=device)
        outputs = [torch.empty_like(source)]
        splits, outer_sizes, num_chunks = [8], [2], 2
        if invalid == "chunks":
            num_chunks = 0
        elif invalid == "split_count":
            splits = []
        elif invalid == "outer_size_count":
            outer_sizes = []
        elif invalid == "negative_split":
            splits = [-8]
        elif invalid == "split_sum":
            splits = [10]
        elif invalid == "zero_outer_size":
            outer_sizes = [0]
        elif invalid == "indivisible_outer_size":
            outer_sizes = [3]
        elif invalid == "input_size":
            source = source[:-1]
        elif invalid == "output_size":
            outputs = [outputs[0][:-1]]
        elif invalid == "output_outer_size":
            outputs = [outputs[0][:-2]]
        elif invalid == "dtype":
            outputs = [outputs[0].to(torch.float64)]
        elif invalid == "output_contiguity":
            outputs = [torch.empty(32, device=device)[::2]]
        with self.assertRaisesRegex(RuntimeError, match):
            torch.ops.fsdp._all_gather_copy_out_(
                outputs, source, splits, outer_sizes, num_chunks
            )

    @parametrize("num_chunks", [1, 4])
    @parametrize("outer_size", [1, 128])
    @parametrize("noncontiguous", [False, True])
    @parametrize("nonzero_shards", [False, True])
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
        self,
        device,
        num_chunks,
        outer_size,
        noncontiguous,
        nonzero_shards,
        in_dtype,
        out_dtype,
    ):
        shapes = [
            (num_chunks + 1, 5),
            (outer_size, num_chunks * 3, 5),
            (2, 3, num_chunks * 2, 5),
        ]
        tensors = [make_tensor(s, device=device, dtype=in_dtype) for s in shapes]
        if noncontiguous:
            tensors[0] = tensors[0].t().contiguous().t()
        dims = [0, 1, 2] if nonzero_shards else [0, 0, 0]
        rank_outputs = []
        for rank in range(num_chunks):
            rank_shards = []
            for dim, tensor in zip(dims, tensors):
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

        result = torch.ops.fsdp._reduce_scatter_copy_in_(
            output, tensors, dims, num_chunks
        )

        self.assertIs(result, output)
        self.assertGreater(output._version, version)
        self.assertEqual(output, expected, atol=0, rtol=0)
        self.assertEqual(buffer[:5], buffer.new_full((5,), 7))
        self.assertEqual(buffer[-5:], buffer.new_full((5,), 7))

    @parametrize("num_leading_dims", [1, 2])
    def test_chunk_cat_empty_outer_size(self, device, num_leading_dims):
        tensor = make_tensor((2, 8, 3), device=device, dtype=torch.float32)
        if num_leading_dims == 2:
            tensor = tensor.unsqueeze(0)
        expected = torch.stack(
            [shard.flatten() for shard in torch.chunk(tensor, 4, num_leading_dims)]
        )
        output = torch.empty_like(expected)
        torch.ops.fsdp._reduce_scatter_copy_in_(
            output, [tensor[:0], tensor], [num_leading_dims] * 2, 4
        )
        self.assertEqual(output, expected, atol=0, rtol=0)

    @parametrize("dim", [0, 1])
    def test_chunk_cat_strided_output(self, device, dim):
        tensor = torch.arange(32, device=device, dtype=torch.bfloat16).view(4, 8)
        expected = torch.stack([t.flatten() for t in tensor.chunk(2, dim)]).float()
        buffer = expected.new_full((expected.numel() * 2,), 7)
        output = buffer[::2].view_as(expected)
        version = output._version
        result = torch.ops.fsdp._reduce_scatter_copy_in_(output, [tensor], [dim], 2)
        self.assertIs(result, output)
        self.assertGreater(output._version, version)
        self.assertEqual(output, expected)
        self.assertEqual(buffer[1::2], buffer.new_full((expected.numel(),), 7))

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
            ("input_contiguity", "contiguous"),
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
        elif invalid == "input_contiguity":
            tensors = [torch.zeros(8, 2, device=device).t()]
        with self.assertRaisesRegex(RuntimeError, match):
            torch.ops.fsdp._reduce_scatter_copy_in_(output, tensors, dims, num_chunks)

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
                torch.ops.fsdp._all_gather_copy_out_(
                    [output], packed, [tensor.numel() // 4], [2], 4
                )
            return
        if operation == "split":
            torch.ops.fsdp._all_gather_copy_out_(
                [output], packed, [tensor.numel() // 4], [2], 4
            )
        else:
            torch.ops.fsdp._reduce_scatter_copy_in_(output, [tensor], [1], 4)
        self.assertEqual(output, expected, atol=0, rtol=0)

    @parametrize("operation", ["split", "chunk"])
    @parametrize("outer_size", [1, 2])
    def test_functionalize(self, device, operation, outer_size):
        tensor = make_tensor((outer_size, 8, 3), device=device, dtype=torch.float32)
        packed = torch.stack([t.flatten() for t in torch.chunk(tensor, 4, dim=1)])
        expected = tensor if operation == "split" else packed
        output = torch.empty_like(expected)

        def copy(destination):
            if operation == "split":
                torch.ops.fsdp._all_gather_copy_out_(
                    [destination], packed, [tensor.numel() // 4], [outer_size], 4
                )
            else:
                torch.ops.fsdp._reduce_scatter_copy_in_(destination, [tensor], [1], 4)
            return destination

        result = torch.func.functionalize(copy)(output)
        self.assertEqual(output, expected, atol=0, rtol=0)
        self.assertEqual(result, expected, atol=0, rtol=0)

    def test_all_gather_schema(self, device):
        source = torch.arange(8, device=device, dtype=torch.float32)
        torch.library.opcheck(
            torch.ops.fsdp._all_gather_copy_out_.default,
            ([torch.empty_like(source)], source, [4], [2], 2),
            test_utils="test_schema",
        )

    def test_all_gather_fake_constant(self, device):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            source = torch.tensor([1.0], device=device)
            output = torch.empty_like(source)
            self.assertIsNotNone(source.constant)
            torch.ops.fsdp._all_gather_copy_out_([output], source, [1], [1], 1)
            self.assertEqual(output.size(), (1,))

    @onlyCUDA
    def test_device_mismatch(self, device):
        tensor = torch.zeros(2, 8, device=device)
        output = torch.empty(16)
        with self.assertRaisesRegex(RuntimeError, "same device"):
            torch.ops.fsdp._all_gather_copy_out_([output], tensor, [4], [2], 4)
        with self.assertRaisesRegex(RuntimeError, "same device"):
            torch.ops.fsdp._reduce_scatter_copy_in_(output, [tensor], [1], 4)

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
                torch.ops.fsdp._reduce_scatter_copy_in_(packed, [tensor], [1], 4)
                torch.ops.fsdp._all_gather_copy_out_([output], packed, [12], [2], 4)
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
        torch.ops.fsdp._reduce_scatter_copy_in_(packed, tensors, [0, 1], num_chunks)
        torch.ops.fsdp._all_gather_copy_out_(
            outputs, packed, splits, [1, 128], num_chunks
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            torch.ops.fsdp._reduce_scatter_copy_in_(packed, tensors, [0, 1], num_chunks)
            torch.ops.fsdp._all_gather_copy_out_(
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
