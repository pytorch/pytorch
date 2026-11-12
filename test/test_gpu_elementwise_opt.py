import math
import os

import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


SHAPES = [(4096,), (64, 128), (4, 16, 64), (2, 4, 8, 64), (2, 3, 4, 8, 32)]


def values(shape, dtype):
    data = (torch.arange(math.prod(shape)) % 17 - 8).reshape(shape)
    if dtype == torch.bool:
        return data.remainder(3) == 0
    if dtype.is_complex:
        return torch.complex(data.float(), data.remainder(5).float()).to(dtype)
    return data.to(dtype)


def view_pair(shape, dtype, layout, device):
    if layout == "gapped":
        base_shape = tuple(2 * size + 1 for size in shape)
        cpu = values(base_shape, dtype)
        index = tuple(slice(1, 2 * size + 1, 2) for size in shape)
        return cpu[index], cpu.to(device)[index]
    cpu = values(shape, dtype)
    gpu = cpu.to(device)
    if layout == "permuted":
        order = tuple(reversed(range(len(shape))))
        return cpu.permute(order), gpu.permute(order)
    if layout == "broadcast":
        base_shape = tuple(1 if axis % 2 == 0 else size for axis, size in enumerate(shape))
        cpu = values(base_shape, dtype)
        return cpu.expand(shape), cpu.to(device).expand(shape)
    return cpu, gpu


class TestGPUElementwiseOpt(TestCase):
    @dtypes(torch.float32, torch.float64, torch.float16, torch.bfloat16,
            torch.int32, torch.int64, torch.bool, torch.complex64)
    @parametrize("shape", SHAPES)
    @parametrize("layout", ["gapped", "permuted", "broadcast"])
    def test_copy_layouts(self, device, dtype, shape, layout):
        cpu, gpu = view_pair(shape, dtype, layout, device)
        expected = torch.empty(cpu.shape, dtype=dtype).copy_(cpu)
        actual = torch.empty(cpu.shape, dtype=dtype, device=device).copy_(gpu)
        self.assertEqual(actual.cpu(), expected)
        out_cpu, out_gpu = view_pair(tuple(cpu.shape), dtype, "gapped", device)
        out_cpu.copy_(cpu)
        out_gpu.copy_(gpu)
        self.assertEqual(out_gpu.cpu(), out_cpu)

    @parametrize("dtype_pair", [
        (torch.bfloat16, torch.float32), (torch.float16, torch.float32),
        (torch.int64, torch.float32), (torch.float64, torch.float32),
        (torch.bool, torch.float32), (torch.int32, torch.float32),
        (torch.float64, torch.bfloat16), (torch.uint8, torch.bfloat16),
        (torch.bfloat16, torch.float64), (torch.complex128, torch.complex64),
        (torch.float32, torch.int64), (torch.bool, torch.int64),
        (torch.int32, torch.int64), (torch.float32, torch.int32),
        (torch.int64, torch.int32), (torch.bfloat16, torch.int32),
        (torch.bool, torch.int32), (torch.int32, torch.bool),
        (torch.int64, torch.bool),
    ])
    @parametrize("length", [0, 1, 7, 16, 255, 256, 1025, 4096])
    def test_contiguous_cast(self, device, dtype_pair, length):
        source_dtype, target_dtype = dtype_pair
        cpu = values((length + 2,), source_dtype)
        gpu = cpu.to(device)
        for offset in (0, 1):
            actual = gpu[offset:offset + length].to(target_dtype)
            expected = cpu[offset:offset + length].to(target_dtype)
            self.assertEqual(actual.cpu(), expected)

    @parametrize("dtype_pair", [
        (torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16),
        (torch.float32, torch.float16), (torch.int64, torch.float32),
    ])
    @parametrize("shape", SHAPES)
    @parametrize("layout", ["gapped", "permuted", "broadcast"])
    def test_cast_layouts(self, device, dtype_pair, shape, layout):
        source_dtype, target_dtype = dtype_pair
        cpu, gpu = view_pair(shape, source_dtype, layout, device)
        actual = torch.empty(cpu.shape, dtype=target_dtype, device=device).copy_(gpu)
        expected = torch.empty(cpu.shape, dtype=target_dtype).copy_(cpu)
        self.assertEqual(actual.cpu(), expected)

    @dtypes(torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32)
    @parametrize("shape", SHAPES)
    @parametrize("pattern", ["scalar", "row", "column", "cross"])
    def test_binary_broadcast(self, device, dtype, shape, pattern):
        if pattern == "scalar":
            shape_a, shape_b = shape, ()
        elif pattern == "row":
            shape_a, shape_b = shape, (shape[-1],)
        elif pattern == "column":
            shape_a, shape_b = shape, shape[:-1] + (1,)
        else:
            shape_a = tuple(size if axis % 2 == 0 else 1 for axis, size in enumerate(shape))
            shape_b = tuple(1 if axis % 2 == 0 else size for axis, size in enumerate(shape))
        a, b = values(shape_a, dtype), values(shape_b, dtype)
        ga, gb = a.to(device), b.to(device)
        for op in (torch.add, torch.sub, torch.mul, torch.gt):
            for left, right, gleft, gright in ((a, b, ga, gb), (b, a, gb, ga)):
                self.assertEqual(op(gleft, gright).cpu(), op(left, right))

    @dtypes(torch.float32, torch.bfloat16, torch.float16, torch.float64)
    @parametrize("shape", [(37, 7), (32768, 8), (32769, 8), (2051, 128),
                          (3, 8193, 8), (5, 4099, 16), (3, 29, 13)])
    @parametrize("layout", ["cross", "middle", "gather"])
    def test_packed_layout_boundaries(self, device, dtype, shape, layout):
        if layout == "cross":
            sa = tuple(n if i % 2 == 0 else 1 for i, n in enumerate(shape))
            sb = tuple(1 if i % 2 == 0 else n for i, n in enumerate(shape))
            a, b = values(sa, dtype).abs() + 1, values(sb, dtype).abs() + 1
        elif layout == "middle":
            a = values(shape, dtype).abs() + 1
            sb = list(shape)
            sb[-2] = 1
            b = values(tuple(sb), dtype).abs() + 1
        else:
            source = values(shape[:-1] + (2 * shape[-1] + 1,), dtype).abs() + 1
            a = source[..., 1:2 * shape[-1] + 1:2]
            b = values(shape[:-1] + (1,), dtype).abs() + 1
        ga, gb = a.to(device), b.to(device)
        if layout == "gather":
            ga = source.to(device)[..., 1:2 * shape[-1] + 1:2]
        for offset, step in ((0, 1), (1, 1), (1, 2)):
            strides = [step]
            for n in reversed(shape[1:]):
                strides.insert(0, strides[0] * n + 3)
            span = offset + 1 + sum((n - 1) * s for n, s in zip(shape, strides))
            for op in (torch.sub, torch.div):
                for left, right, gl, gr in ((a, b, ga, gb), (b, a, gb, ga)):
                    expected = torch.full((span + 7,), -99, dtype=dtype)
                    actual = expected.to(device)
                    op(left, right, out=expected.as_strided(shape, strides, offset))
                    op(gl, gr, out=actual.as_strided(shape, strides, offset))
                    self.assertEqual(actual.cpu(), expected)

    @dtypes(torch.float64, torch.int64)
    @parametrize("shape", [(129, 127), (128, 128), (512, 512), (1024, 1024), (1025, 1024)])
    def test_packed_wide_cross(self, device, dtype, shape):
        rows, width = shape
        a = values((rows + 2,), dtype)[1:rows + 1, None].abs() + 1
        b = values((1, width), dtype).abs() + 1
        ga, gb = a.to(device), b.to(device)
        ops = (torch.add, torch.sub, torch.mul)
        if dtype.is_floating_point:
            ops += (torch.div, torch.pow)
        for offset, gap in ((0, 0), (0, 2), (1, 1)):
            strides = (width + gap, 1)
            span = offset + (rows - 1) * strides[0] + width
            for op in ops:
                for left, right, gl, gr in ((a, b, ga, gb), (b, a, gb, ga)):
                    expected = torch.full((span + 7,), -99, dtype=dtype)
                    actual = expected.to(device)
                    op(left, right, out=expected.as_strided(shape, strides, offset))
                    op(gl, gr, out=actual.as_strided(shape, strides, offset))
                    self.assertEqual(actual.cpu(), expected)

    @dtypes(torch.float32, torch.bfloat16)
    @parametrize("broadcast", [False, True])
    def test_four_dim_fast_divmod(self, device, dtype, broadcast):
        base = values((7, 6, 14, 33), dtype).abs() + 1
        a, ga = base[:, :5, :13, :32], base.to(device)[:, :5, :13, :32]
        b = values((7, 5, 13, 1) if broadcast else (7, 5, 13, 32), dtype).abs() + 1
        gb = b.to(device)
        for op in (torch.sub, torch.div, torch.pow):
            self.assertEqual(op(ga, gb).cpu(), op(a, b))

    def test_pcb_direct_tile_tail(self, device):
        dtype = torch.bfloat16
        base = values((129, 64, 4, 4, 16), dtype).abs() + 1
        a, ga = base.permute(4, 0, 2, 1, 3), base.to(device).permute(4, 0, 2, 1, 3)
        for offset in (0, 1):
            for op in (torch.neg, torch.log, torch.ops.aten.gelu.out):
                expected = torch.full((a.numel() + 7,), -99, dtype=dtype)
                actual = expected.to(device)
                op(a, out=expected[offset:a.numel() + offset].view(a.shape))
                op(ga, out=actual[offset:a.numel() + offset].view(a.shape))
                self.assertEqual(actual.cpu(), expected)

    @parametrize("width", [8193, 65537, 262144])
    def test_packed_unary_row_gap(self, device, width):
        dtype = torch.bfloat16
        base = values((2, 3, width + 4), dtype).abs() + 1
        a, ga = base[:, :2, :width], base.to(device)[:, :2, :width]
        for offset in (0, 1):
            for op in (torch.neg, torch.log, torch.ops.aten.gelu.out):
                expected = torch.full((a.numel() + 7,), -99, dtype=dtype)
                actual = expected.to(device)
                op(a, out=expected[offset:a.numel() + offset].view(a.shape))
                op(ga, out=actual[offset:a.numel() + offset].view(a.shape))
                self.assertEqual(actual.cpu(), expected)

    @dtypes(torch.float32, torch.bfloat16, torch.float64, torch.int64)
    @parametrize("length", [262143, 262144, 262145, 262147, 1048579])
    def test_strided_work_tail(self, device, dtype, length):
        base = values((length * 3 + 5,), dtype)
        gpu = base.to(device)
        source, gsource = base[1:1 + 3 * length:3], gpu[1:1 + 3 * length:3]
        expected = torch.full((length + 7,), -99, dtype=dtype)
        actual = expected.to(device)
        expected[1:length + 1].copy_(source)
        actual[1:length + 1].copy_(gsource)
        self.assertEqual(actual.cpu(), expected)
        self.assertEqual((gsource + 2).cpu(), source + 2)

    @dtypes(torch.float32, torch.bfloat16)
    def test_packed_special_values_and_mask(self, device, dtype):
        data = torch.tensor([float("nan"), float("inf"), -float("inf"),
                             0., -0., 1., -1., 2.], dtype=dtype)
        a = data.expand(3, 11001, 8)
        b = torch.ones((3, 1, 8), dtype=dtype)
        ga, gb = a.to(device), b.to(device)
        self.assertEqual((ga - gb).cpu(), a - b)
        mask = torch.arange(8).remainder(2).bool().expand(3, 1, 8)
        self.assertEqual(ga.masked_fill(mask.to(device), 0).cpu(), a.masked_fill(mask, 0))

    @parametrize("shape", SHAPES)
    @parametrize("dtype_pair", [
        (torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16),
        (torch.float16, torch.float32), (torch.float32, torch.float16),
    ])
    def test_mixed_binary(self, device, shape, dtype_pair):
        da, db = dtype_pair
        a, b = values(shape, da), values(shape[:-1] + (1,), db)
        ga, gb = a.to(device), b.to(device)
        for op in (torch.add, torch.sub, torch.mul):
            self.assertEqual(op(ga, gb).cpu(), op(a, b))
            self.assertEqual(op(gb, ga).cpu(), op(b, a))
            for out_dtype in (da, db):
                expected = torch.empty(shape, dtype=out_dtype)
                actual = torch.empty(shape, dtype=out_dtype, device=device)
                op(a, b, out=expected)
                op(ga, gb, out=actual)
                self.assertEqual(actual.cpu(), expected)

    @dtypes(torch.float32, torch.bfloat16, torch.int64, torch.bool)
    @parametrize("length", [1, 255, 256, 257, 1023, 1024, 1025, 4096])
    def test_where(self, device, dtype, length):
        a, b = values((length,), dtype), values((), dtype)
        mask = torch.arange(length).remainder(3) == 0
        self.assertEqual(
            torch.where(mask.to(device), a.to(device), b.to(device)).cpu(),
            torch.where(mask, a, b),
        )
        self.assertEqual(
            torch.where(torch.tensor(True, device=device), a.to(device), a.to(device)).cpu(), a,
        )

    @dtypes(torch.float32, torch.float64, torch.float16, torch.bfloat16)
    @parametrize("length", [7, 255, 256, 257, 1024, 4096])
    def test_output_offset_guards(self, device, dtype, length):
        a = values((length,), dtype)
        ga = a.to(device)
        scalar = torch.tensor(2, dtype=dtype, device=device)
        base = torch.full((2 * length + 7,), -99, dtype=dtype, device=device)
        expected = base.cpu()
        expected[1:1 + 2 * length:2] = a - 2
        torch.sub(ga, scalar, out=base[1:1 + 2 * length:2])
        self.assertEqual(base.cpu(), expected)
        base.fill_(-99)
        expected.fill_(-99)
        expected[1:1 + length] = a - 2
        torch.sub(ga, scalar, out=base[1:1 + length])
        self.assertEqual(base.cpu(), expected)

    @dtypes(torch.float32, torch.float64)
    def test_autograd(self, device, dtype):
        a = (values((8, 32, 64), dtype) / 16).requires_grad_()
        b = (values((8, 1, 64), dtype) / 16).requires_grad_()
        ga = a.detach().to(device).requires_grad_()
        gb = b.detach().to(device).requires_grad_()
        result = torch.sigmoid(a.transpose(1, 2) * b.transpose(1, 2) + b.transpose(1, 2))
        actual = torch.sigmoid(ga.transpose(1, 2) * gb.transpose(1, 2) + gb.transpose(1, 2))
        result.sum().backward()
        actual.sum().backward()
        self.assertEqual(actual.cpu(), result)
        self.assertEqual(ga.grad.cpu(), a.grad)
        self.assertEqual(gb.grad.cpu(), b.grad)

    @dtypes(torch.float32, torch.bfloat16)
    def test_nondefault_stream_and_graph(self, device, dtype):
        a, b = values((16, 2049, 8), dtype).to(device), values((16, 1, 8), dtype).to(device)
        expected = a.cpu() - b.cpu()
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            out = a - b
        stream.synchronize()
        self.assertEqual(out.cpu(), expected)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = a - b
        a.add_(1)
        graph.replay()
        torch.cuda.synchronize(device)
        self.assertEqual(out.cpu(), expected + 1)

    def test_device_switch(self, device):
        if torch.cuda.device_count() < 2:
            self.skipTest("requires two visible CUDA devices")
        for index in (0, 1, 0):
            with torch.cuda.device(index):
                source = torch.arange(4096, dtype=torch.int64, device=f"cuda:{index}")
                result = source.to(torch.float32)
                self.assertEqual(result.cpu(), torch.arange(4096, dtype=torch.float32))

    def test_special_float_copy(self, device):
        cpu = torch.tensor([float("nan"), float("inf"), -float("inf"), 0., -0., 1.])
        actual = cpu.to(device).expand(64, 6).to(torch.bfloat16)
        self.assertEqual(actual.cpu(), cpu.expand(64, 6).to(torch.bfloat16))
        self.assertEqual(torch.signbit(actual.cpu()), torch.signbit(cpu.expand(64, 6)))

    @parametrize("kind", ["packed", "strided", "wide", "mixed", "unary"])
    def test_new_dispatch_probe(self, device, kind):
        if os.getenv("GPU_ELEMENTWISE_EXPECT_OPT") is None:
            self.skipTest("kernel-path check only runs in the dedicated toggle verification process")
        expected = os.environ["GPU_ELEMENTWISE_EXPECT_OPT"] == "1"
        if kind == "packed":
            a = values((3, 11001, 8), torch.float32).to(device)
            b = values((3, 1, 8), torch.float32).to(device)
            kernel = "elementwise_kernel_binary_packed"
        elif kind == "wide":
            a = values((512, 1), torch.float64).to(device)
            b = values((1, 512), torch.float64).to(device)
            kernel = "elementwise_kernel_binary_packed"
        elif kind == "mixed":
            a = values((1024, 64), torch.float32).to(device)
            b = values((1, 64), torch.bfloat16).to(device)
            kernel = "elementwise_kernel_binary_packed"
        elif kind == "unary":
            base = values((8, 9, 16392), torch.bfloat16).to(device)
            a, b = base[:, :8, :16384], 2
            kernel = "elementwise_kernel_unary_packed"
        else:
            base = values((3 * 262147,), torch.float32).to(device)
            a, b = base[::3], 2
            kernel = "elementwise_kernel_1_1_strided"
        a + b
        torch.cuda.synchronize(device)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            actual = a + b
            torch.cuda.synchronize(device)
        names = [event.name for event in prof.events() if event.device_type == torch.autograd.DeviceType.CUDA]
        self.assertEqual(any(kernel in name for name in names), expected, msg=str(names))
        self.assertEqual(len(names), 1, msg=str(names))
        self.assertEqual(actual.cpu(), a.cpu() + (b.cpu() if isinstance(b, torch.Tensor) else b))

    def test_dispatch_probe(self, device):
        if os.getenv("GPU_ELEMENTWISE_EXPECT_OPT") is None:
            self.skipTest("kernel-path check only runs in the dedicated toggle verification process")
        expected = os.environ["GPU_ELEMENTWISE_EXPECT_OPT"] == "1"
        source = torch.arange(4096, dtype=torch.int64, device=device)
        source.to(torch.float32)
        torch.cuda.synchronize(device)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            result = source.to(torch.float32)
            torch.cuda.synchronize(device)
        names = [event.name for event in prof.events() if event.device_type == torch.autograd.DeviceType.CUDA]
        hit = any("unrolled_elementwise_kernel_1_1_contiguous_castin1" in name for name in names)
        self.assertEqual(hit, expected, msg=str(names))
        self.assertEqual(result.cpu(), torch.arange(4096, dtype=torch.float32))


instantiate_device_type_tests(TestGPUElementwiseOpt, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
