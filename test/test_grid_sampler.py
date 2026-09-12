# Owner(s): ["module: nn"]

"""Public CUDA grid-sampler backward correctness and determinism coverage."""

import unittest
from functools import partial

import torch
import torch.nn.functional as F
from torch.testing._internal.common_cuda import TEST_CUDNN
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    skipCUDAIfRocm,
)
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    parametrize,
    run_tests,
    TestCase,
)


DIMENSION_MODES = [
    (2, "bilinear"),
    (2, "nearest"),
    (2, "bicubic"),
    (3, "bilinear"),
    (3, "nearest"),
]


def gradients(input, grid, grad_output, options, required=(True, True)):
    """Return fresh requested gradients through the public functional operator."""
    input = input.detach().requires_grad_(required[0])
    grid = grid.detach().requires_grad_(required[1])
    output = F.grid_sample(input, grid, **options)
    targets = tuple(t for t in (input, grid) if t.requires_grad)
    return torch.autograd.grad(output, targets, grad_output)


class TestGridSampler(TestCase):
    """Exercise both public autograd and direct backward output-mask contracts."""

    def assert_gradient_accuracy(
        self, value: torch.Tensor, expected: torch.Tensor, baseline: torch.Tensor
    ) -> None:
        """Use CPU FP64 as the oracle, with CPU eager's low-precision rounding baseline."""
        value = value.cpu()
        if value.dtype in (torch.float16, torch.bfloat16):
            error = (value.double() - expected).abs()
            baseline_error = (baseline.double() - expected).abs()
            eps = torch.finfo(value.dtype).eps
            self.assertTrue(torch.isfinite(value).all())
            self.assertLessEqual(
                error.max().item(),
                baseline_error.max().item() + 2 * eps * expected.abs().max().item(),
            )
            self.assertLessEqual(
                error.mean().item(),
                baseline_error.mean().item() + eps * expected.abs().mean().item(),
            )
        else:
            tolerance = 1e-10 if value.dtype == torch.float64 else 2e-5
            self.assertEqual(value.double(), expected, atol=tolerance, rtol=tolerance)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("dimension_mode", DIMENSION_MODES)
    @parametrize("padding", ["zeros", "border", "reflection"])
    @parametrize("align", [False, True])
    def test_backward(self, device, dtype, dimension_mode, padding, align):
        """Check numerics, gradient masks, and identical bits with determinism on/off."""
        dims, mode = dimension_mode
        rng = torch.Generator().manual_seed(20260911)
        spatial = (4, 8) if dims == 2 else (2, 4, 8)
        output_shape = (3, 7) if dims == 2 else (2, 3, 5)
        input = (
            (torch.rand((2, 3) + spatial, generator=rng) * 2 - 1)
            .to(dtype)
            .transpose(-1, -2)
        )
        # Dyadic coordinates isolate accumulation error from low-precision
        # coordinate rounding; arbitrary-coordinate coverage remains in test_nn.
        grid = (
            torch.randint(-6, 7, (2,) + output_shape + (dims,), generator=rng) / 4
        ).to(dtype)
        grid = grid.transpose(1, dims)
        grad = (
            (torch.rand((2,) + tuple(grid.shape[1:-1]) + (3,), generator=rng) * 2 - 1)
            .to(dtype)
            .movedim(-1, 1)
        )
        options = dict(mode=mode, padding_mode=padding, align_corners=align)
        data = tuple(t.to(device) for t in (input, grid, grad))
        with torch.backends.cudnn.flags(enabled=False):
            for required in ((True, True), (True, False), (False, True)):
                expected = gradients(
                    input.double(), grid.double(), grad.double(), options, required
                )
                cpu = gradients(input, grid, grad, options, required)
                with DeterministicGuard(False):
                    first = gradients(*data, options, required)
                with DeterministicGuard(True):
                    actual = gradients(*data, options, required)
                for value, prior, ref, baseline in zip(actual, first, expected, cpu):
                    self.assertEqual(value.view(torch.uint8), prior.view(torch.uint8))
                    self.assert_gradient_accuracy(value, ref, baseline)
            backward = (
                torch.ops.aten.grid_sampler_2d_backward
                if dims == 2
                else torch.ops.aten.grid_sampler_3d_backward
            )
            with DeterministicGuard(True):
                no_input, grid_grad = backward(
                    data[2],
                    data[0],
                    data[1],
                    F.GRID_SAMPLE_INTERPOLATION_MODES[mode],
                    F.GRID_SAMPLE_PADDING_MODES[padding],
                    align,
                    (False, False),
                )
            self.assertIsNone(no_input)
            # Native backward historically returns a computed grid gradient even
            # when the second output-mask entry is false.
            self.assertEqual(grid_grad, first[0], atol=0, rtol=0)

    @unittest.skipIf(not TEST_CUDNN, "cuDNN not available")
    @skipCUDAIfRocm
    def test_cudnn_backward(self, device):
        """A cuDNN-selected forward must use deterministic backward as well."""
        input = torch.randn(2, 3, 5, 7, device=device, requires_grad=True)
        grid = torch.rand(2, 3, 4, 2, device=device, requires_grad=True)
        grad = torch.randn(2, 3, 3, 4, device=device)
        with torch.backends.cudnn.flags(enabled=True), DeterministicGuard(True):
            output = F.grid_sample(input, grid, align_corners=True)
            self.assertEqual(type(output.grad_fn).__name__, "CudnnGridSamplerBackward0")
            first = torch.autograd.grad(output, (input, grid), grad)
            second = gradients(input, grid, grad, dict(align_corners=True))
        expected = gradients(
            input.cpu().double(),
            grid.cpu().double(),
            grad.cpu().double(),
            dict(align_corners=True),
        )
        for a, b, ref in zip(first, second, expected):
            self.assertEqual(a.view(torch.int32), b.view(torch.int32))
            self.assertEqual(a.cpu().double(), ref, atol=2e-5, rtol=2e-5)

    @parametrize("dimension_mode", [(2, "bilinear"), (2, "bicubic"), (3, "bilinear")])
    def test_gradcheck(self, device, dimension_mode):
        """Preserve first- and second-order autograd formulas through the new kernel."""
        dims, mode = dimension_mode
        input = torch.randn(
            (1, 2) + (3,) * dims, device=device, dtype=torch.double, requires_grad=True
        )
        grid = (
            torch.rand((1,) + (2,) * dims + (dims,), device=device, dtype=torch.double)
            * 0.6
            - 0.3
        ).requires_grad_()
        with torch.backends.cudnn.flags(enabled=False), DeterministicGuard(True):
            fn = partial(F.grid_sample, mode=mode, align_corners=False)
            self.assertTrue(torch.autograd.gradcheck(fn, (input, grid)))
            self.assertTrue(torch.autograd.gradgradcheck(fn, (input, grid)))

    def test_expanded_streams_and_graph(self, device):
        """Cover offsets/zero strides, singleton dimensions, channel tails, and stream isolation."""
        rng = torch.Generator().manual_seed(91)
        t = 513
        bases = [
            torch.rand(1, 1, 1, 18, generator=rng),
            torch.rand(1, 1, 2 * t, 4, generator=rng) * 0.02 - 0.01,
            torch.rand(2, 1, 1, 2 * t, generator=rng),
        ]
        bases = [t.to(device) for t in bases]
        data = (
            bases[0][..., 1::2].expand(2, 65, 1, 9),
            bases[1][:, :, 1::2, ::2].expand(2, 1, t, 2),
            bases[2][..., 1::2].expand(2, 65, 1, t),
        )
        options = dict(padding_mode="reflection", align_corners=False)
        expected = gradients(*(t.cpu().double() for t in data), options)
        current = torch.cuda.current_stream()
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        results = []
        with DeterministicGuard(True), torch.backends.cudnn.flags(enabled=False):
            for stream in streams:
                stream.wait_stream(current)
                with torch.cuda.stream(stream):
                    results.append(gradients(*data, options))
            for stream in streams:
                current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = torch.ops.aten.grid_sampler_2d_backward(
                    data[2], data[0], data[1], 0, 2, False, (True, True)
                )
            for _ in range(3):
                graph.replay()
                for value, other, graph_value, ref in zip(*results, captured, expected):
                    self.assertEqual(value.view(torch.int32), other.view(torch.int32))
                    self.assertEqual(
                        value.view(torch.int32), graph_value.view(torch.int32)
                    )
                    self.assertEqual(value.cpu().double(), ref, atol=2e-4, rtol=2e-5)
            # Replaying fixed pointers must rebuild geometry from current values.
            bases[1].add_(0.25)
            graph.replay()
            eager = gradients(*data, options)
            expected = gradients(*(t.cpu().double() for t in data), options)
            for value, other, ref in zip(captured, eager, expected):
                self.assertEqual(value.view(torch.int32), other.view(torch.int32))
                self.assertEqual(value.cpu().double(), ref, atol=2e-4, rtol=2e-5)

    @dtypes(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @parametrize("dimension_mode", DIMENSION_MODES)
    def test_long_segments(self, device, dtype, dimension_mode):
        """Exercise fixed partial reductions, including low-precision product rounding."""
        dims, mode = dimension_mode
        rng = torch.Generator().manual_seed(62)
        output_shape = (1,) * (dims - 1) + (513,)
        cpu = (
            (torch.rand((2, 3) + (4,) * dims, generator=rng) * 2 - 1).to(dtype),
            torch.zeros((2,) + output_shape + (dims,), dtype=dtype),
            (torch.rand((2, 3) + output_shape, generator=rng) * 2 - 1).to(dtype),
        )
        options = dict(mode=mode, align_corners=False)
        expected = gradients(*(t.double() for t in cpu), options)
        baseline = gradients(*cpu, options)
        data = tuple(t.to(device) for t in cpu)
        with DeterministicGuard(True), torch.backends.cudnn.flags(enabled=False):
            first = gradients(*data, options)
            second = gradients(*data, options)
        for value, other, ref, low in zip(first, second, expected, baseline):
            value, other = value.cpu(), other.cpu()
            self.assertEqual(value.view(torch.uint8), other.view(torch.uint8))
            self.assert_gradient_accuracy(value, ref, low)

    def test_chunk_boundary(self, device):
        """Fixed workspace chunks must accumulate across a batch boundary without overwrite."""
        rng = torch.Generator().manual_seed(18)
        cpu = (
            torch.rand(2, 1, 2, 2, generator=rng, dtype=torch.double),
            torch.rand(2, 1, 131075, 2, generator=rng, dtype=torch.double) * 0.5 - 0.25,
            torch.rand(2, 1, 1, 131075, generator=rng, dtype=torch.double) * 2 - 1,
        )
        expected = gradients(*cpu, dict(align_corners=False))
        data = tuple(t.to(device) for t in cpu)
        with DeterministicGuard(True), torch.backends.cudnn.flags(enabled=False):
            first = gradients(*data, dict(align_corners=False))
            second = gradients(*data, dict(align_corners=False))
        for value, other, ref in zip(first, second, expected):
            self.assertEqual(value.view(torch.int64), other.view(torch.int64))
            self.assertEqual(value.cpu(), ref, atol=1e-9, rtol=1e-10)

    @parametrize("dims", [2, 3])
    @parametrize("empty", ["batch", "channel", "output"])
    def test_empty(self, device, dims, empty):
        """Empty cases retain defined zero gradients without launching invalid grids."""
        n = 0 if empty == "batch" else 2
        c = 0 if empty == "channel" else 3
        output_shape = (0 if empty == "output" else 2,) + (2,) * (dims - 1)
        input = torch.empty((n, c) + (3,) * dims, device=device, requires_grad=True)
        grid = torch.zeros(
            (n,) + output_shape + (dims,), device=device, requires_grad=True
        )
        with DeterministicGuard(True), torch.backends.cudnn.flags(enabled=False):
            output = F.grid_sample(input, grid, align_corners=False)
            gi, gg = torch.autograd.grad(
                output, (input, grid), torch.empty_like(output)
            )
        self.assertEqual(gi, torch.zeros_like(input))
        self.assertEqual(gg, torch.zeros_like(grid))


instantiate_device_type_tests(TestGridSampler, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
