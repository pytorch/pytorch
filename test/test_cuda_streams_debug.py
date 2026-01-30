# Owner(s): ["module: cuda"]

import gc
import sys
import threading
import warnings

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda import warn_on_null_stream_use
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    NoTest,
    parametrize,
    run_tests,
    TEST_CUDA,
    TestCase,
)


if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811


_warning_count = 0
_original_warn = warnings.warn


def _counting_warn(msg, *args, **kwargs):
    global _warning_count
    if "launched on NULL stream" in str(msg):
        _warning_count += 1
        return
    return _original_warn(msg, *args, **kwargs)


def _reset_warning_count():
    global _warning_count
    _warning_count = 0


def _get_warning_count():
    return _warning_count


def _count_warnings(op, stream):
    _reset_warning_count()
    with torch.cuda.stream(stream):
        op()
    torch.cuda.synchronize()
    return _get_warning_count()


def _warns(op, stream):
    return _count_warnings(op, stream) > 0


OPS = [
    # Tensor creation
    ("zeros", lambda: torch.zeros(100, device="cuda")),
    ("ones", lambda: torch.ones(100, device="cuda")),
    ("empty_fill", lambda: torch.empty(100, device="cuda").fill_(0)),
    ("full", lambda: torch.full((10,), 5, device="cuda")),
    ("arange", lambda: torch.arange(100, device="cuda")),
    ("linspace", lambda: torch.linspace(0, 1, 100, device="cuda")),
    ("eye", lambda: torch.eye(10, device="cuda")),
    ("randn", lambda: torch.randn(100, device="cuda")),
    ("rand", lambda: torch.rand(100, device="cuda")),
    ("randint", lambda: torch.randint(0, 10, (100,), device="cuda")),
    ("randperm", lambda: torch.randperm(100, device="cuda")),
    # Unary ops
    ("abs", lambda: torch.randn(100, device="cuda").abs()),
    ("neg", lambda: torch.randn(100, device="cuda").neg()),
    ("exp", lambda: torch.randn(100, device="cuda").exp()),
    ("log", lambda: torch.randn(100, device="cuda").log()),
    ("sqrt", lambda: torch.randn(100, device="cuda").sqrt()),
    ("sin", lambda: torch.randn(100, device="cuda").sin()),
    ("cos", lambda: torch.randn(100, device="cuda").cos()),
    ("tanh", lambda: torch.randn(100, device="cuda").tanh()),
    ("sigmoid", lambda: torch.randn(100, device="cuda").sigmoid()),
    ("relu", lambda: torch.randn(100, device="cuda").relu()),
    ("floor", lambda: torch.randn(100, device="cuda").floor()),
    ("ceil", lambda: torch.randn(100, device="cuda").ceil()),
    ("round", lambda: torch.randn(100, device="cuda").round()),
    ("sign", lambda: torch.randn(100, device="cuda").sign()),
    ("reciprocal", lambda: torch.randn(100, device="cuda").reciprocal()),
    # Binary ops
    ("add", lambda: torch.randn(100, device="cuda") + torch.randn(100, device="cuda")),
    ("sub", lambda: torch.randn(100, device="cuda") - torch.randn(100, device="cuda")),
    ("mul", lambda: torch.randn(100, device="cuda") * torch.randn(100, device="cuda")),
    ("div", lambda: torch.randn(100, device="cuda") / torch.randn(100, device="cuda")),
    ("pow", lambda: torch.randn(100, device="cuda") ** 2),
    (
        "matmul_vec",
        lambda: torch.randn(100, device="cuda") @ torch.randn(100, device="cuda"),
    ),
    (
        "maximum",
        lambda: torch.maximum(
            torch.randn(100, device="cuda"), torch.randn(100, device="cuda")
        ),
    ),
    (
        "minimum",
        lambda: torch.minimum(
            torch.randn(100, device="cuda"), torch.randn(100, device="cuda")
        ),
    ),
    # Reductions
    ("sum", lambda: torch.randn(100, device="cuda").sum()),
    ("mean", lambda: torch.randn(100, device="cuda").mean()),
    ("std", lambda: torch.randn(100, device="cuda").std()),
    ("var", lambda: torch.randn(100, device="cuda").var()),
    ("max", lambda: torch.randn(100, device="cuda").max()),
    ("min", lambda: torch.randn(100, device="cuda").min()),
    ("prod", lambda: torch.randn(100, device="cuda").prod()),
    ("norm", lambda: torch.randn(100, device="cuda").norm()),
    ("argmax", lambda: torch.randn(10, 10, device="cuda").argmax()),
    ("argmin", lambda: torch.randn(10, 10, device="cuda").argmin()),
    # Matrix ops
    (
        "matmul",
        lambda: torch.randn(10, 10, device="cuda") @ torch.randn(10, 10, device="cuda"),
    ),
    (
        "mm",
        lambda: torch.mm(
            torch.randn(10, 10, device="cuda"), torch.randn(10, 10, device="cuda")
        ),
    ),
    (
        "bmm",
        lambda: torch.bmm(
            torch.randn(5, 10, 10, device="cuda"), torch.randn(5, 10, 10, device="cuda")
        ),
    ),
    (
        "addmm",
        lambda: torch.addmm(
            torch.randn(10, 10, device="cuda"),
            torch.randn(10, 10, device="cuda"),
            torch.randn(10, 10, device="cuda"),
        ),
    ),
    ("t", lambda: torch.randn(10, 10, device="cuda").t()),
    (
        "transpose",
        lambda: torch.randn(10, 10, device="cuda").transpose(0, 1).contiguous(),
    ),
    (
        "permute",
        lambda: torch.randn(5, 10, 10, device="cuda").permute(2, 0, 1).contiguous(),
    ),
    # Linear algebra
    (
        "inv",
        lambda: torch.linalg.inv(
            torch.randn(5, 5, device="cuda") + 5 * torch.eye(5, device="cuda")
        ),
    ),
    ("det", lambda: torch.linalg.det(torch.randn(5, 5, device="cuda"))),
    ("svd", lambda: torch.linalg.svd(torch.randn(5, 5, device="cuda"))),
    ("eig", lambda: torch.linalg.eig(torch.randn(5, 5, device="cuda"))),
    ("qr", lambda: torch.linalg.qr(torch.randn(5, 5, device="cuda"))),
    (
        "cholesky",
        lambda: torch.linalg.cholesky(
            (a := torch.randn(5, 5, device="cuda")) @ a.t()
            + 5 * torch.eye(5, device="cuda")
        ),
    ),
    # Memory ops
    ("clone", lambda: torch.randn(100, device="cuda").clone()),
    ("contiguous", lambda: torch.randn(100, device="cuda").contiguous()),
    (
        "copy_",
        lambda: torch.zeros(100, device="cuda").copy_(torch.ones(100, device="cuda")),
    ),
    ("fill_", lambda: torch.randn(100, device="cuda").fill_(0)),
    ("zero_", lambda: torch.randn(100, device="cuda").zero_()),
    # Indexing
    ("slice", lambda: torch.randn(100, device="cuda")[10:50]),
    (
        "index_select",
        lambda: torch.randn(100, device="cuda").index_select(
            0, torch.tensor([1, 2, 3], device="cuda")
        ),
    ),
    (
        "gather",
        lambda: torch.randn(10, 10, device="cuda").gather(
            1, torch.zeros(10, 5, dtype=torch.long, device="cuda")
        ),
    ),
    (
        "scatter_",
        lambda: torch.zeros(10, 10, device="cuda").scatter_(
            1,
            torch.zeros(10, 5, dtype=torch.long, device="cuda"),
            torch.ones(10, 5, device="cuda"),
        ),
    ),
    (
        "masked_select",
        lambda: torch.randn(100, device="cuda").masked_select(
            torch.randn(100, device="cuda") > 0
        ),
    ),
    # Sorting
    ("sort", lambda: torch.randn(100, device="cuda").sort()),
    ("argsort", lambda: torch.randn(100, device="cuda").argsort()),
    ("topk", lambda: torch.randn(100, device="cuda").topk(10)),
    ("unique", lambda: torch.randperm(100, device="cuda").unique()),
    # Comparison
    ("gt", lambda: torch.randn(100, device="cuda") > 0),
    (
        "eq_tensor",
        lambda: torch.randn(100, device="cuda") == torch.randn(100, device="cuda"),
    ),
    ("eq", lambda: torch.randn(100, device="cuda").eq(0)),
    ("ne", lambda: torch.randn(100, device="cuda").ne(0)),
    (
        "where",
        lambda: torch.where(
            torch.randn(100, device="cuda") > 0,
            torch.ones(100, device="cuda"),
            torch.zeros(100, device="cuda"),
        ),
    ),
    # Shape ops
    ("view", lambda: torch.randn(100, device="cuda").view(10, 10)),
    ("reshape", lambda: torch.randn(100, device="cuda").reshape(10, 10)),
    ("flatten", lambda: torch.randn(10, 10, device="cuda").flatten()),
    ("unsqueeze", lambda: torch.randn(100, device="cuda").unsqueeze(0)),
    ("squeeze", lambda: torch.randn(1, 100, device="cuda").squeeze()),
    (
        "cat",
        lambda: torch.cat(
            [torch.randn(10, device="cuda"), torch.randn(10, device="cuda")]
        ),
    ),
    (
        "stack",
        lambda: torch.stack(
            [torch.randn(10, device="cuda"), torch.randn(10, device="cuda")]
        ),
    ),
    ("split", lambda: torch.randn(100, device="cuda").split(10)),
    ("chunk", lambda: torch.randn(100, device="cuda").chunk(10)),
    # NN ops
    ("F.relu", lambda: F.relu(torch.randn(100, device="cuda"))),
    ("F.sigmoid", lambda: F.sigmoid(torch.randn(100, device="cuda"))),
    ("F.tanh", lambda: F.tanh(torch.randn(100, device="cuda"))),
    ("F.softmax", lambda: F.softmax(torch.randn(10, 10, device="cuda"), dim=1)),
    ("F.log_softmax", lambda: F.log_softmax(torch.randn(10, 10, device="cuda"), dim=1)),
    ("F.gelu", lambda: F.gelu(torch.randn(100, device="cuda"))),
    ("F.silu", lambda: F.silu(torch.randn(100, device="cuda"))),
    ("F.leaky_relu", lambda: F.leaky_relu(torch.randn(100, device="cuda"))),
    (
        "F.dropout",
        lambda: F.dropout(torch.randn(100, device="cuda"), p=0.5, training=True),
    ),
    (
        "F.linear",
        lambda: F.linear(
            torch.randn(10, 10, device="cuda"), torch.randn(5, 10, device="cuda")
        ),
    ),
    (
        "F.conv1d",
        lambda: F.conv1d(
            torch.randn(1, 3, 100, device="cuda"), torch.randn(5, 3, 3, device="cuda")
        ),
    ),
    (
        "F.conv2d",
        lambda: F.conv2d(
            torch.randn(1, 3, 32, 32, device="cuda"),
            torch.randn(5, 3, 3, 3, device="cuda"),
        ),
    ),
    ("F.max_pool2d", lambda: F.max_pool2d(torch.randn(1, 3, 32, 32, device="cuda"), 2)),
    ("F.avg_pool2d", lambda: F.avg_pool2d(torch.randn(1, 3, 32, 32, device="cuda"), 2)),
    (
        "F.batch_norm",
        lambda: F.batch_norm(
            torch.randn(10, 10, device="cuda"),
            torch.zeros(10, device="cuda"),
            torch.ones(10, device="cuda"),
            training=True,
        ),
    ),
    ("F.layer_norm", lambda: F.layer_norm(torch.randn(10, 10, device="cuda"), [10])),
    (
        "F.embedding",
        lambda: F.embedding(
            torch.randint(0, 100, (10,), device="cuda"),
            torch.randn(100, 10, device="cuda"),
        ),
    ),
    (
        "F.cross_entropy",
        lambda: F.cross_entropy(
            torch.randn(10, 5, device="cuda"), torch.randint(0, 5, (10,), device="cuda")
        ),
    ),
    (
        "F.mse_loss",
        lambda: F.mse_loss(
            torch.randn(100, device="cuda"), torch.randn(100, device="cuda")
        ),
    ),
    (
        "F.interpolate",
        lambda: F.interpolate(torch.randn(1, 3, 32, 32, device="cuda"), scale_factor=2),
    ),
    ("F.pad", lambda: F.pad(torch.randn(1, 3, 32, 32, device="cuda"), (1, 1, 1, 1))),
    # FFT
    ("fft", lambda: torch.fft.fft(torch.randn(100, device="cuda"))),
    (
        "ifft",
        lambda: torch.fft.ifft(torch.randn(100, device="cuda", dtype=torch.complex64)),
    ),
    ("rfft", lambda: torch.fft.rfft(torch.randn(100, device="cuda"))),
    ("fft2", lambda: torch.fft.fft2(torch.randn(10, 10, device="cuda"))),
    # Events and streams
    ("event_record", lambda: torch.cuda.Event().record()),
    ("stream_sync", lambda: torch.cuda.current_stream().synchronize()),
]

LEGACY_STREAM_OPS = {
    "fft",
    "fft2",
    "ifft",
    "rfft",
}


class TestCudaStreamsDebug(TestCase):
    def setUp(self):
        super().setUp()
        warnings.warn = _counting_warn

    def tearDown(self):
        torch.cuda.synchronize()
        warnings.warn = _original_warn
        gc.collect()
        super().tearDown()

    @parametrize("name,op", OPS)
    def test_null_stream_warns(self, name, op):
        with warn_on_null_stream_use():
            self.assertTrue(
                _warns(op, torch.cuda.default_stream()),
                f"{name} should warn on NULL stream",
            )

    @parametrize("name,op", OPS)
    def test_non_null_stream_no_warn(self, name, op):
        with warn_on_null_stream_use():
            if name in LEGACY_STREAM_OPS:
                self.assertTrue(
                    _warns(op, torch.cuda.Stream()),
                    f"{name} should warn on legacy stream",
                )
            else:
                self.assertFalse(
                    _warns(op, torch.cuda.Stream()),
                    f"{name} should not warn on non-NULL stream",
                )


class TestCudaStreamsDebugMultithreaded(TestCase):
    def setUp(self):
        super().setUp()
        warnings.warn = _counting_warn

    def tearDown(self):
        torch.cuda.synchronize()
        warnings.warn = _original_warn
        gc.collect()
        super().tearDown()

    @parametrize("use_null_stream", [True, False])
    def test_backward(self, use_null_stream):
        warned = [False]
        fwd_tid = [None]
        bwd_tid = [None]

        class NullStreamBackward(Function):
            @staticmethod
            def forward(ctx, x):
                fwd_tid[0] = threading.get_ident()
                return x.clone()

            @staticmethod
            def backward(ctx, gO):
                bwd_tid[0] = threading.get_ident()
                _reset_warning_count()
                stream = (
                    torch.cuda.default_stream()
                    if use_null_stream
                    else torch.cuda.Stream()
                )
                with torch.cuda.stream(stream):
                    result = gO.clone()
                torch.cuda.synchronize()
                warned[0] = _get_warning_count() > 0
                return result

        with warn_on_null_stream_use():
            inp = torch.rand(100, device="cuda", requires_grad=True)
            NullStreamBackward.apply(inp).sum().backward()

        torch.cuda.synchronize()
        self.assertNotEqual(fwd_tid[0], bwd_tid[0])
        self.assertEqual(warned[0], use_null_stream)

    def test_local_context(self):
        counts = []

        def worker():
            with warn_on_null_stream_use():
                counts.append(
                    _count_warnings(
                        lambda: torch.randn(100, device="cuda"),
                        torch.cuda.default_stream(),
                    )
                )

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        torch.cuda.synchronize()
        self.assertGreaterEqual(sum(counts), 3)

    def test_global_context(self):
        counts = []

        def worker():
            counts.append(
                _count_warnings(
                    lambda: torch.randn(100, device="cuda"),
                    torch.cuda.default_stream(),
                )
            )

        with warn_on_null_stream_use():
            threads = [threading.Thread(target=worker) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            torch.cuda.synchronize()

        self.assertGreaterEqual(sum(counts), 3)

    def test_outside_context(self):
        results = []
        start_bar = threading.Barrier(2)
        end_bar = threading.Barrier(2)

        def worker():
            try:
                start_bar.wait()
                results.append(
                    _warns(
                        lambda: torch.randn(100, device="cuda"),
                        torch.cuda.default_stream(),
                    )
                )
                end_bar.wait()
            except threading.BrokenBarrierError:
                pass

        t = threading.Thread(target=worker)
        t.start()
        try:
            with warn_on_null_stream_use():
                start_bar.wait()
                torch.randn(10, device="cuda")
                end_bar.wait()
                torch.cuda.synchronize()
        finally:
            start_bar.abort()
            end_bar.abort()
            t.join()

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0])

    def test_reentrance_same_thread(self):
        def op():
            return torch.randn(100, device="cuda")

        null_stream = torch.cuda.default_stream()

        with warn_on_null_stream_use():
            self.assertTrue(_warns(op, null_stream))
            with warn_on_null_stream_use():
                self.assertTrue(_warns(op, null_stream))
            self.assertTrue(_warns(op, null_stream))
            torch.cuda.synchronize()


instantiate_parametrized_tests(TestCudaStreamsDebug)
instantiate_parametrized_tests(TestCudaStreamsDebugMultithreaded)


if __name__ == "__main__":
    run_tests()
