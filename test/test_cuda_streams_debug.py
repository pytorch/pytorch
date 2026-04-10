# Owner(s): ["module: cuda"]

import re
import sys
import threading
import unittest
import warnings

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.profiler import profile, ProfilerActivity
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    NoTest,
    parametrize,
    run_tests,
    TEST_CUDA,
    TestCase,
)
from torch.utils._cuda_debug import warn_on_null_stream_use


if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811


NULL_STREAM_RE = re.compile(r"(Runtime|Driver)::\w+ launched on NULL stream")


def _null_stream_warnings(caught):
    return [w for w in caught if NULL_STREAM_RE.search(str(w.message))]


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


LEGACY_STREAM_OPS = {"fft", "fft2", "ifft", "rfft"}


class TestCudaStreamsDebug(TestCase):
    @parametrize("name,op", OPS)
    def test_null_stream_warns(self, name, op):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with warn_on_null_stream_use():
                with torch.cuda.stream(torch.cuda.default_stream()):
                    op()
        self.assertGreater(
            len(_null_stream_warnings(caught)),
            0,
            f"{name}: expected NULL stream warning",
        )

    @parametrize("name,op", OPS)
    def test_non_null_stream_no_warn(self, name, op):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with warn_on_null_stream_use():
                with torch.cuda.stream(torch.cuda.Stream()):
                    op()
        null_warns = _null_stream_warnings(caught)
        if name in LEGACY_STREAM_OPS:
            self.assertGreater(len(null_warns), 0, f"{name}: expected legacy warning")
        else:
            self.assertEqual(len(null_warns), 0, f"{name}: unexpected warning")

    @unittest.expectedFailure
    def test_profiler_subscriber_conflict(self):
        from cupti import cupti

        cupti.unsubscribe(cupti.subscribe(lambda: None, None))

        with profile(activities=[ProfilerActivity.CUDA]):
            torch.randn(1, device="cuda")
            with self.assertRaisesRegex(
                RuntimeError,
                r"CUPTI subscriber already exists .*existing subscriber:",
            ):
                with warn_on_null_stream_use():
                    torch.randn(1, device="cuda")


class TestCudaStreamsDebugMultithreaded(TestCase):
    @parametrize("use_null_stream", [True, False])
    def test_backward(self, use_null_stream):
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
                stream = (
                    torch.cuda.default_stream()
                    if use_null_stream
                    else torch.cuda.Stream()
                )
                with torch.cuda.stream(stream):
                    result = gO.clone()
                return result

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with warn_on_null_stream_use():
                inp = torch.rand(100, device="cuda", requires_grad=True)
                NullStreamBackward.apply(inp).sum().backward()

        self.assertNotEqual(fwd_tid[0], bwd_tid[0])
        null_warns = _null_stream_warnings(caught)
        if use_null_stream:
            self.assertGreater(len(null_warns), 0)
        else:
            self.assertEqual(len(null_warns), 0)

    def test_local_context(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(3):

                def worker():
                    with warn_on_null_stream_use():
                        with torch.cuda.stream(torch.cuda.default_stream()):
                            torch.randn(100, device="cuda")

                t = threading.Thread(target=worker)
                t.start()
                t.join()

        self.assertGreaterEqual(len(_null_stream_warnings(caught)), 3)

    def test_global_context(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with warn_on_null_stream_use():
                for _ in range(3):

                    def worker():
                        with torch.cuda.stream(torch.cuda.default_stream()):
                            torch.randn(100, device="cuda")

                    t = threading.Thread(target=worker)
                    t.start()
                    t.join()

        self.assertGreaterEqual(len(_null_stream_warnings(caught)), 3)

    def test_outside_context(self):
        start_bar = threading.Barrier(2)
        end_bar = threading.Barrier(2)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            def worker():
                try:
                    start_bar.wait()
                    with torch.cuda.stream(torch.cuda.default_stream()):
                        torch.randn(100, device="cuda")
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
            finally:
                start_bar.abort()
                end_bar.abort()
                t.join()

        self.assertGreater(len(_null_stream_warnings(caught)), 0)

    def test_reentrance_same_thread(self):
        null_stream = torch.cuda.default_stream()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with warn_on_null_stream_use():
                with torch.cuda.stream(null_stream):
                    torch.randn(100, device="cuda")
                with warn_on_null_stream_use():
                    with torch.cuda.stream(null_stream):
                        torch.randn(100, device="cuda")
                with torch.cuda.stream(null_stream):
                    torch.randn(100, device="cuda")

        self.assertGreaterEqual(len(_null_stream_warnings(caught)), 3)


instantiate_parametrized_tests(TestCudaStreamsDebug)
instantiate_parametrized_tests(TestCudaStreamsDebugMultithreaded)


if __name__ == "__main__":
    run_tests()
