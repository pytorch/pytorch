# Owner(s): ["module: dynamo"]
import io, logging, subprocess, sys, tempfile, torch, torch._logging.structured

from torch._inductor.test_case import TestCase


class _FxGraphRunnableFilter(logging.Filter):
    def filter(self, record):
        return (
            "artifact" in record.metadata
            and record.metadata["artifact"]["name"] == "fx_graph_runnable"
        )


class _PayloadFormatter(logging.Formatter):
    def format(self, record):
        # The structured-trace payload already contains the complete script
        return record.payload.strip()


_LOG = logging.getLogger("torch.__trace")


class FxGraphRunnableTest(TestCase):
    """
    Everything lives in the same file; just keep adding more ``test_*`` methods.
    Each method:
        1. torch.compile(...) a tiny program
        2. Runs it once so the graph is emitted
        3. Captures the fx_graph_runnable payload from structured-trace
        4. Drops the payload into a tmp-file and executes it as a standalone script
    """

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        torch._logging.structured.INTERN_TABLE.clear()

        self._old_level = _LOG.level
        _LOG.setLevel(logging.DEBUG)

        self._buf = io.StringIO()
        self._handler = logging.StreamHandler(self._buf)
        self._handler.setFormatter(_PayloadFormatter())
        self._handler.addFilter(_FxGraphRunnableFilter())
        _LOG.addHandler(self._handler)

    def tearDown(self):
        _LOG.removeHandler(self._handler)
        _LOG.setLevel(self._old_level)

    # ---------- helpers ----------
    def _exec_payload(self):
        """Write captured payload to disk & run it in a fresh Python proc."""
        payload = self._buf.getvalue().strip()
        self.assertTrue(payload, "Expected fx_graph_runnable payload but got nothing")
        self.assertIn("def forward", payload)  # sanity-check for actual FX code

        with tempfile.NamedTemporaryFile("w", suffix=".py") as tmp:
            tmp.write(payload)
            tmp.flush()
            res = subprocess.run(
                [sys.executable, tmp.name], capture_output=True, text=True, timeout=30
            )
            self.assertEqual(
                res.returncode,
                0,
                f"Standalone fx_graph_runnable failed:\nSTDERR:\n{res.stderr}",
            )

    # ---------- actual tests ----------
    def test_basic_tensor_add(self):
        def f(x):
            return x + 1

        torch.compile(f)(torch.randn(4))
        self._exec_payload()

    def test_two_inputs_matmul(self):
        def f(a, b):
            return (a @ b).relu()

        a, b = torch.randn(2, 3), torch.randn(3, 4)
        torch.compile(f)(a, b)
        self._exec_payload()

    def test_module_subclass(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.sin(x) * 3.14

        mod_opt = torch.compile(M())
        mod_opt(torch.randn(8))
        self._exec_payload()

    def test_inplace_op(self):
        def f(x):
            y = x.clone()
            y.add_(2)
            return y

        torch.compile(f)(torch.zeros(5))
        self._exec_payload()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
