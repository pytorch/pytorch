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
        return record.payload.strip()


traceLOG = logging.getLogger("torch.__trace")


class FxGraphRunnableTest(TestCase):

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        torch._logging.structured.INTERN_TABLE.clear()

        self._old_level = traceLOG.level
        traceLOG.setLevel(logging.DEBUG)

        self._buf = io.StringIO()
        self._handler = logging.StreamHandler(self._buf)
        self._handler.setFormatter(_PayloadFormatter())
        self._handler.addFilter(_FxGraphRunnableFilter())
        traceLOG.addHandler(self._handler)

    def tearDown(self):
        traceLOG.removeHandler(self._handler)
        traceLOG.setLevel(self._old_level)

    #helper function
    def _exec_payload(self):
        #Write captured payload & run it in a fresh Python process
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

    # basic tests
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

    def test_scalar_multiply(self):
        def f(x):
            return x * 2

        torch.compile(f)(torch.randn(5))
        self._exec_payload()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
