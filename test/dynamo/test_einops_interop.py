# Owner(s): ["module: dynamo"]

import os
import subprocess
import sys
import unittest

import torch
import torch._dynamo.test_case
from torch._dynamo.exc import Unsupported
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


try:
    import einops

    has_einops = True
except ImportError:
    has_einops = False


# einops.rearrange
# einops.reduce
# einops.repeat (v0.2.0)
# einops.einsum (v0.5.0)
# einops.pack (v0.6.0)
# einops.unpack (v0.6.0)


class TestEinops(torch._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        if not has_einops:
            raise unittest.SkipTest("Test requires einops")
        return super().setUpClass()

    def _run_in_subprocess(self, flag, method, einops_method, snippet):
        # run in a different process
        script = f"""
import torch
from torch._dynamo.testing import EagerAndRecordGraphs, normalize_gm

torch._dynamo.config.enable_einops_tracing={flag}

import einops

backend = EagerAndRecordGraphs()

def f(x):
    {snippet}
    return y.sin()

x = torch.randn(3, 4, 5)
expected = f(x)
got = torch.compile(f, backend=backend, fullgraph=True)(x)

assert torch.allclose(expected, got)
assert len(backend.graphs) == 1, len(backend.graphs)
graph = backend.graphs[0]
print(normalize_gm(graph.print_readable(print_output=False)))
"""
        script = script.strip()
        try:
            output = (
                subprocess.check_output(
                    [sys.executable, "-c", script],
                    stderr=subprocess.STDOUT,
                    cwd=os.path.dirname(os.path.realpath(__file__)),
                    # env=os.environ.copy(),
                )
                .decode("utf-8")
                .strip()
            )
        except subprocess.CalledProcessError as e:
            self.fail(
                msg=(f"Subprocess exception {method}:\n" + e.output.decode("utf-8"))
            )
        else:
            if flag:
                self.assertNotIn(einops_method, output)
            else:
                self.assertIn(einops_method, output)

    @parametrize(
        "method",
        ["reduce", "repeat", "pack", "unpack", "einsum", "rearrange"],
        name_fn=lambda f: f,
    )
    @parametrize("flag", [True, False], name_fn=lambda f: f)
    def test_einops_method(self, flag, method):
        if not hasattr(einops, method):
            self.skipTest(f"Needs einops.{method}")

        if method == "reduce":
            einops_method = f"einops_einops_{method}"
            snippet = "y = einops.reduce(x, 'a b c -> a b', 'min')"
        elif method == "repeat":
            einops_method = f"einops_einops_{method}"
            snippet = "y = einops.repeat(x, 'a b c -> a b c d', d=2)"
        elif method == "rearrange":
            einops_method = f"einops_einops_{method}"
            snippet = "y = einops.rearrange(x, 'a b c -> a c b')"
        elif method == "einsum":
            einops_method = f"einops_einops_{method}"
            snippet = "y = einops.einsum(x, 'a b c -> a c b')"
        elif method == "pack":
            einops_method = f"einops_packing_{method}"
            snippet = "y, meta = einops.pack([x], '* b')"
        elif method == "unpack":
            einops_method = f"einops_packing_{method}"
            snippet = "x_packed, meta = einops.pack([x], '* b'); y = einops.unpack(x_packed, meta, '* b')[0]"
        else:
            self.fail(method)
        self._run_in_subprocess(flag, method, einops_method, snippet)

    @torch._dynamo.config.patch(enable_einops_tracing=True)
    def test_graph_break_message(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            y = einops.reduce(x, "a b -> a", "min")
            return y.sin()

        x = torch.randn(2, 3, 4)  # force a graph break due to a shape mismatch
        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: fn(x),
            """\
Failed to trace einops function 'reduce'.
  Explanation: Failed to trace builtin operator
      Explanation: Dynamo does not know how to trace builtin operator `add` with argument types ['str', '<unknown type>'] (has_kwargs False)
      Hint: Avoid calling builtin `add` with argument types ['str', '<unknown type>']. Consider using an equivalent alternative function/method to `add`.
      Hint: If you are attempting to call a logging function (e.g. `print`), you can try adding it to `torch._dynamo.config.reorderable_logging_functions`.
      Hint: Please report an issue to PyTorch.

      Developer debug context: builtin add [<class 'torch._dynamo.variables.constant.ConstantVariable'>, <class 'torch._dynamo.variables.misc.StringFormatVariable'>] False

  Hint: Tracing through einops functions is experimental and may not be fully supported.
    To disable einops tracing, set `torch._dynamo.config.enable_einops_tracing = False`.
    Alternatively, explicitly allow this function in the graph with `torch._dynamo.allow_in_graph(reduce)`
  Hint: This is likely to be a Dynamo bug. Please report an issue to PyTorch.

  Developer debug context: einops function 'reduce'


from user code:
   File "test_einops.py", line N, in fn
    y = einops.reduce(x, "a b -> a", "min")""",  # noqa: B950
        )

    @torch._dynamo.config.patch(enable_einops_tracing=True)
    def test_trace_einops(self):
        if einops.__version__ < "0.7":
            self.skipTest(f"Needs einops 0.7 or newer, got {einops.__version__}")

        from einops import einsum, pack, rearrange, reduce, repeat, unpack

        # Test copied from arogozhnikov/einops at
        # https://github.com/arogozhnikov/einops/blob/5dac4043970e0a74c81fcc5a73d7386ca696113e/einops/tests/test_other.py#L254-L301
        from torch import nn

        class TorchModuleWithOperations(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x_abc, suffix=""):
                a, b, c = x_abc.shape

                def suf(pattern):
                    parts = pattern.split()
                    return " ".join(
                        [p if p[-1] not in "acd" else p + suffix for p in parts]
                    )

                # patterns look a bit strange because names a, c, d will be modified on every run
                # by suf function
                x_abcd = repeat(x_abc, suf("a b c -> a b c 4"))
                x_abc = reduce(x_abcd, suf("a b c d -> a b c"), "min")
                x_abdc, ps = pack([x_abc] * (2 + len(suffix)), suf("a b * c"))
                x_array = unpack(
                    rearrange(x_abdc, suf("a b d c -> (a b ) 1 c d")), ps, "ab one1 c *"
                )
                x1 = x_array[0] + len(x_array)
                x1 = rearrange(x1, suf("(a b ) 1 c -> a b c"), b=b)
                addition = einsum(x_abc, x_abcd, suf("a b c , a b c d -> d"))[0]
                return x1 + addition

        original = TorchModuleWithOperations()
        compiled = torch.compile(original, fullgraph=True, backend="eager")
        for size in [10, 20, 40]:
            x = torch.rand([size, size + 1, size + 2])
            for suffix in ["", "suf1", "other_suffix"]:
                result1 = compiled(x, suffix)
                result2 = original(x, suffix)
                assert torch.allclose(result1, result2)


instantiate_parametrized_tests(TestEinops)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
