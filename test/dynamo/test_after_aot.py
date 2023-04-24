# Owner(s): ["module: dynamo"]

import io

import torch._dynamo.test_case

from torch._dynamo.repro.after_aot import save_graph_repro

from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._traceback import report_compile_source_on_error


def strip_trailing_whitespace(r):
    return "\n".join([l.rstrip() for l in r.split("\n")])


class TestAfterAot(torch._dynamo.test_case.TestCase):
    def tearDown(self):
        assert not torch.cuda.is_initialized()

    def test_save_graph_repro(self):
        return
        buf = io.StringIO()
        args = [torch.randn(4)]

        def f(x):
            return (x * x,)

        gm = make_fx(f)(*args)
        save_graph_repro(buf, gm, args, "inductor_accuracy", stable_output=True)
        r = strip_trailing_whitespace(buf.getvalue())
        self.assertExpectedInline(
            r,
            """\
import torch._inductor.overrides

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx

# config omitted due to stable_output=True

# REPLACEABLE COMMENT FOR TESTING PURPOSES



from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()



    def forward(self, x_1):
        mul = torch.ops.aten.mul.Tensor(x_1, x_1);  x_1 = None
        return (mul,)

args = []
args.append(rand_strided((4,), (1,), torch.float32, 'cpu'))  # shape (4,), stride (1,)
mod = make_fx(Repro(), tracing_mode='real')(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
class AccuracyError(Exception):
    pass
if not same_two_models(mod, compiled, args, only_fwd=True):
    raise AccuracyError("Bad accuracy detected")
""",
        )
        with report_compile_source_on_error():
            exec(r, {"__compile_source__": r})


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
