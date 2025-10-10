# Owner(s): ["module: dynamo"]

import torch
import torch._inductor.test_case
import torch.fx.traceback as fx_traceback
import torch.utils.checkpoint
from torch._dynamo.backends.common import aot_autograd
from torch._inductor.test_case import run_tests

# from torch._inductor.utils import run_and_get_code
from torch._inductor.utils import run_fw_bw_and_get_code
from torch.fx.passes.regional_inductor import compile_fx_annotated_nodes_with_inductor


# Some issues raised in the HOP meeting
# 1) CSE will not differentiate different meta custom nodes and do wrong thing.
# 2) SAC - The recomputed forward will be smaller than the forward. Will we
# compile a smaller region than?
# 3) What happens if you have a op in the middle whcih does not disturb
# topology, is it still 1 subgraph?
# 4) What happens with the nesting of fx_traceback.annotate? Are there ordering
# requirements?
# 5) What are we going to use the annotations for?
#   a) compile flex
#   b) streams
#   c) nn.MOdule info to organize MoE runtime
#   d) PP stages
#   e) rename graph nodes for more debugging.


def checkpoint_wrapper(fn):
    def inner(*args):
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=True)

    return inner


def aot_eager_regional_inductor():
    return aot_autograd(
        fw_compiler=compile_fx_annotated_nodes_with_inductor,
        bw_compiler=compile_fx_annotated_nodes_with_inductor,
    )


class RegionalInductorTests(torch._inductor.test_case.TestCase):
    def test_simple(self):
        def fn(x, y):
            sin = torch.sin(x)

            with fx_traceback.annotate({"compile_with_inductor": 0}):
                mul = sin * y
                add = mul + 1

            return torch.sin(add)

        opt_fn = torch.compile(
            fn, backend=aot_eager_regional_inductor(), fullgraph=True
        )
        x = torch.randn(10, requires_grad=True)
        y = torch.randn(10, requires_grad=True)

        # Check that inductor compilation is called twicw
        _, codes = run_fw_bw_and_get_code(lambda: opt_fn(x, y))
        self.assertEqual(len(codes), 2)

    def test_repeated_blocks(self):
        def fn(x, y):
            sin = torch.sin(x)

            with fx_traceback.annotate({"compile_with_inductor": 0}):
                mul = sin * y
                add = mul + 1

            return torch.sin(add)

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                a = fn(x, y)
                return fn(a, y)

        mod = Mod()

        opt_mod = torch.compile(
            mod, backend=aot_eager_regional_inductor(), fullgraph=True
        )
        x = torch.randn(10, requires_grad=True)
        y = torch.randn(10, requires_grad=True)

        # Check that inductor compilation is called 4 times
        # there will be 2 partitions in the fwd and 2 in the bwd, totalling 4
        _, codes = run_fw_bw_and_get_code(lambda: opt_mod(x, y))
        self.assertEqual(len(codes), 4)

    def test_invoke_subgraph(self):
        @torch.compiler.nested_compile_region
        def gn(x):
            return torch.sin(x)

        def fn(x):
            x = x + 1
            with fx_traceback.annotate({"compile_with_inductor": 0}):
                ac = gn(x)
            return torch.sigmoid(ac)

        opt_fn = torch.compile(
            fn, backend=aot_eager_regional_inductor(), fullgraph=True
        )
        x = torch.randn(10, requires_grad=True)

        _, codes = run_fw_bw_and_get_code(lambda: opt_fn(x))
        self.assertEqual(len(codes), 2)


if __name__ == "__main__":
    run_tests()
