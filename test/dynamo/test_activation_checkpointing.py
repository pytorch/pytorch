# Owner(s): ["module: dynamo"]
import functools
import unittest
from importlib import import_module

import torch

import torch._dynamo.test_case
import torch._functorch.config
import torch.utils.checkpoint
from torch._dynamo.backends.common import aot_autograd
from torch.testing._internal.inductor_utils import HAS_CUDA


requires_cuda = functools.partial(unittest.skipIf, not HAS_CUDA, "requires cuda")


def count_ops(gm, args, freq, op):
    assert [node.target for node in gm.graph.nodes].count(op) == freq
    return gm


class ActivationCheckpointingViaTagsTests(torch._dynamo.test_case.TestCase):
    def _validate(self, fn, backend, *args, skip_check=False, fullgraph=True):
        cloned_args = []
        for arg in args:
            cloned_args.append(arg.clone().detach().requires_grad_(arg.requires_grad))

        torch.manual_seed(0)
        expected = fn(*args)
        expected.sum().backward()

        torch.manual_seed(0)
        result = torch.compile(fn, fullgraph=fullgraph, backend=backend)(*cloned_args)
        result.sum().backward()

        if not skip_check:
            self.assertEqual(result, expected)
            for arg, cloned_arg in zip(args, cloned_args):
                self.assertEqual(arg.grad, cloned_arg.grad)

    @requires_cuda()
    def test_tags_function(self):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(gn, torch.sin(x), y)

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops, freq=3, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @requires_cuda()
    def test_tags_function_with_kwargs(self):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=True, preserve_rng_state=False
            )

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops, freq=3, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @requires_cuda()
    def test_tags_multiple_checkpoints(self):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            x = torch.sin(x)
            z = torch.utils.checkpoint.checkpoint(gn, x, y)
            x = torch.sin(z)
            z = torch.utils.checkpoint.checkpoint(gn, x, y)
            return z

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops, freq=6, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @requires_cuda()
    def test_tags_module(self):
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return torch.sigmoid(self.linear(x))

        mod = MockModule().cuda()

        def fn(x):
            return torch.utils.checkpoint.checkpoint(mod, torch.sin(x))

        x = torch.randn(10, 10, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.sigmoid.default
        )
        bw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.sigmoid.default
        )
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x)

    @requires_cuda()
    def test_tags_decomps(self):
        # Ensures that tags are passed on through decompositions as well
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return torch.nn.functional.gelu(self.linear(x))

        mod = MockModule().cuda()

        def fn(x):
            return torch.utils.checkpoint.checkpoint(mod, torch.sin(x))

        x = torch.randn(10, 10, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.erf.default
        )
        bw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.erf.default
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            decompositions=lambda: import_module(
                "torch._inductor.compile_fx"
            ).select_decomp_table(),
        )
        self._validate(fn, backend, x)

    @requires_cuda()
    @torch._inductor.config.patch(fallback_random=True)
    def test_tags_recomputed_rand(self):
        def gn(x, y):
            return torch.sigmoid(torch.rand_like(x) * y) * x

        def fn(x, y):
            x = torch.sin(x)
            x = torch.utils.checkpoint.checkpoint(gn, x, y)
            x = torch.sin(x)
            z = torch.utils.checkpoint.checkpoint(gn, x, y)
            return z

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        # fw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        # bw_compiler = functools.partial(
        #     count_ops, freq=6, op=torch.ops.aten.mm.default
        # )  # mm recomputed in the bwd
        # backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        backend = "inductor"
        self._validate(fn, backend, x, y)

    @requires_cuda()
    @torch._inductor.config.patch(fallback_random=True)
    def test_tags_rand(self):
        def gn(x, y):
            x = torch.mm(x, y)
            x = torch.mm(x, y)
            return x

        def fn(x, y):
            x = torch.sin(x)
            x = torch.utils.checkpoint.checkpoint(gn, x, y)
            x = torch.sin(x)
            # x = torch.utils.checkpoint.checkpoint(gn, x, y)
            return x

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        # fw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        # bw_compiler = functools.partial(
        #     count_ops, freq=6, op=torch.ops.aten.mm.default
        # )  # mm recomputed in the bwd
        # backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        # backend = "aot_eager"
        backend = "inductor"
        self._validate(fn, backend, x, y)

    @requires_cuda()
    @torch._inductor.config.patch(fallback_random=True)
    def test_tags_dropout(self):
        # Figure out a way to test the number of inductor_random calls
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.dropout = torch.nn.Dropout(0.2)

            def forward(self, x):
                return self.dropout(self.linear(x))

        mod = MockModule().cuda()

        def fn(x):
            return torch.utils.checkpoint.checkpoint(mod, x)

        x = torch.randn(10, 10, device="cuda", requires_grad=True)
        backend = "inductor"
        # rand decomps do not have have numerical results as eager
        self._validate(fn, backend, x, skip_check=True)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
