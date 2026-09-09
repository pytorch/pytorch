import functools
import unittest
import torch
from torch.testing._internal.common_cuda import requires_cuda_and_triton
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.utils.checkpoint import checkpoint
from torch._functorch.aot_autograd import (
    aot_autograd,
    min_cut_rematerialization_partition,
)


def count_ops(g, args, freq, op):
    count = 0
    for node in g.nodes:
        if node.target == op:
            count += 1
    assert count == freq, f"Expected {freq} occurrences of {op}, but found {count}"
    return g


class TestActivationCheckpointing(TestCase):
    def _validate(self, fn, backend, *args):
        # Eager execution reference
        cloned_args_eager = [
            a.clone().detach().requires_grad_(a.requires_grad) for a in args
        ]
        ref_out = fn(*cloned_args_eager)
        ref_out.sum().backward()
        ref_grads = [a.grad for a in cloned_args_eager]

        # Compiled execution path
        cloned_args_compiled = [
            a.clone().detach().requires_grad_(a.requires_grad) for a in args
        ]
        opt_fn = torch.compile(fn, backend=backend)
        opt_out = opt_fn(*cloned_args_compiled)
        opt_out.sum().backward()
        opt_grads = [a.grad for a in cloned_args_compiled]

        self.assertEqual(ref_out, opt_out)
        for rg, og in zip(ref_grads, opt_grads):
            if rg is not None:
                self.assertEqual(rg, og)

    @requires_cuda_and_triton
    @parametrize("use_reentrant", [True, False])
    def test_kwargs(self, device, use_reentrant):
        def gn(x, y, z=None):
            a = torch.matmul(x, y)
            if z is not None:
                return torch.matmul(a, z)
            return a

        def fn(x, y, z):
            return torch.cos(
                checkpoint(gn, x, y, use_reentrant=use_reentrant, z=z)
            )

        x = torch.randn(4, 4, requires_grad=True, device=device)
        y = torch.randn(4, 4, requires_grad=True, device=device)
        z = torch.randn(4, 4, requires_grad=True, device=device)

        fw_compiler = functools.partial(
            count_ops, freq=2, op=torch.ops.aten.mm.default
        )
        bw_compiler = functools.partial(
            count_ops, freq=6, op=torch.ops.aten.mm.default
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=min_cut_rematerialization_partition,
        )
        self._validate(fn, backend, x, y, z)


instantiate_parametrized_tests(TestActivationCheckpointing)

if __name__ == "__main__":
    run_tests()
