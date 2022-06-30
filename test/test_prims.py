# Owner(s): ["module: primTorch"]

from functools import partial
from itertools import product
import unittest

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase, TEST_SCIPY
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
    skipCUDAIfRocm,
    dtypes,
)
from torch.testing._internal.logging_tensor import LoggingTensor, capture_logs, log_input
import torch._prims as prims
from torch._prims.executor import make_traced

if TEST_SCIPY:
    import scipy.special


class TestPrims(TestCase):
    @onlyCUDA
    @skipCUDAIfRocm
    @dtypes(torch.float32)
    def test_broadcast_in_dim(self, device, dtype):
        # nvfuser is not currently capable of realizing a broadcasted tensor
        # when the broadcast is the only operation.  Another op is needed.
        def _wrapper(a, b, broadcast_dimensions):
            a_bc = prims.broadcast_in_dim(a, b.shape, broadcast_dimensions)
            return prims.add(a_bc, b)

        traced = make_traced(_wrapper)
        make_arg = partial(make_tensor, device=device, dtype=dtype)

        for executor in ('aten', 'nvfuser'):
            fn = partial(traced, executor=executor)
            # Same shape
            shape = (5, 5)
            a = make_arg(shape)
            b = make_arg(shape, low=0.0, high=0.0)
            result = fn(a, b, (0, 1))

            self.assertEqual(result.shape, a.shape)
            self.assertTrue(result.is_contiguous)
            self.assertEqual(a, result)

            # Error input: reordering dims
            with self.assertRaises(Exception):
                result = fn(a, b, (1, 0))

            # Adding outermost dimensions
            a = make_arg((5, 5))
            b = make_arg((3, 3, 5, 5), low=0.0, high=0.0)
            result = fn(a, b, (2, 3))

            self.assertEqual(result.shape, b.shape)
            self.assertEqual(a.broadcast_to(b.shape), result)

            # Expands
            a = make_arg((1, 5, 1))
            b = make_arg((3, 5, 7), low=0.0, high=0.0)
            result = fn(a, b, (0, 1, 2))

            self.assertEqual(result.shape, b.shape)
            self.assertEqual(a.expand_as(result), result)

            # Unsqueezes
            a = make_arg((1, 2, 3))
            b = make_arg((1, 2, 1, 3), low=0.0, high=0.0)
            result = fn(a, b, (0, 1, 3))

            self.assertEqual(result.shape, b.shape)
            self.assertEqual(a.unsqueeze(2), result)

            # FIXME: This test exposes an issue in nvfuser
            # Adds outermost, expands, and unsqueezes
            """
            a = make_arg((1, 2, 3))
            b = make_arg((4, 1, 7, 2, 3, 3), low=0.0, high=0.0)
            result = fn(a, b, (1, 3, 4))

            self.assertEqual(result.shape, b.shape)
            a.unsqueeze_(3)
            a.unsqueeze_(1)
            a.unsqueeze_(0)
            self.assertEqual(a.expand_as(result), result)
            """

    @onlyCUDA
    @skipCUDAIfRocm
    @dtypes(torch.float32)
    def test_broadcast_in_dim_sum(self, device, dtype):
        def _wrapper(a):
            a_sum = prims.sum(a, [0, 1])
            a_bc = prims.broadcast_in_dim(a_sum, [], [])
            return a_bc

        traced = make_traced(_wrapper)
        make_arg = partial(make_tensor, device=device, dtype=dtype)

        for executor in ('aten', 'nvfuser'):
            fn = partial(traced, executor=executor)
            shape = (5, 5)
            a = make_arg(shape)
            result = fn(a)

            self.assertEqual(result.shape, ())
            self.assertTrue(result.is_contiguous)
            self.assertEqual(_wrapper(a), result)

    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    @dtypes(torch.float64, torch.long)
    def test_cbrt_prim(self, device, dtype):
        make_arg = partial(make_tensor, device=device, dtype=dtype)
        batches = [(), (1,), (2,), (0, 1), (1, 1), (2, 2)]
        shapes = [(), (0,), (1,), (5,)]

        try:
            # Sets the default dtype to NumPy's default dtype of double
            cur_default = torch.get_default_dtype()
            torch.set_default_dtype(torch.double)

            # Tested here, as this OP is not currently exposed or tested in ATen
            for b, s in product(batches, shapes):
                x = make_arg(b + s)
                y = prims.cbrt(x)

                x_np = x.cpu().numpy()
                y_np = scipy.special.cbrt(x_np)

                self.assertEqual(y, y_np, exact_device=False)
        finally:
            torch.set_default_dtype(cur_default)

    @onlyCUDA
    @skipCUDAIfRocm
    def test_nvfuser_impl_is_used(self, device):
        # This test is to ensure that when the nvfuser implementation exists it is used
        # Assuming one-to-one mapping between prims and nvfuser implementations
        # This test is not intended to test the correctness of the nvfuser implementation
        from torch._C._nvfuser import FusionDefinition as fd

        prim_nvfuser_ops = set(torch._prims.__all__).intersection(dir(fd.Ops))
        ops_without_nvfuser_impl = {
            name
            for name in prim_nvfuser_ops
            if getattr(torch.ops.prims, name).default.impl_nvfuser is None
        }
        assert (
            len(ops_without_nvfuser_impl) == 0
        ), (f"The following prims do not have 'impl_nvfuser' defined: {ops_without_nvfuser_impl} ",
            "while there exists nvfuser implementations for them.")

    @onlyCUDA
    @skipCUDAIfRocm
    @dtypes(torch.float32)
    @parametrize("correction", [0, 1])
    def test_var(self, device, dtype, correction):
        def _wrapper(a):
            return prims.var(a, [0, 1], correction=correction)

        traced = make_traced(_wrapper)
        make_arg = partial(make_tensor, device=device, dtype=dtype)

        for executor in ('aten', 'nvfuser'):
            fn = partial(traced, executor=executor)
            shape = (5, 5)
            a = make_arg(shape)
            result = fn(a)

            self.assertEqual(result.shape, ())
            self.assertTrue(result.is_contiguous)
            self.assertEqual(_wrapper(a), result)

    @onlyCUDA
    @skipCUDAIfRocm
    @dtypes(torch.float32)
    def test_pytree_output(self, device, dtype):
        @make_traced
        def fn(a, b):
            d = {}
            d["c"] = torch.add(a, b)
            return (d, torch.add(a, d["c"]))

        make_arg = partial(make_tensor, device=device, dtype=dtype)
        a = make_arg((5, 5))
        b = make_arg((1, 5))

        result_aten = fn(a, b, executor="aten")
        result_nvfuser = fn(a, b, executor="nvfuser")
        self.assertEqual(result_aten, result_nvfuser)


class TestPrimsBasic(TestCase):
    def test_torch_ops(self):
        r = make_tensor((2,), device='cpu', dtype=torch.float)
        self.assertEqual(torch.ops.prims.sin(r), torch.sin(r))

        r = LoggingTensor(r)
        with capture_logs() as logs:
            log_input("input", r)
            prims.sin(r)
        self.assertExpectedInline('\n'.join(logs), """\
$0 = input('input')
$1 = torch._ops.prims.sin.default($0)""")

    def test_mul_complex(self):
        prims.mul(torch.randn(2), 1 + 1j)


instantiate_device_type_tests(TestPrims, globals())

if __name__ == "__main__":
    run_tests()
