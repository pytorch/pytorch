# Owner(s): ["module: primTorch"]

from functools import partial
from itertools import product
from warnings import catch_warnings
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
import torch._refs as refs


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
    def test_nvfuser_executor_cached_noncontiguous(self, device):
        # This test is to ensure that nvfuser computes correct results for noncontiguous tensors
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._prims.context import TorchRefsMode
        from torch._prims.executor import execute

        a = torch.randn(3, 3, device=device)

        def func(a):
            return torch.sigmoid(a)

        with TorchRefsMode():
            gm = make_fx(func)(a)

        # First run to create the cache
        execute(gm, a, executor="nvfuser")

        # a.mT is noncontiguous, but it shouldn't affect correctness
        expected = execute(gm, a.mT, executor="aten")
        actual = execute(gm, a.mT, executor="nvfuser")
        self.assertEqual(expected, actual)

    @onlyCUDA
    @skipCUDAIfRocm
    def test_nvfuser_executor_partitioned(self, device):
        # This test is to ensure that nvfuser partitioned executor works correctly
        # It's assumed that digamma is not supported by nvfuser
        # If it's ever supported, this test will need to be updated
        self.assertTrue(torch.ops.prims.digamma.default.impl_nvfuser is None)

        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._prims.context import TorchRefsMode
        from torch._prims.executor import execute

        a = torch.randn(3, 4, device=device)
        b = torch.rand(3, 1, device=device)
        c = torch.rand(3, 4, device=device)

        def func(a, b, c):
            aa = torch.digamma(a)  # not supported by nvfuser
            d = torch.add(b, c)
            dd = torch.sqrt(d)
            return torch.mul(aa, dd.digamma())

        with TorchRefsMode():
            gm = make_fx(func)(a, b, c)

        expected = execute(gm, a, b, c, executor="aten")
        actual = execute(gm, a, b, c, executor="nvfuser")
        self.assertEqual(expected, actual)

    @onlyCUDA
    @skipCUDAIfRocm
    def test_nvfuser_executor_partitioned_no_partitions_error(self, device):
        # This test is to ensure that nvfuser partitioned executor works correctly
        # It's assumed that digamma is not supported by nvfuser
        # If it's ever supported, this test will need to be updated
        self.assertTrue(torch.ops.prims.digamma.default.impl_nvfuser is None)

        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._prims.context import TorchRefsMode
        from torch._prims.executor import execute

        a = torch.randn(3, 4, device=device)

        def func(a):
            return torch.digamma(a)  # not supported by nvfuser

        with TorchRefsMode():
            gm = make_fx(func)(a)

        with catch_warnings(record=True) as w:
            # Trigger warning
            execute(gm, a, executor="nvfuser")
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue("is not supported by nvFuser" in str(w[-1].message))

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
    def test_pytree_input_output(self, device, dtype):
        @make_traced
        def fn(a, b_dict):
            b = b_dict["b"]
            d = {}
            d["c"] = torch.add(a, b)
            return (d, torch.add(a, d["c"]))

        make_arg = partial(make_tensor, device=device, dtype=dtype)
        a = make_arg((5, 5))
        b = make_arg((1, 5))
        b_dict = {"b": b}

        result_aten = fn(a, b_dict, executor="aten")
        result_nvfuser = fn(a, b_dict, executor="nvfuser")
        self.assertEqual(result_aten, result_nvfuser)

    @dtypes(torch.float32)
    def test_memory_format_strides(self, device, dtype):
        shapes = (
            (),
            (0,),
            (1,),
            (5),
            (1, 0),
            (1, 1),
            (3, 7),
            (3, 0, 2),
            (1, 1, 2),
            (4, 1, 1),
            (7, 8, 9),
        )

        channels_last_shapes = (
            (0, 0, 0, 0),
            (1, 0, 3, 0),
            (0, 2, 3, 5),
            (2, 2, 2, 0),
            (5, 4, 3, 2),
            (8, 8, 7, 2),
            (9, 1, 3, 1),
            (4, 5, 8, 7)
        )

        channels_last_3d_shapes = (
            (0, 8, 7, 9, 2),
            (5, 0, 7, 9, 2),
            (5, 0, 7, 9, 0),
            (5, 8, 7, 9, 2),
            (5, 1, 7, 9, 2),
            (5, 1, 7, 9, 1),
        )

        pairs = (
            (shapes, torch.contiguous_format),
            (channels_last_shapes, torch.contiguous_format),
            (channels_last_3d_shapes, torch.contiguous_format),
            (channels_last_shapes, torch.channels_last),
            (channels_last_3d_shapes, torch.channels_last_3d),
        )

        for shapes, memory_format in pairs:
            for shape in shapes:
                # tests empty
                expected = torch.empty(shape, device=device, dtype=dtype, memory_format=memory_format)
                actual = refs.empty(shape, device=device, dtype=dtype, memory_format=memory_format)
                self.assertEqual(expected.stride(), actual.stride())

                # tests clone
                a = torch.testing.make_tensor(shape, device=device, dtype=dtype)
                expected = torch.clone(a, memory_format=memory_format)
                actual = torch.clone(a, memory_format=memory_format)
                self.assertEqual(expected.stride(), actual.stride())

                # tests contiguous
                a = torch.testing.make_tensor(shape, device=device, dtype=dtype, noncontiguous=True)
                expected = a.contiguous(memory_format=memory_format)
                actual = refs.contiguous(a, memory_format=memory_format)
                self.assertEqual(expected.stride(), actual.stride())


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


class TestRefs(TestCase):
    @dtypes(torch.float32)
    def test_constant_pad_nd_memory_format(self, device, dtype):
        # Test memory format is preserved in unambiguous cases
        for mf, ndim in (
                (torch.channels_last, 4),
                (torch.contiguous_format, 4),
                (torch.channels_last_3d, 5),
                (torch.contiguous_format, 5),
        ):
            a = torch.zeros([2] * ndim).to(memory_format=mf)
            res = refs.constant_pad_nd(a, pad=[1] * (2 * ndim))
            self.assertTrue(res.is_contiguous(memory_format=mf))

        # Ambiguous cases

        # is_channels_last_ and is_contiguous_, results in channels_last output
        a = torch.empty_strided((2, 1, 2, 2), stride=(4, 1, 2, 1))
        self.assertTrue(a.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(a.is_contiguous())
        actual = refs.constant_pad_nd(a, pad=[1] * 8)
        expect = torch.constant_pad_nd(a, pad=[1] * 8)
        self.assertEqual(actual.stride(), expect.stride())
        self.assertTrue(actual.is_contiguous(memory_format=torch.channels_last))

        # is_channels_last_contiguous_ but not is_channels_last_, results in
        # contiguous output
        a = torch.empty_strided((2, 1, 2, 2), stride=(4, 4, 2, 1))
        self.assertTrue(a.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(a.is_contiguous())
        actual = refs.constant_pad_nd(a, pad=[1] * 8)
        expect = torch.constant_pad_nd(a, pad=[1] * 8)
        self.assertEqual(actual.stride(), expect.stride())
        self.assertTrue(actual.is_contiguous())


instantiate_device_type_tests(TestRefs, globals())


if __name__ == "__main__":
    run_tests()
