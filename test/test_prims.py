# Owner(s): ["module: primTorch"]

from functools import partial
from itertools import product
import unittest

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_utils import (parametrize, run_tests, TestCase, TEST_SCIPY,
                                                  set_default_dtype)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
    dtypes,
    OpDTypes,
)
from torch.testing._internal.common_methods_invocations import (
    op_db,
)
from torch.testing._internal.common_device_type import (
    ops,
)

from torch.testing._internal.logging_tensor import LoggingTensor, capture_logs, log_input
import torch._prims as prims
from torch._prims_common import CUDARngStateHelper
from torch._prims.executor import make_traced
import torch._refs as refs


if TEST_SCIPY:
    import scipy.special

NVPRIM_ATEN_FALLBACK_WARNING = "fallback to aten executor"
GET_ISOLATED_GRAPHMODULE_ERROR = "get_isolated_graphmodule failed on decomposition"

class TestPrims(TestCase):
    @onlyCUDA
    @dtypes(torch.float32)
    def test_broadcast_in_dim(self, device, dtype):
        def _wrapper(a, b, broadcast_dimensions):
            return prims.broadcast_in_dim(a, b.shape, broadcast_dimensions)

        traced = make_traced(_wrapper)
        make_arg = partial(make_tensor, device=device, dtype=dtype)

        for executor in ('aten',):
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

    @onlyCUDA
    @dtypes(torch.float32)
    def test_broadcast_in_dim_sum(self, device, dtype):
        def _wrapper(a):
            a_sum = prims.sum(a, [0, 1])
            a_bc = prims.broadcast_in_dim(a_sum, [], [])
            return a_bc

        traced = make_traced(_wrapper)
        make_arg = partial(make_tensor, device=device, dtype=dtype)

        for executor in ('aten',):
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

        # Sets the default dtype to NumPy's default dtype of double
        with set_default_dtype(torch.double):
            # Tested here, as this OP is not currently exposed or tested in ATen
            for b, s in product(batches, shapes):
                x = make_arg(b + s)
                y = prims.cbrt(x)

                x_np = x.cpu().numpy()
                y_np = scipy.special.cbrt(x_np)

                self.assertEqual(y, y_np, exact_device=False)

    @dtypes(torch.float32)
    def test_collapse(self, device, dtype):
        t = torch.rand(2, 2, 2)
        dim_ranges = [(0, 0), (0, 1), (1, 2), (0, 2)]
        expected_shapes = [(2, 2, 2), (4, 2), (2, 4), (8,)]

        for (start, end), shape in zip(dim_ranges, expected_shapes):
            expect = t.reshape(shape)

            copy = prims.collapse(t, start, end)
            self.assertEqual(copy, expect)
            self.assertFalse(copy._is_view())

            view = prims.collapse_view(t, start, end)
            self.assertEqual(view, expect)
            self.assertTrue(view._is_view())

        t_discontig = t.transpose(0, 1)
        with self.assertRaises(ValueError, msg="no such view exists"):
            view = prims.collapse_view(t_discontig, 0, 2)

        copy = prims.collapse(t_discontig, 0, 1)
        self.assertEqual(copy, t_discontig.reshape(4, 2))

        error_dims = [(-1, 1), (0, 3), (1, -1)]
        for start, end in error_dims:
            for fn in [prims.collapse, prims.collapse_view]:
                with self.assertRaises(AssertionError):
                    fn(t, start, end)


    def test_aten_overload_to_prims(self, device):
        # This test is to ensure that the torch.ops.aten calls are replaced with refs
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._prims.context import TorchRefsMode

        a = torch.randn(3, 3, device=device)

        def func(a):
            return torch.ops.aten.sigmoid.default(torch.ops.aten.digamma.default(a))

        with TorchRefsMode():
            gm = make_fx(func)(a)

        # Check that all call_function nodes are prims
        call_function_nodes = list(filter(lambda n: n.op == "call_function", gm.graph.nodes))
        all_prims_namespace = all(
            node.target.name().startswith("prims") for node in call_function_nodes
        )
        self.assertTrue(all_prims_namespace)

    @onlyCUDA
    @dtypes(torch.float32)
    @parametrize("correction", [0, 1])
    def test_var(self, device, dtype, correction):
        def _wrapper(a):
            return prims.var(a, [0, 1], correction=correction)

        traced = make_traced(_wrapper)
        make_arg = partial(make_tensor, device=device, dtype=dtype)

        for executor in ('aten',):
            fn = partial(traced, executor=executor)
            shape = (5, 5)
            a = make_arg(shape)
            result = fn(a)

            self.assertEqual(result.shape, ())
            self.assertTrue(result.is_contiguous)
            self.assertEqual(_wrapper(a), result)

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

    @dtypes(torch.float32)
    def test_reshape_view_method(self, device, dtype):
        make_arg = partial(make_tensor, device=device, dtype=dtype)
        a = make_arg((5, 5))
        new_shape = 1, 5, 1, 5
        result_eager = a.reshape(*new_shape)
        result_refs = refs.reshape(a, *new_shape)
        self.assertEqual(result_eager, result_refs)

        result_eager = a.view(*new_shape)
        result_refs = refs.view(a, *new_shape)
        self.assertEqual(result_eager, result_refs)


    @onlyCUDA
    @dtypes(torch.float32)
    def test_philox_rand(self, device, dtype):
        sizes = (1000, 1000000)  # offsets of 4 and 8
        repeats = 2  # Checks multiple rand calls results with multiple philox_rand calls
        for size in sizes:
            torch.cuda.manual_seed(123)
            references = []
            results = []
            rng_states = []
            for _ in range(repeats):
                rng_states.append(CUDARngStateHelper.get_torch_state_as_tuple())
                references.append(torch.rand(size, device=device, dtype=dtype))

            torch.cuda.manual_seed(123)
            for idx in range(repeats):
                seed, offset = rng_states[idx]
                result, _ = torch.ops.rngprims.philox_rand((size,),
                                                           seed=seed,
                                                           offset=offset,
                                                           stride=None,
                                                           device=device,
                                                           dtype=dtype)
                results.append(result)

            for a, b in zip(references, results):
                self.assertEqual(a, b)


    @dtypes(torch.float32)
    def test_functional_rng_wrappers(self, device, dtype):

        torch.manual_seed(123)
        ref1 = torch.rand(10, device=device, dtype=dtype)
        ref2 = torch.rand(10, device=device, dtype=dtype)


        torch.manual_seed(123)
        rng_state1, res1 = torch._prims.rng_prims.run_and_save_rng_state(torch.rand, 10, device=device, dtype=dtype)
        rng_state2, res2 = torch._prims.rng_prims.run_and_save_rng_state(torch.rand, 10, device=device, dtype=dtype)

        res3 = torch._prims.rng_prims.run_with_rng_state(rng_state1, torch.rand, 10, device=device, dtype=dtype)
        res4 = torch._prims.rng_prims.run_with_rng_state(rng_state2, torch.rand, 10, device=device, dtype=dtype)

        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)
        self.assertEqual(ref1, res3)
        self.assertEqual(ref2, res4)

class TestPrimsBasic(TestCase):
    def test_torch_ops(self):
        r = make_tensor((2,), device='cpu', dtype=torch.float)
        self.assertEqual(torch.ops.prims.sin(r), torch.sin(r))

        r = LoggingTensor(r)
        with capture_logs() as logs:
            log_input("input", r)
            prims.sin(r)
        self.assertExpectedInline('\n'.join(logs), """\
$0: f32[2] = input('input')
$1: f32[2] = torch._ops.prims.sin.default($0)""")

    def test_mul_complex(self):
        prims.mul(torch.randn(2), 1 + 1j)

    def test_check_deprecation_warning(self):
        with self.assertWarnsRegex(DeprecationWarning, 'will be removed in the future'):
            torch._prims_common.check(True, lambda: 'message')


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

    def test_unbind(self):
        # If unbind returns empty tuple, it breaks some assumptions in some backward tests in test_ops.py.
        # So can't put this test into common_methods_invocations.py.
        a = torch.rand([3, 0, 4])
        actual = refs.unbind(a, 1)
        expect = torch.unbind(a, 1)
        self.assertEqual(actual, expect)

    def test_logspace_with_complex_input(self):
        actual = refs.logspace(2, 10 + 5j, steps=5)
        expect = torch.logspace(2, 10 + 5j, steps=5)
        self.assertEqual(actual, expect)

    def test_linspace_with_complex_input(self):
        actual = refs.linspace(2, 10 + 5j, steps=5)
        expect = torch.linspace(2, 10 + 5j, steps=5)
        self.assertEqual(actual, expect)

    # From https://github.com/pytorch/pytorch/issues/109558
    def test_infinite_loop_from_py_dispatcher(self):
        # enables prim decomps
        with torch._dispatch.python.enable_python_dispatcher():
            x = torch.ones(4)
            y = x.to(device="meta")


instantiate_device_type_tests(TestRefs, globals())


class TestDecomp(TestCase):
    @ops([op for op in op_db if op.supports_varargs], dtypes=OpDTypes.any_one)
    def test_decomposition_method_vararg(self, device, dtype, op):
        # some ops have vararg variants for the methods. this tests it.
        # we don't have tests for varargs in OpInfo, so we need to
        # improvise this a bit.
        # The rule for general functions (the special cases being e.g. tensor
        # creation functions taking shapes) is that things can be vararg
        # if the method has only one argument of sequence type.
        # e.g. permute can be called on a 3d tensor t as t.permute(0, 2, 1)
        #      as well as t.permute([0, 2, 1])
        #      when the signature in native_functions.yaml
        #      shows arguments Tensor self, IntList dims
        # we might need to adjust things for the factory functions or
        # have them do their own test
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._prims.context import TorchRefsMode

        # filter out empty tuple as that cannot be the varargs
        sample_inputs = (si for si in op.sample_inputs(device, dtype, requires_grad=False)
                         if (si.args[-1] if si.args else si.input))

        # just run one test, we assume there is a suitable one in the tests
        sample_input = next(sample_inputs)
        all_args = (sample_input.input,) + sample_input.args

        # in general, the methods take varargs and not (always?) the function
        # variants, the exception to this rule are the factory functions
        if op.is_factory_function:
            fn = op.op
        else:
            fn = op.method_variant
        with TorchRefsMode():
            gm = make_fx(fn)(*all_args[:-1], *all_args[-1])

        # in case we add random factory functions
        torch.manual_seed(1)
        res = gm(*all_args[:-1], *all_args[-1])
        torch.manual_seed(1)
        expected = fn(*all_args[:-1], *all_args[-1])
        self.assertEqual(res, expected)


instantiate_device_type_tests(TestDecomp, globals())


if __name__ == "__main__":
    run_tests()
