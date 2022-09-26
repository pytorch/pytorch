# Owner(s): ["module: primTorch"]

from functools import partial
from itertools import product
import warnings
from warnings import catch_warnings
import unittest

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase, TEST_SCIPY, skipCUDAMemoryLeakCheckIf
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

NVPRIM_ATEN_FALLBACK_WARNING = "fallback to aten executor"

class TestPrims(TestCase):
    @onlyCUDA
    @skipCUDAIfRocm
    @dtypes(torch.float32)
    def test_broadcast_in_dim(self, device, dtype):
        def _wrapper(a, b, broadcast_dimensions):
            return prims.broadcast_in_dim(a, b.shape, broadcast_dimensions)

        traced = make_traced(_wrapper)
        make_arg = partial(make_tensor, device=device, dtype=dtype)

        for executor in ('aten', 'strictly_nvfuser'):
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

        for executor in ('aten', 'strictly_nvfuser'):
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

        prim_nvfuser_ops = set(torch._prims.__all__).intersection(dir(fd.ops))
        ops_without_nvfuser_impl = {
            name
            for name in prim_nvfuser_ops
            if getattr(torch.ops.nvprims, name, None) is None
        }
        assert (
            len(ops_without_nvfuser_impl) == 0
        ), (f"The following prims do not have 'impl_nvfuser' defined: {ops_without_nvfuser_impl} ",
            "while there exists nvfuser implementations for them.")

    @onlyCUDA
    @skipCUDAIfRocm
    def test_nvfuser_empty_fusion(self, device):
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._prims.executor import execute

        a = torch.randn(3, 3, device=device)

        def func(a, b, c):
            return (a, b, c)

        gm = make_fx(func)(a, a, a)

        with self.assertRaisesRegex(AssertionError, "Graph must contain at least one call_function node"):
            execute(gm, a, a, a, executor="strictly_nvfuser")

        # Should pass with partitioned executor
        out = execute(gm, a, a, a, executor="nvfuser")
        self.assertEqual(out, (a, a, a))

    @onlyCUDA
    @skipCUDAIfRocm
    def test_nvfuser_rand_like_fusion(self, device):
        from torch._prims.context import TorchRefsNvfuserCapabilityMode
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._prims.executor import execute

        a = torch.randn(3, 3, device=device)

        def func(a):
            return torch.rand_like(a)

        with TorchRefsNvfuserCapabilityMode():
            gm = make_fx(func)(a)

        out = execute(gm, a, executor="strictly_nvfuser")
        self.assertEqual(out.size(), a.size())

    @skipCUDAMemoryLeakCheckIf(True)  # https://github.com/pytorch/pytorch/issues/84529
    @onlyCUDA
    @skipCUDAIfRocm
    def test_nvfuser_no_args(self, device):
        from torch._prims.context import TorchRefsNvfuserCapabilityMode
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._prims.executor import execute
        from torch._prims.nvfuser_executor import make_nvfuser_fusion

        a = torch.randn(3, 3, device=device)

        def func():
            return torch.sigmoid(a)

        with TorchRefsNvfuserCapabilityMode():
            gm = make_fx(func)()

        with warnings.catch_warnings(record=True) as caught:
            execute(gm, executor="strictly_nvfuser")
        # fusion execute with no cuda input is handled by nvprim aten fallback
        self.assertTrue(any(NVPRIM_ATEN_FALLBACK_WARNING in str(w.message) for w in caught))

        with self.assertRaisesRegex(AssertionError, "There must be at least one argument"):
            make_nvfuser_fusion(gm)

        with self.assertRaisesRegex(AssertionError, "Number of placeholder nodes in the graph must match"):
            execute(gm, a, executor="strictly_nvfuser")

        # Should pass with partitioned executor
        out = execute(gm, executor="nvfuser")
        self.assertEqual(out, func())

    @onlyCUDA
    @skipCUDAIfRocm
    def test_nvfuser_constant_tensors(self, device):
        from torch._prims.context import TorchRefsNvfuserCapabilityMode
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._prims.executor import execute

        a = torch.randn(3, 3, device=device)
        b = torch.randn(3, 3, device=device)

        def func(b):
            return a + b

        with TorchRefsNvfuserCapabilityMode():
            gm = make_fx(func)(b)

        with self.assertRaisesRegex(AssertionError, "not supported yet"):
            execute(gm, b, executor="strictly_nvfuser")

        # Should pass with partitioned executor
        out = execute(gm, b, executor="nvfuser")
        self.assertEqual(out, gm(b))

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

    def test_nvfuser_capability_context(self, device):
        # This test is to ensure that the torch calls are replaced with refs
        # based on the nvfuser+prims capability
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._prims.context import TorchRefsNvfuserCapabilityMode

        # It's assumed that digamma is not supported by nvfuser
        # If it's ever supported, this test will need to be updated
        self.assertTrue(getattr(torch.ops.nvprims, "digamma", None) is None)

        a = torch.randn(3, 3, device=device)

        def func(a):
            return torch.digamma(a)

        with TorchRefsNvfuserCapabilityMode():
            gm = make_fx(func)(a)

        # Check that the torch.digamma is not replaced with torch.ops.prims.digamma
        call_function_nodes = list(filter(lambda n: n.op == "call_function", gm.graph.nodes))
        includes_aten_digamma = any(
            torch.ops.aten.digamma.default == node.target
            for node in call_function_nodes
        )
        includes_prims_digamma = any(
            torch.ops.prims.digamma.default == node.target
            for node in call_function_nodes
        )
        self.assertTrue(includes_aten_digamma)
        self.assertFalse(includes_prims_digamma)

        # Check mixed case, sigmoid is replaced with refs, but digamma is not
        def func(a):
            return torch.sigmoid(torch.digamma(a))

        with TorchRefsNvfuserCapabilityMode():
            gm = make_fx(func)(a)

        call_function_nodes = list(filter(lambda n: n.op == "call_function", gm.graph.nodes))
        includes_aten_sigmoid = any(
            torch.ops.aten.sigmoid.default == node.target
            for node in call_function_nodes
        )
        includes_prims_digamma = any(
            torch.ops.prims.digamma.default == node.target
            for node in call_function_nodes
        )
        includes_nvprims_exp = any(
            torch.ops.nvprims.exp.default == node.target
            for node in call_function_nodes
        )
        self.assertFalse(includes_aten_sigmoid)
        self.assertFalse(includes_prims_digamma)
        self.assertTrue(includes_nvprims_exp)


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
    @skipCUDAIfRocm
    def test_nvfuser_executor_parameters(self, device):
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._prims.executor import execute

        a = torch.randn(3, 4, device=device)

        def func(a):
            return torch.ops.nvprims.add(a, a)

        gm = make_fx(func)(a)

        expected = execute(gm, a, executor="aten")
        # Shouldn't raise an error because unuseful parameters are ignored
        params_dicts = [None, {}, {"none": None}]
        for params in params_dicts:
            actual = execute(gm, a, executor="nvfuser", executor_parameters=params)
            self.assertEqual(expected, actual)

        # Check caching parameter
        for use_cache in [True, False]:
            params = {"use_python_fusion_cache": use_cache}
            actual = execute(gm, a, executor="nvfuser", executor_parameters=params)
            self.assertEqual(expected, actual)

        # Check allow_single_op_fusion parameter
        for allow_single_op_fusion in [True, False]:
            params = {"allow_single_op_fusion": allow_single_op_fusion}
            actual = execute(gm, a, executor="nvfuser", executor_parameters=params)
            self.assertEqual(expected, actual)


    @onlyCUDA
    @skipCUDAIfRocm
    def test_nvfuser_executor_partitioned(self, device):
        # This test is to ensure that nvfuser partitioned executor works correctly
        # It's assumed that digamma is not supported by nvfuser
        # If it's ever supported, this test will need to be updated
        self.assertTrue(getattr(torch.ops.nvprims, "digamma", None) is None)

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
        self.assertTrue(getattr(torch.ops.nvprims, "digamma", None) is None)

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

    def test_nvprims(self, device):
        # This test is to ensure that nvfuser specific prims are exposed
        # and can be traced with make_fx
        from torch.fx.experimental.proxy_tensor import make_fx

        def func(a):
            return torch.ops.nvprims.add(a, a)

        a = torch.randn(3, 4, device=device)
        gm = make_fx(func)(a)

        for node in gm.graph.nodes:
            if node.op == "call_function":
                self.assertTrue(node.name == "add")
                self.assertTrue(node.target == torch.ops.nvprims.add.default)
                self.assertFalse(node.target == torch.ops.prims.add.default)
                self.assertFalse(node.target == torch.ops.aten.add.default)

    @dtypes(torch.float32, torch.float16)
    def test_batch_norm_backward_nvprims(self, device, dtype):
        # This test verifies that the backward pass of batch norm is correctly decomposed into nvprims
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._prims.context import TorchRefsNvfuserCapabilityMode
        from torch.testing._internal.common_methods_invocations import sample_inputs_batch_norm

        samples_iter = sample_inputs_batch_norm(None, device, dtype, requires_grad=True)
        sample = next(samples_iter)
        grad = torch.randn_like(sample.input)

        def func(grad, input, weight, rm, rv, eps, train):
            return torch.ops.aten.native_batch_norm_backward.default(
                grad, input, weight, rm, rv, rm, rv, train, eps, [True, True, True]
            )

        args = sample.args
        kwargs = sample.kwargs
        all_args = [grad, sample.input, args[2], args[0], args[1], kwargs['eps'], kwargs['training']]
        with TorchRefsNvfuserCapabilityMode():
            gm = make_fx(func)(*all_args)

        call_function_nodes = list(filter(lambda n: n.op == "call_function", gm.graph.nodes))
        includes_batch_norm_backward = any(
            torch.ops.aten.native_batch_norm_backward.default == node.target
            for node in call_function_nodes
        )
        self.assertFalse(includes_batch_norm_backward)

    @onlyCUDA
    @skipCUDAIfRocm
    @dtypes(torch.float32)
    @parametrize("correction", [0, 1])
    def test_var(self, device, dtype, correction):
        def _wrapper(a):
            return prims.var(a, [0, 1], correction=correction)

        traced = make_traced(_wrapper)
        make_arg = partial(make_tensor, device=device, dtype=dtype)

        for executor in ('aten', 'strictly_nvfuser'):
            fn = partial(traced, executor=executor)
            shape = (5, 5)
            a = make_arg(shape)
            result = fn(a)

            self.assertEqual(result.shape, ())
            self.assertTrue(result.is_contiguous)
            self.assertEqual(_wrapper(a), result)

    @onlyCUDA
    @skipCUDAIfRocm
    @dtypes(torch.float16, torch.float32)
    @parametrize("correction", [0, 1])
    @parametrize("keepdim", [True, False])
    def test_var_mean(self, device, dtype, correction, keepdim):
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._prims.context import TorchRefsNvfuserCapabilityMode


        def _wrapper(a):
            return torch.var_mean(a, [0, 1], correction=correction, keepdim=keepdim)

        make_arg = partial(make_tensor, device=device, dtype=dtype)

        with TorchRefsNvfuserCapabilityMode():
            gm = make_fx(_wrapper)(make_arg((5, 5)))

        call_function_nodes = list(filter(lambda n: n.op == "call_function", gm.graph.nodes))
        includes_nvprims_var_mean = any(
            torch.ops.nvprims.var_mean.main == node.target
            for node in call_function_nodes
        )
        self.assertTrue(includes_nvprims_var_mean)

    @onlyCUDA
    @skipCUDAIfRocm
    @dtypes(torch.float32, torch.float16)
    def test_cpu_tensor(self, device, dtype):
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._prims.context import TorchRefsNvfuserCapabilityMode
        from torch._prims.executor import execute

        def _wrapper(t0, t1, cpu_scalar):
            return t0 + t1 + cpu_scalar

        make_arg = partial(make_tensor, device=device, dtype=dtype)
        a = make_arg((12, 1))
        b = make_arg((12, 12))
        c = torch.tensor(0.5)

        with TorchRefsNvfuserCapabilityMode():
            gm = make_fx(_wrapper)(a, b, c)

        with warnings.catch_warnings(record=True) as caught:
            actual = execute(gm, a, b, c, executor="nvfuser")
        # cpu scalar tensor is handled by nvfuser codegen, so it shouldn't fallback
        self.assertFalse(any(NVPRIM_ATEN_FALLBACK_WARNING in str(w.message) for w in caught))

        expected = execute(gm, a, b, c, executor="aten")
        self.assertEqual(expected, actual)

        call_function_nodes = list(filter(lambda n: n.op == "call_function", gm.graph.nodes))
        includes_aten_add = any(
            torch.ops.aten.add.default == node.target
            for node in call_function_nodes
        )
        self.assertFalse(includes_aten_add)

        with warnings.catch_warnings(record=True) as caught:
            nvprim_aten_fallback = execute(gm, a.cpu(), b.cpu(), c, executor="nvfuser")
        # cpu tensor is handled by nvprim aten fallback, assert that it's indeed in warning
        self.assertTrue(any(NVPRIM_ATEN_FALLBACK_WARNING in str(w.message) for w in caught))

        self.assertEqual(expected, nvprim_aten_fallback)

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
        result_nvfuser = fn(a, b_dict, executor="strictly_nvfuser")
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


class TestDecomp(TestCase):
    @onlyCUDA
    @skipCUDAIfRocm
    @dtypes(torch.float16, torch.float32)
    def test_decomposition_type_promotion_nvprim_amp(self, device, dtype):
        x = torch.rand(5, device=device).to(dtype)
        y = torch.rand(5, device=device).to(dtype)

        from torch._prims.context import TorchRefsNvfuserCapabilityMode, _is_func_unsupported_nvfuser
        from torch.fx.experimental.proxy_tensor import make_fx
        op = torch._decomp.decomposition_table.get(torch.ops.aten.leaky_relu_backward.default)

        def fn0(*arg):
            return _is_func_unsupported_nvfuser(TorchRefsNvfuserCapabilityMode(), op, arg, {})

        def fn1(x):
            x = x * 2
            x = x @ x
            x = x * 2
            return x

        self.assertFalse(fn0(x, y, 0.3, False))
        with TorchRefsNvfuserCapabilityMode():

            # Autocast context has C++ level ATen calls that are hidden from
            # TorchRefsNvfuserCapabilityMode that works only on Python level.
            # The first call to make_fx records autocast C++ calls directly and
            # doesn't have the chance to translate to nvprims. After the first
            # call, "gm" contains explicit calls to torch.ops.aten and nothing
            # is hidden, so the second call to make_fx actually translates
            # recorded autocast dtype conversions to nvprims.
            with torch.autocast("cuda"):
                gm = make_fx(fn1)(x)
            gm = make_fx(gm)(x)
            call_function_nodes = list(filter(lambda n: n.op == "call_function", gm.graph.nodes))
            includes_aten_to_copy = any(
                torch.ops.aten._to_copy.default == node.target
                for node in call_function_nodes
            )
            self.assertFalse(includes_aten_to_copy)


instantiate_device_type_tests(TestDecomp, globals())


if __name__ == "__main__":
    run_tests()
