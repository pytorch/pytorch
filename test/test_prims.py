# Owner(s): ["module: primTorch"]

from functools import partial
from itertools import product
import warnings
from warnings import catch_warnings
import unittest

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_utils import (parametrize, run_tests, TestCase, TEST_SCIPY,
                                                  set_default_dtype, skipCUDAMemoryLeakCheckIf)
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
from torch.fx.experimental.proxy_tensor import make_fx


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

    @onlyCUDA
    def test_nvfuser_impl_is_used(self, device):
        # This test is to ensure that when the nvfuser implementation exists it is used
        # Assuming one-to-one mapping between prims and nvfuser implementations
        # This test is not intended to test the correctness of the nvfuser implementation
        try:
            from nvfuser import FusionDefinition as fd
        except ImportError:
            from nvfuser._C import FusionDefinition as fd


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

    def test_skip_ops_nvfuser_prims_mode(self, device):
        # This test verifies that the NvfuserPrimsMode skips the specified
        # functions. Skipping a function means that it's not converted into
        # nvprims counterparts.
        from torch._prims.context import NvfuserPrimsMode

        a = make_tensor(5, 5, device=device, dtype=torch.float32)

        def func(a):
            return torch.ops.prims.sin.default(a)

        skip_ops = {"prims.sin.default", }
        with NvfuserPrimsMode(skip_ops=skip_ops):
            gm = make_fx(func)(a)

        includes_any_prims_sin = any(
            node.target == torch.ops.prims.sin.default for node in gm.graph.nodes
        )
        self.assertTrue(includes_any_prims_sin)
        include_any_nvprims_sin = any(
            node.target == torch.ops.nvprims.sin.default for node in gm.graph.nodes
        )
        self.assertFalse(include_any_nvprims_sin)

    def test_skip_ops_nvfuser_capability_mode(self, device):
        # This test verifies that the NvfuserCapabilityMode skips the specified
        # functions. Skipping a function means that specific
        # reference/decomposition is not traced and there's no attempt to lower
        # it to nvprims.
        from torch._prims.context import TorchRefsNvfuserCapabilityMode

        a = make_tensor(5, 5, device=device, dtype=torch.float32)

        def func(a):
            return torch.sin(a)

        skip_ops = {"torch.sin", }
        with TorchRefsNvfuserCapabilityMode(skip_ops=skip_ops):
            gm = make_fx(func)(a)

        includes_any_aten_sin = any(
            node.target == torch.ops.aten.sin.default for node in gm.graph.nodes
        )
        self.assertTrue(includes_any_aten_sin)
        include_any_nvprims_sin = any(
            node.target == torch.ops.nvprims.sin.default for node in gm.graph.nodes
        )
        self.assertFalse(include_any_nvprims_sin)

    def test_partitioner_tuple_output(self, device):
        # This test verifies that the partitioner doesn't segment on nodes with
        # tuple outputs.
        from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
        from torch._prims.nvfuser_executor import NvfuserPrimOperatorSupport

        a = make_tensor(5, 3, 3, device=device, dtype=torch.float32)

        def func(x):
            xx = torch.ops.nvprims.add(x, 1)
            var, mean = torch.ops.nvprims.var_mean(x, correction=0)
            var_cos = torch.ops.nvprims.cos(var)
            mean_sin = torch.ops.nvprims.sin(mean)
            return torch.ops.nvprims.add(var_cos, mean_sin)

        gm = make_fx(func)(a)
        supported_ops = NvfuserPrimOperatorSupport()
        partitioner = CapabilityBasedPartitioner(
            gm, supported_ops, allows_single_node_partition=False
        )
        partitions = partitioner.propose_partitions()
        self.assertEqual(len(partitions), 1)

    @onlyCUDA
    @dtypes(torch.float32)
    def test_full(self, device, dtype):
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._prims.context import TorchRefsNvfuserCapabilityMode
        from torch._prims.executor import execute

        def func1(size, value, b):
            return (torch.full(size, value, dtype=dtype, device=device),)

        def func2(size, value, b):
            a = torch.full(size, value, dtype=dtype, device=device)
            b_sin = b.sin()
            return (torch.add(a, b_sin),)

        def func3(size, value, b):
            return (torch.full(size, value, dtype=dtype, device=device), b)

        def func4(size, value, b):
            b_sin = b.sin()
            return (torch.full(size, value, dtype=dtype, device=device), b_sin)

        def func5(size, value, b):
            b_sin = b.sin()
            a = torch.full(size, value, dtype=dtype, device=device)
            a_sin = a.sin()
            return (a, b_sin, a_sin)

        for func in (func1, func3, func2, func3, func4, func5):
            size = (3, 3)
            value = 10
            b = torch.randn(*size, dtype=dtype, device=device)

            with TorchRefsNvfuserCapabilityMode():
                gm = make_fx(func)(size, value, b)

            out = execute(gm, size, value, b, executor="strictly_nvfuser")
            self.assertEqual(out, func(size, value, b))

    @onlyCUDA
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
    @dtypes(torch.float16, torch.uint8)
    def test_nvprim_convert_element_type(self, device, dtype):
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._prims.executor import execute
        from torch._prims.context import TorchRefsNvfuserCapabilityMode
        from torch._prims_common import _torch_dtype_to_nvfuser_dtype_map

        # initialize input as float32, which is different from `dtype` in the argument.
        # this ensures that tracing will have a _to_copy node.
        a = torch.randn(3, 3, device=device, dtype=torch.float32)

        def func(x, dtype):
            return x.to(dtype).to(x.dtype)

        with TorchRefsNvfuserCapabilityMode():
            gm = make_fx(func)(a, dtype)
            execute(gm, a, dtype, executor="nvfuser")

        call_function_nodes = list(filter(lambda n: n.op == "call_function", gm.graph.nodes))
        includes_aten_to_copy = any(
            torch.ops.aten._to_copy.default == node.target
            for node in call_function_nodes
        )
        includes_nvprim_convert_element_type = any(
            torch.ops.nvprims.convert_element_type.default == node.target
            for node in call_function_nodes
        )
        nvprim_support_flag = _torch_dtype_to_nvfuser_dtype_map.get(dtype) is not None
        self.assertEqual(includes_aten_to_copy, not nvprim_support_flag)
        self.assertEqual(includes_nvprim_convert_element_type, nvprim_support_flag)

    @onlyCUDA
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
    def test_nvfuser_executor_cached_noncontiguous(self, device):
        # This test is to ensure that nvfuser computes correct results for noncontiguous tensors
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._prims.context import TorchRefsNvfuserCapabilityMode
        from torch._prims.executor import execute

        a = torch.randn(3, 3, device=device)

        def func(a):
            return torch.sigmoid(a)

        with TorchRefsNvfuserCapabilityMode():
            gm = make_fx(func)(a)

        # First run to create the cache
        execute(gm, a, executor="strictly_nvfuser")

        # a.mT is noncontiguous, but it shouldn't affect correctness
        expected = execute(gm, a.mT, executor="aten")
        for use_python_cache in [True, False]:
            params = {"use_python_fusion_cache": use_python_cache}
            actual = execute(gm, a.mT, executor="strictly_nvfuser", executor_parameters=params)
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
    def test_nvfuser_executor_partitioned(self, device):
        # This test is to ensure that nvfuser partitioned executor works correctly
        # It's assumed that digamma is not supported by nvfuser
        # If it's ever supported, this test will need to be updated
        self.assertTrue(getattr(torch.ops.nvprims, "digamma", None) is None)

        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._prims.context import TorchRefsNvfuserCapabilityMode
        from torch._prims.executor import execute

        a = torch.randn(3, 4, device=device)
        b = torch.rand(3, 1, device=device)
        c = torch.rand(3, 4, device=device)

        def func(a, b, c):
            aa = torch.digamma(a)  # not supported by nvfuser
            d = torch.add(b, c)
            dd = torch.sqrt(d)
            return torch.mul(aa, dd.digamma())

        with TorchRefsNvfuserCapabilityMode():
            gm = make_fx(func)(a, b, c)

        expected = execute(gm, a, b, c, executor="aten")
        actual = execute(gm, a, b, c, executor="nvfuser")
        self.assertEqual(expected, actual)

    @onlyCUDA
    def test_nvfuser_executor_partitioned_no_partitions_error(self, device):
        # This test is to ensure that nvfuser partitioned executor works correctly
        # It's assumed that digamma is not supported by nvfuser
        # If it's ever supported, this test will need to be updated
        self.assertTrue(getattr(torch.ops.nvprims, "digamma", None) is None)

        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._prims.context import TorchRefsNvfuserCapabilityMode
        from torch._prims.executor import execute

        a = torch.randn(3, 4, device=device)

        def func(a):
            return torch.digamma(a)  # not supported by nvfuser

        with TorchRefsNvfuserCapabilityMode():
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

    @onlyCUDA
    @dtypes(torch.float32, torch.float64)
    def test_native_batch_norm_nvprims(self, device, dtype):
        from torch._prims.context import TorchRefsNvfuserCapabilityMode
        from torch._prims.executor import execute

        # This test verifies that native_batch_norm is translated into nvprims
        # and can be executed with nvFuser
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.testing._internal.common_methods_invocations import (
            sample_inputs_native_batch_norm,
        )

        samples = sample_inputs_native_batch_norm(
            None, device, dtype, requires_grad=False
        )
        batch_norms = [
            torch.native_batch_norm,
            torch.ops.aten.native_batch_norm,
            torch.ops.aten.native_batch_norm.default,
            torch.ops.nvprims.native_batch_norm.default,
        ]
        for sample, batch_norm in product(samples, batch_norms):
            if sample.input.numel() == 0:
                continue

            def func(
                input, weight, bias, running_mean, running_var, training, momentum, eps
            ):
                return batch_norm(
                    input,
                    weight,
                    bias,
                    running_mean,
                    running_var,
                    training,
                    momentum,
                    eps,
                )

            with TorchRefsNvfuserCapabilityMode():
                gm = make_fx(func)(sample.input, *sample.args)

            call_function_nodes = list(
                filter(lambda n: n.op == "call_function", gm.graph.nodes)
            )
            includes_aten_batch_norm = any(
                torch.ops.aten.native_batch_norm.default == node.target
                for node in call_function_nodes
            )
            self.assertFalse(includes_aten_batch_norm)

            includes_nvprims_batch_norm = any(
                torch.ops.nvprims.native_batch_norm.default == node.target
                for node in call_function_nodes
            )
            self.assertTrue(includes_nvprims_batch_norm)

            # Check that the graph can be executed with nvFuser
            out = execute(gm, sample.input, *sample.args, executor="strictly_nvfuser")
            self.assertEqual(out, gm(sample.input, *sample.args))

    @onlyCUDA
    @dtypes(torch.float32, torch.float64)
    def test_cudnn_batch_norm_nvprims(self, device, dtype):
        from torch._prims.context import TorchRefsNvfuserCapabilityMode
        from torch._prims.executor import execute

        # This test verifies that cudnn_batch_norm is translated into nvprims
        # and can be executed with nvFuser
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.testing._internal.common_methods_invocations import (
            sample_inputs_native_batch_norm,
        )

        samples = sample_inputs_native_batch_norm(
            None, device, dtype, requires_grad=False
        )
        for sample in samples:
            if sample.input.numel() == 0:
                continue

            def func(
                input, weight, bias, running_mean, running_var, training, momentum, eps
            ):
                return torch.ops.aten.cudnn_batch_norm.default(
                    input,
                    weight,
                    bias,
                    running_mean,
                    running_var,
                    training,
                    momentum,
                    eps,
                )

            with TorchRefsNvfuserCapabilityMode():
                gm = make_fx(func)(sample.input, *sample.args)

            call_function_nodes = list(
                filter(lambda n: n.op == "call_function", gm.graph.nodes)
            )
            includes_aten_batch_norm = any(
                torch.ops.aten.cudnn_batch_norm.default == node.target
                for node in call_function_nodes
            )
            self.assertFalse(includes_aten_batch_norm)

            includes_nvprims_batch_norm = any(
                torch.ops.nvprims.native_batch_norm.default == node.target
                for node in call_function_nodes
            )
            self.assertTrue(includes_nvprims_batch_norm)

            # Check that the graph can be executed with nvFuser
            out = execute(gm, sample.input, *sample.args, executor="nvfuser")
            ref_out = gm(sample.input, *sample.args)
            for idx, (left, right) in enumerate(zip(out, ref_out)):
                # Nvfuser does not support torch.uint8 dtype so check reserve output against 0 scalar
                if idx == 3:
                    self.assertTrue(torch.all(torch.eq(left, 0)))
                else:
                    self.assertEqual(left, right)

    # decomposition of native_batch_norm_backward uses a casting, which prevents nvprim lowering on CPU build
    @onlyCUDA
    @dtypes(torch.float32, torch.float16)
    def test_batch_norm_backward_nvprims(self, device, dtype):
        # This test verifies that the backward pass of batch norm is correctly decomposed into nvprims
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._prims.context import TorchRefsNvfuserCapabilityMode
        from torch.testing._internal.common_methods_invocations import sample_inputs_batch_norm

        samples_iter = sample_inputs_batch_norm(None, device, dtype, requires_grad=True)
        sample = next(samples_iter)
        grad = torch.randn_like(sample.input)

        def func1(grad, input, weight, rm, rv, eps, train):
            return torch.ops.aten.native_batch_norm_backward.default(
                grad, input, weight, rm, rv, rm, rv, train, eps, [True, True, True]
            )

        def func2(grad, input, weight, rm, rv, eps, train):
            return torch.ops.aten.cudnn_batch_norm_backward.default(
                input, grad, weight, rm, rv, rm, rv, eps, grad
            )

        args = sample.args
        kwargs = sample.kwargs
        all_args = [grad, sample.input, args[2], args[0], args[1], kwargs['eps'], kwargs['training']]

        for func in (func1, func2):
            with TorchRefsNvfuserCapabilityMode():
                gm = make_fx(func)(*all_args)

            call_function_nodes = list(filter(lambda n: n.op == "call_function", gm.graph.nodes))
            includes_batch_norm_backward = any(
                torch.ops.aten.native_batch_norm_backward.default == node.target
                for node in call_function_nodes
            )
            self.assertFalse(includes_batch_norm_backward)
            all_nvprims = all(
                str(node.target).startswith("nvprims") for node in call_function_nodes
            )
            self.assertTrue(all_nvprims)

    @onlyCUDA
    @dtypes(torch.float32)
    def test_silu_backward_no_filled_tensor(self, device, dtype):
        # This test verifies a workaround for
        # https://github.com/pytorch/pytorch/issues/86612
        from torch.fx.experimental.proxy_tensor import make_fx
        from functorch import functionalize
        from torch._prims.nvfuser_executor import _remove_empty_like_fill
        from torch._prims.context import TorchRefsNvfuserCapabilityMode

        def func(a):
            out = torch.nn.functional.silu(a)
            grad = torch.ones_like(out)
            return torch.autograd.grad([out], [a], [grad])

        make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=True)
        a = make_arg((3, 4))
        gm = make_fx(func)(a)
        # functionalize(gm) doesn't work with non-detached inputs
        gm = make_fx(functionalize(gm))(a.detach())

        # replace aten.sub with nvprims.sub
        with TorchRefsNvfuserCapabilityMode():
            gm = make_fx(gm)(a)

        # Check that the graph contains empty_like
        any_aten_empty_like = any(
            node.target == torch.ops.aten.empty_like.default for node in gm.graph.nodes
        )
        self.assertTrue(any_aten_empty_like)
        any_aten_fill = any(
            node.target == torch.ops.aten.fill.Scalar for node in gm.graph.nodes
        )
        self.assertTrue(any_aten_fill)

        # Now remove the empty_like and fill
        gm = _remove_empty_like_fill(gm)
        any_aten_empty_like = any(
            node.target == torch.ops.aten.empty_like.default for node in gm.graph.nodes
        )
        self.assertFalse(any_aten_empty_like)
        any_aten_fill = any(
            node.target == torch.ops.aten.fill.Scalar for node in gm.graph.nodes
        )
        self.assertFalse(any_aten_fill)
        self.assertEqual(gm(a), func(a))


    @onlyCUDA
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
    @dtypes(torch.float16, torch.float32)
    def test_nvprims_view(self, device, dtype):
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._prims.context import TorchRefsNvfuserCapabilityMode
        from torch._prims.executor import execute

        make_arg = partial(make_tensor, device=device, dtype=dtype)
        a = make_arg((3, 4, 5))

        def func1(a):
            return a.view(tuple(reversed(a.shape)))

        def func2(a):
            return a.reshape(tuple(reversed(a.shape)))

        def func3(a):
            return torch.view_copy(a, tuple(reversed(a.shape)))

        def func4(a):
            return torch.reshape(a, tuple(reversed(a.shape)))

        def func5(a):
            return torch.ops.aten.view.default(a, tuple(reversed(a.shape)))

        def func6(a):
            return torch.ops.aten._unsafe_view.default(a, tuple(reversed(a.shape)))

        def func7(a):
            return torch.ops.aten.view_copy.default(a, tuple(reversed(a.shape)))

        for func in (func1, func2, func3, func4, func5, func6, func7):
            with TorchRefsNvfuserCapabilityMode():
                gm = make_fx(func)(a)

            call_function_nodes = list(filter(lambda n: n.op == "call_function", gm.graph.nodes))
            includes_nvprims_view = any(
                torch.ops.nvprims.view.default == node.target
                for node in call_function_nodes
            )
            self.assertTrue(includes_nvprims_view)

            # Try executing the graph
            out = execute(gm, a, executor="strictly_nvfuser")
            self.assertEqual(out, func(a))

    @onlyCUDA
    @dtypes(torch.float16, torch.float32)
    def test_nvprims_view_partitioner(self, device, dtype):
        # This test verifies that views that are not fused with other ops are
        # correctly overriden to call aten implementation.
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._prims.context import TorchRefsNvfuserCapabilityMode
        from torch._prims.nvfuser_executor import maybe_partition_graph

        make_arg = partial(make_tensor, device=device, dtype=dtype)
        a = make_arg((4, 5))
        b = make_arg((5, 4))

        def func(a, b):
            aa = a.view(b.shape)
            aa = aa.view(a.shape)
            return aa.digamma()

        with TorchRefsNvfuserCapabilityMode():
            gm = make_fx(func)(a, b)
        gm, _ = maybe_partition_graph(gm, False, False)

        out = gm(a, b)
        self.assertEqual(out, func(a, b))

    @onlyCUDA
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

    def test_unbind(self):
        # If unbind returns empty tuple, it breaks some assumptions in some backward tests in test_ops.py.
        # So can't put this test into common_methods_invocations.py.
        a = torch.rand([3, 0, 4])
        actual = refs.unbind(a, 1)
        expect = torch.unbind(a, 1)
        self.assertEqual(actual, expect)


instantiate_device_type_tests(TestRefs, globals())


class TestDecomp(TestCase):
    @onlyCUDA
    @dtypes(torch.float16, torch.float32)
    def test_decomposition_type_promotion_nvprim_amp(self, device, dtype):
        x = torch.rand(5, device=device).to(dtype)
        y = torch.rand(5, device=device).to(dtype)

        from torch._prims.context import TorchRefsNvfuserCapabilityMode, _is_func_unsupported_nvfuser
        from torch.fx.experimental.proxy_tensor import make_fx
        op = torch.ops.aten.leaky_relu_backward.default
        op_decomp = torch._decomp.decomposition_table.get(op)

        def fn0(*arg):
            return _is_func_unsupported_nvfuser(TorchRefsNvfuserCapabilityMode(), op, op_decomp, arg, {})

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

    @onlyCUDA
    @dtypes(torch.float16, torch.float32)
    def test_masked_fill_decomposition_under_nvprim_context(self, device, dtype):
        # Test masked_fill decomposition doesn't trigger data-dependent control flow
        # on TorchRefsNvfuser speculative lowering.
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch._prims.context import TorchRefsNvfuserCapabilityMode

        x = torch.empty(2, 3, device=device).to(dtype=dtype)
        mask = torch.ones_like(x).bool()
        y = torch.tensor(0.3)  # cpu scalar tensor

        def func(x, mask, y):
            return torch.masked_fill(x, mask, y)

        # mimics real use-case for TorchRefsNvfuserCapabilityMode context
        gm = make_fx(func, decomposition_table={})(x, mask, y)

        with warnings.catch_warnings(record=True) as caught:
            with TorchRefsNvfuserCapabilityMode():
                gm = make_fx(gm)(x, mask, y)
        # masked_fill decomposition fails inside `get_isolated_graphmodule`
        self.assertFalse(any(GET_ISOLATED_GRAPHMODULE_ERROR in str(w.message) for w in caught))

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
