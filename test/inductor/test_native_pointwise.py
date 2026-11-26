# Owner(s): ["module: inductor"]
from unittest.mock import patch

import torch
from torch._dynamo.backends.registry import lookup_backend
from torch._inductor import config, lowering
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map


class NativeCalls(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func._schema.name in (
            "aten::special_laguerre_polynomial_l",
            "aten::special_legendre_polynomial_p",
        ):

            def encode(value):
                if isinstance(value, torch.Tensor):
                    return (
                        value.dtype,
                        value.device,
                        tuple(value.shape),
                        value.stride(),
                        value.storage_offset(),
                        value.data_ptr(),
                        value.untyped_storage().data_ptr(),
                        value.is_neg(),
                        value.is_conj(),
                    )
                return value

            self.calls.append((func, tree_map(encode, args), tree_map(encode, kwargs)))
        return func(*args, **kwargs)


class TestNativePointwise(TestCase):
    hw_classification = HardwareClassification.CUDA

    @dtypes(torch.float32, torch.float64)
    @parametrize("op_name", ("laguerre_polynomial_l", "legendre_polynomial_p"))
    @parametrize("layout", ("contiguous", "strided", "scalar"))
    @parametrize("numerics", ("default", "strict"))
    @parametrize("mixed", (False, True))
    def test_reused_native_arguments(
        self, device, dtype, op_name, layout, numerics, mixed
    ):
        compiled_calls = []
        compilations = []

        def fn(x, degree):
            result = getattr(torch.special, op_name)(x, degree)
            return (result, x + 1) if mixed else result

        def backend(gm, inputs):
            compilations.append(gm)
            compiled = lookup_backend("inductor")(gm, inputs)

            def execute(*args):
                calls = NativeCalls()
                with calls:
                    output = compiled(*args)
                compiled_calls.extend(calls.calls)
                return output

            return execute

        torch._dynamo.reset()
        with config.patch(
            numerics=numerics, fx_graph_cache=False, fx_graph_remote_cache=False
        ):
            compiled = torch.compile(fn, backend=backend, fullgraph=True)
            for offset in (0, 1, 2):
                base = torch.arange(16, dtype=dtype, device=device)
                x = base[offset : offset + 6]
                if layout == "strided":
                    x = x[::2]
                elif layout == "scalar":
                    x = x[0]
                degree = torch.full_like(x, 2.75)
                eager_calls = NativeCalls()
                with eager_calls:
                    expected = fn(x, degree)
                actual = compiled(x, degree)
                self.assertEqual(actual, expected, atol=0, rtol=0)
                self.assertEqual(len(eager_calls.calls), 1)
                self.assertEqual(compiled_calls, eager_calls.calls)
                compiled_calls.clear()
        self.assertEqual(len(compilations), 1)

    @dtypes(torch.bool, torch.float32, torch.float64, torch.int64, torch.uint64)
    @parametrize("op_name", ("laguerre_polynomial_l", "legendre_polynomial_p"))
    @parametrize("layout", ("contiguous", "strided", "broadcast"))
    @parametrize("parameter_kind", ("tensor", "scalar", "cpu_scalar"))
    @parametrize("numerics", ("default", "strict"))
    def test_native_arguments(
        self, device, dtype, op_name, layout, parameter_kind, numerics
    ):
        x = torch.tensor([[0, 1, 2], [3, 4, 5]], device=device, dtype=dtype)
        degree = torch.tensor(
            [[1, 2, 3], [1, 2, 3]], device=device, dtype=torch.float32
        )
        if layout == "strided":
            base = torch.empty((2, 6), device=device, dtype=dtype)
            base[:, ::2] = x
            x = base[:, ::2]
        elif layout == "broadcast":
            x = x[:1]
            degree = degree[:, :1]
        if parameter_kind == "scalar":
            degree = 2.75
        elif parameter_kind == "cpu_scalar":
            degree = torch.tensor(2.75, dtype=torch.float64)

        def fn(x, degree):
            return getattr(torch.special, op_name)(x, degree)

        eager_calls = NativeCalls()
        with eager_calls:
            expected = fn(x, degree)
        compiled_calls = []

        def backend(gm, inputs):
            compiled = lookup_backend("inductor")(gm, inputs)

            def execute(*args):
                calls = NativeCalls()
                with calls:
                    output = compiled(*args)
                compiled_calls.extend(calls.calls)
                return output

            return execute

        torch._dynamo.reset()
        with config.patch(
            numerics=numerics, fx_graph_cache=False, fx_graph_remote_cache=False
        ):
            actual = torch.compile(fn, backend=backend, fullgraph=True)(x, degree)
        self.assertEqual(actual, expected, atol=0, rtol=0)
        self.assertEqual(len(eager_calls.calls), 1)
        self.assertEqual(compiled_calls, eager_calls.calls)

    def test_custom_callback_receives_promoted_arguments(self, device):
        x = torch.tensor([[0, 1, 2], [3, 4, 5]], device=device, dtype=torch.int64)
        degree = torch.tensor([[1, 2, 3]], device=device, dtype=torch.float32)
        op = torch.ops.aten.special_laguerre_polynomial_l.default
        received = []

        def callback(*args):
            received.extend((arg.get_dtype(), tuple(arg.get_size())) for arg in args)
            return lowering.fallback_handler(op, add_to_fallback_set=False)(*args)

        def backend(gm, inputs):
            with patch.dict(lowering.lowerings):
                lowering.register_pointwise(
                    op,
                    name="laguerre_polynomial_l",
                    triton_fallback=callback,
                    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
                )
                return lookup_backend("inductor")(gm, inputs)

        torch._dynamo.reset()
        with config.patch(fx_graph_cache=False, fx_graph_remote_cache=False):
            actual = torch.compile(op, backend=backend, fullgraph=True)(x, degree)
        self.assertEqual(actual, op(x, degree), atol=0, rtol=0)
        self.assertEqual(received, [(torch.float32, (2, 3)), (torch.float32, (2, 3))])


class TestNativePointwiseNumerics(TestCase):
    hw_classification = HardwareClassification.CUDA

    @dtypes(torch.float32, torch.float64, torch.int64)
    @parametrize("op_name", ("laguerre_polynomial_l", "legendre_polynomial_p"))
    @parametrize("numerics", ("default", "strict"))
    def test_pointwise_numerics(self, device, dtype, op_name, numerics):
        x = torch.tensor([[0, 1, 2], [3, 4, 5]], device=device, dtype=dtype).t()
        degree = torch.tensor([1, 2], device=device, dtype=torch.float32)
        calls = NativeCalls()

        def fn(x, degree):
            return getattr(torch.special, op_name)(x, degree)

        def backend(gm, inputs):
            compiled = lookup_backend("inductor")(gm, inputs)

            def execute(*args):
                with calls:
                    return compiled(*args)

            return execute

        torch._dynamo.reset()
        with config.patch(
            numerics=numerics, fx_graph_cache=False, fx_graph_remote_cache=False
        ):
            actual = torch.compile(fn, backend=backend, fullgraph=True)(x, degree)
        self.assertEqual(actual, fn(x, degree), atol=0, rtol=0)
        if torch.device(device).type == "cpu":
            self.assertEqual(calls.calls, [])
        else:
            self.assertEqual(len(calls.calls), 1)


instantiate_device_type_tests(TestNativePointwise, globals(), only_for=("cuda",))
instantiate_device_type_tests(
    TestNativePointwiseNumerics, globals(), only_for=("cpu", "cuda")
)

if __name__ == "__main__":
    run_tests()
