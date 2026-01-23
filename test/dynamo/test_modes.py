# Owner(s): ["module: dynamo"]

import operator
from unittest.mock import patch

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._C import (
    _len_torch_function_stack,
    _pop_torch_function_stack,
    _push_on_torch_function_stack,
)
from torch._dynamo.testing import normalize_gm
from torch._dynamo.utils import counters
from torch.overrides import (
    _get_current_function_mode_stack,
    BaseTorchFunctionMode,
    TorchFunctionMode,
)
from torch.testing._internal.common_utils import skipIfXpu
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.triton_utils import requires_gpu
from torch.utils._device import DeviceContext
from torch.utils._python_dispatch import TorchDispatchMode


device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)


class TestMode(BaseTorchFunctionMode):
    def __torch_function__(self, func, types, args, kwargs=None):
        if not kwargs:
            kwargs = {}

        if func == torch.add:
            return torch.zeros(2, 2)

        return super().__torch_function__(func, types, args, kwargs)


class HopDetectionError(Exception):
    pass


class TestModeRaises(BaseTorchFunctionMode):
    def __torch_function__(self, func, types, args, kwargs=None):
        if not kwargs:
            kwargs = {}

        import torch._higher_order_ops

        if func == torch._higher_order_ops.flex_attention:
            raise HopDetectionError("test")

        return super().__torch_function__(func, types, args, kwargs)


class TorchDispatchModeTests(torch._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def test_torch_dispatch_ignore_compile_internals(self):
        counters.clear()
        from torch.utils._python_dispatch import TorchDispatchMode

        @torch.library.custom_op("mylib::modes_checksum", mutates_args=())
        def foo(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        def checksum(x):
            return x.abs().sum()

        _checksums = []

        class ChecksumFoo(TorchDispatchMode):
            @classmethod
            def ignore_compile_internals(cls):
                return True

            def __init__(self) -> None:
                super().__init__()

            def __torch_dispatch__(self, func, types, args, kwargs=None):
                kwargs = kwargs or {}

                if func is torch.ops.mylib.modes_checksum.default:
                    # Do some compute, smoketest to see if there's a bad interaction
                    _checksums.append(args[0].abs().sum())

                return func(*args, **kwargs)

        # test e2e, with Inductor, as smoketest.
        @torch._dynamo.error_on_graph_break(True)
        @torch.compile(backend="inductor")
        def g(x):
            return 2 * x.sin().cos()

        x = torch.randn(3)

        with ChecksumFoo():
            foo(x)
            g(x)
            foo(x)

        self.assertEqual(len(_checksums), 2)
        # The correct result here is 1: Dynamo should capture the `g` frame.
        self.assertEqual(counters["frames"]["total"], 1)
        self.assertEqual(counters["frames"]["ok"], 1)

    def test_skip_torch_dispatch_modes(self):
        class RewriteAddToMul(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func is torch.ops.aten.add.Tensor:
                    func = torch.ops.aten.mul.Tensor
                return func(*args, **kwargs)

        def fn(x):
            return x + x

        cnt = torch._dynamo.testing.CompileCounter()

        x = torch.tensor([3.0])
        with RewriteAddToMul():
            eager_res = fn(x)
            compiled_res = torch.compile(fn, backend=cnt)(x)

        self.assertEqual(eager_res, compiled_res)
        self.assertEqual(cnt.frame_count, 0)


class TorchFunctionModeTests(torch._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.default_device_old = torch.get_default_device()
        except AttributeError:
            cls.default_device_old = torch.device("cpu")
        global_default_ctx = getattr(
            getattr(torch, "_GLOBAL_DEVICE_CONTEXT", None), "device_context", None
        )
        cls._had_global_default_device = global_default_ctx is not None
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        if cls._had_global_default_device:
            torch.set_default_device(cls.default_device_old)
        super().tearDownClass()

    def setUp(self):
        torch.set_default_device(None)
        torch._dynamo.reset()

    def tearDown(self):
        torch.set_default_device(None)
        torch._dynamo.reset()

    def _run_torch_function_mode_guard_test(self):
        class TestMode1(BaseTorchFunctionMode):
            pass

        class TestMode2(BaseTorchFunctionMode):
            pass

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt.__call__)
        def fn(x):
            return x + 1

        inp = torch.ones(2, 2)
        fn(inp)
        self.assertEqual(cnt.frame_count, 1)

        with TestMode1():
            fn(inp)
        self.assertEqual(cnt.frame_count, 2)

        with TestMode1(), TestMode2():
            fn(inp)
        self.assertEqual(cnt.frame_count, 3)

        with TestMode2(), TestMode1():
            fn(inp)
        self.assertEqual(cnt.frame_count, 4)

        with TestMode1():
            fn(inp)
        self.assertEqual(cnt.frame_count, 4)

    @torch._dynamo.config.patch("enable_cpp_guard_manager", False)
    def test_torch_function_mode_guards_py(self):
        self._run_torch_function_mode_guard_test()

    def test_torch_function_mode_guards_cpp(self):
        self._run_torch_function_mode_guard_test()

    @requires_gpu
    def test_torch_function_mode_preserves_cuda_rng_state(self):
        class ConstantReturnMode(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return -42

        @torch._dynamo.optimize("eager")
        def fn():
            with ConstantReturnMode():
                return 123

        self.assertEqual(fn(), 123)

    def test_stack_state_mutation_default_device(self):
        m = BaseTorchFunctionMode()
        m1 = BaseTorchFunctionMode()
        with m, m1:

            @torch.compile(fullgraph=True)
            def fn(x):
                torch.set_default_device("cpu")
                _pop_torch_function_stack()

            fn(torch.ones(2, 2))
            _push_on_torch_function_stack(m1)

            stack = _get_current_function_mode_stack()
            self.assertIsInstance(stack[0], DeviceContext)
            self.assertEqual(stack[0].device, torch.device("cpu"))
            self.assertIs(stack[1], m)
            self.assertIs(stack[2], m1)

    def test_stack_state_clear_default_device(self):
        @torch.compile(fullgraph=True)
        def fn(x):
            torch.set_default_device(None)
            return x + 1

        fn(torch.ones(2, 2))
        stack = _get_current_function_mode_stack()
        self.assertEqual(len(stack), 0)

        m = BaseTorchFunctionMode()
        m1 = BaseTorchFunctionMode()

        # Stack populated, add device
        with m, m1:

            @torch.compile(fullgraph=True)
            def fn(x):
                torch.set_default_device("cpu")
                torch.set_default_device(None)
                torch.set_default_device("cpu")
                return x + 1

            fn(torch.ones(2, 2))
            stack = _get_current_function_mode_stack()
            self.assertEqual(stack[0].device, torch.device("cpu"))
            self.assertIs(stack[1], m)
            self.assertIs(stack[2], m1)

        # Stack populated, remove device
        torch.set_default_device("cpu")
        with m, m1:

            @torch.compile(fullgraph=True)
            def fn(x):
                torch.set_default_device(None)
                return x + 1

            fn(torch.ones(2, 2))
            stack = _get_current_function_mode_stack()
            self.assertIs(stack[0], m)
            self.assertIs(stack[1], m1)

        @torch.compile(fullgraph=True)
        def fn(x):
            torch.set_default_device("cpu")
            torch.set_default_device("cpu")
            return x + 1

        fn(torch.ones(2, 2))
        stack = _get_current_function_mode_stack()
        self.assertEqual(stack[0].device, torch.device("cpu"))
        torch.set_default_device(None)

    def test_pop_torch_function_mode(self):
        m = BaseTorchFunctionMode()
        with m:

            @torch.compile(fullgraph=True)
            def fn(x):
                _pop_torch_function_stack()
                return x + 1

            fn(torch.ones(2, 2))

            self.assertEqual(_len_torch_function_stack(), 0)
            # reset stack so __exit__ doesn't crash
            _push_on_torch_function_stack(m)

        self.assertEqual(_len_torch_function_stack(), 0)

    def test_is_torch_function_all_disabled(self):
        @torch.compile(fullgraph=True)
        def fn(x):
            return (
                torch._C._is_torch_function_all_disabled(),
                torch.add(x, 1.0),
            )

        input = torch.ones(2, 2)
        res, _ = fn(input)
        self.assertFalse(res)

    def test_error_empty_stack_pop_torch_function_mode(self):
        @torch.compile(fullgraph=True)
        def fn(x):
            _pop_torch_function_stack()
            return x + 1

        self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Attempted to pop from empty torch function mode stack",
            lambda: fn(torch.ones(2, 2)),
        )

    def test_push_torch_function_mode(self):
        m = BaseTorchFunctionMode()
        with m:

            @torch.compile(fullgraph=True)
            def fn(x, m):
                _push_on_torch_function_stack(m)
                return x + 1

            fn(torch.ones(2, 2), m)

            self.assertEqual(_len_torch_function_stack(), 2)
            # reset stack state
            _pop_torch_function_stack()

        self.assertEqual(_len_torch_function_stack(), 0)

    def test_len_torch_function_mode(self):
        m = BaseTorchFunctionMode()
        with m:

            @torch.compile(fullgraph=True)
            def fn(x):
                z = _len_torch_function_stack()
                return x + z

            res = fn(torch.ones(2, 2))
            self.assertEqual(res, torch.ones(2, 2) + 1)
            self.assertEqual(_len_torch_function_stack(), 1)

    def test_intermedate_torch_function_mode_construction_mutation(self):
        class TestMode(BaseTorchFunctionMode):
            def __init__(self, x):
                self.x = x

        @torch.compile(fullgraph=True)
        def fn(x):
            z = TestMode(2)
            z.y = 2
            return x + 1, z

        fn(torch.ones(2, 2))

    def test_torch_function_mode_enabled_guard(self):
        cnt = torch._dynamo.testing.CompileCounter()
        inp = torch.ones(2, 2)

        @torch.compile(backend=cnt.__call__)
        def fn(x):
            return x + 1

        with BaseTorchFunctionMode(), torch._C.DisableTorchFunctionSubclass():
            with torch._C.DisableTorchFunction():
                fn(inp)
            fn(inp)
        self.assertEqual(cnt.frame_count, 2)

    def test_nested_torch_function_mode(self):
        mode_1_called = False
        mode_2_called = False

        def reset_state():
            nonlocal mode_1_called
            nonlocal mode_2_called
            mode_1_called = False
            mode_2_called = False

        ones = torch.ones(2, 2)
        zeros = torch.zeros(2, 2)

        class TestMode1(BaseTorchFunctionMode):
            def __torch_function__(self, func, types, args, kwargs=None):
                if not kwargs:
                    kwargs = {}

                nonlocal mode_1_called

                mode_1_called = True

                if func == torch.add:
                    return zeros

                return super().__torch_function__(func, types, args, kwargs)

        class TestMode2(BaseTorchFunctionMode):
            def __torch_function__(self, func, types, args, kwargs=None):
                if not kwargs:
                    kwargs = {}

                nonlocal mode_2_called

                mode_2_called = True

                if func == torch.mul:
                    return ones

                return super().__torch_function__(func, types, args, kwargs)

        def fn(x):
            return torch.add(x, 3)

        def fn_2(x):
            return torch.mul(x, 3) + torch.add(x, 3)

        inp = torch.ones(2, 2) + 1

        for fn_i in [fn, fn_2]:
            fn_opt = torch.compile(fn_i, fullgraph=True)
            with TestMode1(), TestMode2():
                expected = fn_i(inp), mode_1_called, mode_2_called
                reset_state()
                actual = fn_opt(inp), mode_1_called, mode_2_called
                reset_state()

            self.assertEqual(expected, actual)

    def test_torch_function_mode_disable(self):
        class TestSubclass(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args, kwargs=None):
                if not kwargs:
                    kwargs = {}
                if func == torch.add:
                    return torch.ones(2, 2)
                return super().__torch_function__(func, types, args, kwargs)

        class TestMode(BaseTorchFunctionMode):
            def __torch_function__(self, func, types, args, kwargs=None):
                if not kwargs:
                    kwargs = {}

                if func == torch.add:
                    return torch.zeros(2, 2)

                return super().__torch_function__(func, types, args, kwargs)

        def fn(x):
            return torch.add(x, 3)

        inp = (torch.ones(2, 2) + 1).as_subclass(TestSubclass)

        fn_opt = torch.compile(fn, fullgraph=True)
        with TestMode():
            with torch._C.DisableTorchFunctionSubclass():
                expected = fn(inp)
                actual = fn_opt(inp)

            self.assertEqual(expected, actual)

            with torch._C.DisableTorchFunction():
                expected = fn(inp)
                actual = fn_opt(inp)

            self.assertEqual(expected, actual)

    def test_torch_function_mode_highest_priority(self):
        class TestSubclass(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args, kwargs=None):
                if not kwargs:
                    kwargs = {}
                if func == torch.add:
                    return torch.ones(2, 2)
                return super().__torch_function__(func, types, args, kwargs)

        def fn(x):
            return torch.add(x, 3)

        inp = (torch.ones(2, 2) + 1).as_subclass(TestSubclass)

        fn_opt = torch.compile(fn, fullgraph=True)
        with TestMode():
            expected = fn(inp)
            actual = fn_opt(inp)

        self.assertEqual(expected, actual)

    def test_torch_function_mode_enter_exit(self):
        def fn(x, y):
            with TestMode():
                o = torch.add(x, 3)

            return torch.add(o, y)

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2) + 2)
        fn_opt = torch.compile(fn, fullgraph=True)

        expected = fn(*inp)
        actual = fn_opt(*inp)

        self.assertEqual(expected, actual)

    def test_torch_function_mode_graph_break(self):
        def fn(x, y):
            with TestMode():
                torch._dynamo.graph_break()
                o = torch.add(x, 3)

            return torch.add(o, y)

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2) + 2)
        fn_opt = torch.compile(fn)

        expected = fn(*inp)
        actual = fn_opt(*inp)

        self.assertEqual(expected, actual)

    def test_torch_function_mode_and_pop_graph_break(self):
        def fn(x, y):
            with TestMode():
                z = _pop_torch_function_stack()
                torch._dynamo.graph_break()
                _push_on_torch_function_stack(z)
                o = torch.add(x, 3)

            return torch.add(o, y)

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2) + 2)
        fn_opt = torch.compile(fn)

        expected = fn(*inp)
        actual = fn_opt(*inp)

        self.assertEqual(expected, actual)

    def test_torch_function_mode_restore_on_exc(self):
        @torch._dynamo.disable()
        def err():
            raise RuntimeError("test")

        @torch.compile()
        def fn(x):
            with TestMode():
                x += 1
                err()
                x += 2
                return x

        try:
            fn(torch.ones(2, 2))
        except RuntimeError:
            pass
        self.assertEqual(_len_torch_function_stack(), 0)

    def test_torch_function_mode_and_pop_graph_break_mutation(self):
        def fn(x, y):
            with TestMode():
                z = _pop_torch_function_stack()
                z.y = 5
                torch._dynamo.graph_break()
                _push_on_torch_function_stack(z)
                o = torch.add(x, 3)
                o = torch.mul(o, z.y)

            return torch.add(o, y)

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2) + 2)
        fn_opt = torch.compile(fn)

        expected = fn(*inp)
        actual = fn_opt(*inp)

        self.assertEqual(expected, actual)

    # Needs larger cache size since we recompile for each op
    @patch.object(torch._dynamo.config, "recompile_limit", 48)
    def test_builtin_equivalent_funcs(self):
        from torch._dynamo.variables.builtin import (
            BUILTIN_TO_TENSOR_FN_MAP,
            BUILTIN_TO_TENSOR_RFN_MAP,
        )
        from torch._dynamo.variables.torch_function import (
            bin_int_ops,
            bin_ops,
            tensor_and_int_ops,
            un_int_ops,
            un_ops,
        )

        expected_func = None
        valid = False

        class FuncEquivMode(BaseTorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                nonlocal expected_func
                nonlocal valid
                if not kwargs:
                    kwargs = {}
                if torch._dynamo.is_compiling():
                    valid = expected_func == func
                return super().__torch_function__(func, types, args, kwargs)

        inp0 = torch.ones(1, 1)
        inp1 = torch.ones(1, 1)
        inp0_int = torch.ones(1, 1, dtype=torch.int32)
        inp1_int = torch.ones(1, 1, dtype=torch.int32)

        @torch.compile(fullgraph=True)
        def fn_un(op, inp):
            return op(inp)

        @torch.compile(fullgraph=True)
        def fn_un_int(op, inp):
            return op(inp)

        @torch.compile(fullgraph=True)
        def fn_bin(op, inp0, inp1):
            return op(inp0, inp1)

        @torch.compile(fullgraph=True)
        def fn_bin_int(op, inp0, inp1):
            return op(inp0, inp1)

        @torch.compile(fullgraph=True)
        def fn_tensor_and_int(op, inp0, inp1):
            return op(inp0, inp1)

        setups_and_oplists = [
            (lambda o: fn_un(o, inp0), un_ops),
            (lambda o: fn_un_int(o, inp0_int), un_int_ops),
            (lambda o: fn_bin(o, inp0, inp1), bin_ops),
            (lambda o: fn_bin_int(o, inp0_int, inp1_int), bin_int_ops),
            (lambda o: fn_tensor_and_int(o, inp0_int, 0), tensor_and_int_ops),
        ]

        # gather the reverse functions
        rsetups_and_oplists = [
            (
                lambda o: fn_bin(o, 1, inp1),
                bin_ops,
            ),  # Get r* ops, (ex. __sub__(int, Tensor) -> __rsub__(Tensor, int))
            (lambda o: fn_bin_int(o, 1, inp1_int), bin_int_ops),
            (lambda o: fn_tensor_and_int(o, 0, inp0_int), tensor_and_int_ops),
        ]

        skips = {operator.not_}  # Has local scalar dense call which graph breaks
        rskips = {
            operator.matmul,
            operator.imatmul,
            operator.getitem,
        }  # Doesn't type check with reversed args

        def run_checks(setups_and_oplists, skips, ref_map):
            nonlocal valid
            nonlocal expected_func
            for setup_fn, op_list in setups_and_oplists:
                for op in op_list:
                    if op in skips or op not in ref_map:
                        continue
                    with FuncEquivMode():
                        expected_func = ref_map[op]
                        setup_fn(op)
                        self.assertTrue(valid)

                    expected_func = None
                    valid = False

        run_checks(setups_and_oplists, skips, BUILTIN_TO_TENSOR_FN_MAP)
        run_checks(rsetups_and_oplists, rskips, BUILTIN_TO_TENSOR_RFN_MAP)

    def test_expand(self):
        from torch.distributions import (
            AffineTransform,
            ComposeTransform,
            Normal,
            TanhTransform,
            TransformedDistribution,
        )

        # https://github.com/pytorch/pytorch/issues/141232
        with torch.device("cpu"):

            @torch.compile(fullgraph=True)
            def func(a):
                d = TransformedDistribution(
                    Normal(a, 1),
                    ComposeTransform([TanhTransform(), AffineTransform(2, 2)]),
                )
                b = d.log_prob(d.rsample((10,)))
                return b

            func(torch.randn(3))

    @requires_gpu
    def test_flex_attention(self):
        import torch
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        torch.set_default_device(device_type)

        flex_attention = torch.compile(flex_attention, dynamic=False)

        prefix_lengths = torch.arange(8)

        def prefix_lm(b, h, q, kv):
            return prefix_lengths[b] >= kv

        # This runs in fullgraph already
        create_block_mask(
            prefix_lm, 8, None, 512, 512, _compile=True, device=device_type
        )

    def test_register_hook(self):
        import functools

        def my_hook(grad, *, k=0):
            return grad + k

        hook = functools.partial(my_hook, k=3)

        class MyMod(torch.nn.Module):
            def forward(self, x):
                x.register_hook(hook)
                y = x.mul(2)
                z = y.mul(3)
                return (z,)

        mod = MyMod()
        x = torch.ones(4, requires_grad=True)

        with torch.device("cpu"):
            torch.compile(mod, fullgraph=True)(x)

    @requires_gpu
    @skipIfXpu(msg="XPU does not support flex attention")
    def test_hop(self):
        import torch
        import torch._higher_order_ops
        from torch.nn.attention.flex_attention import (
            flex_attention as flex_attention_eager,
        )

        with torch.device(GPU_TYPE):
            flex_attention = torch.compile(flex_attention_eager, dynamic=False)

            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported,
                "raised exception HopDetectionError([ConstantVariable(str: 'test')])",
            ):
                # This runs in fullgraph already
                with TestModeRaises():
                    flex_attention(
                        torch.ones(2, 2, 2, 2),
                        torch.ones(2, 2, 2, 2),
                        torch.ones(2, 2, 2, 2),
                    )

    @requires_gpu
    @skipIfXpu(msg="XPU does not support flex attention")
    def test_hop_eager(self):
        import torch
        import torch._higher_order_ops
        from torch.nn.attention.flex_attention import (
            flex_attention as flex_attention_eager,
        )

        with torch.device(GPU_TYPE):
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported,
                "raised exception HopDetectionError([ConstantVariable(str: 'test')])",
            ):
                with TestModeRaises():
                    flex_attention_eager(
                        torch.ones(2, 2, 2, 2),
                        torch.ones(2, 2, 2, 2),
                        torch.ones(2, 2, 2, 2),
                    )


class InvokeSubgraphBackendTests(torch._dynamo.test_case.TestCase):
    @torch._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_make_fx_over_compiled_function(self):
        """Test that make_fx can trace over torch.compile'd functions using invoke_subgraph backend.

        When force_compile_during_fx_trace=True, the invoke_subgraph backend should
        emit an invoke_subgraph HOP in the traced graph instead of inlining the subgraph.
        """
        from torch.fx.experimental.proxy_tensor import make_fx

        torch._dynamo.reset()  # Clear any cached graphs

        def simple_fn(x, y):
            return x * 2 + y

        compiled_fn = torch.compile(simple_fn, backend="invoke_subgraph")

        def outer_fn(x, y):
            z = x + 1
            result = compiled_fn(z, y)
            return result * 2

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        # Trace with make_fx - the compiled_fn should appear as invoke_subgraph HOP
        traced = make_fx(
            outer_fn, tracing_mode="fake", _disable_torch_fn_metadata_mode=True
        )(x, y)

        self.assertExpectedInline(
            normalize_gm(traced.print_readable(print_output=False)),
            """\
class outer_fn(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]", y_1: "f32[3, 3]"):
        add: "f32[3, 3]" = torch.ops.aten.add.Tensor(x_1, 1);  x_1 = None
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', add, y_1);  repeated_subgraph0 = add = y_1 = None
        getitem: "f32[3, 3]" = invoke_subgraph[0];  invoke_subgraph = None
        mul: "f32[3, 3]" = torch.ops.aten.mul.Tensor(getitem, 2);  getitem = None
        return mul

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]", arg1_1: "f32[3, 3]"):
            mul: "f32[3, 3]" = torch.ops.aten.mul.Tensor(arg0_1, 2);  arg0_1 = None
            add: "f32[3, 3]" = torch.ops.aten.add.Tensor(mul, arg1_1);  mul = arg1_1 = None
            return (add,)
""",  # noqa: B950
        )

    @torch._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_same_compiled_fn_called_twice_shares_subgraph(self):
        """Test that calling the same compiled function twice uses the same subgraph.

        When the same compiled function is called multiple times with inputs that
        don't cause guard failures, both calls should reference the same subgraph.
        """
        from torch._guards import tracing, TracingContext
        from torch.fx.experimental.proxy_tensor import make_fx

        torch._dynamo.reset()

        def simple_fn(x):
            return x * 2

        compiled_fn = torch.compile(simple_fn, backend="invoke_subgraph")

        def outer_fn(x, y):
            # Call the same compiled function twice
            a = compiled_fn(x)
            b = compiled_fn(y)
            return a + b

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        # Set up TracingContext so invoke_subgraph cache works
        tracing_ctx = TracingContext(fake_mode=None)
        with tracing(tracing_ctx):
            traced = make_fx(
                outer_fn, tracing_mode="fake", _disable_torch_fn_metadata_mode=True
            )(x, y)

        self.assertExpectedInline(
            normalize_gm(traced.print_readable(print_output=False)),
            """\
class outer_fn(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]", y_1: "f32[3, 3]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', x_1);  repeated_subgraph0 = x_1 = None
        getitem: "f32[3, 3]" = invoke_subgraph[0];  invoke_subgraph = None
        repeated_subgraph0_1 = self.repeated_subgraph0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0_1, 'invoke_subgraph_0', y_1);  repeated_subgraph0_1 = y_1 = None
        getitem_1: "f32[3, 3]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        add: "f32[3, 3]" = torch.ops.aten.add.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
        return add

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]"):
            mul: "f32[3, 3]" = torch.ops.aten.mul.Tensor(arg0_1, 2);  arg0_1 = None
            return (mul,)
""",  # noqa: B950
        )

    @torch._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_guard_failure_creates_separate_subgraphs(self):
        """Test that guard failures create separate subgraphs.

        When the same compiled function is called with inputs that cause guard
        failures (e.g., different bool values), each compilation should result
        in a separate invoke_subgraph with a different identifier.
        """
        from torch.fx.experimental.proxy_tensor import make_fx

        torch._dynamo.reset()

        def conditional_fn(x, flag: bool):
            if flag:
                return x * 2
            else:
                return x * 3

        compiled_fn = torch.compile(conditional_fn, backend="invoke_subgraph")

        def outer_fn(x):
            # Call with flag=True, then flag=False - should trigger recompilation
            a = compiled_fn(x, True)
            b = compiled_fn(x, False)
            return a + b

        x = torch.randn(3, 3)

        traced = make_fx(
            outer_fn, tracing_mode="fake", _disable_torch_fn_metadata_mode=True
        )(x)

        self.assertExpectedInline(
            normalize_gm(traced.print_readable(print_output=False)),
            """\
class outer_fn(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', x_1);  repeated_subgraph0 = None
        getitem: "f32[3, 3]" = invoke_subgraph[0];  invoke_subgraph = None
        repeated_subgraph1 = self.repeated_subgraph1
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(repeated_subgraph1, 'invoke_subgraph_1', x_1);  repeated_subgraph1 = x_1 = None
        getitem_1: "f32[3, 3]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        add: "f32[3, 3]" = torch.ops.aten.add.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
        return add

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]"):
            mul: "f32[3, 3]" = torch.ops.aten.mul.Tensor(arg0_1, 2);  arg0_1 = None
            return (mul,)

    class repeated_subgraph1(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]"):
            mul: "f32[3, 3]" = torch.ops.aten.mul.Tensor(arg0_1, 3);  arg0_1 = None
            return (mul,)
""",  # noqa: B950
        )

    @torch._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_multiple_inputs(self):
        """Test invoke_subgraph with multiple tensor inputs.

        Verifies that the invoke_subgraph HOP correctly handles functions
        that take more than 2 tensor inputs.
        """
        from torch.fx.experimental.proxy_tensor import make_fx

        torch._dynamo.reset()

        def multi_input_fn(a, b, c, d):
            return a * b + c * d

        compiled_fn = torch.compile(multi_input_fn, backend="invoke_subgraph")

        def outer_fn(w, x, y, z):
            return compiled_fn(w, x, y, z)

        w = torch.randn(3, 3)
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.randn(3, 3)

        traced = make_fx(
            outer_fn, tracing_mode="fake", _disable_torch_fn_metadata_mode=True
        )(w, x, y, z)

        self.assertExpectedInline(
            normalize_gm(traced.print_readable(print_output=False)),
            """\
class outer_fn(torch.nn.Module):
    def forward(self, w_1: "f32[3, 3]", x_1: "f32[3, 3]", y_1: "f32[3, 3]", z_1: "f32[3, 3]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', w_1, x_1, y_1, z_1);  repeated_subgraph0 = w_1 = x_1 = y_1 = z_1 = None
        getitem: "f32[3, 3]" = invoke_subgraph[0];  invoke_subgraph = None
        return getitem

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]", arg1_1: "f32[3, 3]", arg2_1: "f32[3, 3]", arg3_1: "f32[3, 3]"):
            mul: "f32[3, 3]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
            mul_1: "f32[3, 3]" = torch.ops.aten.mul.Tensor(arg2_1, arg3_1);  arg2_1 = arg3_1 = None
            add: "f32[3, 3]" = torch.ops.aten.add.Tensor(mul, mul_1);  mul = mul_1 = None
            return (add,)
""",  # noqa: B950
        )

    @torch._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_multiple_outputs(self):
        """Test invoke_subgraph with multiple tensor outputs.

        Verifies that the invoke_subgraph HOP correctly handles functions
        that return multiple tensors.
        """
        from torch.fx.experimental.proxy_tensor import make_fx

        torch._dynamo.reset()

        def multi_output_fn(x, y):
            return x + y, x - y, x * y

        compiled_fn = torch.compile(multi_output_fn, backend="invoke_subgraph")

        def outer_fn(a, b):
            sum_out, diff_out, prod_out = compiled_fn(a, b)
            return sum_out * diff_out + prod_out

        a = torch.randn(3, 3)
        b = torch.randn(3, 3)

        traced = make_fx(
            outer_fn, tracing_mode="fake", _disable_torch_fn_metadata_mode=True
        )(a, b)

        self.assertExpectedInline(
            normalize_gm(traced.print_readable(print_output=False)),
            """\
class outer_fn(torch.nn.Module):
    def forward(self, a_1: "f32[3, 3]", b_1: "f32[3, 3]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', a_1, b_1);  repeated_subgraph0 = a_1 = b_1 = None
        getitem: "f32[3, 3]" = invoke_subgraph[0]
        getitem_1: "f32[3, 3]" = invoke_subgraph[1]
        getitem_2: "f32[3, 3]" = invoke_subgraph[2];  invoke_subgraph = None
        mul: "f32[3, 3]" = torch.ops.aten.mul.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
        add: "f32[3, 3]" = torch.ops.aten.add.Tensor(mul, getitem_2);  mul = getitem_2 = None
        return add

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]", arg1_1: "f32[3, 3]"):
            add: "f32[3, 3]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1)
            sub: "f32[3, 3]" = torch.ops.aten.sub.Tensor(arg0_1, arg1_1)
            mul: "f32[3, 3]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
            return (add, sub, mul)
""",  # noqa: B950
        )

    @torch._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_multiple_inputs_and_outputs(self):
        """Test invoke_subgraph with both multiple inputs and outputs.

        Verifies that the invoke_subgraph HOP correctly handles functions
        that have both multiple tensor inputs and multiple tensor outputs.
        """
        from torch.fx.experimental.proxy_tensor import make_fx

        torch._dynamo.reset()

        def multi_io_fn(a, b, c):
            # Multiple inputs, multiple outputs
            return a + b + c, a * b * c

        compiled_fn = torch.compile(multi_io_fn, backend="invoke_subgraph")

        def outer_fn(x, y, z):
            sum_out, prod_out = compiled_fn(x, y, z)
            return sum_out - prod_out

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.randn(3, 3)

        traced = make_fx(
            outer_fn, tracing_mode="fake", _disable_torch_fn_metadata_mode=True
        )(x, y, z)

        self.assertExpectedInline(
            normalize_gm(traced.print_readable(print_output=False)),
            """\
class outer_fn(torch.nn.Module):
    def forward(self, x_1: "f32[3, 3]", y_1: "f32[3, 3]", z_1: "f32[3, 3]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'invoke_subgraph_0', x_1, y_1, z_1);  repeated_subgraph0 = x_1 = y_1 = z_1 = None
        getitem: "f32[3, 3]" = invoke_subgraph[0]
        getitem_1: "f32[3, 3]" = invoke_subgraph[1];  invoke_subgraph = None
        sub: "f32[3, 3]" = torch.ops.aten.sub.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
        return sub

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[3, 3]", arg1_1: "f32[3, 3]", arg2_1: "f32[3, 3]"):
            add: "f32[3, 3]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1)
            add_1: "f32[3, 3]" = torch.ops.aten.add.Tensor(add, arg2_1);  add = None
            mul: "f32[3, 3]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
            mul_1: "f32[3, 3]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
            return (add_1, mul_1)
""",  # noqa: B950
        )

    @torch._dynamo.config.patch(force_compile_during_fx_trace=True)
    def test_aot_autograd_over_dynamo_with_requires_grad(self):
        """Test AOTAutograd tracing over a torch.compile'd function with requires_grad inputs.

        This tests the scenario where:
        1. An outer aot_function traces a function with requires_grad inputs
        2. Inside that function, a torch.compile'd function with invoke_subgraph backend is called
        3. AOTAutograd partitions into forward/backward graphs
        4. The inner Dynamo region should only be compiled once
        """
        from torch._dynamo.testing import CompileCounterWithBackend
        from torch._functorch.aot_autograd import aot_function
        from torch._functorch.compilers import nop

        torch._dynamo.reset()

        # Use a compile counter to track how many times Dynamo compiles
        compile_counter = CompileCounterWithBackend("invoke_subgraph")

        def inner_fn(x):
            return x * 2 + 1

        compiled_fn = torch.compile(inner_fn, backend=compile_counter)

        def outer_fn(x):
            y = x + 1
            z = compiled_fn(y)
            return z.sum()

        x = torch.randn(3, 3, requires_grad=True)

        # Track forward graph to verify invoke_subgraph appears
        fw_graph = None

        def fw_compiler(gm, example_inputs):
            nonlocal fw_graph
            fw_graph = gm
            return gm

        aot_fn = aot_function(
            outer_fn,
            fw_compiler=fw_compiler,
            bw_compiler=nop,
            _disable_torch_fn_metadata_mode=True,
        )

        # Run forward and backward
        result = aot_fn(x)
        result.backward()

        # Check that we got a forward graph with invoke_subgraph
        self.assertIsNotNone(fw_graph, "Expected a forward graph")
        fw_graph_code = fw_graph.print_readable(print_output=False)
        self.assertIn("invoke_subgraph", fw_graph_code)

        # Check compile count - should be 1 (compiled once during tracing)
        self.assertEqual(
            compile_counter.frame_count,
            1,
            f"Expected 1 compilation, got {compile_counter.frame_count}",
        )


class TorchFunctionModeLifecycleTests(torch._dynamo.test_case.TestCase):
    def test_default_device_restored_after_mode_tests(self):
        case = TorchFunctionModeTests("test_stack_state_mutation_default_device")
        TorchFunctionModeTests.setUpClass()
        try:
            case.setUp()
            try:
                case.test_stack_state_mutation_default_device()
            finally:
                case.tearDown()
        finally:
            TorchFunctionModeTests.tearDownClass()

        stack = _get_current_function_mode_stack()
        self.assertFalse(any(isinstance(mode, DeviceContext) for mode in stack))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
