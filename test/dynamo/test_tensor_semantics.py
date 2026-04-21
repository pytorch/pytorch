# Owner(s): ["module: dynamo"]
# flake8: noqa: B001,B006,B020,B021,B950,C405,C416,E711,E721,E722,E731,F401,F403,F405,F541,F821,F823
# ruff: noqa: F403,F405,F841,PERF102,PIE808,SIM118,UP008
try:
    from .test_misc import *
except ImportError:
    from test_misc import *


class TensorSemanticsTests(torch._inductor.test_case.TestCase):
    def test_scalar_device_movement(self):
        if not torch._dynamo.config.assume_static_by_default:
            self.skipTest("Doesn't work with symints")

        def add_fn(a, b, out):
            res = torch.add(a, b, out=out)
            return res

        res = add_fn(2, 3, torch.tensor(0.0))
        add_fn = torch.compile(add_fn, backend="eager", fullgraph=True)
        res_compiled = add_fn(2, 3, torch.tensor(0.0))
        self.assertEqual(res, res_compiled)

    def test_inplace(self):
        def inplace1(a, b):
            o = torch.empty((10, 10))
            o.copy_(a)
            o -= b
            return o

        torch._dynamo.testing.standard_test(self, inplace1, 2, expected_ops=3)

    def test_inplace_desugaring(self):
        def inplace_on_literals(y):
            x0 = 1
            x0 += y
            x1 = 1
            x1 -= y
            return x0, x1

        torch._dynamo.testing.standard_test(
            self, inplace_on_literals, 1, expected_ops=2
        )

    def test_tensor_setattr_getset_descriptor(self):
        # Tensor attribute `real` has special getter/setter for complex dtype.
        def f(x):
            x.real = 10
            return x + 1

        opt_f = torch.compile(f, backend="eager", fullgraph=False)
        x = torch.ones(5, dtype=torch.cfloat)

        res = opt_f(x)
        ref = f(x)
        self.assertEqual(res, ref)

    def test_newly_constructed_tensor_attr_mutation(self):
        def f(x):
            y = x + 10
            y.grad = x
            y.foo = 42
            return y

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.ones(5)

        res = opt_f(x)
        ref = f(x)
        self.assertEqual(res, ref)
        self.assertEqual(res.grad, ref.grad)
        self.assertEqual(res.foo, ref.foo)

    def test_input_tensor_custom_attr_mutation(self):
        def f(x, flag):
            x.offloading_activation = flag
            return x + 1

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.ones(5)

        res = opt_f(x, True)
        self.assertEqual(res, torch.ones(5) + 1)
        self.assertTrue(x.offloading_activation)

    def test_intermediate_tensor_custom_attr_mutation(self):
        def f(x, flag):
            y = x + 1
            y.offloading_activation = flag
            return y

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.ones(5)

        res = opt_f(x, True)
        self.assertEqual(res, torch.ones(5) + 1)
        self.assertTrue(res.offloading_activation)

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
        "requires Hopper+ (SM >= 9.0) for TMA",
    )
    @unittest.skipIf(
        not torch.utils._triton.has_triton()
        or not hasattr(__import__("triton"), "set_allocator"),
        "requires triton with set_allocator support",
    )
    def test_triton_set_allocator_no_graph_break(self):
        """set_allocator inside torch.compile does not graph break and
        replays correctly at runtime (including cache hits)."""
        import triton
        import triton.language as tl
        from triton.runtime._allocation import NullAllocator

        @triton.jit
        def tma_copy_kernel(
            x_ptr,
            out_ptr,
            M,
            N,
            stride_m,
            stride_n,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
        ):
            pid = tl.program_id(0)
            desc = tl.make_tensor_descriptor(
                x_ptr,
                shape=[M, N],
                strides=[stride_m, stride_n],
                block_shape=[BLOCK_M, BLOCK_N],
            )
            block = tl.load_tensor_descriptor(desc, [pid * BLOCK_M, 0])
            out_desc = tl.make_tensor_descriptor(
                out_ptr,
                shape=[M, N],
                strides=[stride_m, stride_n],
                block_shape=[BLOCK_M, BLOCK_N],
            )
            tl.store_tensor_descriptor(out_desc, [pid * BLOCK_M, 0], block)

        M, N, BLOCK_M, BLOCK_N = 128, 64, 64, 64

        def run_kernel(x):
            out = torch.empty_like(x)
            tma_copy_kernel[(M // BLOCK_M,)](
                x,
                out,
                M,
                N,
                x.stride(0),
                x.stride(1),
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
            return out

        x = torch.randn(M, N, device="cuda")

        from contextlib import contextmanager

        from triton.runtime._allocation import _allocator

        @contextmanager
        def triton_allocator(allocator):
            prev = _allocator.get()
            triton.set_allocator(allocator)
            try:
                yield
            finally:
                triton.set_allocator(prev)

        def fn_with_set_allocator(x):
            triton.set_allocator(
                lambda size, alignment, stream: torch.empty(
                    size, device="cuda", dtype=torch.int8
                )
            )
            return run_kernel(x)

        opt_fn = torch.compile(
            fn_with_set_allocator, backend="aot_eager", fullgraph=True
        )

        # set_allocator inside compiled region does NOT graph break
        with triton_allocator(NullAllocator()):
            out = opt_fn(x)
            self.assertEqual(out, x)

            # Verify set_allocator replays on cache hit (not just tracing)
            triton.set_allocator(NullAllocator())
            out2 = opt_fn(x)
            self.assertEqual(out2, x)

    def test_tensor_hasattr(self):
        @torch.compile(fullgraph=True, backend="eager")
        def fn(x):
            if hasattr(x, "test"):
                return x + 2
            else:
                return x + 1

        self.assertEqual(torch.ones(2, 2) + 1, fn(torch.ones(2, 2)))

        inp = torch.ones(2, 2)
        inp.test = None
        self.assertEqual(torch.ones(2, 2) + 2, fn(inp))

    def test_tensor_call_obj_hasattr_view(self):
        @torch.compile(fullgraph=True, backend="eager")
        def fn(x):
            output3 = getattr(x, "view", None)(10)
            return output3

        x = torch.randn(10)
        self.assertEqual(x.view(10), fn(x))

    def test_tensor_dynamic_method(self):
        def add_one(x):
            return x + 1

        t = torch.nn.Parameter(torch.ones(1))
        t.add_one = add_one

        @torch.compile(fullgraph=True, backend="eager")
        def fn(x):
            return t.add_one(t) + x

        result = fn(torch.ones(1))
        self.assertEqual(torch.ones(1) + 2, result)

    def test_known_tensor_methods_traced(self):
        # Verify that known tensor methods (in all_tensor_attrs) are still
        # traced into the graph via the generic proxy path.
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            return x.abs().cos()

        result = fn(torch.randn(4))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)

    def test_tensor_subclass_method_traced(self):
        # Methods defined on the actual tensor class (including dynamically
        # added ones) should be proxied through the generic call_method path,
        # not graph-broken.  This validates that the guard uses the concrete
        # class_type rather than the static all_tensor_attrs dict.
        def _dynamo_test_method(self):
            return self + 1

        with unittest.mock.patch.object(
            torch.Tensor, "_dynamo_test_method", _dynamo_test_method, create=True
        ):
            cnt = CompileCounterWithBackend("eager")

            @torch.compile(backend=cnt)
            def fn(x):
                y = x._dynamo_test_method()
                return y + 1

            result = fn(torch.randn(4))
            self.assertEqual(cnt.frame_count, 1)
            # Verify _dynamo_test_method appears as a call_method in the FX graph
            call_method_targets = [
                n.target for n in cnt.graphs[0].graph.nodes if n.op == "call_method"
            ]
            self.assertIn("_dynamo_test_method", call_method_targets)

    def test_unknown_tensor_method_graph_break(self):
        # Truly unknown methods raise AttributeError during tracing at
        # LOAD_ATTR time (dynamic_getattr), ensuring dynamo does not
        # silently proxy them into the compiled graph.
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            y = x._nonexistent_test_method_xyz()
            return y + 1

        with self.assertRaises(AttributeError):
            fn(torch.randn(4))

    def test_tensor__iter__(self):
        def fn(x):
            it = x.__iter__()
            for y in it:
                y.add_(1.0)
            return y

        torch._dynamo.testing.standard_test(
            self,
            fn,
            1,
            expected_ops=20,
        )

    def test_tensor_iter(self):
        def fn(x):
            for y in x:
                y.add_(1.0)
            return y

        torch._dynamo.testing.standard_test(
            self,
            fn,
            1,
            expected_ops=20,
        )

    def test_tensor_share_memory(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_size = 64
                self.num_layers = 2

            def forward(self, x):
                batch_size = x.size(0)
                h = torch.zeros(
                    self.num_layers, batch_size, self.hidden_size
                ).share_memory_()
                c = torch.zeros(self.num_layers, batch_size, self.hidden_size)
                return x + h.sum() + c.sum()

        model = Model()
        x = torch.randn(4, 10)
        expected = model(x)
        compiled_model = torch.compile(model, fullgraph=False, backend="eager")
        actual = compiled_model(x)
        self.assertEqual(expected, actual)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_arange_length_with_float32_dtype(self):
        @torch.compile(fullgraph=True, backend="eager")
        def f(x):
            y = x.item()
            r = torch.arange(y, dtype=torch.float32)

            if r.size(0) == y:
                return r + 1

            return r

        x = torch.tensor([300])
        r = f(x)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_torch_check(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            y = x.item()
            torch._check(y >= 0)
            return torch.arange(0, y)

        f(torch.tensor([3]))
        f(torch.tensor([4]))
        self.assertEqual(cnts.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_torch_check_symbolic_shape_rel(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            y = x.item()
            torch._check(x.shape[0] == 1)
            torch._check(x.shape[0] != 2)
            torch._check(x.shape[0] >= 0)
            torch._check(x.shape[0] > 0)
            torch._check(x.shape[0] < 4)
            torch._check(x.shape[0] <= 3)
            return torch.arange(0, y)

        f(torch.tensor([3]))
        f(torch.tensor([4]))
        self.assertEqual(cnts.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    # Translation validation changes the exception type, don't run with it
    @torch.fx.experimental._config.patch(translation_validation=False)
    def test_torch_check_nonnegative(self):
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            y = x.item()
            torch._check(y >= 0)
            # Cannot conditional on unbacked SymInt
            if y == 0:
                assert False  # noqa: B011, S101
            else:
                return torch.arange(0, y)

        self.assertRaises(torch._dynamo.exc.UserError, lambda: f(torch.tensor([3])))

    def test_assert(self):
        @torch.compile(backend="eager")
        def fn1(x):
            assert x.shape != x.shape  # noqa: S101

        with self.assertRaises(AssertionError):
            a = torch.randn(10)
            fn1(a)

        def fn2(x):
            assert x.shape == x.shape  # noqa: S101
            return x.abs()

        torch._dynamo.testing.standard_test(self, fn=fn2, nargs=1, expected_ops=1)

    def test_size_input(self):
        def fn(x, s):
            a, b = s
            return x + (a - b)

        v = torch.zeros(10, 20)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(v, v.size())[0, 0], -10)
        self.assertEqual(opt_fn(v, (10, 20))[0, 0], -10)
        self.assertEqual(opt_fn(v, [10, 20])[0, 0], -10)
        # One recompile per differing input type
        self.assertEqual(cnts.frame_count, 3)

    def test_tensor_dict1(self):
        def fn(inputs):
            return inputs["a"] - inputs["b"] * 1.5

        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        self.assertEqual(opt_fn({"a": v1, "b": v2})[0], -200)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_tensor_dict3(self):
        def fn(inputs_a, inputs_b):
            total = torch.zeros(1)
            input_keys = inputs_a.keys() | inputs_b.keys()
            for k in input_keys:
                if k in inputs_a:
                    total += inputs_a[k]
                if k in inputs_b:
                    total += inputs_b[k]
            return total

        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        self.assertEqual(
            opt_fn({"a": v1, "b": v2}, {"b": v1, "c": v2}),
            fn({"a": v1, "b": v2}, {"b": v1, "c": v2}),
        )
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 5)

    def test_tensor_dict2(self):
        def fn1(inputs):
            total = torch.zeros(1)
            for k, v in inputs.items():
                total += v
            return total

        def fn2(inputs):
            total = torch.zeros(1)
            for v in inputs.values():
                total += v
            return total

        def fn3(inputs):
            total = torch.zeros(1)
            for k in inputs.keys():
                total += inputs[k]
            return total

        v1 = torch.Tensor([100])
        v2 = torch.Tensor([200])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn1 = torch.compile(fn1, backend=cnts, fullgraph=True)
        opt_fn2 = torch.compile(fn2, backend=cnts, fullgraph=True)
        opt_fn3 = torch.compile(fn3, backend=cnts, fullgraph=True)
        self.assertEqual(opt_fn1({"a": v1, "b": v2})[0], 300)
        self.assertEqual(opt_fn2({"a": v1, "b": v2})[0], 300)
        self.assertEqual(opt_fn3({"a": v1, "b": v2})[0], 300)
        self.assertEqual(cnts.frame_count, 3)
        self.assertEqual(cnts.op_count, 9)

    def test_is_floating_point(self):
        def fn(a, b):
            x = a + 1.0
            if torch.is_floating_point(b):
                x = x + b
            return x + 2.0

        return torch._dynamo.testing.standard_test(self, fn=fn, nargs=2, expected_ops=3)

    def test_is_floating_point2(self):
        def fn(a, b):
            x = a + 1.0
            if b.is_floating_point():
                x = x + b
            return x + 2.0

        return torch._dynamo.testing.standard_test(self, fn=fn, nargs=2, expected_ops=3)

    def test_is_tensor(self):
        def fn(a, b):
            x = a + 1.0
            if torch.is_tensor(b):
                x = x + b
            return x + 2.0

        return torch._dynamo.testing.standard_test(self, fn=fn, nargs=2, expected_ops=3)

    def test_is_tensor2(self):
        def fn(x):
            if torch.is_tensor(x):
                return x + 1
            else:
                return torch.ones([2, 3])

        x1 = {"input": torch.rand(2, 3)}
        x2 = torch.rand(2, 3)
        ref1 = fn(x1)
        ref2 = fn(x2)
        opt_fn = torch.compile(fn, backend="eager")
        res1 = opt_fn(x1)
        res2 = opt_fn(x2)
        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)

    def test_numel(self):
        def fn(a):
            return (a + a.numel() + torch.numel(a), a + a.nelement())

        return torch._dynamo.testing.standard_test(
            self,
            fn=fn,
            nargs=1,
            expected_ops=3,
            expected_ops_dynamic=ifdynstaticdefault(3, 4),
        )

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_tensor_item_capture(self):
        def fn(a, b):
            return (a + b).sum().item()

        v1 = torch.randn((10, 10))
        v2 = torch.randn((10, 10))
        correct = fn(v1, v2)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(v1, v2), correct)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 4)

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", False)
    def test_tensor_item_no_capture(self):
        def fn(a, b):
            return (a + b).sum().item()

        v1 = torch.randn((10, 10))
        v2 = torch.randn((10, 10))
        correct = fn(v1, v2)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(opt_fn(v1, v2), correct)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_data_access_in_inference_mode(self):
        @torch.compile(fullgraph=True, backend="eager")
        def f(x):
            y = x.data
            return y

        with torch.inference_mode():
            x = torch.randn(3)
            y = f(x)
        self.assertEqual(y, x)

    def test_tensor_build_list_unpack(self):
        def fn(x):
            # seen in fastNLP_Bert
            return torch.cat([*x], dim=-1)

        val = torch.randn([1, 1, 473, 768])
        correct = fn(val)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(opt_fn(val), correct))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_tensor_interacts_with_numpy_ndarray(self):
        def fn(x, y):
            a = x.numpy()
            b = y.numpy()
            c = np.ones_like(a)
            d = np.ones_like(b)
            torch._dynamo.graph_break()
            return np.add(a, c), np.add(b, d)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        for _ in range(10):
            x = torch.randn([1, 3])
            y = torch.randn([1, 3])
            ref = fn(x, y)
            res = opt_fn(x, y)
            self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)

    def test_inplace_view_on_graph_input(self):
        # graph break when calling methods with inplace_view tag on graph input
        func_args_map = {
            lambda x: x.resize_(6).mul_(2): torch.ones(4),
            lambda x: x.t_().mul_(2): torch.rand(2, 3),
            lambda x: x.transpose_(0, 1).mul_(2): torch.rand(2, 3),
            lambda x: x.squeeze_().mul_(2): torch.rand(1, 2, 3),
            lambda x: x.unsqueeze_(0).mul_(2): torch.rand(2, 3),
            lambda x: x.resize_as_(torch.rand(200, 300)): torch.rand(2, 3),
            lambda x: x.swapaxes_(0, 1).mul_(2): torch.rand(2, 3),
            lambda x: x.swapdims_(0, 1).mul_(2): torch.rand(2, 3),
            lambda x: x.rename_("N", "C").mul_(2): torch.zeros(2, 3),
            lambda x: x.as_strided_((3, 2), (2, 1)).mul_(2): torch.zeros(2, 3),
            lambda x: x.detach_().mul_(2): torch.zeros(2, 3),
        }
        for func, args in func_args_map.items():
            args_clone = args.clone()
            cnts = torch._dynamo.testing.CompileCounter()
            opt_f = torch.compile(func, backend=cnts)
            self.assertTrue(same(func(args).shape, opt_f(args_clone).shape))
            self.assertEqual(cnts.frame_count, 1)
            self.assertEqual(cnts.op_count, 1)  # mul_

    def test_out_variants_with_resizing_on_graph_inputs(self):
        def fn(x, y):
            return torch.cosh(x, out=y) + 1

        x = torch.rand(2, 3)
        y = torch.rand(4)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertTrue(same(fn(x, y), opt_fn(x.clone(), y.clone())))
        self.assertEqual(cnts.frame_count, 1)

    def test_out_variants_with_resizing_on_graph_inputs_with_dynamic(self):
        # https://github.com/pytorch/pytorch/issues/120482
        class CustomModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, inputs):
                return torch.outer(**inputs)

        compile_fn = torch.compile(CustomModel(), backend="eager", fullgraph=True)

        shapes = [(2, 1), (6, 1), (4, 1)]
        for shape in shapes:
            vec1, vec2 = shape
            input_tensor1 = torch.randn(vec1)
            input_tensor2 = torch.randn(vec2)
            out_tensor = torch.empty(shape)
            args = {"input": input_tensor1, "vec2": input_tensor2, "out": out_tensor}
            res = compile_fn(args)
            opt_res = res.clone()  # cuz this is out and we mutate it
            res = CustomModel()(args)
            self.assertEqual(res, opt_res)

    def test_out_variants_with_resizing_on_graph_inputs_with_dynamic1(self):
        mv_op = torch.mv

        def mv_out_op(a, b, c):
            torch.mv(b, c, out=a)
            return a

        def fn(op, *args):
            return op(*args)

        opt_fn = torch.compile(fn, backend="eager")

        ref = fn(mv_op, torch.ones(3, 3), torch.ones(3))
        res = opt_fn(mv_op, torch.ones(3, 3), torch.ones(3))
        self.assertEqual(ref, res)

        ref = fn(mv_out_op, torch.empty(0), torch.ones(3, 3), torch.ones(3))
        res = opt_fn(mv_out_op, torch.empty(0), torch.ones(3, 3), torch.ones(3))
        self.assertEqual(ref, res)

    def test_input_cell_mutation(self):
        def fn(x):
            x = x.cos()

            def inner():
                return x.sin()

            return inner()

        x = torch.ones(10)
        opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    def test_torch_size(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            output_size = torch.Size([10, 10])
            x = x.view(*output_size)
            return (x,)

        x = torch.randn(100, requires_grad=True)
        x_clone = x.clone()
        ref = fn(x)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x_clone)

        self.assertTrue(same(ref, res))

    def test_torch_size_numel(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn():
            return torch.Size([10, 8]).numel()

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        num = torch.Size([10, 8]).numel()
        self.assertEqual(opt_fn(), num)

    def test_torch_size_numel_dynamic(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x):
            return x.size().numel()

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        x = torch.rand(10, 1, 8, 1)
        expect = fn(x)
        self.assertEqual(opt_fn(x), expect)

    def test_size_dim(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x, dim):
            return x.size(dim=dim)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        x = torch.empty([4, 9, 8])
        self.assertEqual(opt_fn(x, 1), 9)
        self.assertEqual(opt_fn(x, -2), 9)

    def test_torch_size_tensor_index_scalar_constant(self):
        def fn(x):
            idx = torch.tensor(1)
            dim_size = x.shape[idx]
            return x.reshape(-1, dim_size)

        x = torch.randn(2, 3, 4)
        ref = fn(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x)

        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 1)

    def test_torch_size_tensor_index_non_scalar(self):
        def fn(x):
            idx = torch.tensor([1, 1])
            try:
                dim_size = x.shape[idx]
                return x * dim_size
            except TypeError:
                return x.sum()

        x = torch.randn(2, 3, 4)
        ref = fn(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(x)

        self.assertTrue(same(ref, res))

    def test_stride_dim(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x, dim):
            return x.stride(dim=dim)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        x = torch.empty([4, 9, 8])
        self.assertEqual(opt_fn(x, 0), 72)
        self.assertEqual(opt_fn(x, -2), 8)

    def test_torch_seed(self):
        from torch._dynamo.utils import counters

        cnts = torch._dynamo.testing.CompileCounter()
        counters.clear()

        def fn(x):
            attention_seed = int(torch.seed() % sys.maxsize)
            torch.manual_seed(attention_seed)
            return (x,)

        x = torch.randn(10, requires_grad=True)
        ref = fn(x)

        # Python code is needed here, since torch.manual_seed graph-breaks.
        # Refs: https://github.com/pytorch/pytorch/issues/107187
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=False)
        res = opt_fn(x)

        self.assertTrue(same(ref, res))
        # Only the torch.seed call is turned into an FX graph.
        self.assertEqual(cnts.op_count, 1)
        self.assertEqual(cnts.frame_count, 1)
        # Graph breaks at manual_seed.
        self.assertEqual(len(counters["graph_break"]), 1)

    def test_torch_generator_manual_seed(self):
        from torch._dynamo.utils import counters

        cnts = torch._dynamo.testing.CompileCounter()
        counters.clear()

        def fn(x, gen):
            gen.manual_seed(3)
            return x + 1

        x = torch.randn(10)
        ref = fn(x, torch.Generator())

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=False)
        res = opt_fn(x, torch.Generator())

        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.op_count, 1)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(len(counters["graph_break"]), 1)

    def test_torch_generator_initial_seed(self):
        from torch._dynamo.utils import counters

        cnts = torch._dynamo.testing.CompileCounter()
        counters.clear()

        def fn(x):
            return x + 1, torch.default_generator.initial_seed()

        x = torch.randn(10)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=False)
        res = opt_fn(x)

        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.op_count, 1)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(len(counters["graph_break"]), 1)

    def test_torch_generator_get_state_fullgraph(self):
        def fn():
            return torch.default_generator.get_state()

        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)()

    def test_is_tensor_like(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def f(x):
            if torch.overrides.is_tensor_like(x):
                return (x * 2,)
            return (torch.ones(10) + x,)

        x = torch.randn(10)
        ref0 = f(x)
        ref1 = f(4)
        opt_f = torch.compile(f, backend=cnts, fullgraph=True)
        res0 = opt_f(x)
        res1 = opt_f(4)
        self.assertTrue(same(ref0, res0))
        self.assertTrue(same(ref1, res1))

    def test_is_tensor_like2(self):
        class MyTensor:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}

                if func is torch.max:
                    return torch.tensor(123)
                return func(*args, **kwargs)

        def fn(x):
            if torch.overrides.is_tensor_like(x):
                return torch.max(x)
            else:
                return torch.zeros(1)

        x = MyTensor()
        ref0 = fn(x)
        ref1 = fn(4)
        opt_fn = torch.compile(fn, backend="eager")
        res0 = opt_fn(x)
        res1 = opt_fn(4)
        self.assertTrue(same(ref0, res0))
        self.assertTrue(same(ref1, res1))

    def test_tensor_data(self):
        def fn(x, y):
            return x[y.data]

        x = torch.rand(8)
        y = torch.ones(8).to(torch.int)
        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_tensor_layout(self):
        def fn(x):
            return torch.zeros(
                [x.size()[0], x.size()[1]],
                dtype=x.dtype,
                layout=x.layout,
                device=x.device,
            )

        x = torch.rand(2, 3)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_version_ci(self):
        # temporary test to check that the ci torch version is set correctly
        self.assertTrue(hasattr(torch, "_subclasses"))

    def test_slice_input(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def getitem(a, idx):
            if isinstance(idx, slice):
                return (
                    torch.zeros(1),
                    a[idx]
                    + [
                        100,
                    ],
                )
            else:
                return (torch.zeros(1), a[idx])

        layers = list(range(10))
        ref0 = getitem(layers, slice(0, 2, 1))
        ref1 = getitem(layers, 2)
        ref2 = getitem(layers, slice(3, 8, 2))
        opt_getitem = torch.compile(getitem, backend=cnts, fullgraph=True)
        res0 = opt_getitem(layers, slice(0, 2, 1))
        res1 = opt_getitem(layers, 2)
        res2 = opt_getitem(layers, slice(3, 8, 2))

        self.assertTrue(ref0 == res0)
        self.assertTrue(ref1 == res1)
        self.assertTrue(ref2 == res2)

    def test_grad(self):
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(a, b):
            out = a * b
            out.sum().backward()
            real_out = torch.sigmoid(a.grad + b)
            return real_out

        inps = [torch.randn(4, requires_grad=True) for _ in range(2)]
        for inp in inps:
            inp.grad = None
        ref = fn(*inps)

        for inp in inps:
            inp.grad = None
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(*inps)

        self.assertTrue(same(ref, res))

    @torch._dynamo.config.patch(guard_nn_modules=True)
    def test_source_non_input_grad_access(self):
        # This test creates a model, and accesses the grads
        # from its parameter. This means that within dynamo,
        # the tensor we are reading the grad from HAS a source,
        # but is not known to graphargs.
        cnts = torch._dynamo.testing.CompileCounter()

        class TrivialModel(torch.nn.Module):
            def __init__(self) -> None:
                super(TrivialModel, self).__init__()
                self.linear = torch.nn.Linear(2, 1)

            def forward(self, x):
                return self.linear(x)

        def fn(a, b):
            outs = []
            for param in model.parameters():
                outs.append(torch.ones(param.grad.size()))
            return outs, param.grad + 1

        model = TrivialModel()
        # Eager
        a = torch.ones([2, 2], requires_grad=True)
        b = torch.ones([2, 2])
        out = model(a)
        out_sum = out.sum()
        out_sum.backward()
        ref = fn(a, b)

        # Compiled
        model = TrivialModel()
        a = torch.ones([2, 2], requires_grad=True)
        b = torch.ones([2, 2])
        out = model(a)
        out_sum = out.sum()
        out_sum.backward()

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(a, b)

        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 3)

    def test_intermediary_tensor_grad_access(self):
        # This test creates a model, and accesses the grads
        # from its parameters and an entirely intermediary tensor.
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(a, b):
            intermediary = torch.ones(2, 2)
            c = a + intermediary
            outs = []
            outs.append(intermediary.grad)
            return outs

        # Eager
        a = torch.ones([2, 2], requires_grad=True)
        b = torch.ones([2, 2])
        ref = fn(a, b)

        # Compiled
        a = torch.ones([2, 2], requires_grad=True)
        b = torch.ones([2, 2])
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(a, b)
        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    def test_clone_sparse_input(self):
        for layout in [
            torch.sparse_coo,
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        ]:
            for sparse_input in self.generate_simple_inputs(
                layout,
                device="cpu",
                dtype=torch.float64,
                index_dtype=torch.int64,
            ):
                # Invoke the dynamo clone input method directly.
                sparse_copy = torch._dynamo.utils.clone_input(sparse_input)
                # Make sure sparse clone is successful.
                self.assertEqual(sparse_input, sparse_copy)

    def test_tensor_is_contiguous(self):
        def fn(x):
            input = torch.randn((1, 16, 1, 1))
            weight = torch.randn((8, 16, 3, 3))
            weight = weight.to(memory_format=x)
            output = torch.conv2d(input, weight, None, (2, 1), (1, 1), (1, 1), 1)
            return output.is_contiguous(memory_format=x)

        opt_fn = torch.compile(fn, backend="eager")
        for x in [torch.contiguous_format, torch.channels_last]:
            self.assertEqual(fn(x), opt_fn(x))

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_tensor_ctor_list_of_tensor(self):
        def fn(x):
            return torch.tensor([x], dtype=torch.int64)

        x = torch.tensor(20)
        res = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res2 = opt_fn(x)
        self.assertEqual(res, res2)
        self.assertEqual(cnts.frame_count, 1)

    def test_tensor_types(self):
        def fn(dtype, tensor_type):
            x = torch.empty(4, dtype=dtype)
            assert isinstance(x, tensor_type)  # noqa: S101

        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(torch.float32, torch.FloatTensor)
        opt_fn(torch.float64, torch.DoubleTensor)
        opt_fn(torch.float16, torch.HalfTensor)
        opt_fn(torch.bfloat16, torch.BFloat16Tensor)
        opt_fn(torch.uint8, torch.ByteTensor)
        opt_fn(torch.int8, torch.CharTensor)
        opt_fn(torch.int64, torch.LongTensor)
        opt_fn(torch.int, torch.IntTensor)
        opt_fn(torch.int16, torch.ShortTensor)
        opt_fn(torch.bool, torch.BoolTensor)

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_item(self):
        class MyMod(torch.nn.Module):
            def forward(self, x):
                z = torch.max(x)
                return z.int().item()

        x = torch.tensor([[10.6763, 11.7445, -2.2369]])
        model = MyMod()
        y = torch.compile(model, backend="eager", fullgraph=True)(x)

        self.assertEqual(y, 11)

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_item_changes(self):
        class MyMod(torch.nn.Module):
            def forward(self, x):
                z = torch.max(x)
                return z.int().item()

        x = torch.tensor([[10.6763, 11.7445, -2.2369]])
        model = MyMod()
        opt_model = torch.compile(model, backend="eager", fullgraph=True)
        y = opt_model(x)
        z = opt_model(torch.tensor([[y - 5, y + 10, y + 50]]))

        self.assertEqual(y, 11)
        self.assertEqual(z, 61)

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_item_changes_new_shape(self):
        class MyMod(torch.nn.Module):
            def forward(self, x):
                z = torch.max(x)
                return z.int().item()

        x = torch.tensor([[10.6763, 11.7445, -2.2369]])
        model = MyMod()
        opt_model = torch.compile(model, backend="eager", fullgraph=True)
        y = opt_model(x)
        z = opt_model(torch.tensor([[y - 5, y + 50], [y + 5, y - 50]]))

        self.assertEqual(y, 11)
        self.assertEqual(z, 61)

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_foreach_tensor_scalar(self):
        # Test that foreach ops with tensor scalar values can be traced fullgraph
        # when capture_scalar_outputs is enabled. The scalar's .item() creates an
        # unbacked symbol that should be properly ignored since it only affects
        # tensor values, not shapes.
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        c = torch.randn(4, 4)
        val = torch.tensor(0.5, dtype=torch.float64)

        # Test _foreach_addcmul with value arg
        def fn_addcmul(a, b, c, val):
            return torch._foreach_addcmul([a], [b], [c], value=1 - val)

        expected = fn_addcmul(a, b, c, val)
        actual = torch.compile(fn_addcmul, backend="eager", fullgraph=True)(
            a, b, c, val
        )
        self.assertEqual(expected, actual)

        # Test _foreach_addcdiv with value arg
        def fn_addcdiv(a, b, c, val):
            return torch._foreach_addcdiv([a], [b], [c], value=1 - val)

        expected = fn_addcdiv(a, b, c, val)
        actual = torch.compile(fn_addcdiv, backend="eager", fullgraph=True)(
            a, b, c, val
        )
        self.assertEqual(expected, actual)

        # Test _foreach_add with scalar arg
        def fn_add(a, val):
            return torch._foreach_add([a], 1 - val)

        expected = fn_add(a, val)
        actual = torch.compile(fn_add, backend="eager", fullgraph=True)(a, val)
        self.assertEqual(expected, actual)

        # Test _foreach_mul with scalar arg
        def fn_mul(a, val):
            return torch._foreach_mul([a], 1 - val)

        expected = fn_mul(a, val)
        actual = torch.compile(fn_mul, backend="eager", fullgraph=True)(a, val)
        self.assertEqual(expected, actual)

        # Test _foreach_pow with scalar exponent
        def fn_pow(a, val):
            return torch._foreach_pow([a.abs()], 1 - val.item())

        expected = fn_pow(a, val)
        actual = torch.compile(fn_pow, backend="eager", fullgraph=True)(a, val)
        self.assertEqual(expected, actual)

    @unittest.skip("https://github.com/pytorch/pytorch/issues/99726")
    def test_cross_entropy_loss_fancy_ctor1(self):
        rand_5 = torch.randn(5)
        rand_3_5 = torch.randn(3, 5)
        target = torch.empty(3, dtype=torch.long).random_(5)

        loss = torch.nn.CrossEntropyLoss(
            weight=rand_5, reduce=False, label_smoothing=0.5
        )
        opt_loss = torch.compile(loss, backend="eager", fullgraph=True)
        input = rand_3_5
        dynamo_output = opt_loss(input, target)

        loss = torch.nn.CrossEntropyLoss(
            weight=rand_5, reduce=False, label_smoothing=0.5
        )
        input = rand_3_5
        output = loss(input, target)

        self.assertTrue(torch.allclose(dynamo_output, output))

    def test_cross_entropy_loss_fancy_ctor2(self):
        rand_3_5 = torch.randn(3, 5)
        target = torch.empty(3, dtype=torch.long).random_(5)

        loss = torch.nn.CrossEntropyLoss(reduce=False, label_smoothing=0.5)
        opt_loss = torch.compile(loss, backend="eager", fullgraph=True)
        input = rand_3_5
        dynamo_output = opt_loss(input, target)

        loss = torch.nn.CrossEntropyLoss(reduce=False, label_smoothing=0.5)
        input = rand_3_5
        output = loss(input, target)

        self.assertTrue(torch.allclose(dynamo_output, output))

    def test_cross_entropy_loss_simple_ctor(self):
        output = None
        rand_3_5 = torch.randn(3, 5)
        target = torch.empty(3, dtype=torch.long).random_(5)

        loss = torch.nn.CrossEntropyLoss()
        opt_loss = torch.compile(loss, backend="eager", fullgraph=True)
        input = rand_3_5
        dynamo_output = opt_loss(input, target)

        loss = torch.nn.CrossEntropyLoss()
        input = rand_3_5
        output = loss(input, target)

        self.assertTrue(torch.allclose(dynamo_output, output))

    def test_named_parameters(self):
        n_embd = 768
        block_size = 128
        vocab_size = 65
        embd_pdrop = 0.1

        class MyModel2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.tok_emb = torch.nn.Embedding(vocab_size, n_embd)
                self.pos_emb = torch.nn.Parameter(torch.zeros(1, block_size, n_embd))
                self.drop = torch.nn.Dropout(embd_pdrop)

            def forward(self, x):
                return x

        class MyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.tok_emb = torch.nn.Embedding(vocab_size, n_embd)
                self.pos_emb = torch.nn.Parameter(torch.zeros(1, block_size, n_embd))
                self.drop = torch.nn.Dropout(embd_pdrop)
                self.submod2 = MyModel2()

            def forward(self, x):
                return x

        # Regular
        params = []
        mod = MyModel()
        actual_params = list(mod.named_parameters())

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            return list(mod.named_parameters())

        params = fn()

        self.assertEqual(len(actual_params), len(params))
        for idx in range(len(params)):
            k_a, v_a = actual_params[idx]
            k, v = params[idx]
            self.assertEqual(k_a, k)
            self.assertTrue(torch.allclose(v_a, v))

        # Prefix
        params = []
        mod = MyModel()
        actual_params = list(mod.named_parameters(prefix="foo"))

        @torch.compile(backend="eager", fullgraph=True)
        def fn1():
            return list(mod.named_parameters(prefix="foo"))

        params = fn1()

        self.assertEqual(len(actual_params), len(params))
        for idx in range(len(params)):
            k_a, v_a = actual_params[idx]
            k, v = params[idx]
            self.assertEqual(k_a, k)
            self.assertTrue(torch.allclose(v_a, v))

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    def test_tensor_dot_grad_no_graph_break(self):
        def fn(a, b):
            y = 3 * a**3 - b**2
            y.backward(gradient=torch.tensor([1.0, 1.0]))
            b.grad.zero_()
            return a.grad, b.grad

        a = torch.tensor([2.0, 3.0], requires_grad=True)
        b = torch.tensor([6.0, 4.0], requires_grad=True)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        _, b_grad = opt_fn(a, b)
        self.assertTrue(same(b_grad, torch.tensor([0.0, 0.0])))
        self.assertEqual(cnts.frame_count, 1)

    def test_torch_nn_parameter_isinstance(self):
        def fn(x):
            a = torch.nn.Parameter(torch.rand(2, 3))
            if isinstance(a, torch.Tensor):
                return x + 1
            else:
                return x - 1

        x = torch.tensor([2.5])
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_onnx_shape_as_tensor(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            return 1 + torch._shape_as_tensor(x)[0]

        gm, _ = torch._dynamo.export(f)(torch.ones(6))

        input_one_dim = torch.ones(6)
        input_two_dims = torch.ones(7, 4)
        self.assertEqual(f(input_one_dim), 7)
        self.assertEqual(f(input_two_dims), 8)
        self.assertEqual(f(input_two_dims), 8)

        @torch.compile(backend="eager", fullgraph=True)
        def f_onnx(x):
            return 1 + torch.onnx.operators.shape_as_tensor(x)[0]

        self.assertEqual(f_onnx(input_one_dim), 7)
        self.assertEqual(f_onnx(input_two_dims), 8)
        self.assertEqual(f_onnx(input_two_dims), 8)

    def test_inplace_param_update(self):
        def fn(param, y):
            prev_grad = torch.is_grad_enabled()
            try:
                torch.set_grad_enabled(False)
                torch.set_grad_enabled(True)
                torch.set_grad_enabled(False)
                param.add_(y)
            finally:
                torch.set_grad_enabled(prev_grad)

        y = torch.randn(4)
        x = torch.nn.Parameter(torch.randn(4))
        fn(x, y)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        opt_fn(x, y)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 3)

    def test_torch_generator_set_state(self):
        def fn():
            default_state = torch.default_generator.get_state()
            x = torch.rand([2, 3])
            if default_state.dtype != "float32":
                x = x * 2
            torch._dynamo.graph_break()
            torch.default_generator.set_state(default_state)
            y = torch.rand([2, 3])
            return x, y

        opt_fn = torch.compile(fn, backend="eager")
        x, y = opt_fn()
        self.assertEqual(x, y * 2)

    def test_torch_distributions_lazy_property(self):
        def fn(x):
            return torch.distributions.Categorical(probs=x).entropy()

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.rand([4, 4])
        self.assertEqual(opt_fn(x), fn(x))

    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "Test requires CUDA or XPU.")
    def test_symint_as_device_kwarg_non_strict_export(self):
        class Mod(torch.nn.Module):
            def forward(self, x):
                # -2 to make device id 0 for easier testing on CI
                return torch.ones(10, device=x.size(0) - 2)

        x = torch.randn(2)
        m = Mod()
        d1 = torch.export.Dim("d1", max=2048)
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError, r"Constraints violated \(d1\)"
        ):
            ep = torch.export.export(
                m, (x,), dynamic_shapes={"x": {0: d1}}, strict=False
            )

    def test_real_imag_tensor_attribute(self):
        def fn(x, y):
            a = x.real
            b = x.imag
            return torch.mul(torch.add(a, y), b)

        x_real = torch.rand((4, 4))
        x_imag = torch.rand((4, 4))
        x = torch.complex(x_real, x_imag)
        y = torch.rand((4, 4))

        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_cast(self):
        from typing import cast

        def fn(x):
            return cast(torch.Tensor, torch.add(x, 1.0))

        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        ref = fn(torch.ones(2, 2))
        res = opt_fn(torch.ones(2, 2))

        self.assertTrue(same(ref, res))

    def test_cast_with_different_module_types(self):
        # typing.cast works correctly when used in a mixin pattern with
        # different module types, producing correct results without
        # graph breaks.
        from typing import cast

        class Mixin:
            def get_self_as_module(self):
                return cast(torch.nn.Module, self)

        class ModuleA(Mixin, torch.nn.Module):
            def forward(self, x):
                self.get_self_as_module()
                return x + 1

        class ModuleB(Mixin, torch.nn.Module):
            def forward(self, x):
                self.get_self_as_module()
                return x + 2

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(mod, x):
            mod.get_self_as_module()
            return x + 1

        x = torch.randn(4)
        ref_a = fn.__wrapped__(ModuleA(), x)
        ref_b = fn.__wrapped__(ModuleB(), x)
        res_a = fn(ModuleA(), x)
        res_b = fn(ModuleB(), x)

        self.assertEqual(ref_a, res_a)
        self.assertEqual(ref_b, res_b)
        self.assertEqual(cnt.frame_count, 2)

    def test_cast_fullgraph_with_non_tensor(self):
        # Verify typing.cast works with non-tensor values under fullgraph=True
        from typing import cast

        def fn(x):
            val = cast(int, x.shape[0])
            return x + val

        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        ref = fn(torch.ones(3, 4))
        res = opt_fn(torch.ones(3, 4))

        self.assertTrue(same(ref, res))

    @torch._dynamo.config.patch(nested_graph_breaks=False)
    def test_cast_no_recompile_after_graph_break(self):
        # In FSDP, cast(nn.Module, self) can be called after a
        # graph break. Without the polyfill + skip_code fix, PEP 523 compiles
        # typing.cast as a standalone frame with TYPE_MATCH guards on val,
        # causing recompilation when different module types pass through.
        # https://github.com/pytorch/pytorch/blob/0feb90404fbeb9b1594ae194f8fd47bbe7f5f245/torch/distributed/fsdp/_fully_shard/_fully_shard.py#L376
        from typing import cast

        from torch._dynamo.utils import counters

        counters.clear()

        class Base(torch.nn.Module):
            def get_state(self):
                torch._dynamo.decorators.graph_break()
                return cast(torch.nn.Module, self)

        class ModuleA(Base):
            pass

        class ModuleB(Base):
            pass

        cnt = torch._dynamo.testing.CompileCounter()
        a, b = ModuleA(), ModuleB()

        @torch.compile(backend=cnt)
        def fn(mod, x):
            mod.get_state()
            return x + 1

        x = torch.randn(4)
        fn(a, x)
        fn(b, x)
        self.assertEqual(cnt.frame_count, 1)
        # 5 frames: fn (x2), get_state before graph_break (x2),
        # get_state resume after graph_break (x1, no recompile).
        # Without skip_code, typing.cast would add 2 more frames (7 total).
        self.assertEqual(counters["frames"]["total"], 5)

    def test_T_tensor_attribute(self):
        def fn(x, y):
            a = x.T
            return torch.add(a, y)

        x = torch.rand((4, 4))
        y = torch.rand((4, 4))

        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_recursive_tensor_attribute(self):
        def fn(x, y):
            a = x.real.T
            b = x.imag
            return torch.mul(torch.add(a, y), b)

        x_real = torch.rand((4, 4))
        x_imag = torch.rand((4, 4))
        x = torch.complex(x_real, x_imag)
        y = torch.rand((4, 4))

        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_tagging_tensors_simple(self):
        def foo(x, y):
            return x * y, x, y

        a = torch.randn([3, 3])
        a.tag = "a"
        b = torch.randn([3, 3])
        b.tag = "b"

        exported = torch._dynamo.export(foo)(a, b)
        out_graph = exported[0]

        nodes = list(out_graph.graph.nodes)
        placeholders = [node for node in nodes if node.op == "placeholder"]
        all_tags = []
        for placeholder in placeholders:
            if "tensor_dict" in placeholder.meta:
                all_tags.append(placeholder.meta["tensor_dict"]["tag"])

        self.assertEqual(all_tags, ["a", "b"])

    def test_tagging_tensors_mix_used_unused_structure(self):
        def pre_attention_state_ops(input, mems, state):
            lc_key = state[0]
            lc_val = state[1]
            bar = []
            for i in range(0, 4):
                bar2 = []
                for j in range(0, 3):
                    bar2.append(
                        lc_key + lc_val + torch.tensor([0.1, 0.25, 0.4, 0.5, 0.1])
                    )
                bar.append(bar2)

            return bar

        mems = torch.tensor([[[1.8364, 0.2724, -1.4917, -0.4367, 0.8640]]])
        state = [
            torch.tensor([[[1.0517, 0.3848, -0.6472, 0.0823, 0.9116]]]),
            torch.tensor([[[1.0517, 0.3848, -0.6472, 0.0823, 0.9116]]]),
        ]
        i = torch.tensor(
            [
                [0.0313, -0.1487, -0.3846, -0.5321],
                [-1.7073, 1.3331, -0.0890, -1.4935],
                [-0.8314, -0.1862, -0.5935, 1.5232],
            ]
        )

        mems.tag = "MEMS"
        i.tag = "FOO"
        state[0].tag = "STATE_0"
        state[1].tag = "HMMM"

        exported = torch._dynamo.export(pre_attention_state_ops)(i, mems, state)
        out_graph = exported[0]

        nodes = list(out_graph.graph.nodes)
        placeholders = [node for node in nodes if node.op == "placeholder"]
        all_tags = []
        for placeholder in placeholders:
            if "tensor_dict" in placeholder.meta:
                all_tags.append(placeholder.meta["tensor_dict"]["tag"])

        self.assertEqual(all_tags, ["STATE_0", "HMMM"])

    def test_get_custom_tensor_attribute(self):
        def fn(x):
            return x.custom_attr * x

        x = torch.rand((2, 2))
        x.custom_attr = 3.14
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_sparse_output_inductor_should_break(self) -> None:
        # See https://github.com/pytorch/pytorch/issues/164823
        # We want consistent semantics here
        def forward(x: torch.Tensor) -> torch.Tensor:
            x_sparse = x.to_sparse()
            return x_sparse * 2

        test_tensor = torch.randn(10, 10)
        pt = forward(test_tensor)
        aot_eager = torch.compile(forward, backend="aot_eager")(test_tensor)
        self.assertEqual(pt, aot_eager)
        inductor = torch.compile(forward, backend="inductor")(test_tensor)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    @torch._dynamo.config.patch(assume_static_by_default=True)
    def test_symint_copy_into_unbacked_slice(self):
        @torch.compile(backend="eager")
        def fn(a, x):
            u0 = torch.tensor(x[0].to(torch.int64).item()).item()
            B, H, T, D = a.shape
            a_padding = torch.zeros((B, H, u0, D), dtype=torch.float64)
            b = torch.cat([a, a_padding], dim=2)
            c = torch.randn(B, H, 152, D)
            b[:, :, :152, :] = c
            return b

        x = torch.tensor([0])
        torch._dynamo.decorators.mark_unbacked(x, 0)
        a = torch.zeros((1, 16, 152, 96))

        # Previously would crash with guard on data dependent error
        fn(a, x)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_symint_fold_nontrivial_product_modulo(self):
        @torch.compile(fullgraph=True, backend="eager")
        def f(x):
            u0, u1 = x.tolist()
            # The condition should fold to true.
            if ((u0 + 10) * (u0 + 10)) % (u0 + 10) == 0:
                return torch.tensor(True)
            return torch.tensor(False)

        res = f(torch.tensor([20, 21]))
        self.assertEqual(torch.tensor(True), res)

    def test_tolist(self):
        # This should compile with no faluire.
        cnt = CompileCounterWithBackend("inductor")

        @torch.compile(fullgraph=False, backend=cnt)
        def func(a):
            a = a * 100
            u0, u1, u2, u3, u4 = a.tolist()
            return a * u0 * u1

        func(torch.tensor([1, 2, 3, 4, 5]))
        self.assertEqual(cnt.frame_count, 2)

    def test_grad_state_mutated(self):
        prior = torch.is_grad_enabled()
        value = None
        cnt = CompileCounter()

        @torch._dynamo.allow_in_graph
        def check_state():
            nonlocal value
            value = torch.is_grad_enabled()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            check_state()
            torch.set_grad_enabled(False)
            return x + 1

        try:
            torch.set_grad_enabled(True)
            fn(torch.randn(10))
            if value is not True:
                raise AssertionError(f"Expected value is True, got {value}")
            if torch.is_grad_enabled() is not False:
                raise AssertionError("Expected grad disabled after fn()")

            value = None
            torch.set_grad_enabled(True)
            fn(torch.randn(10))
            if value is not True:
                raise AssertionError(f"Expected value is True, got {value}")
            if torch.is_grad_enabled() is not False:
                raise AssertionError("Expected grad disabled after fn()")

            if cnt.frame_count != 1:
                raise AssertionError(f"Expected frame_count 1, got {cnt.frame_count}")
        finally:
            torch.set_grad_enabled(prior)

    def test_torch_guards_stack_frame_register_inlining(self):
        x = torch.tensor([0.5, 0.5])
        y = torch.tensor([0.75, 0.75, 0.75, 0.75])
        z = torch.tensor([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])

        def uwu_inline_me(x, y, z):
            r = torch.cat((x, x)) + y
            r2 = torch.cat((y, y)) + z
            return r, r2

        def fn(x, y, z):
            r, r2 = uwu_inline_me(x, y, z)
            return torch.mul(r, r), torch.mul(r2, r2)

        seen_frames = []
        import contextlib

        @contextlib.contextmanager
        def global_context_capture_fn(frame_summary):
            if frame_summary is not None:
                seen_frames.append(frame_summary)
            yield

        with mock.patch(
            "torch._guards.TracingContext.current_frame",
            side_effect=global_context_capture_fn,
        ):
            torch.compile(fn, backend="eager")(x, y, z)

        self.assertEqual(len(seen_frames), 1)
        self.assertEqual(seen_frames[0].name, "fn")
        self.assertEqual(seen_frames[0].line, "r, r2 = uwu_inline_me(x, y, z)")

    def test_fullgraph_capture(self):
        from torch._dynamo.convert_frame import fullgraph_capture
        from torch._dynamo.utils import dynamo_timed, get_metrics_context

        def foo(x):
            if x.shape[1] >= 3:
                return x + x.shape[0]
            else:
                return x - x.shape[0]

        x = torch.randn(4, 3)
        with (
            get_metrics_context(),
            dynamo_timed(""),
        ):
            capture_output = fullgraph_capture(foo, (x,))
            graph_capture_output = capture_output.graph_capture_output
            fn = graph_capture_output.build_guards(foo.__code__)

            for guard in graph_capture_output.output_graph.guards:
                if guard.source == torch._guards.GuardSource.SHAPE_ENV:
                    dynamic = guard.code_list is not None
                    if dynamic:
                        self.assertEqual(
                            guard.code_list,
                            [
                                "L['x'].stride()[0] == L['x'].size()[1]",
                                "2 <= L['x'].size()[0]",
                                "3 <= L['x'].size()[1]",
                            ],
                        )
                        self.assertTrue(
                            fn.guard_manager.check({"x": torch.randn(3, 3)})
                        )
                        self.assertTrue(
                            fn.guard_manager.check({"x": torch.randn(4, 4)})
                        )
                    else:
                        self.assertFalse(
                            fn.guard_manager.check({"x": torch.randn(3, 3)})
                        )
                        self.assertFalse(
                            fn.guard_manager.check({"x": torch.randn(4, 4)})
                        )
                    self.assertFalse(fn.guard_manager.check({"x": torch.randn(4, 2)}))
                    self.assertFalse(fn.guard_manager.check({"x": torch.randn(1, 3)}))
                    break

            backend_input = capture_output.backend_input
            self.assertTrue(fn.guard_manager.check({"x": x}))
        import_sources = {
            alias: importlib.import_module(module_name)
            for alias, module_name in graph_capture_output.import_sources.items()
        }
        self.assertEqual(
            foo(x),
            types.FunctionType(
                graph_capture_output.bytecode,
                {
                    **import_sources,
                    backend_input.backend_id: backend_input.graph_module,
                },
            )(x),
        )

    def test_torch_guards_stack_frame_register_inlining_deep(self):
        x = torch.tensor([0.5, 0.5])
        y = torch.tensor([0.75, 0.75, 0.75, 0.75])
        z = torch.tensor([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])

        def uwu_inline_me_deep(x, y):
            return torch.cat((x, x)) + y

        def uwu_inline_me(x, y, z):
            r = uwu_inline_me_deep(x, y)
            r2 = uwu_inline_me_deep(y, z)
            return r, r2

        def fn(x, y, z):
            r, r2 = uwu_inline_me(x, y, z)
            return torch.mul(r, r), torch.mul(r2, r2)

        seen_frames = []
        import contextlib

        @contextlib.contextmanager
        def global_context_capture_fn(frame_summary):
            if frame_summary is not None:
                seen_frames.append(frame_summary)
            yield

        with mock.patch(
            "torch._guards.TracingContext.current_frame",
            side_effect=global_context_capture_fn,
        ):
            torch.compile(fn, backend="eager")(x, y, z)

        self.assertEqual(len(seen_frames), 3)
        self.assertEqual(seen_frames[0].name, "fn")
        self.assertEqual(seen_frames[1].name, "uwu_inline_me")
        self.assertEqual(seen_frames[2].line, "r2 = uwu_inline_me_deep(y, z)")

    def test_scalar_tensor_is_equivalent_to_symint_argument(self):
        class GumbelTopKSampler(torch.nn.Module):
            def __init__(self, T, k):
                super().__init__()
                self.T = torch.nn.Parameter(
                    torch.tensor(T, dtype=torch.float32), requires_grad=False
                )
                self.k = torch.nn.Parameter(
                    torch.tensor(k, dtype=torch.int32), requires_grad=False
                )

            def sample_discrete(self, logits):
                threshold = torch.topk(logits, self.k, sorted=True)[0][..., -1]
                samples = torch.ge(logits.squeeze(1), threshold).float()
                return samples

            def forward(self, logits):
                dsamples = self.sample_discrete(logits)
                return dsamples

        x = torch.rand([4, 4, 4, 4])
        m = GumbelTopKSampler(T=4, k=4)
        orig_out = m(x)
        opt_m = torch.compile(backend="eager")(m)
        opt_out = opt_m(x)
        self.assertTrue(same(orig_out, opt_out))

    def test_scalar_tensor_is_equivalent_to_symint_list_argument(self):
        class Jitter(torch.nn.Module):
            def __init__(self, jitter_val):
                super().__init__()
                self.jitter_val = jitter_val

            def roll_tensor(self, input):
                h_shift = self.jitter_val - 1
                w_shift = self.jitter_val + 1
                return torch.roll(
                    torch.roll(input, shifts=h_shift, dims=2), shifts=w_shift, dims=3
                )

            def forward(self, input):
                return self.roll_tensor(input)

        x = torch.rand([4, 4, 4, 4])
        m = Jitter(jitter_val=4)
        orig_out = m(x)
        opt_m = torch.compile(backend="eager")(m)
        opt_out = opt_m(x)
        self.assertTrue(same(orig_out, opt_out))

    def test_scalar_tensor_is_equivalent_to_int_list_argument(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                permute = torch.tensor([0, 2, 1])
                x = input.permute(*permute)
                return x

        x = torch.randn(2, 3, 4)
        m = MyModel()
        orig_out = m(x)
        opt_m = torch.compile(backend="eager")(m)
        opt_out = opt_m(x)
        self.assertTrue(same(orig_out, opt_out))

    def test_torch_variable_hasattr(self):
        def fn(x):
            if hasattr(torch.nn, "Module"):
                return x * x
            return x + 1

        compiled_fn = torch.compile(backend="eager", fullgraph=True)(fn)

        x = torch.rand([4, 4])
        fn_out = fn(x)
        compiled_out = compiled_fn(x)
        self.assertTrue(same(fn_out, compiled_out))

    def test_jacfwd_one_hot_dynamic_compile(self):
        import torch.nn.functional as F

        MAX, BATCH = 3, 37

        def func(x, idxs):
            return x.square() * F.one_hot(idxs, MAX)

        def jacfunc(x, idxs):
            return torch.func.jacfwd(func, argnums=(0,))(x, idxs)

        idxs = torch.randint(MAX, (BATCH,), dtype=torch.int64)
        x = torch.rand((BATCH, MAX), dtype=torch.float64)
        eager = jacfunc(x, idxs)

        compiled = torch.compile(jacfunc, backend="eager", dynamic=True)
        out_comp = compiled(x, idxs)
        self.assertEqual(eager[0], out_comp[0])

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_out_variant_custom_op(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define(
                "split_with_sizes_copy(Tensor all_gather_output, SymInt[] all_gather_input_split_sizes, int dim=0, *, Tensor(a!)[] out) -> ()"
            )

            @torch.library.impl(lib, "split_with_sizes_copy", "Meta")
            @torch.library.impl(lib, "split_with_sizes_copy", "CPU")
            def split_with_sizes_copy(
                all_gather_output: torch.Tensor,
                all_gather_input_split_sizes: typing.List[int],
                dim: int,
                out: typing.List[torch.Tensor],
            ) -> None:
                torch.split_with_sizes_copy(
                    all_gather_output, all_gather_input_split_sizes, dim=dim, out=out
                )

            @torch.compile(backend="eager", fullgraph=True)
            def f1(all_gather_output, all_gather_input_split_sizes, dim, out):
                return torch.ops.mylib.split_with_sizes_copy(
                    all_gather_output, all_gather_input_split_sizes, dim, out=out
                )

            all_gather_output = torch.randn(2, 272)
            all_gather_input_split_sizes = [128, 8, 128, 8]
            dim = 1
            out = [
                torch.empty(2, 128),
                torch.empty(2, 8),
                torch.empty(2, 128),
                torch.empty(2, 8),
            ]
            f1(all_gather_output, all_gather_input_split_sizes, dim, out)

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define(
                "chunk_cat(Tensor[] tensors, int dim, int num_chunks, *, Tensor(a!) out) -> ()"
            )

            @torch.library.impl(lib, "chunk_cat", "Meta")
            @torch.library.impl(lib, "chunk_cat", "CPU")
            def chunk_cat(
                tensors: typing.List[torch.Tensor],
                dim: int,
                num_chunks: int,
                out: torch.Tensor,
            ) -> None:
                torch._chunk_cat(tensors, dim, num_chunks, out=out)

            @torch.compile(backend="eager", fullgraph=True)
            def f2(tensors, dim, num_chunks, out):
                return torch.ops.mylib.chunk_cat(tensors, dim, num_chunks, out=out)

            x = torch.zeros(100, dtype=torch.int64)
            tensors = [
                torch.randn(16, 16),
                torch.randn(16),
                torch.randn(16, 16),
                torch.randn(16),
            ]
            dim = 0
            num_chunks = 2
            out = torch.empty(2, 272)
            f2(tensors, dim, num_chunks, out)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_dim_order(self):
        @torch.compile(dynamic=False, fullgraph=True, backend="eager")
        def f(x):
            x = x.permute(3, 0, 2, 1)
            return x, x.dim_order()

        @torch.compile(dynamic=False, fullgraph=True, backend="eager")
        def g(x):
            return x.dim_order()

        @torch.compile(dynamic=False, fullgraph=True, backend="eager")
        def h0(xs, ambiguity_check=False):
            u0, u1, u2 = xs.tolist()
            torch._check(u2 >= u0)
            torch._check(u1 >= u0)
            # stride ordering still isn't unique here, should raise
            y = torch.empty_strided([4, 4, 4], [u0, u1, u2])
            return y.dim_order(ambiguity_check=ambiguity_check)

        @torch.compile(dynamic=False, fullgraph=True, backend="eager")
        def h1(xs, ambiguity_check=False):
            u0, u1, u2 = xs.tolist()
            y = torch.empty_strided([4, 4, 4], [u0, u0, u0])  # no ordering
            return y.dim_order(ambiguity_check=ambiguity_check)

        # check that for functions permuting contiguous input, the original stride is recovered with dim_order.
        def test(x):
            stride_inp = tuple(x.stride())
            f_out, f_order = f(x)
            self.assertEqual(stride_inp, tuple(f_out.stride(i) for i in f_order))

        # shape: [4, u0, 5, u1]
        x0 = torch.randn(4, 1, 5, 2)
        torch._dynamo.decorators.mark_unbacked(x0, 1)
        torch._dynamo.decorators.mark_unbacked(x0, 3)
        test(x0)

        # shape: [u0, u1, u2, u3]
        x1 = torch.randn(4, 1, 5, 2)
        for i in range(x1.ndim):
            torch._dynamo.decorators.mark_unbacked(x1, i)
        test(x1)

        # custom strides (all integers)
        x2 = torch.randn(10000)
        x2 = x2.as_strided([4, 4, 4, 4], [1, 2, 4, 8])
        if g(x2) != (3, 2, 1, 0):
            raise AssertionError(f"Expected g(x2) == (3, 2, 1, 0), got {g(x2)}")

        # custom unbacked strides with no ordering: ambiguity check should raise
        xs = torch.tensor([2, 3, 4])
        h0(xs)
        with self.assertRaisesRegex(
            torch._dynamo.exc.TorchRuntimeError,
            r"The tensor does not have unique dim order.",
        ):
            h0(xs, ambiguity_check=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.TorchRuntimeError,
            r"The tensor does not have unique dim order.",
        ):
            h1(xs, ambiguity_check=True)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_tolist_scalar(self):
        def fn(x):
            new_list = []
            for i in x.tolist():
                new_list.append(i * 4)
            return new_list

        x = torch.tensor([3])
        eager = fn(x)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)(x)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_tolist_1d(self):
        def fn(x):
            new_list = []
            for i in x.tolist():
                new_list.append(i * 4)
            return new_list

        x = torch.tensor([2, 1])
        eager = fn(x)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)(x)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_tolist_kd(self):
        def fn(x):
            new_list = []
            for i in x.tolist():
                new_list.append(i * 4)
            return new_list

        x = torch.tensor([[[2, 1], [2, 1], [2, 1]], [[2, 1], [2, 1], [2, 1]]])
        eager = fn(x)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)(x)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    @patch.object(torch._dynamo.config, "specialize_int", True)
    def test_tolist_0d(self):
        def fn(x):
            new_list = []
            i = x.tolist()
            new_list.append(i * 4)
            return new_list

        x = torch.tensor(42)
        eager = fn(x)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)(x)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

    @patch.object(torch._dynamo.config, "assume_static_by_default", False)
    @patch.object(torch._dynamo.config, "automatic_dynamic_shapes", False)
    def test_tolist_kd_dynamic(self):
        def fn(x):
            new_list = []
            i = x.tolist()
            new_list.append(i * 4)
            return new_list, x * 10

        x = torch.randint(3, 5, [5, 5])
        eager = fn(x)
        counter = CompileCounter()
        compiled_fn = torch.compile(fn, backend=counter, fullgraph=False)
        compiled = compiled_fn(x)
        self.assertEqual(eager, compiled)
        self.assertEqual(counter.frame_count, 1)

        # Value change, no recompiles
        x = torch.randint(7, 9, [5, 5])
        compiled_fn(x)
        self.assertEqual(counter.frame_count, 1)

        # Size change, forced recompiles
        x = torch.randint(3, 5, [3, 3])
        compiled_fn(x)
        self.assertEqual(counter.frame_count, 2)

    def test_tolist_float(self):
        def fn(x):
            new_list = []
            for i in x.tolist():
                new_list.append(i * 4)
            return new_list

        x = torch.tensor(
            [[[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]], [[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]]]
        )
        eager = fn(x)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter)(x)
        self.assertEqual(eager, compiled)
        # Nothing to compile here
        self.assertEqual(counter.frame_count, 0)

    def test_default_args_device_dtype(self):
        class Foo:
            def __init__(
                self,
                dtype: torch.dtype = torch.float16,
                device: torch.device = torch.device("cpu"),
            ) -> None:
                self.value = torch.tensor(10, dtype=dtype, device=device)

        def fn():
            return Foo().value + 1

        opt_func = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn()
        res = opt_func()
        self.assertEqual(ref, res)

    def test_torch_dtype_python_type(self):
        def fn(target):
            target_dtype = target.dtype
            a = torch.zeros(2, 3, dtype=target_dtype)
            # Constant assert at trace time
            assert isinstance(target_dtype, torch.dtype)  # noqa: S101
            b = torch.zeros(2, 3, dtype=target_dtype)
            c = torch.zeros(2, 3, dtype=target_dtype)
            return a + b + c

        from torch._dynamo.variables import ConstantVariable

        dtype = torch.float16
        expected_variable = ConstantVariable(dtype)
        self.assertEqual(expected_variable.python_type(), type(dtype))

        opt_func = torch.compile(fn, backend="eager", fullgraph=True)
        a = torch.tensor([2, 3], dtype=dtype)
        res = opt_func(a)
        self.assertIsInstance(res, torch.Tensor)

    def test_storage_return(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            y = torch.sin(x + 1)
            storage = x.untyped_storage()
            storage.resize_(0)
            y = torch.cos(y)
            return y, storage

        x = torch.randn(10)
        expected = torch.cos(torch.sin(x + 1))
        y, s = fn(x)
        self.assertEqual(y, expected)
        self.assertEqual(x.untyped_storage().size(), 0)
        self.assertIs(s, x.untyped_storage())

    def test_default_dtype_change(self):
        @torch.compile(backend="eager")
        def foo():
            def inner(a, b, res_dtype):
                print(a, b, res_dtype)
                self.assertEqual(torch.result_type(a, b), res_dtype)

            inner(torch.tensor(1, device="cpu"), 1.0, torch.get_default_dtype())

        with set_default_dtype(torch.float):
            foo()
        with set_default_dtype(torch.double):
            foo()

    @wrapDeterministicFlagAPITest
    def test_backward_deterministic_mode_mismatch_warning(self):
        @torch.compile(backend="aot_eager")
        def func(a, b):
            return a + b

        for forward_deterministic, backward_deterministic in itertools.product(
            [True, False], [True, False]
        ):
            torch.use_deterministic_algorithms(forward_deterministic)
            a = torch.randn(10, requires_grad=True)
            res = func(a, 1)
            grad = torch.ones_like(res)
            torch.use_deterministic_algorithms(backward_deterministic)

            if not forward_deterministic and backward_deterministic:
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^This compiled backward function is being run with torch\.use_deterministic_algorithms",
                ):
                    res.backward(grad)

            else:
                res.backward(grad)

    @skipIfWindows(
        msg="AssertionError: False is not true : Encountered an unexpected fallback to 'aten pow' in dynamo compiled code"
    )
    @unittest.skipIf(
        torch._inductor.config.cpu_backend != "cpp",
        "Skip for non cpp backend CPU as comments contain 'aten.pow' ",
    )
    def test_torch_dynamo_codegen_pow(self):
        def pow(x):
            return x**2

        x = np.arange(8)
        pow_opt = torch.compile(pow, backend="eager")

        actual, source_code = run_and_get_code(pow_opt, x)
        expect = pow(x)

        self.assertEqual(expect, actual)

        self.assertTrue(
            all("aten.pow" not in code for code in source_code),
            msg="Encountered an unexpected fallback to 'aten pow' in dynamo compiled code",
        )

    def test_grad_none(self):
        def fn(x, y):
            x.grad = torch.abs(y)
            x.grad.add_(y)
            return torch.abs(y)

        y = torch.arange(4).reshape(2, 2).to(torch.float)
        x = torch.randn(2, 2)
        x.grad = None

        z = fn(x, y)
        ref_y = torch.clone(z).detach()
        ref_x_grad = torch.clone(x.grad).detach()

        y = torch.arange(4).reshape(2, 2).to(torch.float)
        x = torch.randn(2, 2)
        x.grad = None

        opt_fn = torch.compile(fn, backend="eager")
        z = opt_fn(x, y)
        self.assertEqual(z, ref_y)
        self.assertEqual(x.grad, ref_x_grad)

    def test_grad_non_none(self):
        def fn(x, y):
            x.grad.add_(y)
            return torch.abs(y)

        y = torch.ones(2, 2)
        x = torch.randn(2, 2)
        x.grad = torch.arange(4).reshape(2, 2).to(torch.float)

        z = fn(x, y)
        ref_y = torch.clone(z).detach()
        ref_x_grad = torch.clone(x.grad).detach()

        y = torch.ones(2, 2)
        x = torch.randn(2, 2)
        x.grad = torch.arange(4).reshape(2, 2).to(torch.float)

        cnt = torch._dynamo.testing.CompileCounterWithBackend("eager")
        opt_fn = torch.compile(fn, backend=cnt)
        z = opt_fn(x, y)

        # Ensure that the generated graph returns only one output. We want the
        # add_ on the grad to be part of the graph itself, so that inductor can
        # theoretically move the add_ and resulting copy_ nodes at the right
        # place to free memory.
        self.assertEqual(len(list(cnt.graphs[0].graph.nodes)[-1].all_input_nodes), 1)
        self.assertEqual(z, ref_y)
        self.assertEqual(x.grad, ref_x_grad)

    def test_new_with_int_list(self):
        # Make sure torch.Tensor.new(int argument list) behaves the same on dynamo.
        def fn(x):
            return x.new(*x.size()) + 5

        optfn = torch.compile(backend="eager")(fn)

        x = torch.arange(10).view(2, 5)

        expected = fn(x)
        actual = optfn(x)

        self.assertEqual(expected.dtype, actual.dtype)
        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(expected.stride(), actual.stride())
        self.assertEqual(expected.storage_offset(), actual.storage_offset())

    def test_dynamic_shapes_as_strided(self):
        def fn(t, new_size, new_stride):
            tmp = t.as_strided(new_size, new_stride)
            tmp = tmp.view(-1)
            return t * tmp.sum()

        optfn = torch.compile(backend="eager", dynamic=True)(fn)

        x = torch.randn(3)
        new_size = [0, 3]
        new_stride = [3, 1]

        expected = fn(x, new_size, new_stride)
        actual = optfn(x, new_size, new_stride)

        self.assertEqual(expected.dtype, actual.dtype)
        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(expected.stride(), actual.stride())
        self.assertEqual(expected.storage_offset(), actual.storage_offset())

    def test_data_descriptor_priority_over_instance_dict(self):
        # CPython: data descriptors on the type take priority over instance
        # dict values. Verify Dynamo follows this ordering.
        class Foo:
            @property
            def x(self):
                return 10

        foo = Foo()
        # Manually put a different value in the instance dict.
        # The property (data descriptor) should still win.
        foo.__dict__["x"] = 999

        def fn(t):
            return t + foo.x

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        t = torch.randn(4)
        ref = fn(t)
        res = opt_fn(t)
        self.assertEqual(ref, res)
        self.assertEqual(ref, t + 10)

    def test_assert_size_stride(self):
        x = torch.randn(2, 3, 4)
        with self.assertRaisesRegex(
            AssertionError,
            "expected size 2==5, stride 12==9 at dim=0; expected size 3==6, stride 4==9 at dim=1; expected size 4==7, stride 1==10 at dim=2",
        ):
            torch._C._dynamo.guards.assert_size_stride(x, (5, 6, 7), (9, 9, 10))

    def test_data_ptr_graph_break_builtin(self):
        def f(a, b):
            # builtin + not implemented for DataPtrVariable
            return a.data_ptr() + b.data_ptr()

        a = torch.randn(4)
        b = torch.randn(5)

        # make sure there is a graph break
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(f, backend="eager", fullgraph=True)(a, b)

        torch._dynamo.reset()

        expected = f(a, b)
        actual = torch.compile(f, backend="eager")(a, b)

        self.assertEqual(expected, actual)

    def test_data_ptr_graph_break_aten(self):
        def f(a):
            # torch.add not implemented for DataPtrVariable
            return torch.add(a, a.data_ptr())

        a = torch.randn(4)

        counters.clear()

        expected = f(a)
        actual = torch.compile(f, backend="eager")(a)

        self.assertEqual(expected, actual)
        self.assertTrue(len(counters["graph_break"]) > 0)
        counters.clear()

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    def test_new_tensor_break(self):
        a = torch.tensor([1, 0, 0, 5])

        cases = {
            "scalar": lambda a: a.new_tensor([a.nonzero().squeeze(-1).numel()]),
            "multi": lambda a: (
                n := a.nonzero().squeeze(-1).numel(),
                a.new_tensor([n, n + 1, n * 2]),
            )[-1],
            "mixed_shape": lambda a: (
                n := a.nonzero().squeeze(-1).numel(),
                a.new_tensor([n * a.shape[0], n + a.shape[0], a.shape[0] - n]),
            )[-1],
            "nested": lambda a: (
                n := a.nonzero().squeeze(-1).numel(),
                a.new_tensor([[n, n + 1], [n * 2, n - 1]]),
            )[-1],
            "with_zero": lambda a: (
                n := a.nonzero().squeeze(-1).numel(),
                a.new_tensor([0, n, n * n]),
            )[-1],
        }

        for name, fn in cases.items():
            with self.subTest(case=name):
                self.assertEqual(
                    torch.compile(fn, fullgraph=True, backend="eager")(a),
                    fn(a),
                )

    def test_make_contiguous_strides_for_under_compile(self):
        # is_nested_int and sym_max must be traceable under Dynamo.
        from torch._prims_common import make_contiguous_strides_for

        def fn(x):
            strides = make_contiguous_strides_for(x.shape)
            return x.as_strided(x.shape, strides)

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)
        x = torch.randn(4, 8)
        result = compiled_fn(x)
        self.assertEqual(result, x)

        x2 = torch.randn(7, 8)
        result2 = compiled_fn(x2)
        self.assertEqual(result2, x2)

    def test_requires_grad_changes_dynamo_graph(self):
        # requires_grad_() on a graph input graph-breaks, so no fullgraph
        def fn(x):
            x.requires_grad_()
            if x.requires_grad:
                return x * 2
            return x + 1

        x = torch.randn(3, 3)
        opt_fn = torch.compile(fn)
        result = opt_fn(x)
        self.assertEqual(result, x * 2)

    def test_requires_grad_backward_outside_compile(self):
        # requires_grad_() on a graph input graph-breaks, but eager fallback
        # produces correct results.
        def fn(x):
            x.requires_grad_()
            return (x * 2).sum()

        x_ref = torch.randn(3, 3)
        x_test = x_ref.clone()

        fn(x_ref).backward()
        torch.compile(fn)(x_test).backward()

        self.assertEqual(x_ref.grad, x_test.grad)

    def test_detach_inplace_on_intermediate_updates_metadata(self):
        def fn(x):
            y = x * 2
            y.detach_()
            return y + 1, y.requires_grad, y.grad_fn is None

        x = torch.randn(3, 3, requires_grad=True)
        ref = fn(x.clone())
        result = torch.compile(fn, backend="eager", fullgraph=True)(x.clone())

        self.assertEqual(ref, result)
        self.assertFalse(result[1])
        self.assertTrue(result[2])

    def test_requires_grad_on_intermediate(self):
        def fn(x):
            y = x * 2
            y.requires_grad_()
            return y

        x = torch.randn(3, 3)

        # fullgraph=True should error with actionable message
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            r"requires_grad_\(\)(.|\n)*\.detach\(\)",
        ):
            torch.compile(fn, fullgraph=True)(x)

        # Without fullgraph, falls back to eager and is correct
        result = torch.compile(fn)(x)
        self.assertTrue(result.requires_grad)
        self.assertEqual(fn(x), result)

    def test_requires_grad_on_intermediate_derived_returned(self):
        def fn(x):
            y = x * 2
            y.requires_grad_()
            return y * 3

        x = torch.randn(3, 3)

        # Derived tensor also loses requires_grad — should error with message
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            r"requires_grad_\(\)(.|\n)*\.detach\(\)",
        ):
            torch.compile(fn, fullgraph=True)(x)

        # Without fullgraph, falls back to eager and is correct
        result = torch.compile(fn)(x)
        ref = fn(x)
        self.assertTrue(result.requires_grad)
        self.assertEqual(ref, result)

    def test_requires_grad_on_intermediate_partial_graph(self):
        # When requires_grad_() on a source-less intermediate leaks as output,
        # Dynamo should restart and graph break at requires_grad_(), capturing
        # ops before it in a compiled graph (partial acceleration).
        def fn(x):
            a = x.sin()
            b = a.cos()
            b.requires_grad_()
            return b

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        x = torch.randn(3, 3)
        result = torch.compile(fn, backend=backend)(x)
        self.assertEqual(result, fn(x))
        self.assertTrue(result.requires_grad)
        # The graph should capture the ops before requires_grad_()
        self.assertEqual(len(backend.graphs), 1)
        # Dynamic shapes adds shape guards to the graph, skip the exact check
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(
                backend.graphs[0].code.strip(),
                """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    a = l_x_.sin();  l_x_ = None
    b = a.cos();  a = None
    return (b,)""",
            )

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    def test_requires_grad_on_intermediate_not_returned(self):
        def fn(x):
            y = x * 2
            y.requires_grad_()
            loss = (y * 3).sum()
            loss.backward()
            return y.grad

        x = torch.randn(3, 3)

        ref = fn(x.clone())
        result = torch.compile(fn, fullgraph=True)(x.clone())
        self.assertEqual(ref, result)

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    def test_requires_grad_intermediate_backward_grad_used_in_compute(self):
        # Use the grad result in further computation within compile
        def fn(x):
            y = x * 2
            y.requires_grad_()
            loss = (y**2).sum()
            loss.backward()
            return y.grad * 2 + 1

        x = torch.randn(3, 3)

        ref = fn(x.clone())
        result = torch.compile(fn, fullgraph=True)(x.clone())
        self.assertEqual(ref, result)

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    def test_requires_grad_intermediate_chunked_loss_backward(self):
        # Mirrors the TxtUnembedding pattern: forward compute, detach, make
        # new leaf, chunked loss with per-chunk backward, then propagate
        # accumulated grad back to the original input via h.backward().
        def fn(x, targets):
            # Forward computation before detach (e.g. transformer layers)
            h = x * 2 + 1
            x_detached = h.detach().requires_grad_()
            chunksz = x_detached.shape[0] // 2
            total_loss = torch.tensor(0.0)
            for start in range(0, x_detached.shape[0], chunksz):
                chunk = x_detached[start : start + chunksz]
                chunk_targets = targets[start : start + chunksz]
                logits = chunk @ torch.eye(chunk.shape[-1])
                loss = torch.nn.functional.cross_entropy(logits, chunk_targets)
                loss.backward()
                total_loss = total_loss + loss.detach()
            # Propagate chunked grad back through the forward computation
            h.backward(x_detached.grad)
            return x.grad, total_loss

        x_ref = torch.randn(4, 8, requires_grad=True)
        targets = torch.randint(0, 8, (4,))

        x_test = x_ref.clone().detach().requires_grad_(True)
        ref_grad, ref_loss = fn(x_ref, targets)
        compiled_grad, compiled_loss = torch.compile(fn, fullgraph=True)(
            x_test, targets
        )
        self.assertEqual(ref_grad, compiled_grad)
        self.assertEqual(ref_loss, compiled_loss)
        # Verify grad propagated to the input
        self.assertEqual(x_ref.grad, x_test.grad)

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    def test_requires_grad_intermediate_backward_and_return_detached(self):
        # Returning a detached version of the tainted tensor is safe — detach()
        # strips requires_grad so AOTAutograd functionalization can't lose anything.
        def fn(x):
            y = x * 2
            y.requires_grad_()
            out = y * 3
            loss = out.sum()
            loss.backward()
            return y.grad, out.detach()

        x = torch.randn(3, 3)

        ref_grad, ref_out = fn(x.clone())
        compiled_grad, compiled_out = torch.compile(fn, fullgraph=True)(x.clone())
        self.assertEqual(ref_grad, compiled_grad)
        self.assertEqual(ref_out, compiled_out)
        self.assertFalse(compiled_out.requires_grad)

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    def test_requires_grad_intermediate_metadata_checks(self):
        # After requires_grad_() on an intermediate, requires_grad and is_leaf
        # should report correctly and be usable in control flow.
        def fn(x):
            y = x * 2
            y.requires_grad_()
            if y.requires_grad and y.is_leaf:
                loss = (y * 3).sum()
                loss.backward()
                return y.grad
            return y

        x = torch.randn(3, 3)
        ref = fn(x.clone())
        result = torch.compile(fn, fullgraph=True)(x.clone())
        self.assertEqual(ref, result)

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    def test_requires_grad_intermediate_side_effect_global(self):
        # requires_grad_() on intermediate, then store grad in a global
        saved = {}

        def fn(x):
            y = x * 2
            y.requires_grad_()
            loss = (y**2).sum()
            loss.backward()
            saved["grad"] = y.grad
            return y.grad.clone()

        x = torch.randn(3, 3)
        ref = fn(x.clone())
        saved_ref = saved["grad"].clone()
        saved.clear()

        result = torch.compile(fn, fullgraph=True)(x.clone())
        self.assertEqual(ref, result)
        self.assertEqual(saved_ref, saved["grad"])


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
