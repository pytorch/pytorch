try:
    from ._repros_common import (
        AuxRequest,
        create_block_mask,
        defaultdict,
        E4M3_MAX_POS,
        e4m3_type,
        flex_attention,
        functools,
        inspect,
        mock,
        nn,
        PLATFORM_SUPPORTS_FLASH_ATTENTION,
        PLATFORM_SUPPORTS_FP8,
        pytree,
        rand_strided,
        requires_cuda,
        same,
        skipIfHpu,
        sys,
        TEST_CUDA,
        TEST_WITH_ROCM,
        torch,
        TorchDispatchMode,
        types,
        unittest,
        warnings,
    )
except ImportError:
    from _repros_common import (
        AuxRequest,
        create_block_mask,
        defaultdict,
        E4M3_MAX_POS,
        e4m3_type,
        flex_attention,
        functools,
        inspect,
        mock,
        nn,
        PLATFORM_SUPPORTS_FLASH_ATTENTION,
        PLATFORM_SUPPORTS_FP8,
        pytree,
        rand_strided,
        requires_cuda,
        same,
        skipIfHpu,
        sys,
        TEST_CUDA,
        TEST_WITH_ROCM,
        torch,
        TorchDispatchMode,
        types,
        unittest,
        warnings,
    )


class ReproTestsDeviceMixin:
    def test_sub_alpha_scalar_repro(self, device):
        @torch.compile(backend="aot_eager")
        def f(x):
            return x.sub(1, alpha=2)

        f(torch.ones(2, device=device, dtype=torch.float64))

    @requires_cuda
    def test_norm_dtype(self, device):
        def foo(_stack0):
            getitem = _stack0[(slice(None, None, None), -1)]
            _stack0 = None
            normalize = torch.nn.functional.normalize(getitem, p=2, dim=1)
            getitem = None
            return (normalize,)

        args = [((2, 50, 256), (1, 256, 1), torch.float16, device, False)]
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]

        torch.compile(foo, backend="aot_eager_decomp_partition")
        with torch.cuda.amp.autocast(enabled=True):
            ref = foo(*args)[0]
            res = foo(*args)[0]
            self.assertEqual(ref.dtype, res.dtype)

            self.assertTrue(same(res, ref))

    def test_guard_default_device(self, device):
        try:
            torch.set_default_device(device)

            counter = torch._dynamo.testing.CompileCounter()

            @torch._dynamo.optimize(counter)
            def f():
                x = torch.randn(3)
                return x * 2

            self.assertEqual(f().device.type + ":0", device)
            self.assertEqual(counter.frame_count, 1)

            torch.set_default_device("cpu")

            self.assertEqual(f().device.type, "cpu")
            self.assertEqual(counter.frame_count, 2)

        finally:
            torch.set_default_device(None)

    @skipIfHpu
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "flash attention not supported",
    )
    def test_flash_attn_backward_mixed_strides(self, device):
        # in this repro, "grad_out" and "value" are transposed tensors,
        # but "key" and "value" are contiguous
        def gen_inputs(device):
            return (
                torch.randn(
                    2, 513, 16, 64, dtype=torch.float16, device=device
                ).transpose(1, 2),
                torch.randn(2, 16, 513, 64, dtype=torch.float16, device=device),
                torch.randn(2, 16, 513, 64, dtype=torch.float16, device=device),
                torch.randn(
                    2, 513, 16, 64, dtype=torch.float16, device=device
                ).transpose(1, 2),
                torch.randn(2, 16, 513, 64, dtype=torch.float16, device=device),
                torch.randn(2, 16, 513, device=device),
                None,
                None,
                513,
                513,
                0.0,
                False,
                torch.tensor(1, dtype=torch.int64),
                torch.tensor(1, dtype=torch.int64),
            )

        inps_device = gen_inputs(device)
        inps_meta = gen_inputs("meta")
        (
            out1_ref,
            out2_ref,
            out3_ref,
        ) = torch.ops.aten._scaled_dot_product_flash_attention_backward(
            *inps_device, scale=0.125
        )
        from torch._meta_registrations import meta__scaled_dot_product_flash_backward

        out1_test, out2_test, out3_test = meta__scaled_dot_product_flash_backward(
            *inps_meta, scale=0.125
        )

        self.assertEqual(out1_ref.shape, out1_test.shape)
        self.assertEqual(out1_ref.stride(), out1_test.stride())
        self.assertEqual(out2_ref.shape, out2_test.shape)
        self.assertEqual(out2_ref.stride(), out2_test.stride())
        self.assertEqual(out3_ref.shape, out3_test.shape)
        self.assertEqual(out3_ref.stride(), out3_test.stride())

    def test_megablocks_moe(self, device):
        try:
            from megablocks.layers import moe
            from megablocks.layers.arguments import Arguments
        except ImportError as e:
            raise unittest.SkipTest("requires megablocks") from e
        bs, sl, hs, num_experts, top_k = (16, 1024, 512, 1, 1)
        args = Arguments(
            hidden_size=hs,
            ffn_hidden_size=hs * 2,
            moe_num_experts=num_experts,
            moe_capacity_factor=1,
            moe_top_k=top_k,
        )
        moe_mlp = moe.MoE(args)
        # moe_mlp.cuda(torch.cuda.current_device()).half()
        moe_mlp.device(torch.device.current_device()).half()
        x = torch.randn(sl, bs, hs).device().half()
        out1, _ = moe_mlp(x)
        out2, _ = torch.compile(moe_mlp, backend="eager")(x)
        self.assertEqual(out1, out2)

    def test_tensor_size_hasattr(self):
        def fn(x):
            if hasattr(x, "size"):
                x = x * 2
            if hasattr(x, "stride"):
                x = x * 3
            return x * 5

        x = torch.ones(4)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    @requires_cuda
    def test_memleak_when_graph_input_has_tensor_attr(self, device):
        @torch.compile(backend="eager")
        def f(x):
            x.add_(1)

        mem_before = torch.cuda.memory_allocated()

        x = torch.ones(2, device=device)
        x.foo = torch.zeros(2, device=device)
        f(x)
        del x.foo
        del x
        mem_after = torch.cuda.memory_allocated()
        self.assertEqual(mem_before, mem_after)

        # check when non-tensor data structure attribute contains a tensor
        @torch.compile(backend="eager")
        def f(x):
            x.add_(1)

        mem_before = torch.cuda.memory_allocated()
        x = torch.ones(2, device=device)
        x.foo = [torch.zeros(2, device=device) for _ in range(5)]
        f(x)
        del x.foo
        del x
        mem_after = torch.cuda.memory_allocated()
        self.assertEqual(mem_before, mem_after)

        # check with tensor refcycle
        @torch.compile(backend="eager")
        def g(x, y):
            return x + y

        mem_before = torch.cuda.memory_allocated()
        x = torch.ones(2, device=device)
        y = torch.zeros(2, device=device)
        x.foo = [y]
        y.foo = [x]
        g(x, y)
        del x.foo
        del y.foo
        del x
        del y
        mem_after = torch.cuda.memory_allocated()
        self.assertEqual(mem_before, mem_after)

    def test_udf_class_source(self):
        class Foo:
            pass

        def fn(x):
            foo = Foo()
            bar = type(foo)()  # noqa: F841
            return torch.cos(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_truthiness_of_symints_no_recompiles(self, device):
        def f(x):
            numel = x.numel()
            if numel:
                return x + 1
            else:
                return x + 2

        cnt = torch._dynamo.testing.CompileCounter()
        f_compiled = torch.compile(f, backend=cnt, dynamic=True)

        x1 = torch.randn(4)
        _ = f_compiled(x1)
        x2 = torch.randn(5)
        _ = f_compiled(x2)

        self.assertEqual(cnt.frame_count, 1)

    @requires_cuda
    def test_sdpa_dynamic_shapes(self, device):
        def f(x, s0, s1, s2):
            q = x.view(2, s0, s2, s0)
            return torch._C._nn.scaled_dot_product_attention(
                q, q, q, attn_mask=None, dropout_p=0.0, is_causal=True
            )

        x = torch.randn(2, 32, 4096, dtype=torch.bfloat16, device=device)
        x_ref = x.clone().detach().requires_grad_()
        s0 = 32
        s1 = 64
        s2 = 128

        f_compiled = torch.compile(f, dynamic=True, backend="eager")

        with torch._dynamo.config.patch(assume_static_by_default=False):
            out_ref = f(x_ref, s0, s1, s2)
            out = f_compiled(x, s0, s1, s2)
            self.assertEqual(out_ref, out)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, "requires gpu with fp8 support")
    @requires_cuda
    def test_partitioner_saves_weights_for_bw(self):
        def mul_tiled(a, *bs):
            for b in bs:
                a = a.unflatten(0, (b.shape[0], -1)).unflatten(-1, (b.shape[-1], -1))
                a = a * b[:, None, :, None]
                a = a.flatten(end_dim=1).flatten(start_dim=-2)
            return a

        def scale(t, amax_t):
            max_v = E4M3_MAX_POS
            scale_t = torch.clamp(amax_t.float(), min=1e-12) / max_v
            t_fp8 = mul_tiled(t, scale_t.reciprocal()).to(e4m3_type)
            return t_fp8, scale_t

        def matmul(first, amax_first, second_t, amax_second_t, bias):
            first_fp8, scale_first = scale(first, amax_first)
            second_t_fp8, scale_second_t = scale(second_t, amax_second_t)
            post_scales = []
            post_bias = None
            post_scales = [scale_first, scale_second_t.t()]
            scale_first = scale_first.new_ones((1, 1))
            scale_second_t = scale_second_t.t().new_ones((1, 1))
            post_bias, bias = bias, None
            res = torch._scaled_mm(
                first_fp8,
                second_t_fp8.t(),
                scale_a=scale_first,
                scale_b=scale_second_t.t(),
                bias=bias,
                out_dtype=torch.bfloat16,
                use_fast_accum=False,
            )
            res = mul_tiled(res, *post_scales).to(torch.bfloat16)
            if post_bias is not None:
                res += post_bias
            return res

        @torch.compiler.allow_in_graph
        class Fp8LinearFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, a, b_t, bias):
                amax_a = a.abs().unflatten(-1, (1, -1)).amax(dim=-1)
                amax_b_t = b_t.abs().unflatten(-1, (1, -1)).amax(dim=-1)
                out = matmul(a, amax_a, b_t, amax_b_t, bias)
                ctx.a_requires_grad = a.requires_grad
                ctx.b_requires_grad = b_t.requires_grad
                ctx.bias_requires_grad = (
                    bias.requires_grad if bias is not None else False
                )
                ctx.save_for_backward(a, b_t, amax_b_t)
                return out

            @staticmethod
            def backward(ctx, grad_out):
                a, b_t, amax_b_t = ctx.saved_tensors
                # Workaround for https://github.com/pytorch/pytorch/issues/141881.
                # The partitioner would pre-compute the transposed scaling of the weight
                # in the forward (as it's most efficient, but it actually uses too much
                # memory). We prevent that by making the scaling depend on the gradient
                # in a way that has no effect and will be optimized away later.
                # Care is needed to support tensor parallelism and circumvent bugs.
                #        b_t = b_t + grad_out[:1, :, None].squeeze(0) * 0
                if ctx.a_requires_grad:
                    b = b_t.t().contiguous()
                    amax_grad_out = grad_out.abs().unflatten(-1, (1, -1)).amax(dim=-1)
                    amax_b = amax_b_t.t().unflatten(-1, (1, -1)).amax(dim=-1)
                    amax_b = amax_b.repeat_interleave(
                        b.shape[0] // amax_b.shape[0], dim=0, output_size=b.shape[0]
                    )
                    grad_a = matmul(grad_out, amax_grad_out, b, amax_b, None)
                else:
                    grad_a = None
                if ctx.b_requires_grad:
                    grad_b = grad_out.t() @ a
                else:
                    grad_b = None
                if ctx.bias_requires_grad:
                    grad_bias = grad_out.sum(dim=0)
                else:
                    grad_bias = None
                return grad_a, grad_b, grad_bias

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Parameter(
                    torch.randn(
                        64, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True
                    )
                )
                self.b = torch.nn.Parameter(
                    torch.randn(
                        64, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True
                    )
                )
                self.bias = torch.nn.Parameter(
                    torch.randn(
                        64, dtype=torch.bfloat16, device="cuda", requires_grad=True
                    )
                )

        class CustomLinear(torch.nn.Linear):
            def forward(self, input: torch.Tensor) -> torch.Tensor:
                out = Fp8LinearFn.apply(
                    input.flatten(end_dim=-2), self.weight, self.bias
                )
                out = out.unflatten(0, input.shape[:-1])
                return out

        m = CustomLinear(64, 64, dtype=torch.bfloat16, device="cuda")
        m = torch.compile(m, backend="aot_eager")

        # simple mode to track how many collective ops we saw in the backward
        class TrackingMode(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.ops_counter = defaultdict(int)

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                rs = func(*args, **kwargs)
                self.ops_counter[func] += 1
                return rs

        a = torch.randn(64, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        out = m(a)
        with TrackingMode() as mode:
            out.sum().backward()
        # If you print out the AOT fw and bw graphs,
        # the main thing to look for is that both weights (primals_1/primals_2)
        # *are* saved for backward, and become back inputs.
        # The easier-to-test thing I'm checking for here is that the recompute
        # on primals_2 happens in the backward. With the recompute,
        # there are 5 _to_copy ops in the backward. Without it, there are 4
        # (aka if you set torch._functorch.config.treat_parameters_as_free_to_save = False)
        self.assertEqual(mode.ops_counter[torch.ops.aten._to_copy.default], 5)

    def test_getattr_return(self):
        _WrapperDescriptor = type(type.__call__)
        _MethodWrapper = type(all.__call__)
        _ClassMethodWrapper = type(int.__dict__["from_bytes"])

        _NonUserDefinedCallables = (
            _WrapperDescriptor,
            _MethodWrapper,
            _ClassMethodWrapper,
            types.BuiltinFunctionType,
        )

        def _signature_get_user_defined_method(cls, method_name):
            try:
                meth = getattr(cls, method_name)
            except AttributeError:
                return
            else:
                if not isinstance(meth, _NonUserDefinedCallables):
                    # Once '__signature__' will be added to 'C'-level
                    # callables, this check won't be necessary
                    return meth

        def fn(x):
            s = _signature_get_user_defined_method(type(torch.nn.Linear), "__call__")
            if s is None:
                return torch.cos(x)

            return torch.sin(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_data_dependent_error_log_no_print(self):
        # This is a regression test case for
        # https://github.com/pytorch/pytorch/pull/149831
        from io import StringIO

        capturedOutput = StringIO()
        sys.stderr = capturedOutput

        @torch.compile(fullgraph=True, backend="eager")
        def func(a):
            if a.sum() > 0:
                return a + 1
            return a + 2

        a = torch.rand(10, 10)
        try:
            func(a)
        except Exception:
            pass
        sys.stderr = sys.__stderr__

        # Make sure we don't _print_ out the graph module.
        output = capturedOutput.getvalue()
        self.assertNotIn("class GraphModule", output)

    def test_deepcopy_constant_tensor_in_aot_bwd(self):
        class Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x + 1

            @staticmethod
            def backward(ctx, grad_out):
                return grad_out * torch.tensor(2) * grad_out.shape[0]

        def f(x):
            return Fn.apply(x)

        x = torch.randn(8, requires_grad=True)
        out = f(x)  # should not raise
        c_out = torch.compile(f, backend="aot_eager", dynamic=True)(x)
        expected = torch.autograd.grad(out.sum(), inputs=(x,))
        actual = torch.autograd.grad(c_out.sum(), inputs=(x,))
        self.assertEqual(expected, actual)

    def test_module_attribute_error(self):
        @torch.compile(backend="eager")
        def f1(x):
            return torch._bar(x)

        @torch.compile(backend="eager")
        def f2(x):
            try:
                return torch._bar(x)
            except AttributeError:
                return x + 1

        with self.assertRaises(AttributeError):
            f1(torch.ones(3))

        self.assertEqual(f2(torch.ones(3)), torch.ones(3) + 1)

    def test_torch_cuda_is_initialized(self):
        @torch.compile(fullgraph=True, backend="eager")
        def f(x):
            if torch.cuda.is_initialized():
                return x + 1
            return x + 2

        inp = torch.randn(3)
        self.assertEqual(f(inp), inp + 1)

        with mock.patch("torch.cuda.is_initialized", lambda: False):
            self.assertEqual(f(inp), inp + 2)

    def test_named_tuple_vt_clone(self):
        # https://github.com/pytorch/pytorch/issues/157945
        class SVDCompressor(nn.Module):
            def __init__(self, k=10):
                super().__init__()
                self.k = k

            def forward(self, x):
                U, S = torch.linalg.svd(x)[:2]
                reduced = U[:, :, : self.k] @ torch.diag_embed(S[:, : self.k])
                return reduced

        input = torch.randn(4, 8, 6)
        model = SVDCompressor(k=5)

        out1 = model(input.clone())
        out2 = torch.compile(model, backend="eager")(input.clone())
        self.assertEqual(out1, out2)

    @requires_cuda
    def test_zero_dim_param_mixed_device_grad(self):
        # cpu 0-dim params with cuda grads
        # https://github.com/pytorch/pytorch/issues/160084
        class RegressionModel(torch.nn.Module):
            def __init__(self, a=0, b=0):
                super().__init__()
                self.a = torch.nn.Parameter(torch.tensor(a).float())
                self.b = torch.nn.Parameter(torch.tensor(b).float())

            def forward(self, x):
                return x * self.a + self.b

        model = RegressionModel()
        model.forward = torch.compile(
            model.forward, backend="aot_eager", fullgraph=True
        )
        inputs = torch.randn(4, 10).to("cuda")
        out = model(inputs)
        out.sum().backward()
        self.assertIsNotNone(model.a.grad)
        self.assertIsNotNone(model.b.grad)
        self.assertEqual(model.a.grad.device, torch.device("cpu"))
        self.assertEqual(model.b.grad.device, torch.device("cpu"))

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    def test_cuda_sync(self):
        def fn(x):
            y = x + 1
            torch.cuda.synchronize()
            return y * 2

        x = torch.ones(2, device="cuda")
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(cnt.frame_count, 1)

    def test_filter_warnings(self):
        x = torch.ones(2, 2, requires_grad=True)

        def call_foobar(x):
            warnings.warn("foobar")

        @torch.compile(backend="eager")
        def f(x):
            call_foobar(x)
            call_foobar(x)
            call_foobar(x)
            call_foobar(x)
            return call_foobar(x)

        with warnings.catch_warnings(record=True) as w:
            f(x)
            self.assertEqual(len(w), 1)
            self.assertEqual(str(w[0].message), "foobar")

    def test_filter_safe_grad_warning(self):
        x = torch.ones(2, 2, requires_grad=True)
        y = x * 5  # non-leaf, .grad should warn
        torch._subclasses.meta_utils.safe_grad(y)  # filters out warning

        def unsafe_grad(y):
            return y.grad

        with warnings.catch_warnings(record=True) as w:
            unsafe_grad(y)  # should still warn, different callsite
            self.assertEqual(len(w), 1)
            self.assertTrue("The .grad attribute of a Tensor" in str(w[0].message))

            unsafe_grad(y)  # should not warn
            self.assertEqual(len(w), 1)

    def test_filter_user_warnings(self):
        x = torch.ones(2, 2, requires_grad=True)
        y = x * 5  # non-leaf, .grad should warn

        @torch._dynamo.eval_frame.TorchPatcher.suppress_torch_distributed_warnings
        def mute_warn(y):
            return y.grad

        mute_warn(y)  # filters out warning

        def unsafe_grad(y):
            return y.grad

        with warnings.catch_warnings(record=True) as w:
            unsafe_grad(y)  # should still warn, different callsite
            self.assertEqual(len(w), 1)
            self.assertTrue("The .grad attribute of a Tensor" in str(w[0].message))

            unsafe_grad(y)  # should not warn
            self.assertEqual(len(w), 1)

    def test_partial_export(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def parallelize(self):
                fn = self._call_impl

                def wrapped_fn(fn, *args, **kwargs):
                    new_args_0 = args[0].to(torch.bfloat16)
                    new_args_1 = args[1].to(torch.bfloat16)
                    return fn(new_args_0, new_args_1)

                fn = functools.partial(wrapped_fn, fn)
                self._call_impl = fn

            def forward(self, a, b):
                return a + b

        from torch._dynamo.functional_export import dynamo_graph_capture_for_export

        foo = Foo()
        foo.parallelize()
        x = torch.randn(4, 4, dtype=torch.float32)
        y = torch.randn(4, 4, dtype=torch.float32)
        ref = foo(x, y)
        gm = dynamo_graph_capture_for_export(foo)(x, y)
        res = gm(x, y)
        self.assertEqual(res, ref)

    def test_current_accelerator(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            torch.accelerator.current_accelerator()
            return x + 1

        self.assertEqual(fn(torch.ones(3)), torch.ones(3) + 1)

    def test_pytree_get_node_type_not_traced(self):
        # Test that torch.utils._pytree._get_node_type is not traced into
        # and doesn't cause excessive trace time overhead
        from torch.utils._pytree import _get_node_type

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x, y):
            # Call _get_node_type which is used internally by pytree operations
            node_type = _get_node_type([x, y])
            assert node_type is list  # noqa: S101
            # Do some work with pytree structures
            data = {"a": x, "b": y}
            flat, spec = pytree.tree_flatten(data)
            result = flat[0] + flat[1]
            return result

        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        result = fn(x, y)
        expected = x + y

        self.assertTrue(torch.allclose(result, expected))
        # Should compile successfully with fullgraph=True
        self.assertEqual(cnt.frame_count, 1)

    def test_pytree_get_node_type_with_namedtuple(self):
        # Test that torch.utils._pytree._get_node_type handles namedtuples correctly
        # without being traced into, even when is_namedtuple_class is True
        from collections import namedtuple

        from torch.utils._pytree import _get_node_type

        Point = namedtuple("Point", ["x", "y"])

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(a, b):
            # Create a namedtuple
            point = Point(a, b)
            # Call _get_node_type with a namedtuple instance
            node_type = _get_node_type(point)
            assert node_type is namedtuple  # noqa: S101
            # Use pytree operations with namedtuples
            flat, spec = pytree.tree_flatten(point)
            result = flat[0] + flat[1]
            return result

        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        result = fn(x, y)
        expected = x + y

        self.assertTrue(torch.allclose(result, expected))
        # Should compile successfully with fullgraph=True
        self.assertEqual(cnt.frame_count, 1)

    def test_pytree_tree_is_leaf_not_traced(self):
        # Test that torch.utils._pytree.tree_is_leaf is not traced into
        # when is_leaf parameter is None (the common case)
        from torch.utils._pytree import tree_is_leaf

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x, y):
            # Test with various types
            # Tensors are leaves
            is_leaf_tensor = tree_is_leaf(x)
            assert is_leaf_tensor is True  # noqa: S101

            # Lists are not leaves (they're in SUPPORTED_NODES)
            is_leaf_list = tree_is_leaf([x, y])
            assert is_leaf_list is False  # noqa: S101

            # Dicts are not leaves
            is_leaf_dict = tree_is_leaf({"a": x, "b": y})
            assert is_leaf_dict is False  # noqa: S101

            return x + y

        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        result = fn(x, y)
        expected = x + y

        self.assertTrue(torch.allclose(result, expected))
        # Should compile successfully with fullgraph=True
        self.assertEqual(cnt.frame_count, 1)

    def test_ordered_set_doesnt_recompile_with_ac(self):
        import torch

        with torch._dynamo.config.patch({"error_on_recompile": True}):
            import functools

            from torch.utils._ordered_set import OrderedSet
            from torch.utils.checkpoint import (
                checkpoint,
                CheckpointPolicy,
                create_selective_checkpoint_contexts,
            )

            def policy(compute_heavy_ops, ctx, func, *args, **kwargs):
                if func in compute_heavy_ops:
                    return CheckpointPolicy.MUST_SAVE
                return CheckpointPolicy.PREFER_RECOMPUTE

            def g(x):
                return torch.mm(x, x).sin().exp()

            @torch.compile(fullgraph=True, backend="eager")
            def f(x, policy):
                return checkpoint(g, x, use_reentrant=False, context_fn=policy)

            x = torch.randn(4, 4, requires_grad=True)
            f(
                x,
                functools.partial(
                    create_selective_checkpoint_contexts,
                    functools.partial(policy, OrderedSet([torch.ops.aten.mm.default])),
                ),
            )
            f(
                x,
                functools.partial(
                    create_selective_checkpoint_contexts,
                    functools.partial(policy, OrderedSet([torch.ops.aten.mm.default])),
                ),
            )

    def test_mro_source_cache_includes_attr_name(self):
        # Base -> Mid -> A hierarchy: two class attributes with the same
        # interned integer value share the same id().  The mro_source_cache
        # must include the attribute name in its key; otherwise the second
        # lookup returns the first attribute's source, installing a guard
        # on the wrong key and missing mutations to the second attribute.
        class Base:
            x = 1
            y = 1

        class Mid(Base):
            pass

        class A(Mid):
            pass

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(obj, t):
            return t * obj.x + t * obj.y

        obj = A()
        t = torch.tensor([1.0])
        result = fn(obj, t)
        self.assertEqual(result, torch.tensor([2.0]))
        self.assertEqual(cnt.frame_count, 1)

        # Changing y on Base must trigger recompilation.
        Base.y = 42
        result = fn(obj, t)
        self.assertEqual(result, torch.tensor([43.0]))
        self.assertEqual(cnt.frame_count, 2)

    def test_pytree_tree_is_leaf_with_namedtuple(self):
        # Test that torch.utils._pytree.tree_is_leaf handles namedtuples correctly
        from collections import namedtuple

        from torch.utils._pytree import tree_is_leaf

        Point = namedtuple("Point", ["x", "y"])

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(a, b):
            # Namedtuples are not leaves (they're in SUPPORTED_NODES)
            point = Point(a, b)
            is_leaf_namedtuple = tree_is_leaf(point)
            assert is_leaf_namedtuple is False  # noqa: S101

            # But individual tensors are leaves
            is_leaf_tensor = tree_is_leaf(a)
            assert is_leaf_tensor is True  # noqa: S101

            return a + b

        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        result = fn(x, y)
        expected = x + y

        self.assertTrue(torch.allclose(result, expected))
        # Should compile successfully with fullgraph=True
        self.assertEqual(cnt.frame_count, 1)

    def test_data_attr_mutation_with_noop_add(self):
        # Regression test: remove_no_ops incorrectly eliminated add(x, 0) -> x
        # when x was subsequently mutated by set_, causing the return value to
        # alias the mutated input instead of being an independent copy.
        def fn(a, b):
            a.data = b
            b.data = torch.zeros_like(b)
            return a + b

        a = torch.tensor([True, False, True, False])
        b = torch.tensor([False, False, True, True])
        a_ = a.clone()
        b_ = b.clone()
        cfunc = torch.compile(fn, backend="inductor")
        res1 = fn(a, b)
        res2 = cfunc(a_, b_)
        self.assertEqual(res1, res2)

    def test_custom_op_mutation_with_noop_add(self):
        @torch.library.custom_op("test_repros::mutate_tensor", mutates_args={"x"})
        def mutate_tensor(x: torch.Tensor, src: torch.Tensor) -> None:
            x.copy_(src)

        def fn(b):
            zeros = torch.zeros_like(b)
            result = b + zeros
            mutate_tensor(b, zeros)
            return result

        b = torch.tensor([4.0, 5.0, 6.0])
        b_ = b.clone()
        cfunc = torch.compile(fn, backend="inductor")
        res1 = fn(b)
        res2 = cfunc(b_)
        self.assertEqual(res1, res2)

    def test_getset_descriptor_objclass_identity(self):
        # GetSetDescriptor.__objclass__ should preserve identity with the class
        # under torch.compile. This is needed for inspect.getattr_static (and
        # therefore inspect.signature) to work on callable class instances.
        class Foo:
            pass

        desc = Foo.__dict__["__dict__"]

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            if desc.__objclass__ is Foo:
                return x + 1.0
            return x + 2.0

        result = f(torch.tensor(0.0))
        self.assertEqual(result.item(), 1.0)

    def test_inspect_signature_callable_class(self):
        # inspect.signature should work on callable class instances under
        # torch.compile, needed by flex_attention's _get_mod_type.
        class MyCallable:
            def __call__(self, b, h, q_idx, kv_idx):
                return q_idx >= kv_idx

        obj = MyCallable()

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            sig = inspect.signature(obj)
            num_params = sum(
                1
                for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
            )
            return x + num_params

        result = f(torch.tensor(0.0))
        self.assertEqual(result.item(), 4.0)

    def test_enum_with_class_values(self):
        from enum import Enum

        class AvgMetric:
            def __init__(self):
                self.sum = None
                self.count = 0

            def append(self, x):
                if self.count > 0:
                    self.sum = self.sum + x
                else:
                    self.sum = x.clone()
                self.count += 1

        class GlobalReduction(Enum):
            AVG = AvgMetric

        class ScalarLogger:
            def __init__(self):
                self.metrics = {}

            def log(self, key, value, global_reduction):
                if key not in self.metrics:
                    self.metrics[key] = global_reduction.value()
                self.metrics[key].append(value)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(logger, x):
            logger.log("test", x, GlobalReduction.AVG)
            return x + 1

        logger = ScalarLogger()
        fn(logger, torch.tensor(1.0))

    def test_class_attr_mutation_recompiles(self):
        class GlobalState:
            factor = 1.0

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            return x * GlobalState.factor

        x = torch.tensor([4.0])

        GlobalState.factor = 1.0
        result1 = fn(x)
        self.assertEqual(result1, torch.tensor([4.0]))
        self.assertEqual(cnt.frame_count, 1)

        GlobalState.factor = 10.0
        result2 = fn(x)
        self.assertEqual(result2, torch.tensor([40.0]))
        self.assertEqual(cnt.frame_count, 2)

    @skipIfHpu
    @requires_cuda
    def test_deterministic_pad_replicate_compile(self, device):
        from torch.testing._internal.common_utils import DeterministicGuard

        pad = torch.nn.ReplicationPad1d(2).to(device)
        compiled_pad = torch.compile(pad, backend="aot_eager", fullgraph=True)
        x = torch.randn(3, 3, device=device, requires_grad=True)
        with DeterministicGuard(True):
            ref = pad(x)
            res = compiled_pad(x)
            self.assertEqual(ref, res)
            grad = torch.autograd.grad(res.sum(), x)
            ref_grad = torch.autograd.grad(ref.sum(), x)
            self.assertEqual(grad, ref_grad)

    @requires_cuda
    @unittest.skipIf(
        TEST_WITH_ROCM or not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "flash attention not supported",
    )
    def test_flex_attention_guard_on_constant_func_defaults(self):
        """
        Dynamo must guard on mask_mod.__defaults__ so that when a
        compiled function is re-invoked with a new BlockMask whose
        mask_mod has the same __code__ but different __defaults__,
        Dynamo recompiles instead of reusing the stale first graph.
        """
        from torch.utils._triton import has_triton

        if not has_triton():
            self.skipTest("requires triton")

        @torch.compile(fullgraph=True)
        def flex_chunk(q, k, v, block_mask, scale):
            out, aux = flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                scale=scale,
                return_aux=AuxRequest(lse=True),
            )
            return out, aux.lse

        def merge(out, lse, new_out, new_lse):
            lse, new_lse = lse.unsqueeze(-1), new_lse.unsqueeze(-1)
            mx = torch.maximum(lse, new_lse)
            e0, e1 = torch.exp(lse - mx), torch.exp(new_lse - mx)
            d = e0 + e1
            return (out * e0 + new_out * e1) / d, (mx + torch.log(d)).squeeze(-1)

        @torch.compile(fullgraph=True)
        def ref_attn(q, k, v, block_mask, scale):
            return flex_attention(q, k, v, block_mask=block_mask, scale=scale)

        torch.manual_seed(42)
        B, H, S, D = 1, 1, 512, 16
        device = "cuda"
        NUM_CHUNKS = 4
        chunk_size = S // NUM_CHUNKS

        q = torch.randn(B, H, S, D, device=device)
        k = torch.randn(B, H, S, D, device=device)
        v = torch.randn(B, H, S, D, device=device)
        scale = D**-0.5

        merged_out = merged_lse = None
        for step in range(NUM_CHUNKS):
            kv_offset = step * chunk_size

            def mask_mod(b, h, q_idx, kv_idx, _offset=kv_offset):
                return q_idx >= kv_idx + _offset

            bm = create_block_mask(
                mask_mod, B=B, H=H, Q_LEN=S, KV_LEN=chunk_size, device=device
            )
            out, lse = flex_chunk(
                q,
                k[:, :, kv_offset : kv_offset + chunk_size],
                v[:, :, kv_offset : kv_offset + chunk_size],
                bm,
                scale,
            )
            if merged_out is None:
                merged_out, merged_lse = out, lse
            else:
                merged_out, merged_lse = merge(merged_out, merged_lse, out, lse)

        def causal(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        ref_bm = create_block_mask(causal, B=B, H=H, Q_LEN=S, KV_LEN=S, device=device)
        ref_out = ref_attn(q, k, v, ref_bm, scale)

        self.assertTrue(
            (merged_out - ref_out).abs().max().item() < 1e-3,
            "flex_attention mask_mod __defaults__ not properly guarded",
        )
