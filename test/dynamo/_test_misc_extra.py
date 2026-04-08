# Owner(s): ["module: dynamo"]
# flake8: noqa: B020, F403, F405, F841
# ruff: noqa: B020,F403,F405,F841,PLW0127
try:
    from ._test_misc_common import *
except ImportError:
    from _test_misc_common import *


class MiscTestsPyTree(torch._inductor.test_case.TestCase):
    @parametrize_pytree_module
    def test_tracing_pytree(self, pytree):
        def fn(xs):
            flat_xs, spec = pytree.tree_flatten(xs)
            res = [x.clone() for x in flat_xs]
            if pytree.__name__ == "optree":
                # The treespec argument comes first in OpTree / JAX PyTree
                return pytree.tree_unflatten(spec, res)
            return pytree.tree_unflatten(res, spec)

        xs = [torch.tensor(i) for i in range(3)]

        counter = CompileCounter()
        torch.compile(fn, backend=counter, fullgraph=True)(xs)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 3)

    @parametrize_pytree_module
    def test_tracing_nested_pytree(self, pytree):
        def fn(xs):
            flat_xs, spec = pytree.tree_flatten(xs)
            res = [x.clone() for x in flat_xs]
            if pytree.__name__ == "optree":
                # The treespec argument comes first in OpTree / JAX PyTree
                return pytree.tree_unflatten(spec, res)
            return pytree.tree_unflatten(res, spec)

        xs = [torch.tensor(i) for i in range(3)]
        xsl = [xs, xs, xs, xs]

        counter = CompileCounter()
        comp_out = torch.compile(fn, backend=counter, fullgraph=True)(xsl)
        real_out = fn(xsl)
        self.assertEqual(comp_out, real_out)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 12)

    @parametrize_pytree_module
    def test_tracing_nested_tuples(self, pytree):
        def fn(xs):
            flat_xs, spec = pytree.tree_flatten(xs)
            res = [x.clone() for x in flat_xs]
            if pytree.__name__ == "optree":
                # The treespec argument comes first in OpTree / JAX PyTree
                return pytree.tree_unflatten(spec, res)
            return pytree.tree_unflatten(res, spec)

        xs = [torch.tensor(i) for i in range(3)]
        xsl = (xs, xs, xs, xs)

        counter = CompileCounter()
        comp_out = torch.compile(fn, backend=counter, fullgraph=True)(xsl)
        real_out = fn(xsl)
        self.assertEqual(comp_out, real_out)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 12)

    @parametrize_pytree_module
    def test_tracing_nested_dicts(self, pytree):
        def fn(xs):
            flat_xs, spec = pytree.tree_flatten(xs)
            res = [x.clone() for x in flat_xs]
            if pytree.__name__ == "optree":
                # The treespec argument comes first in OpTree / JAX PyTree
                return pytree.tree_unflatten(spec, res)
            return pytree.tree_unflatten(res, spec)

        xs = [torch.tensor(i) for i in range(3)]
        xsl = {
            "a": xs,
            "b": xs,
            "c": xs,
        }

        counter = CompileCounter()
        comp_out = torch.compile(fn, backend=counter, fullgraph=True)(xsl)
        real_out = fn(xsl)
        self.assertEqual(comp_out, real_out)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 9)

    @parametrize_pytree_module
    def test_tracing_nested_mixed_all(self, pytree):
        def fn(xs):
            flat_xs, spec = pytree.tree_flatten(xs)
            res = [x.clone() for x in flat_xs]
            if pytree.__name__ == "optree":
                # The treespec argument comes first in OpTree / JAX PyTree
                return pytree.tree_unflatten(spec, res)
            return pytree.tree_unflatten(res, spec)

        xs = [torch.tensor(i) for i in range(3)]
        xsa = (xs, xs)
        xsb = {"aa": xsa, "ab": xs}
        xsl = {
            "a": xs,
            "b": xsa,
            "c": xsb,
        }

        counter = CompileCounter()
        comp_out = torch.compile(fn, backend=counter, fullgraph=True)(xsl)
        real_out = fn(xsl)
        self.assertEqual(comp_out, real_out)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 18)

    @parametrize_pytree_module
    def test_tracing_nested_tensor_subclass(self, pytree):
        from torch.testing._internal.two_tensor import TwoTensor
        from torch.utils.checkpoint import checkpoint

        def fn(xs):
            nested_xs = [[xs]]
            flat_xs, spec = pytree.tree_flatten(xs)
            return flat_xs[0].clone()

        # use checkpoint to trigger a "sourceless" tensor subclass
        def checkpoint_fn(xs):
            return checkpoint(fn, xs, use_reentrant=True)

        xs = TwoTensor(torch.ones(2, 2), torch.ones(2, 2))

        counter = CompileCounter()
        torch.compile(checkpoint_fn, backend=counter, fullgraph=True)(xs)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 2)

    @parametrize_pytree_module
    def test_pytree_tree_leaves(self, pytree):
        def fn(x):
            tree = {
                "a": [x, x - 1],
                "b": x + 2,
                "c": (
                    x,
                    3.0,
                    collections.deque([0.0, -x, 1, 2], maxlen=3),
                ),
                "d": collections.OrderedDict(
                    {
                        "e": torch.return_types.qr((2 * x, None)),
                        "f": MyTuple(x, x + 1, torch.zeros(4, 3)),
                    },
                ),
            }
            leaves = pytree.tree_leaves(tree)
            return leaves

        x = torch.randn(3, 2)
        expected = fn(x)
        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        actual = fn_opt(x)

        self.assertEqual(actual, expected)

    @parametrize_pytree_module
    def test_pytree_tree_flatten_unflatten(self, pytree):
        def fn(x, y):
            tree = {
                "a": [x, x - 1],
                "b": x + 2,
                "c": (
                    x,
                    3.0,
                    collections.deque([0.0, -x, 1, 2], maxlen=3),
                ),
                "d": collections.OrderedDict(
                    {
                        "e": torch.return_types.qr((2 * x, None)),
                        "f": MyTuple(x, x + 1, torch.zeros(4, 3)),
                    },
                ),
            }
            leaves, treespec = pytree.tree_flatten(tree)
            new_leaves = [
                x - 1,
                y,
                x * y,
                3.0,
                y - 2,
                1,
                torch.zeros(2, 2),
                2 * y,
                -y,
                x + y,
                x - y,
                torch.ones(3, 2),
                1,
            ]
            if pytree.__name__ == "optree":
                # `None` is a internal node rather than leaf in default OpTree / JAX PyTree
                new_leaves.pop()
                # The treespec argument comes first in OpTree / JAX PyTree
                new_tree = pytree.tree_unflatten(treespec, new_leaves)
            else:
                new_tree = pytree.tree_unflatten(new_leaves, treespec)
            return leaves, new_tree

        x = torch.randn(3, 2)
        y = torch.randn(3, 2)
        expected = fn(x, y)
        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        actual = fn_opt(x, y)

        self.assertEqual(actual, expected)

    @parametrize_pytree_module
    def test_pytree_tree_map(self, pytree):
        def fn(x, y):
            tree1 = {
                "a": [x, x - 1],
                "b": x + 2,
                "c": (
                    x,
                    3.0,
                    collections.deque([0.0, -x, 1, 2], maxlen=3),
                ),
                "d": collections.OrderedDict(
                    {
                        "e": torch.return_types.qr((2 * x, None)),
                        "f": MyTuple(x, x + 1, torch.zeros(4, 3)),
                    },
                ),
            }
            tree2 = collections.OrderedDict(
                [
                    ("c", (y, 3.0, collections.deque([1, -y, 10.0]))),
                    ("a", [y, y + 1]),
                    ("b", y + 2),
                    (
                        "d",
                        {
                            "f": MyTuple(torch.ones(4, 3), -y, y + 1),
                            "e": torch.return_types.qr((2 * y, None)),
                        },
                    ),
                ],
            )
            return pytree.tree_map(lambda u, v: (u, v), tree1, tree2)

        x = torch.randn(3, 2)
        y = torch.randn(3, 2)
        expected = fn(x, y)
        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        actual = fn_opt(x, y)

        self.assertEqual(actual, expected)

    @parametrize_pytree_module
    def test_pytree_tree_map_dict_order(self, pytree):
        def fn(tree):
            new_tree = pytree.tree_map(lambda x: x, tree)
            return list(new_tree.keys()), list(new_tree.values())

        x = torch.randn(3, 2)
        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)

        tree1 = {"b": x + 2, "a": x, "c": x - 1}
        expected1 = fn(tree1)
        actual1 = fn_opt(tree1)
        self.assertEqual(actual1, expected1)

        tree2 = collections.OrderedDict([("b", x + 2), ("a", x), ("c", x - 1)])
        expected2 = fn(tree2)
        actual2 = fn_opt(tree2)
        self.assertEqual(actual2, expected2)

        tree3 = collections.defaultdict(int, {"b": x + 2, "a": x, "c": x - 1})
        expected3 = fn(tree3)
        actual3 = fn_opt(tree3)
        self.assertEqual(actual3, expected3)

    @parametrize_pytree_module
    def test_pytree_tree_map_only(self, pytree):
        if not callable(getattr(pytree, "tree_map_only", None)):
            # OpTree and JAX PyTree do not have `tree_map_only`
            return

        def fn(xs):
            def mapper(x):
                return x.clone()

            y = pytree.tree_map_only(torch.Tensor, mapper, xs)
            return y

        xs = [torch.tensor(i) for i in range(3)] + ["hi"]
        xsa = (xs, xs)
        xsb = {"aa": xsa, "ab": xs}

        counter = CompileCounter()
        comp_out = torch.compile(fn, backend=counter, fullgraph=True)(xsb)
        real_out = fn(xsb)

        self.assertEqual(comp_out, real_out)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 9)

    def test_pytree_register_constant_with_side_effect(self):
        class Foo:
            pass

        class Bar:
            def __eq__(self, other):
                return super().__eq__(other)

            def __hash__(self):
                return 0

        python_pytree.register_constant(Bar)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, obj):
            obj.attr = {3: Bar()}
            return x + 1

        inp = torch.ones(3)
        self.assertEqual(fn(inp, Foo()), inp + 1)


class TestTracer(JitTestCase):
    def test_jit_save(self):
        def fn():
            class Foo(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.a = 3

                @torch.jit.export
                def __getstate__(self):
                    return (3, self.training)

                @torch.jit.export
                def __setstate__(self, state):
                    self.a = state[0]
                    self.training = state[1]

                def forward(self, x):
                    return x + self.a

            f = Foo()

            return torch.jit.trace(f, (torch.rand(3, 4),))

        fn()
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn()


class TestCustomFunction(torch.testing._internal.common_utils.TestCase):
    def test_autograd_function_with_matmul_folding_at_output(self):
        """
        When tensor folding occurs during matmul operation returned tensor is a view.
        This can cause issues when matmul is used inside a custom function
        and such view is then returned as output. Then it cannot be modified inplace
        and causes errors.
        It can be especially problematic when after such function inplace allreduce
        is performed. This test recreates this behaviour.
        Issue is resolved when unsafe_view is returned from matmul instead.
        """

        class CustomFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp1, inp2):
                ctx.save_for_backward(inp2)
                ctx.output_shape = inp1.size()
                return torch.matmul(inp1, inp2)

            @staticmethod
            def backward(ctx, grad_output):
                output_shape = ctx.output_shape
                (inp2,) = ctx.saved_tensors
                return (
                    torch.mm(grad_output.squeeze(), inp2.t()).view(output_shape),
                    None,
                )

        def outer_function(inp1, inp2):
            res = CustomFunction.apply(inp1, inp2)
            res.add_(1.0)
            return res.sum()

        def usual_function(inp1, inp2) -> torch.Tensor:
            res = torch.matmul(inp1, inp2)
            res.add_(1.0)
            return res.sum()

        inp1_custom = torch.randn(4, 1, 2, requires_grad=True)
        inp1_usual = inp1_custom.detach().clone().requires_grad_(True)

        inp2 = torch.randn(2, 4)
        c_custom_func = torch.compile(outer_function, backend="eager")
        c_usual_func = torch.compile(usual_function, backend="eager")

        result_custom = c_custom_func(inp1_custom, inp2)
        result_custom.backward()
        result_usual = c_usual_func(inp1_usual, inp2)
        result_usual.backward()

        torch.allclose(inp1_custom.grad, inp1_usual.grad)

    def test_retain_grad(self):
        def fn(x, y):
            y.retain_grad()
            return torch.sin(y) + x

        opt_fn = torch.compile(fn, backend="aot_eager")
        x = torch.randn(4, requires_grad=True)
        y = torch.cos(x)
        opt_fn(x, y).sum().backward()
        self.assertTrue(y.grad is not None)


class MiscTestsDevice(torch._inductor.test_case.TestCase):
    def test_rand(self, device):
        cnts = torch._dynamo.testing.CompileCounter()
        device = device

        def fn():
            return torch.randn(10, device=device)

        torch.manual_seed(10)
        ref_run1 = fn()

        torch.manual_seed(10)
        ref_run2 = fn()
        self.assertTrue(same(ref_run1, ref_run2))

        torch.manual_seed(10)
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn()

        self.assertTrue(same(res, ref_run1))

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "Can't run fused SDPA on this platform",
    )
    def test_parsing_sdpa(self, device):
        class MyModule(torch.nn.Module):
            def forward(self, query, key, value):
                out = F.scaled_dot_product_attention(query, key, value, None, 0, True)
                out = F.scaled_dot_product_attention(
                    query, key, value, None, 0, True, scale=8
                )
                out = F.scaled_dot_product_attention(
                    query=query,
                    key=key,
                    value=value,
                    attn_mask=None,
                    dropout_p=0,
                    is_causal=True,
                )
                out = F.scaled_dot_product_attention(
                    query,
                    key=key,
                    value=value,
                    attn_mask=None,
                    dropout_p=0,
                    is_causal=True,
                )
                out = F.scaled_dot_product_attention(
                    query, key, value, None, dropout_p=0, is_causal=True
                )
                out = F.scaled_dot_product_attention(query, key, value, None, scale=8)
                return out

        device = device
        dtype = torch.float16
        seq_len_q = 1
        seq_len_k = 1
        head_dim = 8
        query = torch.ones(
            1, 8, seq_len_q, head_dim, device=device, dtype=dtype, requires_grad=True
        )
        key = torch.ones(
            1, 8, seq_len_k, head_dim, device=device, dtype=dtype, requires_grad=True
        )
        value = torch.ones(
            1, 8, seq_len_k, head_dim, device=device, dtype=dtype, requires_grad=True
        )
        module = MyModule()
        opt_mod = torch.compile(module, backend="inductor")
        opt_mod(query, key, value)

    def test_torch_device_is_available(self, device):
        def fn(x):
            if torch.accelerator.is_available():
                return x + 1
            else:
                return x - 1

        x = torch.rand(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    @unittest.skipIf(not TEST_CUDA, "requires cuda")
    @unittest.skipIf(not torch.backends.cudnn.is_available(), "requires cudnn")
    def test_torch_cudnn_is_acceptable(self, device):
        def fn(x):
            if torch.backends.cudnn.is_acceptable(tensor=x):
                return x + 1
            return x

        x = torch.rand(4).to(device)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    @unittest.skipIf(not TEST_CUDA, "requires cuda")
    @unittest.skipIf(not torch.backends.cudnn.is_available(), "requires cudnn")
    def test_torch_cudnn_is_acceptable_bad_inputs(self, device):
        def fn1(x):
            if torch.backends.cudnn.is_acceptable("invalid"):
                return x + 1
            return x

        def fn2(x):
            if torch.backends.cudnn.is_acceptable(x, 3.14):
                return x + 1
            return x

        with self.assertRaisesRegex(
            AssertionError, "Expect input to cudnn.is_acceptable to be a tensor"
        ):
            x1 = torch.rand(4).to(device)
            opt_fn1 = torch.compile(fn1, backend="eager", fullgraph=True)
            res1 = opt_fn1(x1)

        with self.assertRaisesRegex(
            AssertionError, "Expect 1 input to cudnn.is_acceptable"
        ):
            x2 = torch.rand(4).to(device)
            opt_fn2 = torch.compile(fn2, backend="eager", fullgraph=True)
            res = opt_fn2(x2)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    @torch._dynamo.config.patch(recompile_limit=999)
    def test_legacy_cuda_tensor(self):
        typs = [
            torch.cuda.FloatTensor,
            torch.cuda.DoubleTensor,
            torch.cuda.HalfTensor,
            torch.cuda.BFloat16Tensor,
            torch.cuda.ByteTensor,
            torch.cuda.CharTensor,
            torch.cuda.IntTensor,
            torch.cuda.ShortTensor,
            torch.cuda.LongTensor,
        ]

        def f2(typ):
            return typ([1, 2, 3])

        compiled_f2 = torch.compile(f2, backend="eager", fullgraph=True)
        for typ in typs:
            output = compiled_f2(typ)
            expected = f2(typ)
            self.assertEqual(output, expected)

    def test_get_device(self, device):
        def fn(x, y):
            x = x + 1
            y = y + 1
            return x.get_device(), y.get_device()

        x = torch.rand(4, device=device)
        y = torch.rand(4, device="cpu")
        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, y)
        self.assertTrue(same(ref, res))

    def test_symint_as_device_kwarg(self, device):
        def f(rank):
            # -2 to make device id 0 for easier testing on CI
            return torch.ones(10, device=rank.size(0) - 2)

        x = torch.randn(2)
        out = f(torch.randn(2))
        opt_out = torch.compile(backend="eager", dynamic=True, fullgraph=True)(f)(x)
        self.assertEqual(out, opt_out)

    def test_torch_device_python_type(self, device):
        device_type = torch.device(device).type
        for device, device_type, index in [
            ("cpu", "cpu", None),
            (device, device_type, 0),
        ]:

            def fn(target):
                target_device = target.device
                a = torch.zeros(2, 3, device=target_device)
                # Constant assert at trace time
                assert isinstance(target_device, torch.device)  # noqa: S101
                assert target_device.type == device_type  # noqa: S101
                assert target_device.index == index  # noqa: S101
                b = torch.zeros(2, 3, device=target_device)
                c = torch.zeros(2, 3, device=target_device)
                return a + b + c

            from torch._dynamo.variables import ConstantVariable

            device = torch.device(device)
            expected_variable = ConstantVariable(device)
            self.assertEqual(expected_variable.python_type(), type(device))

            opt_func = torch.compile(fn, backend="eager", fullgraph=True)
            a = torch.tensor([2, 3], device=device)
            res = opt_func(a)
            self.assertIsInstance(res, torch.Tensor)

    @torch._dynamo.config.patch(
        capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True
    )
    @torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True)
    def test_interpolate_propagate_real_tensors(self, device):
        @torch.compile(backend="eager", fullgraph=True)
        def f(mask, box):
            # u0, u1 = mask.tolist()
            mask = torch.randn(1, 1, 30, 30, device=device)
            h, w = box.tolist()
            return torch.nn.functional.interpolate(
                mask, (h, w), mode="bilinear", align_corners=False
            )

        f(torch.tensor([30, 30], device=device), torch.tensor([68, 32], device=device))

    def test_scalar_isin_decomposition(self):
        def f():
            x = torch.tensor(0)
            return torch.isin(x, x)

        opt_f = torch.compile(f, backend="inductor", fullgraph=True)
        ref = f()
        res = opt_f()
        self.assertEqual(ref, res)

    def test_randint_no_graphbreak(self):
        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(actions, n_act, epsilon=0.1):
            actions_random = torch.randint_like(actions, n_act)

            return actions_random

        x = torch.ones([1], dtype=torch.int64)
        y = torch.tensor(5)
        f(x, y)

    def test_full_graph_capture_scalar_outputs(self):
        @torch.compile(fullgraph=True, backend="eager")
        def foo(a):
            return torch.randn(5) * a.item()

        # We expect to no longer raise here
        foo(torch.tensor(2.0))

    def test_full_graph_capture_dynamic_output_shape_ops(self):
        def fn(x):
            nz = torch.nonzero(x)
            squared = nz * nz
            sliced = torch.ops.aten.slice.Tensor(squared, dim=1, start=-2, end=None)
            view = sliced.unsqueeze(dim=0)
            return view.squeeze(dim=0)

        example_inputs = (torch.randn(1, 1, 1, 1),)
        # we expect to no longer raise here
        torch.compile(fn, fullgraph=True, backend="eager")(*example_inputs)

    def test_dynamic_fill_diagonal_(self):
        @torch.compile(dynamic=True, backend="eager")
        def f(x):
            x.fill_diagonal_(True)

        x = torch.zeros(4, 4)
        f(x)

    def test_dynamic_float_scalar_tensor_coersion(self):
        # Minified version of https://github.com/pytorch/pytorch/issues/158376#issuecomment-3079591367
        class Foo:
            def __init__(self):
                self.config = type(
                    "Config", (), {"pad_val": 1123581321.0, "tolerance": 1e-6}
                )

            @torch.compile(fullgraph=True, backend="eager")
            def forward(self, input):
                outputs = torch.where(
                    torch.abs(input - self.config.pad_val) < self.config.tolerance,
                    torch.tensor(
                        self.config.pad_val, dtype=input.dtype, device=input.device
                    ),
                    torch.tensor(
                        self.config.pad_val + 1, dtype=input.dtype, device=input.device
                    ),
                )
                return outputs

        foo = Foo()
        inputs = torch.randn(3, 4)
        result = foo.forward(inputs)

        original_pad_val = foo.config.pad_val
        foo.config.pad_val += 1.0
        result2 = foo.forward(inputs)

        # Previously would crash with:
        #   RuntimeError: value cannot be converted to type at::Half without overflow


class DynamoOpPromotionTests(torch._dynamo.test_case.TestCase):
    @unittest.skipIf(not TEST_CUDA, "This test requires a CUDA device")
    def test_symbool_tensor_mul(self):
        def symbool_mul_fn(x_bool, sentinel):
            result = x_bool * sentinel
            return result

        x_true = torch.tensor([True], device="cuda")
        x_false = torch.tensor([False], device="cuda")
        sentinel = torch.tensor(2.0, requires_grad=True, device="cuda")
        eager_result_true = symbool_mul_fn(x_true, sentinel)
        eager_result_false = symbool_mul_fn(x_false, sentinel)
        compiled_fn = torch.compile(
            symbool_mul_fn, fullgraph=True, dynamic=True, backend="eager"
        )
        compiled_result_true = compiled_fn(x_true, sentinel)
        compiled_result_false = compiled_fn(x_false, sentinel)
        self.assertEqual(eager_result_true, compiled_result_true)
        self.assertEqual(eager_result_false, compiled_result_false)
        self.assertEqual(compiled_result_true.item(), 2.0)
        self.assertEqual(compiled_result_false.item(), 0.0)

    @unittest.skipIf(not TEST_CUDA, "This test requires a CUDA device")
    def test_symbool_guard_or_false(self):
        def symbool_guard_fn(a_bool_tensor, b):
            u0 = a_bool_tensor.item()
            # Make sure guard_or_false still handles SymBool produced by .item()
            if guard_or_false(u0):
                return b * 10
            else:
                return b * 100

        compiled_guard_fn = torch.compile(
            symbool_guard_fn, backend="eager", dynamic=True
        )
        a_true = torch.tensor(True, device="cuda")
        a_false = torch.tensor(False, device="cuda")
        b = torch.randn(6, device="cuda")
        eager_res_true = symbool_guard_fn(a_true, b)
        compiled_res_true = compiled_guard_fn(a_true, b)
        self.assertEqual(eager_res_true, compiled_res_true)
        eager_res_false = symbool_guard_fn(a_false, b)
        compiled_res_false = compiled_guard_fn(a_false, b)
        self.assertEqual(eager_res_false, compiled_res_false)
        self.assertEqual(compiled_res_true, b * 10)
        self.assertEqual(compiled_res_false, b * 100)

    @unittest.skipIf(not TEST_CUDA, "This test requires a CUDA device")
    def test_symbool_tensor_mul_does_not_fail(self):
        def fuzzed_program(arg_0, sentinel):
            var_node_2 = arg_0
            var_node_1 = torch.squeeze(var_node_2)
            var_node_0 = var_node_1.item()
            result = var_node_0 * sentinel
            if result.is_complex():
                result = result.real
            return result

        sentinel = torch.tensor(1.0, requires_grad=True, device="cuda")
        arg_0 = torch.tensor([True], dtype=torch.bool, device="cuda")
        args = (arg_0,) + (sentinel,)
        try:
            compiled_program = torch.compile(
                fuzzed_program, fullgraph=True, dynamic=True, backend="eager"
            )
            compiled_program(*args)
        except Exception as e:
            self.fail(f"torch.compile failed with error: {e}")

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_tensorify_track_item_symint(self):
        def _random_resize(image: torch.Tensor):
            image_metanet = image
            default_patch_size = 14
            rand_cnn_resolution = (224, 256)
            min_nump = rand_cnn_resolution[0] // default_patch_size
            max_nump = rand_cnn_resolution[1] // default_patch_size
            new_nump = torch.randint(min_nump, max_nump + 1, (1,)).item()
            torch._check(new_nump > 0)
            torch._check(new_nump * default_patch_size > 1)

            image_metanet = F.interpolate(
                image_metanet,
                size=(new_nump * default_patch_size, new_nump * default_patch_size),
                mode="bilinear",
                align_corners=True,
            )
            img_h_new, img_w_new = image_metanet.shape[2:]

            return (img_h_new, img_w_new), image_metanet

        _random_resize_compiled = torch.compile(fullgraph=True, backend="eager")(
            _random_resize
        )

        # Test the function
        input_tensor = torch.rand(1, 3, 224, 224)
        (h, w), output = _random_resize_compiled(input_tensor)

        # Verify output properties
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], 3)
        self.assertEqual(output.shape[2], h)
        self.assertEqual(output.shape[3], w)
        self.assertTrue(h % 14 == 0)
        self.assertTrue(w % 14 == 0)
        self.assertTrue(224 <= h <= 256)
        self.assertTrue(224 <= w <= 256)

    @unittest.skipIf(not TEST_CUDA, "This test requires a CUDA device")
    def test_module_to_with_shared_weights_compile(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(num_embeddings=10, embedding_dim=8)

            def forward(self, x):
                token_ids = torch.randint(0, 10, (4,), device=x.device)
                embedded = self.embedding(token_ids).sum()
                return x.sum() + embedded.sum()

        class Container(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mod = Model()

            def forward(self, x):
                if "cuda" in str(x.device):
                    mod = self.mod.to(x.device)
                    return mod(x)
                else:
                    return x.sum()

        container = Container()
        container_eager = copy.deepcopy(container)
        with torch._dynamo.config.patch(graph_break_on_nn_param_ctor=False):
            compiled = torch.compile(container, backend="eager", fullgraph=True)

            inp1 = torch.randn(4, 4, 4, device="cuda")

            # First call with CUDA input
            compiled_result1 = compiled(inp1)
            eager_result1 = container_eager(inp1)
            same(compiled_result1, eager_result1)

            # Second call - weights are now on CUDA from first call
            # This tests that .to(cuda) on already-cuda weights doesn't fail
            compiled_result2 = compiled(inp1)
            eager_result2 = container_eager(inp1)
            same(compiled_result2, eager_result2)

    @unittest.skipIf(not TEST_CUDA, "This test requires a CUDA device")
    def test_module_to_move_compile(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(10, 10)

            def forward(self, x):
                x = self.fc(x)
                self.to("cpu")
                return x

        mod = Model().cuda()
        with torch._dynamo.config.patch(graph_break_on_nn_param_ctor=False):
            fn = torch.compile(mod, backend="aot_eager", fullgraph=True)
            x = torch.randn(10, 10, device="cuda")
            ref = fn(x)
            self.assertEqual(str(mod.fc.weight.device), "cpu")
            mod.cuda()
            ref = fn(
                x
            )  # second time compile runs, we should also move the module to cpu device
            self.assertEqual(str(mod.fc.weight.device), "cpu")
