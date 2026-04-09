# Owner(s): ["module: dynamo"]
# flake8: noqa: F403,F405

try:
    from ._higher_order_ops_test_utils import *
except ImportError:
    from _higher_order_ops_test_utils import *


class ActivationCheckpointingTests(
    torch._dynamo.test_case.TestCaseWithNestedGraphBreaks
):
    def _validate(self, fn, backend, *args, skip_check=False, fullgraph=True):
        cloned_args = []
        for arg in args:
            cloned_args.append(arg.detach().clone().requires_grad_(arg.requires_grad))

        torch.manual_seed(0)
        expected = fn(*args)
        expected.sum().backward()

        opt_fn = torch.compile(fn, fullgraph=fullgraph, backend=backend)
        torch.manual_seed(0)
        result = opt_fn(*cloned_args)
        result.sum().backward()

        if not skip_check:
            self.assertEqual(result, expected)
            for arg, cloned_arg in zip(args, cloned_args):
                self.assertEqual(arg.grad, cloned_arg.grad)

    @requires_cuda_and_triton
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_function(self):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=True
            )

        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @requires_cuda_and_triton
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_function_with_kwargs(self):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn,
                torch.sin(x),
                y,
                use_reentrant=True,
                preserve_rng_state=False,
            )

        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @requires_cuda_and_triton
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_dropout(self):
        def gn(x, y):
            return torch.nn.functional.dropout(torch.matmul(x, y), p=0.2)

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=True
            )

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.rngprims.philox_rand.default
        )
        # philox_rand is passed from fwd
        bw_compiler = functools.partial(
            count_ops, freq=0, op=torch.ops.rngprims.philox_rand.default
        )
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(
            fn, backend, x, y, skip_check=True
        )  # dropout decomp is known to diverge with eager

    @requires_cuda_and_triton
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_dropout_inductor(self):
        def gn(x, y):
            return torch.nn.functional.dropout(torch.matmul(x, y), p=0.2)

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=True
            )

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        backend = "inductor"
        self._validate(
            fn, backend, x, y, skip_check=True
        )  # dropout decomp is known to diverge with eager

    @requires_gpu_and_triton
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_fallback(self):
        def gn(x, y):
            torch._dynamo.graph_break()
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            return torch.cos(
                torch.utils.checkpoint.checkpoint(
                    gn, torch.sin(x), y, use_reentrant=True
                ),
            )

        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        args = (x, y)

        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        expected = fn(*args)
        result = torch.compile(fn, backend=cnt)(*args)

        self.assertEqual(result, expected)

        # One graph for torch.sin on the input, and other for torch.cos.
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 2)
        self.assertEqual(len(backend.graphs), 2)

    @requires_cuda_and_triton
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_module(self):
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return torch.sigmoid(self.linear(x))

        mod = MockModule()

        def fn(x):
            return torch.utils.checkpoint.checkpoint(
                mod, torch.sin(x), use_reentrant=True
            )

        x = torch.randn(10, 10, requires_grad=True)

        fw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.sigmoid.default
        )
        # sigmoid passed from fwd
        bw_compiler = functools.partial(
            count_ops, freq=0, op=torch.ops.aten.sigmoid.default
        )
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x)

    def test_override_fallthrough_dispatch_key(self):
        class _FallthroughTestOnly(torch._ops.HigherOrderOperator):
            def __init__(self):
                super().__init__("_fallthrough_test_only")

            def __call__(self, *args, **kwargs):
                return super().__call__(*args, **kwargs)

        test_op = _FallthroughTestOnly()
        default_keys = torch._ops._HIGHER_ORDER_OP_DEFAULT_FALLTHROUGH_DISPATCH_KEYS
        self.assertTrue(
            not any(test_op.non_fallthrough_keys.has(key) for key in default_keys)
        )

        foos = [lambda x=i: x for i, k in enumerate(default_keys)]
        for foo, fallthrough_key in zip(foos, default_keys):
            test_op.py_impl(fallthrough_key)(foo)

        self.assertTrue(
            all(test_op.non_fallthrough_keys.has(key) for key in default_keys)
        )
        self.assertEqual(
            list(range(len(default_keys))),
            [test_op.py_kernels[key]() for key in default_keys],
        )

    def test_cond_with_kwargs(self):
        from torch._higher_order_ops.cond import cond_op

        def test(pred, x):
            def true_fn(x):
                return x.clone()

            def false_fn(x):
                return -x

            return cond_op(pred=pred, true_fn=true_fn, false_fn=false_fn, operands=[x])

        cnt = CompileCounter()
        opt_test = torch.compile(test, backend=cnt, fullgraph=True)
        inp = torch.ones(3, 3)
        true_pred = torch.Tensor([True])
        false_pred = torch.Tensor([False])
        self.assertTrue(torch.allclose(test(true_pred, inp), opt_test(true_pred, inp)))
        self.assertEqual(cnt.frame_count, 1)
        self.assertTrue(
            torch.allclose(test(false_pred, inp), opt_test(false_pred, inp))
        )
        self.assertEqual(cnt.frame_count, 1)

    def test_cond_with_invalid_kwargs(self):
        from torch._higher_order_ops.cond import cond_op

        def test(pred, mode, x):
            def true_fn(x):
                return x.clone()

            def false_fn(x):
                return -x

            if mode:
                return cond_op(
                    pred=pred,
                    true_fn=true_fn,
                    false_fn=false_fn,
                    operands=[x],
                    invalid=True,
                )
            else:
                return cond_op(
                    pred,
                    pred=pred,
                    true_fn=true_fn,
                    false_fn=false_fn,
                    operands=[x],
                )

        cnt = CompileCounter()
        opt_test = torch.compile(test, backend=cnt)
        inp = torch.ones(3, 3)
        with self.assertRaises(torch._dynamo.exc.UncapturedHigherOrderOpError):
            opt_test(True, True, inp)

        with self.assertRaises(AssertionError):
            opt_test(True, False, inp)

    def test_cond_with_mismatched_output(self):
        def output_mismatch_test(x):
            def true_fn():
                return torch.concat([x, x])

            def false_fn():
                return x.sin()

            return torch.cond(x.sum() > 0, true_fn, false_fn)

        x = torch.randn(2, 3)
        output_mismatch_test(x)

        torch.compile(output_mismatch_test, backend="eager")(x)

    def test_non_aliasing_util(self):
        from torch._dynamo.variables.higher_order_ops import _assert_tensors_nonaliasing

        a = [torch.tensor(1), {"a": torch.tensor(1)}]
        b = (torch.tensor(1),)
        _assert_tensors_nonaliasing(a, b)

        with self.assertRaisesRegex(
            AssertionError, "inputs to function body cannot alias outputs"
        ):
            _assert_tensors_nonaliasing(a, a)

    def test_flop_counter_for_cond(self):
        from torch.utils.flop_counter import FlopCounterMode

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return torch.cond(
                    torch.tensor(True),
                    lambda x: self.linear(x),
                    lambda x: self.linear(self.linear(x)),
                    (x,),
                )

        mod = Mod()
        with FlopCounterMode(mod, display=False) as mode:
            mod(torch.randn(4, 4))

        self.assertEqual(
            mode.get_flop_counts(),
            {
                "Global": {torch.ops.aten.addmm: 256},
                "Mod": {torch.ops.aten.addmm: 256},
                "Mod.linear": {torch.ops.aten.addmm: 256},
            },
        )

    def test_flop_counter_for_nested_cond(self):
        from torch.utils.flop_counter import FlopCounterMode

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(4, 4)
                self.linear2 = torch.nn.Linear(4, 4)

            def forward(self, x):
                def true_branch(x):
                    # Nested cond inside true branch
                    return torch.cond(
                        torch.tensor(True),
                        lambda x: self.linear1(x),
                        lambda x: self.linear2(x),
                        (x,),
                    )

                def false_branch(x):
                    return self.linear1(self.linear2(x))

                return torch.cond(torch.tensor(True), true_branch, false_branch, (x,))

        mod = Mod()
        with FlopCounterMode(mod, display=False) as mode:
            mod(torch.randn(4, 4))

        self.assertEqual(
            mode.get_flop_counts(),
            {
                "Global": {torch.ops.aten.addmm: 256},
                "Mod": {torch.ops.aten.addmm: 256},
                "Mod.linear1": {torch.ops.aten.addmm: 128},
                "Mod.linear2": {torch.ops.aten.addmm: 128},
            },
        )

    def test_flop_counter_for_cond_unbalanced_branches(self):
        from torch.utils.flop_counter import FlopCounterMode

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                def true_branch(x):
                    return self.linear(x)

                def false_branch(x):
                    return x.clone()

                return torch.cond(torch.tensor(True), true_branch, false_branch, (x,))

        mod = Mod()
        with FlopCounterMode(mod, display=False) as mode:
            mod(torch.randn(4, 4))

        self.assertEqual(
            mode.get_flop_counts(),
            {
                "Global": {torch.ops.aten.addmm: 128},
                "Mod": {torch.ops.aten.addmm: 128},
                "Mod.linear": {torch.ops.aten.addmm: 128},
            },
        )
