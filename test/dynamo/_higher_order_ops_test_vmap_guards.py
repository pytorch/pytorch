# Owner(s): ["module: dynamo"]
# flake8: noqa: F403,F405

try:
    from ._higher_order_ops_test_utils import *
except ImportError:
    from _higher_order_ops_test_utils import *


class HigherOrderOpVmapGuardTests(
    torch._dynamo.test_case.TestCaseWithNestedGraphBreaks, LoggingTestCase
):
    @make_logging_test(recompiles=True)
    def test_vmap_grad_guard_ok(self, records):
        vmap = torch.vmap
        grad = torch.func.grad

        def g(x):
            return vmap(grad(torch.sin))(x)

        @torch.compile(backend="eager")
        def fn(x):
            return vmap(g)(x)

        x = torch.randn(4, 5)
        y = fn(x)
        # sanity check
        self.assertEqual(len(records), 0)
        self.assertEqual(x.cos(), y)

        # Calling the same function again won't have any effect on guards
        fn(x)
        self.assertEqual(len(records), 0)

    @xfailIfTorchDynamo
    @make_logging_test(recompiles=True)
    def test_grad_guard_fail(self, records):
        grad = torch.func.grad

        @torch.compile(backend="eager")
        def fn(x):
            return grad(torch.sin)(x.sum())

        x = torch.randn([])
        fn(x)
        self.assertEqual(len(records), 0)

        # calling again should not invalidate the graph
        fn(x)
        self.assertEqual(len(records), 0)

        # call grad should retrigger compilation
        x = torch.randn(3)
        grad(fn)(x)
        self.assertGreater(len(records), 0)
        record = self.getRecord(records, "pyfunctorch")
        self.assertIn(
            """torch._functorch.pyfunctorch.compare_functorch_state([])""",
            munge_exc(record.getMessage()),
        )

    @make_logging_test(recompiles=True)
    def test_dual_level_guard(self, records):
        fwAD = torch.autograd.forward_ad

        @torch.compile(backend="eager", fullgraph=True)
        def fn(foo, tangent):
            with fwAD.dual_level():
                dual = fwAD.make_dual(foo, tangent[1:])
                return dual

        foo = torch.rand(2)
        tangent = torch.rand(3)
        fn(foo, tangent)
        self.assertEqual(len(records), 0)

        # calling again should not invalidate the graph
        fn(foo, tangent)
        self.assertEqual(len(records), 0)

        # assertRaises is only here because Nested forward mode AD is not supported
        with self.assertRaises(torch._dynamo.exc.InternalTorchDynamoError):
            with fwAD.dual_level():
                fn(foo, tangent)
        self.assertGreater(len(records), 0)
        record = self.getRecord(records, "forward_ad")
        self.assertIn(
            """torch.autograd.forward_ad._current_level == -1""",
            munge_exc(record.getMessage()),
        )

    @xfailIfTorchDynamo
    @make_logging_test(recompiles=True)
    def test_jvp_guard_fail(self, records):
        jvp = torch.func.jvp
        vmap = torch.func.vmap

        @torch.compile(backend="eager")
        def fn(x):
            return jvp(torch.sin, (x,), (x,))

        x = torch.randn(3, 4)
        fn(x)
        self.assertEqual(len(records), 0)

        # calling again should not invalidate the graph
        fn(x)
        self.assertEqual(len(records), 0)

        # call jvp should retrigger compilation
        x = torch.randn(3, 4, 5)
        jvp(vmap(fn), (x,), (x,))

        self.assertGreater(len(records), 0)
        if self.hasRecord(records, "pyfunctorch"):
            record = self.getRecord(records, "pyfunctorch")
            self.assertIn(
                """torch._functorch.pyfunctorch.compare_functorch_state([])""",
                munge_exc(record.getMessage()),
            )
        elif self.hasRecord(records, "forward_ad"):
            record = self.getRecord(records, "forward_ad")
            self.assertIn(
                """torch.autograd.forward_ad._current_level == -1""",
                munge_exc(record.getMessage()),
            )

    @make_logging_test(recompiles=True)
    def test_vmap_guard_ok(self, records):
        @torch.compile(backend="eager")
        def fn(x):
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.randn(3, 3, 4, 5)
        y = fn(x)
        # sanity check
        self.assertEqual(len(records), 0)
        self.assertEqual(x.sin(), y)

        # Calling the same function again won't have any effect on guards
        z = fn(x)
        self.assertEqual(len(records), 0)
        self.assertEqual(x.sin(), z)

        # calling with a different object will also not affect guards
        w = fn(z)
        self.assertEqual(len(records), 0)
        self.assertEqual(z.sin(), w)

    @xfailIfTorchDynamo
    @make_logging_test(recompiles=True)
    def test_vmap_guard_fail_different_state(self, records):
        @torch.compile(backend="eager")
        def fn(x):
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.zeros(3, 4)
        y = torch.vmap(fn, randomness="same")(x)
        self.assertEqual(x.sin(), y)
        self.assertEqual(len(records), 0)

        # call vmap(vmap(fn))(x) should retrigger compilation
        y = torch.vmap(fn, randomness="different")(x)
        self.assertEqual(x.sin(), y)
        self.assertGreater(len(records), 0)
        record = self.getRecord(records, "pyfunctorch")
        self.assertIn(
            """torch._functorch.pyfunctorch.compare_functorch_state([('Vmap', 1, 'same')])""",
            record.getMessage(),
        )

    @xfailIfTorchDynamo
    @make_logging_test(recompiles=True)
    def test_vmap_guard_fail(self, records):
        @torch.compile(backend="eager")
        def fn(x):
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.zeros(3, 3, 4, 5)
        y = torch.vmap(fn)(x)
        self.assertEqual(x.sin(), y)
        self.assertEqual(len(records), 0)

        # call vmap(vmap(fn))(x) should retrigger compilation as
        # _functorch.current_level() is not the same
        x = torch.zeros(3, 3, 3, 4, 5)
        y = torch.vmap(torch.vmap(fn))(x)
        self.assertEqual(x.sin(), y)
        self.assertGreater(len(records), 0)
        record = self.getRecord(records, "pyfunctorch")
        self.assertIn(
            """torch._functorch.pyfunctorch.compare_functorch_state([('Vmap', 1, 'error')])""",
            record.getMessage(),
        )

    @xfailIfTorchDynamo
    @make_logging_test(recompiles=True)
    def test_vmap_grad_vmap_guard_fail(self, records):
        vmap = torch.vmap
        grad = torch.func.grad

        def g(x):
            y = vmap(torch.sin, randomness="same")(x)
            return y.sum(0)

        @torch.compile(backend="eager")
        def fn(x):
            return grad(g)(x)

        x = torch.randn(3, 3)
        y = vmap(fn, randomness="error")(x)
        self.assertEqual(x.cos(), y)

        # previous FX graph should be invalidated
        x = torch.randn(3, 3, 4)
        y = vmap(vmap(fn, randomness="different"))(x)
        self.assertGreater(len(records), 0)
        record = self.getRecord(records, "pyfunctorch")
        self.assertIn(
            """torch._functorch.pyfunctorch.compare_functorch_state([('Vmap', 1, 'error')])""",
            munge_exc(record.getMessage()),
        )

    @xfailIfTorchDynamo
    @make_logging_test(recompiles=True)
    def test_vmap_recompile_different_states(self, records):
        @torch.compile(backend="eager")
        def fn(x):
            return torch.vmap(lambda x: x.sin())(x)

        x = torch.zeros(3, 3, 4, 5)
        torch.vmap(fn, randomness="same")(x)
        self.assertEqual(len(records), 0)  # sanity check

        torch.vmap(fn, randomness="different")(x)
        self.assertGreater(len(records), 0)
        record = self.getRecord(records, "pyfunctorch")
        self.assertIn(
            """torch._functorch.pyfunctorch.compare_functorch_state([('Vmap', 1, 'same')])""",
            munge_exc(record.getMessage()),
        )

    @make_logging_test(guards=True)
    def test_emit_functorch_guard_if_active(self, records):
        @torch.compile(backend="eager")
        def fn(x):
            return torch.sin(x)

        x = torch.randn(3, 4)
        _ = fn(x)
        self.assertFalse(self.hasRecord(records, "pyfunctorch"))  # sanity check

        _ = torch.vmap(fn)(x)
        self.assertTrue(self.hasRecord(records, "pyfunctorch"))
        record = self.getRecord(records, "pyfunctorch")
        self.assertIn(
            """torch._functorch.pyfunctorch.compare_functorch_state([('Vmap', 1, 'error')])""",
            munge_exc(record.getMessage()),
        )

    @make_logging_test(recompiles=True)
    def test_linearize_recompiles(self, records):
        @torch.compile(backend="eager")
        def fn(x):
            out, jvp_fn = torch.func.linearize(torch.sin, x)
            return out, jvp_fn(x)

        x = torch.randn(2, 3)
        fn(x)
        self.assertEqual(len(records), 0)

        z = torch.randn(2, 3)
        fn(z)
        self.assertEqual(len(records), 0)

        y = torch.randn(3, 4)
        fn(y)
        self.assertGreater(len(records), 0)
