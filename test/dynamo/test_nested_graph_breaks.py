# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo import config
from torch._dynamo.testing import make_test_cls_with_patches


try:
    # from . import test_ctx_manager
    pass
except ImportError:
    # import test_aot_autograd
    # import test_ctx_manager

    # import test_export
    # import test_functions
    # import test_higher_order_ops
    # import test_misc
    # import test_modules
    # import test_repros
    # import test_sdpa
    # import test_subgraphs
    pass


test_classes = {}


def make_nested_cls(cls):
    suffix = "_nested_graph_breaks"

    cls_prefix = "NestedGraphBreaks"

    test_class = make_test_cls_with_patches(
        cls,
        cls_prefix,
        suffix,
        (config, "debug_force_nested_calls", True),
        (config, "debug_force_graph_break_on_leaf_return", True),
        (config, "debug_disable_compile_counter", True),
        xfail_prop="_expected_failure_nested_graph_breaks",
    )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    # globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__
    return test_class


tests = [
    # test_ctx_manager.CtxManagerTests,
    # test_functions.FunctionTests,
    # test_misc.MiscTests,
    # test_repros.ReproTests,
    # test_modules.NNModuleTests,
    # test_subgraphs.SubGraphTests,
    # test_higher_order_ops.HigherOrderOpTests,
    # test_higher_order_ops.FuncTorchHigherOrderOpTests,
    # test_aot_autograd.AotAutogradFallbackTests,
    # test_sdpa.TestSDPA,
]
test = None
for test in tests:
    make_nested_cls(test)
del test


# for use in test_side_effects_globals
global1, global2, global3, global4 = (torch.zeros(3),) * 4


class NestedGraphBreakTests(torch._dynamo.test_case.TestCaseWithNestedGraphBreaks):
    def test_single_graph_break(self):
        # NOTE marking f1, f2, f3 as global
        # prevents them from being freevars
        global f1, f2, f3

        def f1(x1):
            x1 = x1 + 1
            torch._dynamo.graph_break()
            return x1 + 2

        def f2(x2):
            return f1(x2 + 4) + 8

        def f3(x3):
            return f2(x3 + 16) + 32

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f3)
        x = torch.zeros(3)
        res = f3(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 6)

    def test_single_graph_break_repeat(self):
        global f1, f2, f3

        def f1(x1):
            x1 = x1 + 1
            torch._dynamo.graph_break()
            return x1 + 2

        def f2(x2):
            tmp1 = f1(x2 + 4)
            tmp2 = f1(x2 + 8) << 4
            return tmp1 + tmp2

        def f3(x3):
            return f2(x3 + 256) + 512

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f3)
        x = torch.zeros(3, dtype=torch.long)
        res = f3(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 3)
        self.assertEqual(cnts.op_count, 10)

    def test_doubly_nested_graph_break(self):
        global f1, f2, f3

        def f1(x1):
            x1 = x1 + 1
            torch._dynamo.graph_break()
            return x1 + 2

        def f2(x2):
            x2 = x2 + 4
            torch._dynamo.graph_break()
            return f1(x2 + 8) + 16

        def f3(x3):
            return f2(x3 + 32) + 64

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f3)
        x = torch.zeros(3)
        res = f3(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 3)
        self.assertEqual(cnts.op_count, 7)

    def test_differing_arg_nums(self):
        global f1, f2, f3, f4

        def f1(x1, x2):
            x = x1 + x2
            torch._dynamo.graph_break()
            return x + 1

        def f2(x3, x4, x5, x6):
            return f1(x3 + x4, x5 + x6) + 2

        def f3(x7, x8):
            return f2(x7, x7 + 4, x8, x8 + 8) + 16

        def f4(x9):
            return f3(x9, x9 + 32) + 64

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f4)
        x = torch.zeros(3)
        res = f4(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 10)

    def test_differing_locals_nums(self):
        global f1, f2, f3

        def f1(x1):
            loc1 = x1 + 1
            torch._dynamo.graph_break()
            return loc1 + 2

        def f2(x2):
            loc1 = x2 + 4
            loc2 = x2 + 8
            return f1(x2) + loc1 + loc2

        def f3(x3):
            loc1 = x3 + 16
            loc2 = x3 + 32
            loc3 = x3 + 64
            loc4 = x3 + 128
            return f2(x3) + loc1 + loc2 + loc3 + loc4

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f3)
        x = torch.zeros(3)
        res = f3(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 14)

    def test_counters(self):
        global f1, f2, f3, f4

        def f1(x):
            x = x + 1
            torch._dynamo.graph_break()
            return x + 2

        def f2(x):
            return f1(x + 4) + 8

        def f3(x):
            x = x + 16
            for _ in range(1):
                x = f2(x)
            return x + 32

        @torch.compile(backend="eager")
        def f4(x):
            return f3(x + 64) + 128

        self.assertEqual(f4(torch.zeros(3)), torch.zeros(3) + 255)
        self.assertEqual(len(torch._dynamo.utils.counters["graph_break"]), 2)
        breakpoint()

    def test_supported_ctx_manager(self):
        global check, check_disabled, f1, f2, f3

        @torch._dynamo.disable
        def check_disabled(value):
            assert torch.is_grad_enabled() == value

        def check(value):
            assert torch.is_grad_enabled() == value

        def f1(x):
            with torch.no_grad():
                x = x + 1
                check(False)
                check_disabled(False)
                check(False)
                return x + 2

        def f2(x):
            with torch.enable_grad():
                x = x + 4
                check(True)
                check_disabled(True)
                check(True)
                return f1(x) + 8

        def f3(x):
            with torch.no_grad():
                x = x + 16
                check(False)
                check_disabled(False)
                check(False)
                return f2(x) + 32

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f3)
        x = torch.zeros(3)
        res = f3(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 4)
        # includes set_grad_enabled ops
        self.assertEqual(cnts.op_count, 14)

    def test_inactive_ctx_manager(self):
        global check, f1, f2, f3

        def check(value):
            assert torch.is_grad_enabled() == value

        def f1(x, ctx1):
            x = x + 1
            ctx2 = torch.no_grad()
            # torch.no_grad() is a stack value at the time of graph break
            ctx3 = (torch.no_grad(), torch._dynamo.graph_break())[0]
            x = x + 64
            torch._dynamo.graph_break()
            with ctx1:
                check(False)
            with ctx2:
                check(False)
            with ctx3:
                check(False)
            return x + 2

        def f2(x, ctx1):
            x = x + 4
            ctx2 = torch.no_grad()
            x = f1(x, torch.no_grad())
            with ctx1:
                check(False)
            with ctx2:
                check(False)
            return x + 8

        def f3(x):
            x = x + 16
            ctx = torch.no_grad()
            x = f2(x, torch.no_grad())
            with ctx:
                check(False)
            return x + 32

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f3)
        x = torch.zeros(3)
        res = f3(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 3)
        self.assertEqual(cnts.op_count, 7)

    @torch._dynamo.config.patch(recompile_limit=1, fail_on_recompile_limit_hit=True)
    def test_no_recompiles(self):
        global f1, f2, f3

        def f1(x):
            x = x + 1
            torch._dynamo.graph_break()
            return x + 2

        def f2(x):
            x = x + 4
            x = f1(x)
            torch._dynamo.graph_break()
            return x + 8

        def f3(x):
            x = x + 16
            return f2(x) + 32

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f3)
        x = torch.zeros(3)
        res = f3(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 3)

    def test_cells(self):
        def f1(x1):
            cell1 = x1 + 1
            cell2 = x1 + 2

            def f2(x2, x3):
                nonlocal cell1
                cell3 = x2 + x3 + 4
                cell1 += 8

                def f3(x4):
                    nonlocal cell2, cell3
                    cell2 += 16
                    cell3 += 32
                    torch._dynamo.graph_break()
                    return x4 + cell1 + cell2 + cell3

                return f3(x2 + x3), cell3

            return f2(x1 + 64, x1 + 128) + (cell1, cell2)

        def outer(x):
            return f1(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(outer)
        x = torch.zeros(3)
        res = outer(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 13)

    def test_cells_double_graph_break(self):
        def f1(x1):
            cell1 = x1 + 1

            def f2(x2):
                nonlocal cell1
                cell1 += 2
                torch._dynamo.graph_break()
                torch._dynamo.graph_break()
                return x2 + cell1

            return f2(x1 + 4), cell1

        def outer(x):
            return f1(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(outer)
        x = torch.zeros(3)
        res = outer(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 4)

    def test_side_effects_cells(self):
        cell1, cell2, cell3, cell4 = (torch.zeros(3),) * 4

        def f1():
            nonlocal cell1
            cell1 += 1
            torch._dynamo.graph_break()
            return cell1 + cell2

        def f2():
            nonlocal cell3
            cell3 += 2
            return f1() + cell3 + cell4

        def f3():
            return f2()

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f3)

        cell1 = torch.zeros(3)
        cell2 = torch.zeros(3) + 4
        cell3 = torch.zeros(3)
        cell4 = torch.zeros(3) + 8
        res = f3()
        res = (res,) + tuple(x.clone() for x in (cell1, cell2, cell3, cell4))

        cell1 = torch.zeros(3)
        cell2 = torch.zeros(3) + 4
        cell3 = torch.zeros(3)
        cell4 = torch.zeros(3) + 8
        ref = opt_fn()
        ref = (ref,) + tuple(x.clone() for x in (cell1, cell2, cell3, cell4))

        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 5)

    def test_side_effects_globals(self):
        global f1, f2, f3
        global global1, global2, global3, global4

        def f1():
            global global1
            global1 += 1
            torch._dynamo.graph_break()
            return global1 + global2

        def f2():
            global global3
            global3 += 2
            return f1() + global3 + global4

        def f3(x):
            return x + f2()

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f3)
        x = torch.ones(3)

        global1 = torch.zeros(3)
        global2 = torch.zeros(3) + 4
        global3 = torch.zeros(3)
        global4 = torch.zeros(3) + 8
        res = (f3(x), global1.clone(), global2, global3.clone(), global4)

        global1 = torch.zeros(3)
        global2 = torch.zeros(3) + 4
        global3 = torch.zeros(3)
        global4 = torch.zeros(3) + 8
        ref = (opt_fn(x), global1.clone(), global2, global3.clone(), global4)

        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 6)

    def test_side_effects_globals_different_module(self):
        global f1, f2, _test_nested_graph_breaks_helper
        try:
            from . import _test_nested_graph_breaks_helper
        except ImportError:
            import _test_nested_graph_breaks_helper

        def f1(x):
            x = x + 1
            torch._dynamo.graph_break()
            return x + 1

        def f2(x):
            x = x + 1
            x = _test_nested_graph_breaks_helper.fn(x, f1)
            return x + 1

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f2)

        _test_nested_graph_breaks_helper.reset_state()
        x = torch.zeros(3)
        res = (f2(x), _test_nested_graph_breaks_helper.global1.clone())

        _test_nested_graph_breaks_helper.reset_state()
        ref = (opt_fn(x), _test_nested_graph_breaks_helper.global1.clone())

        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 7)

    def test_nested_graph_break_in_loop(self):
        global f1, f2, f3, f4, f5

        def f1(x, i):
            x = x + 1
            if i == 5:
                torch._dynamo.graph_break()
            return x + 1

        def f2(x, i):
            x = x + 1
            x = f1(x, i)
            return x + 1

        def f3(x):
            for i in range(8):
                x = f2(x, i)
            return x

        def f4(x):
            x = x + 1
            x = f3(x)
            return x + 1

        def f5(x):
            x = x + 1
            x = f4(x)
            return x + 1

        cnts = torch._dynamo.testing.CompileCounter()
        # dynamic=True to prevent unnecessary recompiles
        opt_fn = torch._dynamo.optimize(backend=cnts, dynamic=True)(f5)
        x = torch.zeros(3)
        res = f5(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        # skip frame due to nested graph break in for loop
        # 2 frames from f5+f4, 2 frames from f2+f1 (i == 5), 1 frame from f2+f1 (i != 5)
        self.assertEqual(cnts.frame_count, 5)
        # 4 additions from f5+f4, 2 x 4 additions from f2+f1 (i == 5, i != 5)
        self.assertEqual(cnts.op_count, 12)
        self.assertEqual(torch._dynamo.utils.counters["frames"]["total"], 6)

    def test_nested_graph_break_in_try_block(self):
        # NOTE: this also tests nested step_graph_break
        global f1, f2, f3, f4, f5

        def f1(x):
            x = x + 1
            torch._dynamo.graph_break()
            return x + 1

        def f2(x):
            x = x + 1
            x = f1(x)
            return x + 1

        def f3(x):
            x = x + 1
            try:
                x = x + 1
                x = f2(x)
                x = x + 1
            finally:
                pass
            return x + 1

        def f4(x):
            x = x + 1
            x = f3(x)
            return x + 1

        def f5(x):
            x = x + 1
            x = f4(x)
            return x + 1

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f5)
        x = torch.zeros(3)
        res = f5(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        # skip frame due to graph break in try block
        # 2 frames from f5+f4+(first part of f3), 2 frames from f2+f1
        self.assertEqual(cnts.frame_count, 4)
        # 5 additions from f5+f4+(first part of f3), 4 additions from f2+f1
        self.assertEqual(cnts.op_count, 9)
        self.assertEqual(torch._dynamo.utils.counters["frames"]["total"], 4)

    def test_nested_step_unsupported(self):
        global f1, f2, f3

        def f1(x):
            return x + 1

        def f2(x):
            x = x + 2
            torch._dynamo.step_unsupported()
            return f1(x) + 4

        def f3(x):
            x = x + 8
            return f2(x) + 16

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f3)
        x = torch.zeros(3)
        res = f3(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        # 1 frame from start of f3 + start of f2, 1 frame from f1, 1 frame from the end of f3
        self.assertEqual(cnts.frame_count, 3)
        # all ops except + 4
        self.assertEqual(cnts.op_count, 4)
        self.assertEqual(torch._dynamo.utils.counters["frames"]["total"], 3)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
