# Owner(s): ["module: dynamo"]
import sys

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

    def test_supported_ctx_manager(self):
        global check, check_disabled, f1, f2, f3

        @torch._dynamo.disable
        def check_disabled(value):
            assert torch.is_grad_enabled() == value  # noqa: S101

        def check(value):
            assert torch.is_grad_enabled() == value  # noqa: S101

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
            assert torch.is_grad_enabled() == value  # noqa: S101

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

    def test_ctx_manager_nested_step_graph_break(self):
        global f1, f2

        def f1(x):
            x = x + 1
            torch._dynamo.step_unsupported()
            return x + 2

        def f2(x):
            x = x + 4
            with torch.no_grad():
                x = f1(x)
            return x + 8

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f2)
        x = torch.zeros(3)
        res = f2(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 5)

    def test_ctx_manager_active_in_nested_call(self):
        @torch._dynamo.disable()
        def f1(x):
            assert not torch.is_grad_enabled()  # noqa: S101
            return x + 1

        def f2(x):
            # order matters! should cancel out the enable_grad
            with torch.no_grad():
                return f1(x + 2) + 4

        def f3(x):
            with torch.enable_grad():
                return f2(x + 8) + 16

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f3)
        x = torch.zeros(3)
        res = f3(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 8)

    def test_ctx_manager_active_in_nested_step_graph_break(self):
        def f1(x):
            torch._dynamo.step_unsupported()
            assert not torch.is_grad_enabled()  # noqa: S101
            return x + 1

        def f2(x):
            return f1(x + 2) + 4

        def f3(x):
            with torch.no_grad():
                return f2(x + 8) + 16

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f3)
        x = torch.zeros(3)
        res = f3(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 8)

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

    def test_dead_nested_cells(self):
        global f1, f2, f3

        def f3(x, cell1):
            cell1 += 2
            x = x + cell1
            torch._dynamo.graph_break()
            return x + cell1

        def f1(cell1=0):
            def inner(x):
                x += 4
                x = f3(x, cell1)
                return x + 8

            return inner

        def f2(x):
            return f1()(x + 16) + 32

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f2)
        x = torch.zeros(3)
        res = f2(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        # If we don't handle dead cells in nested functions correctly,
        # frame_count will increase since we also
        # graph break when we attempt to codegen inner.
        # The exact issue was that side_effects was failing to codegen inner's cell's creation.
        # So when we try to codegen cells for resume functions, we end up trying to codegen
        # a CellVariable without a source, which leads to a graph break we can't resume from.
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 6)

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

    def test_nested_step_graph_break_diff_args(self):
        global inner, outer

        def inner(x1, x2):
            torch._dynamo.step_unsupported()
            return x1 + x2

        class Foo:
            def __init__(self):
                self.attr = 1

        def outer(x):
            z = Foo()
            y = inner(x + 1, x + 2)
            y = y + z.attr
            return y

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(outer)
        x = torch.zeros(3)
        res = outer(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 3)

    def test_generator_nested_graph_break(self):
        def gen(x):
            yield x + 1
            torch._dynamo.graph_break()
            yield x + 2

        def fn(x):
            x = x + 4
            return list(gen(x))

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(fn)
        x = torch.zeros(3)
        res = fn(x)
        # NOTE: if we enable nested graph breaks on inlined generators, we expect
        # some sort of internal dynamo failure
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        # fn should be skipped
        self.assertEqual(cnts.frame_count, 0)

        def outer(x):
            x = x + 8
            return fn(x)[0] + 16

        cnts.clear()
        torch.compiler.reset()

        opt_fn = torch._dynamo.optimize(backend=cnts)(outer)
        x = torch.zeros(3)
        res = outer(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        # only outer should be traced
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 2)

    def test_return_after_graph_break_nested(self):
        # With improper implementation, returning immediately after a nested graph
        # break may skip the rest of the top-level frame.
        def f2(inner, x):
            x += 2
            return inner(x)

        @torch.compile(backend="eager")
        def f3(inner, x):
            result = f2(inner, x)
            x += 4
            if result is not None:
                x += result
            return x

        # test normal graph break
        x = torch.zeros(3)

        def inner1(x):
            x += 1
            return torch._dynamo.graph_break()

        ref = f3(inner1, x)
        self.assertEqual(ref, torch.zeros(3) + 7)

        # test step graph break
        x = torch.zeros(3)

        def inner2(x):
            x += 1
            return torch._dynamo.step_unsupported()

        ref = f3(inner2, x)
        self.assertEqual(ref, torch.zeros(3) + 7)

        # test store attr graph break
        # NOTE: we do this manual bytecode generation hack since the only RETURN_*
        # instruction that can follow STORE_ATTR is RETURN_CONST, which was removed in 3.14+.

        # make sure inner3's code options are compatible with the instructions below
        global y

        def y():
            pass

        def inner3(x):
            x.attr = 1000
            y.attr = 2000

        new_inst = torch._dynamo.bytecode_transformation.create_instruction
        insts = [
            new_inst("LOAD_CONST", argval=1000),
            new_inst("LOAD_CONST", argval=2000),
            new_inst("LOAD_GLOBAL", argval="y"),
            # NOTE: this should cause a graph break - change y if it doesn't work!
            new_inst("STORE_ATTR", argval="attr"),
            new_inst("RETURN_VALUE"),
        ]
        if sys.version_info >= (3, 11):
            insts = [new_inst("RESUME", arg=0)] + insts
        code_keys = torch._dynamo.bytecode_transformation.get_code_keys()
        code_options = {k: getattr(inner3.__code__, k) for k in code_keys}
        _, inner3_code = (
            torch._dynamo.bytecode_transformation.clean_and_assemble_instructions(
                insts, code_keys, code_options
            )
        )
        inner3.__code__ = inner3_code

        torch._dynamo.utils.counters.clear()
        x = torch.zeros(3)
        ref = f3(inner3, x)
        self.assertEqual(ref, torch.zeros(3) + 1006)
        # make sure we're actually STORE_ATTR graph breaking
        self.assertEqual(len(torch._dynamo.utils.counters["graph_break"]), 1)

        # dynamic branching is harder to test - the other tests should be enough cover

        # test every function returning
        @torch.compiler.disable
        def inner5(x):
            x += 8
            return x

        def inner4(x):
            x += 1
            return inner5(x)

        @torch.compile(backend="eager")
        def f4(x):
            x += 4
            return f2(inner4, x)

        x = torch.zeros(3)
        ref = f4(x)
        self.assertEqual(ref, torch.zeros(3) + 15)

    def test_return_after_graph_break_deep_nested(self):
        @torch.compiler.disable
        def f1(x):
            return x + 1

        def f2(x):
            return f1(x + 2)

        def f3(x):
            return f2(x + 4)

        def f4(x):
            x = f3(x + 8)
            return x + 16

        def f5(x):
            return f4(x + 32)

        def f6(x):
            return f5(x + 64)

        def f7(x):
            x = f6(x + 128)
            return x + 256

        @torch.compile(backend="eager")
        def f8(x):
            return f7(x + 512)

        x = torch.zeros(3)
        ref = f8(x)
        self.assertEqual(ref, torch.zeros(3) + 1023)

        # check that only 2 resume functions are created
        self.assertEqual(len(torch._dynamo.utils.counters["resumes"]), 2)
        for name in ("resume_in_f4", "resume_in_f7"):
            self.assertTrue(
                any(name in key for key in torch._dynamo.utils.counters["resumes"])
            )

    def test_disable_nested_graph_breaks(self):
        global f1, f2, f3, f4, f5

        def f1(x):
            x = x + 1
            torch._dynamo.graph_break()
            return x + 2

        def f2(x):
            return f1(x + 4) + 8

        # NOTE since the disable_nested_graph_breaks decorator is implemented as a
        # context manager, we don't need to separately test context manager usage.
        @torch._dynamo.disable_nested_graph_breaks
        def f3(x):
            return f2(x + 16) + 32

        def f4(x):
            return f3(x + 64) + 128

        def f5(x):
            return f4(x + 256) + 512

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f5)
        x = torch.zeros(3)
        res = f5(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        # 2 frames from each of f5+f4, f3, f2, f1
        self.assertEqual(cnts.frame_count, 8)
        self.assertEqual(cnts.op_count, 10)

    def test_nested_store_attr_graph_break(self):
        class Foo:
            def __setattr__(self, name, value):
                torch._dynamo.graph_break()
                if not torch.compiler.is_compiling():
                    raise RuntimeError("Expected this to be traced")
                super().__setattr__(name, value + 1)

        def fn(foo, x):
            foo.attr = x + 2
            return x + 4

        foo = Foo()

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(fn)
        x = torch.zeros(3)
        ref = opt_fn(foo, x)
        self.assertEqual(ref, x + 4)
        self.assertEqual(foo.attr, x + 3)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 3)

    def test_nested_store_subscr_graph_break(self):
        class Foo:
            def __setitem__(self, name, value):
                torch._dynamo.graph_break()
                if not torch.compiler.is_compiling():
                    raise RuntimeError("Expected this to be traced")
                super().__setattr__(name, value + 1)

        def fn(foo, x):
            foo["attr"] = x + 2
            return x + 4

        foo = Foo()

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(fn)
        x = torch.zeros(3)
        ref = opt_fn(foo, x)
        self.assertEqual(ref, x + 4)
        self.assertEqual(foo.attr, x + 3)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 3)

    def test_functorch_with_nested_graph_break(self):
        def f1(x):
            x = x * 2
            torch._dynamo.graph_break()
            return x * 4

        def f2(x):
            return (f1(x * 8) * 16).sum()

        def f3(x):
            return torch.func.grad(f2)(x * 32) * 64

        def f4(x):
            return f3(x * 128) * 256

        cnts = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        x = torch.randn(3)
        actual = f4(x)
        expected = torch.compile(f4, backend=cnts, fullgraph=False)(x)
        self.assertEqual(actual, expected)
        self.assertEqual(len(torch._dynamo.utils.counters["graph_break"]), 1)
        # f4 + f3, f3 end + f4 end
        self.assertEqual(cnts.frame_count, 2)
        # multiplication by 32, 64, 128, 256
        self.assertEqual(cnts.op_count, 4)

    def test_error_on_graph_break_nested(self):
        # error_on_graph_break in a nested frame
        cnts = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.error_on_graph_break(False)
        def inner_f5(x):
            x = x + 2
            torch._dynamo.graph_break()
            return x + 4

        @torch._dynamo.error_on_graph_break(True)
        @torch.compile(backend=cnts)
        def f5(x):
            x = x + 1
            return inner_f5(x)

        inp = torch.ones(3)
        self.assertEqual(f5(inp), inp + 7)
        self.assertEqual(cnts.frame_count, 2)

        def inner_f6(x):
            x = x + 2
            with torch._dynamo.error_on_graph_break(False):
                torch._dynamo.graph_break()
            return x + 4

        @torch._dynamo.error_on_graph_break(True)
        @torch.compile(backend=cnts)
        def f6(x):
            x = x + 1
            return inner_f6(x)

        cnts.clear()
        self.assertEqual(f6(inp), inp + 7)
        self.assertEqual(cnts.frame_count, 2)

        def inner_f7(x):
            x = x + 2
            with torch._dynamo.error_on_graph_break(True):
                torch._dynamo.graph_break()
            return x + 4

        @torch._dynamo.error_on_graph_break(False)
        @torch.compile(backend=cnts)
        def f7(x):
            x = x + 1
            return inner_f7(x)

        with self.assertRaises(torch._dynamo.exc.Unsupported):
            f7(inp)

        def inner_f8(x):
            x = x + 2
            with torch._dynamo.error_on_graph_break(True):
                torch._dynamo.graph_break()
            return x + 4

        @torch._dynamo.error_on_graph_break(True)
        @torch.compile(backend=cnts)
        def f8(x):
            x = x + 1
            return inner_f7(x)

        with self.assertRaises(torch._dynamo.exc.Unsupported):
            f8(inp)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
