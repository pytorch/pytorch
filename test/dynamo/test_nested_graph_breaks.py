# Owner(s): ["module: dynamo"]
import torch
import torch._dynamo.test_case
import torch._dynamo.testing

from torch._dynamo import config
from torch._dynamo.testing import make_test_cls_with_patches

try:
    from . import (
        test_aot_autograd,
        test_ctx_manager,
        test_functions,
        test_higher_order_ops,
        test_misc,
        test_modules,
        test_repros,
        test_sdpa,
        test_subgraphs,
    )
except ImportError:
    import test_aot_autograd
    import test_ctx_manager
    import test_export
    import test_functions
    import test_higher_order_ops
    import test_misc

    import test_modules
    import test_repros
    import test_sdpa
    import test_subgraphs


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
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__
    return test_class


tests = [
    test_ctx_manager.CtxManagerTests,
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
for test in tests:
    make_nested_cls(test)
del test

global_val = 0


class CustomizedCtxManager:
    def __init__(self, val):
        self.val = val

    def __enter__(self):
        global global_val
        global_val += self.val

    def __exit__(self, exc_type, exc_value, traceback):
        global global_val
        global_val -= self.val


# for use in test_side_effects_globals
global1, global2, global3, global4 = (torch.zeros(3),) * 4


class NestedGraphBreakTests(torch._dynamo.test_case.TestCase):
    def test_single_graph_break(self):
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

    def test_single_graph_break_repeat(self):
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

    def test_doubly_nested_graph_break(self):
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

    def test_differing_arg_nums(self):
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

    def test_differing_locals_nums(self):
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

    def test_ctx_manager(self):
        global global_val
        global_val = 0

        @torch._dynamo.disable
        def f1():
            return global_val

        def f2(x2):
            with CustomizedCtxManager(8):
                x2 = x2 + (1 << 4)
                x2 = x2 + f1()  # 15
                x2 = x2 + (1 << 5)
            x2 = x2 << 2
            x2 = x2 + global_val  # 3
            with CustomizedCtxManager(4):
                x2 = x2 << 4
                x2 = x2 + f1()  # 7
                x2 = x2 + (1 << 3)
            return x2

        def f3(x3):
            with CustomizedCtxManager(2):
                return f2(x3)

        def f4(x4):
            with CustomizedCtxManager(1):
                return f3(x4)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f4)
        x = torch.zeros(3, dtype=torch.long)
        res = f4(x)
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

    def test_side_effects_globals(self):
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

    def test_side_effects_globals_different_module(self):
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

    def test_nested_graph_break_in_loop(self):
        def f1(x, i):
            if i == 5:
                torch._dynamo.graph_break()
            return x + 1

        def f2(x):
            for i in range(8):
                x = f1(x, i)
            return x

        def f3(x):
            x = x + 1
            x = f2(x)
            x = x + 1

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f3)
        x = torch.zeros(3)
        res = f3(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        # skip frame due to nested graph break in for loop
        self.assertEqual(cnts.frame_count, 0)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
