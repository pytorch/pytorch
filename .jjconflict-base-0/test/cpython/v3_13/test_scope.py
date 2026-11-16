# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_scope.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

import unittest
import weakref

from test.support import check_syntax_error, cpython_only
from test.support import gc_collect


class ScopeTests(CPythonTestCase):

    def testSimpleNesting(self):

        def make_adder(x):
            def adder(y):
                return x + y
            return adder

        inc = make_adder(1)
        plus10 = make_adder(10)

        self.assertEqual(inc(1), 2)
        self.assertEqual(plus10(-2), 8)

    def testExtraNesting(self):

        def make_adder2(x):
            def extra(): # check freevars passing through non-use scopes
                def adder(y):
                    return x + y
                return adder
            return extra()

        inc = make_adder2(1)
        plus10 = make_adder2(10)

        self.assertEqual(inc(1), 2)
        self.assertEqual(plus10(-2), 8)

    def testSimpleAndRebinding(self):

        def make_adder3(x):
            def adder(y):
                return x + y
            x = x + 1 # check tracking of assignment to x in defining scope
            return adder

        inc = make_adder3(0)
        plus10 = make_adder3(9)

        self.assertEqual(inc(1), 2)
        self.assertEqual(plus10(-2), 8)

    def testNestingGlobalNoFree(self):

        def make_adder4(): # XXX add exta level of indirection
            def nest():
                def nest():
                    def adder(y):
                        return global_x + y # check that plain old globals work
                    return adder
                return nest()
            return nest()

        global_x = 1
        adder = make_adder4()
        self.assertEqual(adder(1), 2)

        global_x = 10
        self.assertEqual(adder(-2), 8)

    def testNestingThroughClass(self):

        def make_adder5(x):
            class Adder:
                def __call__(self, y):
                    return x + y
            return Adder()

        inc = make_adder5(1)
        plus10 = make_adder5(10)

        self.assertEqual(inc(1), 2)
        self.assertEqual(plus10(-2), 8)

    def testNestingPlusFreeRefToGlobal(self):

        def make_adder6(x):
            global global_nest_x
            def adder(y):
                return global_nest_x + y
            global_nest_x = x
            return adder

        inc = make_adder6(1)
        plus10 = make_adder6(10)

        self.assertEqual(inc(1), 11) # there's only one global
        self.assertEqual(plus10(-2), 8)

    def testNearestEnclosingScope(self):

        def f(x):
            def g(y):
                x = 42 # check that this masks binding in f()
                def h(z):
                    return x + z
                return h
            return g(2)

        test_func = f(10)
        self.assertEqual(test_func(5), 47)

    def testMixedFreevarsAndCellvars(self):

        def identity(x):
            return x

        def f(x, y, z):
            def g(a, b, c):
                a = a + x # 3
                def h():
                    # z * (4 + 9)
                    # 3 * 13
                    return identity(z * (b + y))
                y = c + z # 9
                return h
            return g

        g = f(1, 2, 3)
        h = g(2, 4, 6)
        self.assertEqual(h(), 39)

    def testFreeVarInMethod(self):

        def test():
            method_and_var = "var"
            class Test:
                def method_and_var(self):
                    return "method"
                def test(self):
                    return method_and_var
                def actual_global(self):
                    return str("global")
                def str(self):
                    return str(self)
            return Test()

        t = test()
        self.assertEqual(t.test(), "var")
        self.assertEqual(t.method_and_var(), "method")
        self.assertEqual(t.actual_global(), "global")

        method_and_var = "var"
        class Test:
            # this class is not nested, so the rules are different
            def method_and_var(self):
                return "method"
            def test(self):
                return method_and_var
            def actual_global(self):
                return str("global")
            def str(self):
                return str(self)

        t = Test()
        self.assertEqual(t.test(), "var")
        self.assertEqual(t.method_and_var(), "method")
        self.assertEqual(t.actual_global(), "global")

    def testCellIsKwonlyArg(self):
        # Issue 1409: Initialisation of a cell value,
        # when it comes from a keyword-only parameter
        def foo(*, a=17):
            def bar():
                return a + 5
            return bar() + 3

        self.assertEqual(foo(a=42), 50)
        self.assertEqual(foo(), 25)

    def testCellIsArgAndEscapes(self):
        # We need to be sure that a cell passed in as an arg still
        # gets wrapped in a new cell if the arg escapes into an
        # inner function (closure).

        def external():
            value = 42
            def inner():
                return value
            cell, = inner.__closure__
            return cell
        cell_ext = external()

        def spam(arg):
            def eggs():
                return arg
            return eggs

        eggs = spam(cell_ext)
        cell_closure, = eggs.__closure__
        cell_eggs = eggs()

        self.assertIs(cell_eggs, cell_ext)
        self.assertIsNot(cell_eggs, cell_closure)

    def testCellIsLocalAndEscapes(self):
        # We need to be sure that a cell bound to a local still
        # gets wrapped in a new cell if the local escapes into an
        # inner function (closure).

        def external():
            value = 42
            def inner():
                return value
            cell, = inner.__closure__
            return cell
        cell_ext = external()

        def spam(arg):
            cell = arg
            def eggs():
                return cell
            return eggs

        eggs = spam(cell_ext)
        cell_closure, = eggs.__closure__
        cell_eggs = eggs()

        self.assertIs(cell_eggs, cell_ext)
        self.assertIsNot(cell_eggs, cell_closure)

    def testRecursion(self):

        def f(x):
            def fact(n):
                if n == 0:
                    return 1
                else:
                    return n * fact(n - 1)
            if x >= 0:
                return fact(x)
            else:
                raise ValueError("x must be >= 0")

        self.assertEqual(f(6), 720)


    def testUnoptimizedNamespaces(self):

        check_syntax_error(self, """if 1:
            def unoptimized_clash1(strip):
                def f(s):
                    from sys import *
                    return getrefcount(s) # ambiguity: free or local
                return f
            """)

        check_syntax_error(self, """if 1:
            def unoptimized_clash2():
                from sys import *
                def f(s):
                    return getrefcount(s) # ambiguity: global or local
                return f
            """)

        check_syntax_error(self, """if 1:
            def unoptimized_clash2():
                from sys import *
                def g():
                    def f(s):
                        return getrefcount(s) # ambiguity: global or local
                    return f
            """)

        check_syntax_error(self, """if 1:
            def f():
                def g():
                    from sys import *
                    return getrefcount # global or local?
            """)

    def testLambdas(self):

        f1 = lambda x: lambda y: x + y
        inc = f1(1)
        plus10 = f1(10)
        self.assertEqual(inc(1), 2)
        self.assertEqual(plus10(5), 15)

        f2 = lambda x: (lambda : lambda y: x + y)()
        inc = f2(1)
        plus10 = f2(10)
        self.assertEqual(inc(1), 2)
        self.assertEqual(plus10(5), 15)

        f3 = lambda x: lambda y: global_x + y
        global_x = 1
        inc = f3(None)
        self.assertEqual(inc(2), 3)

        f8 = lambda x, y, z: lambda a, b, c: lambda : z * (b + y)
        g = f8(1, 2, 3)
        h = g(2, 4, 6)
        self.assertEqual(h(), 18)

    def testUnboundLocal(self):

        def errorInOuter():
            print(y)
            def inner():
                return y
            y = 1

        def errorInInner():
            def inner():
                return y
            inner()
            y = 1

        self.assertRaises(UnboundLocalError, errorInOuter)
        self.assertRaises(NameError, errorInInner)

    def testUnboundLocal_AfterDel(self):
        # #4617: It is now legal to delete a cell variable.
        # The following functions must obviously compile,
        # and give the correct error when accessing the deleted name.
        def errorInOuter():
            y = 1
            del y
            print(y)
            def inner():
                return y

        def errorInInner():
            def inner():
                return y
            y = 1
            del y
            inner()

        self.assertRaises(UnboundLocalError, errorInOuter)
        self.assertRaises(NameError, errorInInner)

    def testUnboundLocal_AugAssign(self):
        # test for bug #1501934: incorrect LOAD/STORE_GLOBAL generation
        exec("""if 1:
            global_x = 1
            def f():
                global_x += 1
            try:
                f()
            except UnboundLocalError:
                pass
            else:
                fail('scope of global_x not correctly determined')
            """, {'fail': self.fail})

    def testComplexDefinitions(self):

        def makeReturner(*lst):
            def returner():
                return lst
            return returner

        self.assertEqual(makeReturner(1,2,3)(), (1,2,3))

        def makeReturner2(**kwargs):
            def returner():
                return kwargs
            return returner

        self.assertEqual(makeReturner2(a=11)()['a'], 11)

    def testScopeOfGlobalStmt(self):
        # Examples posted by Samuele Pedroni to python-dev on 3/1/2001

        exec("""if 1:
            # I
            x = 7
            def f():
                x = 1
                def g():
                    global x
                    def i():
                        def h():
                            return x
                        return h()
                    return i()
                return g()
            self.assertEqual(f(), 7)
            self.assertEqual(x, 7)

            # II
            x = 7
            def f():
                x = 1
                def g():
                    x = 2
                    def i():
                        def h():
                            return x
                        return h()
                    return i()
                return g()
            self.assertEqual(f(), 2)
            self.assertEqual(x, 7)

            # III
            x = 7
            def f():
                x = 1
                def g():
                    global x
                    x = 2
                    def i():
                        def h():
                            return x
                        return h()
                    return i()
                return g()
            self.assertEqual(f(), 2)
            self.assertEqual(x, 2)

            # IV
            x = 7
            def f():
                x = 3
                def g():
                    global x
                    x = 2
                    def i():
                        def h():
                            return x
                        return h()
                    return i()
                return g()
            self.assertEqual(f(), 2)
            self.assertEqual(x, 2)

            # XXX what about global statements in class blocks?
            # do they affect methods?

            x = 12
            class Global:
                global x
                x = 13
                def set(self, val):
                    x = val
                def get(self):
                    return x

            g = Global()
            self.assertEqual(g.get(), 13)
            g.set(15)
            self.assertEqual(g.get(), 13)
            """)

    def testLeaks(self):

        class Foo:
            count = 0

            def __init__(self):
                Foo.count += 1

            def __del__(self):
                Foo.count -= 1

        def f1():
            x = Foo()
            def f2():
                return x
            f2()

        for i in range(100):
            f1()

        gc_collect()  # For PyPy or other GCs.
        self.assertEqual(Foo.count, 0)

    def testClassAndGlobal(self):

        exec("""if 1:
            def test(x):
                class Foo:
                    global x
                    def __call__(self, y):
                        return x + y
                return Foo()

            x = 0
            self.assertEqual(test(6)(2), 8)
            x = -1
            self.assertEqual(test(3)(2), 5)

            looked_up_by_load_name = False
            class X:
                # Implicit globals inside classes are be looked up by LOAD_NAME, not
                # LOAD_GLOBAL.
                locals()['looked_up_by_load_name'] = True
                passed = looked_up_by_load_name

            self.assertTrue(X.passed)
            """)

    def testLocalsFunction(self):

        def f(x):
            def g(y):
                def h(z):
                    return y + z
                w = x + y
                y += 3
                return locals()
            return g

        d = f(2)(4)
        self.assertIn('h', d)
        del d['h']
        self.assertEqual(d, {'x': 2, 'y': 7, 'w': 6})

    def testLocalsClass(self):
        # This test verifies that calling locals() does not pollute
        # the local namespace of the class with free variables.  Old
        # versions of Python had a bug, where a free variable being
        # passed through a class namespace would be inserted into
        # locals() by locals() or exec or a trace function.
        #
        # The real bug lies in frame code that copies variables
        # between fast locals and the locals dict, e.g. when executing
        # a trace function.

        def f(x):
            class C:
                x = 12
                def m(self):
                    return x
                locals()
            return C

        self.assertEqual(f(1).x, 12)

        def f(x):
            class C:
                y = x
                def m(self):
                    return x
                z = list(locals())
            return C

        varnames = f(1).z
        self.assertNotIn("x", varnames)
        self.assertIn("y", varnames)

    @cpython_only
    def testLocalsClass_WithTrace(self):
        # Issue23728: after the trace function returns, the locals()
        # dictionary is used to update all variables, this used to
        # include free variables. But in class statements, free
        # variables are not inserted...
        import sys
        self.addCleanup(sys.settrace, sys.gettrace())
        sys.settrace(lambda a,b,c:None)
        x = 12

        class C:
            def f(self):
                return x

        self.assertEqual(x, 12) # Used to raise UnboundLocalError

    def testBoundAndFree(self):
        # var is bound and free in class

        def f(x):
            class C:
                def m(self):
                    return x
                a = x
            return C

        inst = f(3)()
        self.assertEqual(inst.a, inst.m())

    @cpython_only
    def testInteractionWithTraceFunc(self):

        import sys
        def tracer(a,b,c):
            return tracer

        def adaptgetter(name, klass, getter):
            kind, des = getter
            if kind == 1:       # AV happens when stepping from this line to next
                if des == "":
                    des = "_%s__%s" % (klass.__name__, name)
                return lambda obj: getattr(obj, des)

        class TestClass:
            pass

        self.addCleanup(sys.settrace, sys.gettrace())
        sys.settrace(tracer)
        adaptgetter("foo", TestClass, (1, ""))
        sys.settrace(None)

        self.assertRaises(TypeError, sys.settrace)

    def testEvalExecFreeVars(self):

        def f(x):
            return lambda: x + 1

        g = f(3)
        self.assertRaises(TypeError, eval, g.__code__)

        try:
            exec(g.__code__, {})
        except TypeError:
            pass
        else:
            self.fail("exec should have failed, because code contained free vars")

    def testListCompLocalVars(self):

        try:
            print(bad)
        except NameError:
            pass
        else:
            print("bad should not be defined")

        def x():
            [bad for s in 'a b' for bad in s.split()]

        x()
        try:
            print(bad)
        except NameError:
            pass

    def testEvalFreeVars(self):

        def f(x):
            def g():
                x
                eval("x + 1")
            return g

        f(4)()

    def testFreeingCell(self):
        # Test what happens when a finalizer accesses
        # the cell where the object was stored.
        class Special:
            def __del__(self):
                nestedcell_get()

    def testNonLocalFunction(self):

        def f(x):
            def inc():
                nonlocal x
                x += 1
                return x
            def dec():
                nonlocal x
                x -= 1
                return x
            return inc, dec

        inc, dec = f(0)
        self.assertEqual(inc(), 1)
        self.assertEqual(inc(), 2)
        self.assertEqual(dec(), 1)
        self.assertEqual(dec(), 0)

    def testNonLocalMethod(self):
        def f(x):
            class c:
                def inc(self):
                    nonlocal x
                    x += 1
                    return x
                def dec(self):
                    nonlocal x
                    x -= 1
                    return x
            return c()
        c = f(0)
        self.assertEqual(c.inc(), 1)
        self.assertEqual(c.inc(), 2)
        self.assertEqual(c.dec(), 1)
        self.assertEqual(c.dec(), 0)

    def testGlobalInParallelNestedFunctions(self):
        # A symbol table bug leaked the global statement from one
        # function to other nested functions in the same block.
        # This test verifies that a global statement in the first
        # function does not affect the second function.
        local_ns = {}
        global_ns = {}
        exec("""if 1:
            def f():
                y = 1
                def g():
                    global y
                    return y
                def h():
                    return y + 1
                return g, h
            y = 9
            g, h = f()
            result9 = g()
            result2 = h()
            """, local_ns, global_ns)
        self.assertEqual(2, global_ns["result2"])
        self.assertEqual(9, global_ns["result9"])

    def testNonLocalClass(self):

        def f(x):
            class c:
                nonlocal x
                x += 1
                def get(self):
                    return x
            return c()

        c = f(0)
        self.assertEqual(c.get(), 1)
        self.assertNotIn("x", c.__class__.__dict__)


    def testNonLocalGenerator(self):

        def f(x):
            def g(y):
                nonlocal x
                for i in range(y):
                    x += 1
                    yield x
            return g

        g = f(0)
        self.assertEqual(list(g(5)), [1, 2, 3, 4, 5])

    def testNestedNonLocal(self):

        def f(x):
            def g():
                nonlocal x
                x -= 2
                def h():
                    nonlocal x
                    x += 4
                    return x
                return h
            return g

        g = f(1)
        h = g()
        self.assertEqual(h(), 3)

    def testTopIsNotSignificant(self):
        # See #9997.
        def top(a):
            pass
        def b():
            global a

    def testClassNamespaceOverridesClosure(self):
        # See #17853.
        x = 42
        class X:
            locals()["x"] = 43
            y = x
        self.assertEqual(X.y, 43)
        class X:
            locals()["x"] = 43
            del x
        self.assertFalse(hasattr(X, "x"))
        self.assertEqual(x, 42)

    @cpython_only
    def testCellLeak(self):
        # Issue 17927.
        #
        # The issue was that if self was part of a cycle involving the
        # frame of a method call, *and* the method contained a nested
        # function referencing self, thereby forcing 'self' into a
        # cell, setting self to None would not be enough to break the
        # frame -- the frame had another reference to the instance,
        # which could not be cleared by the code running in the frame
        # (though it will be cleared when the frame is collected).
        # Without the lambda, setting self to None is enough to break
        # the cycle.
        class Tester:
            def dig(self):
                if 0:
                    lambda: self
                try:
                    1/0
                except Exception as exc:
                    self.exc = exc
                self = None  # Break the cycle
        tester = Tester()
        tester.dig()
        ref = weakref.ref(tester)
        del tester
        gc_collect()  # For PyPy or other GCs.
        self.assertIsNone(ref())

    def test_multiple_nesting(self):
        # Regression test for https://github.com/python/cpython/issues/121863
        class MultiplyNested:
            def f1(self):
                __arg = 1
                class D:
                    def g(self, __arg):
                        return __arg
                return D().g(_MultiplyNested__arg=2)

            def f2(self):
                __arg = 1
                class D:
                    def g(self, __arg):
                        return __arg
                return D().g

        inst = MultiplyNested()
        with self.assertRaises(TypeError):
            inst.f1()

        closure = inst.f2()
        with self.assertRaises(TypeError):
            closure(_MultiplyNested__arg=2)

if __name__ == "__main__":
    run_tests()
