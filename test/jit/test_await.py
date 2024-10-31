# Owner(s): ["oncall: jit"]

import io
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch._awaits import _Await as Await
from torch.testing._internal.jit_utils import JitTestCase, make_global


class TestAwait(JitTestCase):
    def test_await_python(self):
        def foo(x: int) -> int:
            return x + 13

        aw: Await[int] = torch.jit._awaitable(foo, 13)
        self.assertTrue(aw.fn()(*aw.args()) == torch.jit._awaitable_wait(aw))
        nw = torch.jit._awaitable_nowait(33)
        self.assertTrue(nw.is_nowait())
        self.assertTrue(nw.args() == (33,))

    def test_await_type_python(self):
        def foo() -> Tensor:
            return torch.randn()

        awaits = torch.jit.annotate(List[Await[Tensor]], [])
        awaits.append(torch.jit._awaitable(foo))

    def test_script(self):
        def delayed(z: int) -> int:
            return z + 3

        def fn(x: Tensor):
            aw: Await[int] = torch.jit._awaitable(delayed, 99)
            a = torch.eye(2)
            b = torch.jit._awaitable_wait(aw)
            return a + b + x

        inp = torch.zeros(2)

        sm = torch.jit.script(fn)
        out = fn(inp)
        script_out = sm(inp)
        self.assertTrue(torch.allclose(torch.eye(2) + 102, script_out))
        self.assertTrue(torch.allclose(script_out, out))

    def test_nowait(self):
        def fn(x: Tensor):
            aw = torch.jit._awaitable_nowait(13)
            a = torch.eye(2)
            b = torch.jit._awaitable_wait(aw)
            return a + b + x

        inp = torch.zeros(2)

        sm = torch.jit.script(fn)
        out = fn(inp)
        script_out = sm(inp)
        self.assertTrue(torch.allclose(torch.eye(2) + 13, script_out))
        self.assertTrue(torch.allclose(script_out, out))

    def test_nowait_class(self):
        class C:
            def __init__(self, a: Tensor, b: Tensor):
                self._a = a
                self._b = b

            def a(self) -> Tensor:
                return self._a

        def fn(x: Tensor):
            aw = torch.jit._awaitable_nowait(C(torch.zeros(2), torch.ones(2)))
            _a = torch.eye(2)
            c = torch.jit._awaitable_wait(aw)
            return _a + c.a() + x

        make_global(C)
        inp = torch.zeros(2)

        sm = torch.jit.script(fn)
        out = fn(inp)
        script_out = sm(inp)
        self.assertTrue(torch.allclose(torch.eye(2), script_out))
        self.assertTrue(torch.allclose(script_out, out))

    def test_await_class_arg(self):
        class C:
            def __init__(self, a: Tensor, b: Tensor):
                self.__a = a
                self.__b = b

            def a(self) -> Tensor:
                return self.__a

        make_global(C)

        def delayed(c: C) -> Tensor:
            return c.a()

        def fn(x: Tensor):
            c = C(torch.zeros(2), torch.ones(2))
            aw = torch.jit._awaitable(delayed, c)
            _a = torch.eye(2)
            c2_t = torch.jit._awaitable_wait(aw)
            return _a + c2_t + x

        inp = torch.zeros(2)

        sm = torch.jit.script(fn)
        out = fn(inp)
        script_out = sm(inp)
        self.assertTrue(torch.allclose(torch.eye(2), script_out))
        self.assertTrue(torch.allclose(script_out, out))

    def test_awaitable_to_await(self):
        class C:
            __slots__ = ["_a", "_b"]

            def __init__(self, a: Tensor, b: Tensor):
                self._a = a
                self._b = b

        make_global(C)

        # Can not stay in the class as Jit does not support Recursive annotations
        # (self in wait_impl can not be annotated as C as C is not defined by this time)
        def C_wait_impl(self: C):
            return self._a + self._b

        def fn(x: Tensor):
            aw = torch.jit._awaitable(C_wait_impl, C(torch.zeros(2), torch.ones(2)))
            _a = torch.eye(2)
            c_wait_impl_res = torch.jit._awaitable_wait(aw)
            return _a + c_wait_impl_res + x

        inp = torch.ones(2)

        sm = torch.jit.script(fn)
        out = fn(inp)
        script_out = sm(inp)
        self.assertTrue(torch.allclose(torch.eye(2) + 2 * torch.ones(2), script_out))
        self.assertTrue(torch.allclose(script_out, out))

    def test_await_class_return(self):
        class C:
            __slots__ = ["a", "b"]

            def __init__(self, a: Tensor, b: Tensor):
                self.a = a
                self.b = b

        make_global(C)

        # Can not stay in the class as Jit does not support Recursive annotations
        # (self in wait_impl can not be annotated as C as C is not defined by this time)
        def C_wait_impl(self: C) -> C:
            return C(self.a * 2, self.b * 3)

        def fn_arg_C(x: C) -> Tensor:
            return x.a + x.b

        def fn(x: Tensor):
            aw: Await[C] = torch.jit._awaitable(C_wait_impl, C(x, x))
            _a = torch.eye(2)
            y = fn_arg_C(torch.jit._awaitable_wait(aw))
            return _a + y + x

        inp = torch.ones(2)

        sm = torch.jit.script(fn)
        out = fn(inp)
        script_out = sm(inp)
        self.assertTrue(torch.allclose(torch.eye(2) + 6 * torch.ones(2), script_out))
        self.assertTrue(torch.allclose(script_out, out))
        self.assertGraphContainsExactly(
            sm.graph, kind="prim::awaitable_wait", num_kind_nodes=1
        )

    def test_await_getattr_implicit_convertion(self):
        class C:
            def __init__(self, a: Tensor, b: Tensor):
                self._a = a
                self._b = b

            def b(self):
                return self._b

        make_global(C)

        # Can not stay in the class as Jit does not support Recursive annotations
        # (self in wait_impl can not be annotated as C as C is not defined by this time)
        def C_wait_impl(self: C) -> C:
            return C(self._a * 2, self._b * 3)

        def fn_arg_C(x: C) -> Tensor:
            return x._a + x._b

        def fn(x: Tensor):
            aw: Await[C] = torch.jit._awaitable(C_wait_impl, C(x, x))
            _a = torch.eye(2)
            ai = aw._a
            awb = aw.b()
            c = C(2 * x, 2 * x)
            return _a + ai + x + c._a + c.b()

        inp = torch.ones(2)

        sm = torch.jit.script(fn)
        out = fn(inp)
        script_out = sm(inp)
        self.assertTrue(torch.allclose(torch.eye(2) + 7 * torch.ones(2), script_out))
        self.assertTrue(torch.allclose(script_out, out))
        self.assertGraphContainsExactly(
            sm.graph, kind="prim::awaitable_wait", num_kind_nodes=2
        )

    def test_await_nested(self):
        class C:
            def __init__(self, a: Tensor, b: Tensor):
                self.__a = a
                self.__b = b

            def a(self) -> Tensor:
                return self.__a

        make_global(C)

        def delayed(c: C) -> Await[Tensor]:
            return torch.jit._awaitable_nowait(3 * c.a())

        def fn(x: Tensor) -> Await[Await[Tensor]]:
            return torch.jit._awaitable(delayed, C(2 * x, x))

        def main(x: Tensor) -> Tensor:
            awaw = fn(x)
            return torch.jit._awaitable_wait(torch.jit._awaitable_wait(awaw))

        inp = torch.eye(2)

        sm = torch.jit.script(main)
        out = main(inp)
        script_out = sm(inp)
        self.assertTrue(torch.allclose(6 * torch.eye(2), script_out))
        self.assertTrue(torch.allclose(script_out, out))

    def test_eager_await_non_scriptable(self):
        # Tree type can not be compiled (Recursive type)
        class Tree:
            def __init__(self, v):
                self.parent = torch.jit.annotate(Optional[Tree], None)
                self.v = v

        make_global(Tree)

        def delayed(t: Tree):
            t.v = t.v + 1
            return t

        aw = torch.jit._awaitable(delayed, Tree(2))
        t = torch.jit._awaitable_wait(aw)
        self.assertTrue(t.v == 3)

    def test_await_isinstance(self):
        def delayed(x: Tensor) -> Tensor:
            return 2 * (x + 1)

        def main(x: Tensor) -> Tensor:
            aw = torch.jit._awaitable(delayed, x)
            if torch.jit.is_scripting():
                assert isinstance(aw, torch.jit._Await)
            return torch.jit._awaitable_wait(aw)

        inp = torch.eye(2)

        sm = torch.jit.script(main)
        out = main(inp)
        script_out = sm(inp)
        self.assertTrue(
            torch.allclose(2 * torch.eye(2) + 2 * torch.ones(2), script_out)
        )
        self.assertTrue(torch.allclose(script_out, out))

    def test_await_eager_lazy(self):
        def delayed(x: Tensor) -> Tensor:
            return 2 * (x + 1)

        t = torch.ones(2, dtype=torch.int64)
        aw = torch.jit._awaitable(delayed, t)
        self.assertTrue(isinstance(aw, torch._C._Await))
        self.assertTrue(t.dtype == aw.dtype)

    def test_await_out_of_interpreter(self):
        def delayed(x: Tensor) -> Tensor:
            return 2 * (x + 1)

        def main(x: Tensor) -> Await[Tensor]:
            aw = torch.jit._awaitable(delayed, x)
            return aw

        inp = torch.eye(2)

        sm = torch.jit.script(main)
        out_aw = main(inp)
        out = torch.jit._awaitable_wait(out_aw)

        script_out_aw = sm(inp)
        script_out = torch.jit._awaitable_wait(script_out_aw)
        self.assertTrue(
            torch.allclose(2 * torch.eye(2) + 2 * torch.ones(2), script_out)
        )
        self.assertTrue(torch.allclose(script_out, out))

    def test_jit_trace(self):
        def gap(x: Tensor):
            return torch.relu(x) + torch.sin(x)

        def delayed(x: Tensor) -> Tensor:
            return 2 * (torch.cos(x) + 1)

        def main(x: Tensor, y: Tensor) -> Tensor:
            aw = torch.jit._awaitable(delayed, x)
            z = gap(y)
            k = torch.jit._awaitable_wait(aw)
            return y + k

        inp = torch.randn(2)
        tm = torch.jit.trace(main, (inp, inp))
        inp_check = torch.ones(2)
        self.assertEqual(main(inp_check, inp_check), tm(inp_check, inp_check))

    def test_await_multiout_save(self):
        def gap(x: Tensor):
            return torch.relu(x) + torch.sin(x)

        def delayed(x: Tensor) -> Tuple[Tensor, List[Tensor]]:
            l = [x * i for i in range(5)]
            return (100 * x, l)

        def main(x: Tensor) -> Tensor:
            aw = torch.jit._awaitable(delayed, x)
            z = gap(x)
            (_, l) = torch.jit._awaitable_wait(aw)
            return l[3] + z

        inp = torch.eye(2)

        sm = torch.jit.script(main)
        out = main(inp)
        script_out = sm(inp)
        expected = 4.8415 * torch.eye(2)
        self.assertTrue(torch.allclose(expected, script_out))
        self.assertTrue(torch.allclose(script_out, out))

        iofile = io.BytesIO()
        torch.jit.save(sm, iofile)
        iofile.seek(0)
        sm = torch.jit.load(iofile)
        script_out_load = sm(inp)
        self.assertTrue(torch.allclose(expected, script_out_load))

    def test_await_func_arg(self):
        def gap(x: Tensor):
            return torch.relu(x) + torch.sin(x)

        def delayed(x: Tensor) -> Tensor:
            return -1 * x

        def fn(aw: Await[Tensor]) -> Tensor:
            return 3 * torch.jit._awaitable_wait(aw)

        def main(x: Tensor) -> Tensor:
            aw = torch.jit._awaitable(delayed, x)
            z = gap(x)
            y = fn(aw)
            return y + x

        inp = torch.eye(2)

        sm = torch.jit.script(main)
        out = main(inp)
        script_out = sm(inp)
        expected = -2 * torch.eye(2)
        self.assertTrue(torch.allclose(expected, script_out))
        self.assertTrue(torch.allclose(script_out, out))

        iofile = io.BytesIO()
        torch.jit.save(sm, iofile)
        iofile.seek(0)
        sm = torch.jit.load(iofile)
        script_out_load = sm(inp)
        self.assertTrue(torch.allclose(expected, script_out_load))
