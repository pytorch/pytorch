import torch
from torch.testing._internal.jit_utils import JitTestCase, _inline_everything
from torch.testing._internal.jit_utils import make_global
from typing import List
from torch import Tensor
from torch.awaits import Await

class TestAwait(JitTestCase):
    def test_await_python(self):
        @torch.jit.script
        def foo():
            return 13
        aw = Await[int](foo)
        #self.assertEqual(aw.wait(), foo())

    def test_await_type_python(self):
        def foo() -> Tensor:
            return torch.randn()
        awaits = torch.jit.annotate(List[Await[Tensor]], [])
        awaits.append(Await[Tensor](foo))

    def test_script(self):
        print("XXX:test_script_START")
        def delayed(z: int) -> int:
            return z + 3

        @torch.jit.script
        def fn(x: Tensor):
            aw: Await[int] = torch.jit.awaitable(delayed, 99)
            a = torch.eye(2)
            b = torch.jit.awaitable_wait(aw)
            return a + b + x

        # print("XXX:test_script_TRACE")
        # tm = torch.jit.trace(fn, (torch.ones([2, 2])))
        # print(f"XXX tm.graph:{tm.graph}")
        inp = torch.zeros(2)

        sm = torch.jit.script(fn)
        print(f"XXX test_await sm.graph:{sm.graph}")
        out = fn(inp)
        print(f"XXX test_await eager out:{out}")
        script_out = sm(inp)
        print(f"XXX test_await script_out:{script_out}")

    def test_nowait(self):
        print("XXX:test_nowait_START")
        @torch.jit.script
        def fn(x: Tensor):
            aw = torch.jit.awaitable_nowait(13)
            a = torch.eye(2)
            b = torch.jit.awaitable_wait(aw)
            return a + b + x

        inp = torch.zeros(2)

        sm = torch.jit.script(fn)
        print(f"XXX test_await TEST_NOWAIT sm.graph:{sm.graph}")
        out = fn(inp)
        print(f"XXX test_await eager out:{out}")
        script_out = sm(inp)
        print(f"XXX test_await script_out:{script_out}")

    def test_nowait_class(self):
        class C(object):
            def __init__(self, a: Tensor, b: Tensor):
                self._a = a
                self._b = b

            def a(self) -> Tensor:
                return self._a

        @torch.jit.script
        def fn(x: Tensor):
            aw = torch.jit.awaitable_nowait(C(torch.zeros(2), torch.ones(2)))
            _a = torch.eye(2)
            c = torch.jit.awaitable_wait(aw)
            return _a + c.a() + x

        make_global(C)
        inp = torch.zeros(2)

        sm = torch.jit.script(fn)
        print(f"XXX test_await TEST_NOWAIT sm.graph:{sm.graph}")
        out = fn(inp)
        print(f"XXX test_await eager out:{out}")
        script_out = sm(inp)
        print(f"XXX test_await script_out:{script_out}")


    def test_await_class_arg(self):

        @torch.jit.script
        class C(object):
            def __init__(self, a: Tensor, b: Tensor):
                self.__a = a
                self.__b = b

            def a(self) -> Tensor:
                return self.__a

        make_global(C)
        @torch.jit.script
        def delayed(c: C) -> Tensor:
            return c.a()

        @torch.jit.script
        def fn(x: Tensor):
            c = C(torch.zeros(2), torch.ones(2))
            aw = torch.jit.awaitable(delayed, c)
            _a = torch.eye(2)
            c2_t = torch.jit.awaitable_wait(aw)
            return _a + c2_t + x
        inp = torch.zeros(2)

        sm = torch.jit.script(fn)
        print(f"XXX test_await TEST_NOWAIT sm.graph:{sm.graph}")
        out = fn(inp)
        print(f"XXX test_await eager out:{out}")
        script_out = sm(inp)
        print(f"XXX test_await script_out:{script_out}")

    def test_awaitable_to_await(self):
        class C(object):
            def __init__(self, a: Tensor, b: Tensor):
                self._a = a
                self._b = b


        make_global(C)
        # Can not stay in the class as Jit does not support Recursive annotations
        # (self in wait_impl can not be annotated as C as C is not defined by this time)
        def C_wait_impl(self: C):
                return self._a + self._b

        @torch.jit.script
        def fn(x: Tensor):
            aw = torch.jit.awaitable(C_wait_impl, C(torch.zeros(2), torch.ones(2)))
            _a = torch.eye(2)
            c_wait_impl_res = torch.jit.awaitable_wait(aw)
            return _a + c_wait_impl_res + x

        inp = torch.zeros(2)

        sm = torch.jit.script(fn)
        print(f"XXX test_await TEST_NOWAIT sm.graph:{sm.graph}")
        out = fn(inp)
        print(f"XXX test_await eager out:{out}")
        script_out = sm(inp)
        print(f"XXX test_await script_out:{script_out}")
