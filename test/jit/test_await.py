import torch
from torch.testing._internal.jit_utils import JitTestCase, _inline_everything
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
