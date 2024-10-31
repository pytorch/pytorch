# Owner(s): ["oncall: jit"]

from typing import NamedTuple, Tuple

import torch
from torch.testing import FileCheck
from torch.testing._internal.jit_utils import JitTestCase


if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


class TestGetDefaultAttr(JitTestCase):
    def test_getattr_with_default(self):
        class A(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.init_attr_val = 1.0

            def forward(self, x):
                y = getattr(self, "init_attr_val")  # noqa: B009
                w: list[float] = [1.0]
                z = getattr(self, "missing", w)  # noqa: B009
                z.append(y)
                return z

        result = A().forward(0.0)
        self.assertEqual(2, len(result))
        graph = torch.jit.script(A()).graph

        # The "init_attr_val" attribute exists
        FileCheck().check('prim::GetAttr[name="init_attr_val"]').run(graph)
        # The "missing" attribute does not exist, so there should be no corresponding GetAttr in AST
        FileCheck().check_not("missing").run(graph)
        # instead the getattr call will emit the default value, which is a list with one float element
        FileCheck().check("float[] = prim::ListConstruct").run(graph)

    def test_getattr_named_tuple(self):
        global MyTuple

        class MyTuple(NamedTuple):
            x: str
            y: torch.Tensor

        def fn(x: MyTuple) -> Tuple[str, torch.Tensor, int]:
            return (
                getattr(x, "x", "fdsa"),
                getattr(x, "y", torch.ones((3, 3))),
                getattr(x, "z", 7),
            )

        inp = MyTuple(x="test", y=torch.ones(3, 3) * 2)
        ref = fn(inp)
        fn_s = torch.jit.script(fn)
        res = fn_s(inp)
        self.assertEqual(res, ref)

    def test_getattr_tuple(self):
        def fn(x: Tuple[str, int]) -> int:
            return getattr(x, "x", 2)

        with self.assertRaisesRegex(RuntimeError, "but got a normal Tuple"):
            torch.jit.script(fn)
