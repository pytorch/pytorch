# Owner(s): ["module: unknown"]

from copy import copy

import torch
from torch import nn
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfTorchDynamo,
    TestCase,
    xfailIfTorchDynamo,
)
from torch.utils.checkpoint import checkpoint
from torch.utils.module_tracker import ModuleTracker


class TestModuleTracker(TestCase):
    # "https://github.com/pytorch/pytorch/issues/127112
    @xfailIfTorchDynamo
    def test_module_hierarchy(self):
        seen_fw = []
        seen_bw = []

        class Foo(nn.Module):
            def forward(self, x):
                x = x["a"].relu_()
                seen_fw.append((copy(tracker.parents), tracker.is_bw))
                x.register_hook(
                    lambda grad: seen_bw.append((copy(tracker.parents), tracker.is_bw))
                )
                return {"a": torch.mm(x, x)}

        class Mod(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = Foo()
                self.b = nn.ModuleDict({"nest": Foo()})
                self.c = nn.ModuleList([Foo()])

            def forward(self, x):
                x = self.c[0](x)
                return self.b["nest"](self.a(x))

        mod = Mod()

        with ModuleTracker() as tracker:
            mod({"a": torch.randn(10, 10, requires_grad=True).clone()})[
                "a"
            ].sum().backward()
            mod({"a": torch.randn(10, 10, requires_grad=True).clone()})[
                "a"
            ].sum().backward()

        self.assertEqual(
            seen_fw,
            [
                ({"Global", "Mod", "Mod.c.0"}, False),
                ({"Global", "Mod", "Mod.a"}, False),
                ({"Global", "Mod", "Mod.b.nest"}, False),
                ({"Global", "Mod", "Mod.c.0"}, False),
                ({"Global", "Mod", "Mod.a"}, False),
                ({"Global", "Mod", "Mod.b.nest"}, False),
            ],
        )

        self.assertEqual(
            seen_bw,
            [
                ({"Global", "Mod", "Mod.b.nest"}, True),
                ({"Global", "Mod", "Mod.a"}, True),
                ({"Global", "Mod", "Mod.c.0"}, True),
                ({"Global", "Mod", "Mod.b.nest"}, True),
                ({"Global", "Mod", "Mod.a"}, True),
                ({"Global", "Mod", "Mod.c.0"}, True),
            ],
        )

    @skipIfTorchDynamo("unexplained 3.13+ recursion error")
    def test_confused_hierarchy(self):
        class MyMod(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = nn.Linear(2, 2)
                self.ran = False

            def forward(self, inp):
                if not self.ran:
                    self.ran = True
                    return self(inp)
                else:
                    self.ran = False
                    return self.inner(inp)

        mod = MyMod()
        inp = torch.rand(1, 2, requires_grad=True)

        # Should not fail
        with ModuleTracker():
            res = mod(inp)
            res.sum().backward()

        # Should not fail
        with ModuleTracker():
            res = checkpoint(lambda inp: mod(inp), inp)
            res.sum().backward()

    def test_bw_detection(self):
        mod = nn.Linear(2, 2)

        with ModuleTracker() as tracker:
            mod(torch.rand(2, requires_grad=True)).sum().backward()
            self.assertFalse(tracker.is_bw)
            self.assertEqual(tracker.parents, {"Global"})


if __name__ == "__main__":
    run_tests()
