# Owner(s): ["module: unknown"]

from copy import copy

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils.module_tracker import ModuleTracker


class TestModuleTracker(TestCase):
    def test_module_hierarchy(self):
        seen_fw = []
        seen_bw = []

        class Foo(torch.nn.Module):
            def forward(self, x):
                x = x["a"].relu_()
                seen_fw.append((copy(tracker.parents), tracker.is_bw))
                x.register_hook(
                    lambda grad: seen_bw.append((copy(tracker.parents), tracker.is_bw))
                )
                return {"a": torch.mm(x, x)}

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = Foo()
                self.b = Foo()

            def forward(self, x):
                return self.b(self.a(x))

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
                ({"Global", "Mod", "Mod.a"}, False),
                ({"Global", "Mod", "Mod.b"}, False),
                ({"Global", "Mod", "Mod.a"}, False),
                ({"Global", "Mod", "Mod.b"}, False),
            ],
        )

        self.assertEqual(
            seen_bw,
            [
                ({"Global", "Mod", "Mod.b"}, True),
                ({"Global", "Mod", "Mod.a"}, True),
                ({"Global", "Mod", "Mod.b"}, True),
                ({"Global", "Mod", "Mod.a"}, True),
            ],
        )


if __name__ == "__main__":
    run_tests()
