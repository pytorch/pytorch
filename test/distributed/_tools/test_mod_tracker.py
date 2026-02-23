# Owner(s): ["oncall: distributed"]

from copy import copy

import torch
from torch.distributed._tools.mod_tracker import ModTracker
from torch.testing._internal.common_utils import run_tests, TestCase, xfailIfTorchDynamo
from torch.utils.checkpoint import checkpoint


class TestModTracker(TestCase):
    # "https://github.com/pytorch/pytorch/issues/127112
    @xfailIfTorchDynamo
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
            def __init__(self) -> None:
                super().__init__()
                self.a = Foo()
                self.b = torch.nn.ModuleDict({"nest": Foo()})
                self.c = torch.nn.ModuleList([Foo()])

            def forward(self, x):
                x = self.c[0](x)
                return self.b["nest"](self.a(x))

        mod = Mod()

        with ModTracker() as tracker:
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

    def test_bw_detection(self):
        mod = torch.nn.Linear(2, 2)

        with ModTracker() as tracker:
            mod(torch.rand(2, requires_grad=True)).sum().backward()
            self.assertFalse(tracker.is_bw)
            self.assertEqual(tracker.parents, {"Global"})

    @xfailIfTorchDynamo
    def test_user_hooks(self):
        class Bar(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = torch.nn.Linear(10, 10)

            def forward(self, x):
                return self.foo(x).relu_()

        mt = ModTracker()
        test_op = []

        def hook(mod, hook_name):
            mfqn = mt.get_known_fqn(mod) if mod is not None else None
            test_op.append((hook_name, mfqn, mfqn in mt.parents, mt.is_bw))

        mod = Bar()

        mt.register_user_hooks(
            lambda m, inp: hook(m, "pre_fw"),
            lambda m, inp, op: hook(m, "post_fw"),
            lambda m, gop: hook(m, "pre_bw"),
            lambda m, ginp: hook(m, "post_bw"),
        )
        with mt:
            mod(torch.rand(10, 10, requires_grad=True)).sum().backward()
        expected_op = [
            ("pre_fw", "Bar", True, False),
            ("pre_fw", "Bar.foo", True, False),
            ("post_fw", "Bar.foo", True, False),
            ("post_fw", "Bar", True, False),
            ("pre_bw", "Bar", True, True),
            ("pre_bw", "Bar.foo", True, True),
            ("post_bw", "Bar", True, True),
            ("post_bw", "Bar.foo", True, True),
        ]
        self.assertEqual(test_op, expected_op)

        with self.assertRaises(AssertionError):
            mt.register_user_hooks(lambda x, y: x, None, None, None)

        test_op.clear()
        with mt:
            loss = mod(torch.rand(10, 10, requires_grad=True)).sum()
            del mod
            loss.backward()
        expected_op = [
            ("pre_fw", "Bar", True, False),
            ("pre_fw", "Bar.foo", True, False),
            ("post_fw", "Bar.foo", True, False),
            ("post_fw", "Bar", True, False),
            ("pre_bw", None, False, True),
            ("pre_bw", None, False, True),
            ("post_bw", None, False, True),
            ("post_bw", None, False, True),
        ]
        self.assertEqual(test_op, expected_op)

    @xfailIfTorchDynamo
    def test_ac(self):
        class Foo(torch.nn.Module):
            def __init__(self, n_layers: int, dim: int, use_ac: bool = False):
                super().__init__()
                self.linears = torch.nn.ModuleList()
                self.use_ac = use_ac
                for _ in range(n_layers):
                    self.linears.append(torch.nn.Linear(dim, dim))

            def forward(self, x):
                for i, block in enumerate(self.linears):
                    if i >= 1 and self.use_ac:
                        x = checkpoint(
                            block, x, preserve_rng_state=True, use_reentrant=False
                        )
                    else:
                        x = block(x)
                    assert x is not None
                    x = torch.nn.functional.relu(x)
                return x

        bsz = 2
        dim = 8
        n_layers = 2
        test_op = []

        def hook(mod, mt, hook_name):
            mfqn = mt.get_known_fqn(mod) if mod is not None else None
            test_op.append((hook_name, mfqn, mfqn in mt.parents, mt.is_bw))

        mt = ModTracker()
        mt.register_user_hooks(
            lambda m, i: hook(m, mt, "pre_fw"),
            lambda m, i, o: hook(m, mt, "post_fw"),
            lambda m, go: hook(m, mt, "pre_bw"),
            lambda m, gi: hook(m, mt, "post_bw"),
        )
        model = Foo(n_layers, dim, True)
        x = torch.randn(bsz, dim)
        with mt:
            model(x).sum().backward()

        expected_op = [
            ("pre_fw", "Foo", True, False),
            ("pre_fw", "Foo.linears.0", True, False),
            ("post_fw", "Foo.linears.0", True, False),
            ("pre_fw", "Foo.linears.1", True, False),
            ("post_fw", "Foo.linears.1", True, False),
            ("post_fw", "Foo", True, False),
            ("pre_bw", "Foo", True, True),
            ("pre_bw", "Foo.linears.1", True, True),
            ("pre_fw", "Foo.linears.1", True, True),
            ("post_fw", "Foo.linears.1", True, True),
            ("post_bw", "Foo.linears.1", True, True),
            ("pre_bw", "Foo.linears.0", True, True),
            ("post_bw", "Foo.linears.0", True, True),
            ("post_bw", "Foo", True, True),
        ]
        self.assertEqual(test_op, expected_op)


class TestCompileSafeHooks(TestCase):
    """Tests for ModTracker.register_compile_safe_hooks."""

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()
        super().tearDown()

    def _make_model(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(4, 8)
                self.act = torch.nn.ReLU()
                self.linear2 = torch.nn.Linear(8, 2)

            def forward(self, x):
                return self.linear2(self.act(self.linear1(x)))

        return TinyModel()

    def test_all_hooks_eager(self):
        model = self._make_model()
        x = torch.randn(3, 4, requires_grad=True)
        log = []

        with ModTracker() as tracker:
            tracker.register_compile_safe_hooks(
                model,
                pre_fw_hook=lambda fqn, t: log.append(("pre_fw", fqn)),
                post_fw_hook=lambda fqn, inp, out: log.append(("post_fw", fqn)),
                pre_bw_hook=lambda fqn, t: log.append(("pre_bw", fqn)),
                post_bw_hook=lambda fqn, t: log.append(("post_bw", fqn)),
            )
            model(x).sum().backward()

        fw = [(k, f) for k, f in log if k.endswith("fw")]
        self.assertEqual(
            fw,
            [
                ("pre_fw", "linear1"),
                ("post_fw", "linear1"),
                ("pre_fw", "act"),
                ("post_fw", "act"),
                ("pre_fw", "linear2"),
                ("post_fw", "linear2"),
            ],
        )
        bw = [(k, f) for k, f in log if k.endswith("bw")]
        self.assertEqual(
            bw,
            [
                ("pre_bw", "linear2"),
                ("post_bw", "linear2"),
                ("pre_bw", "act"),
                ("post_bw", "act"),
                ("pre_bw", "linear1"),
                ("post_bw", "linear1"),
            ],
        )

    def test_all_hooks_compiled(self):
        model = self._make_model()
        x = torch.randn(3, 4, requires_grad=True)
        log = []

        with ModTracker() as tracker:
            tracker.register_compile_safe_hooks(
                model,
                pre_fw_hook=lambda fqn, t: log.append(("pre_fw", fqn)),
                post_fw_hook=lambda fqn, inp, out: log.append(("post_fw", fqn)),
                pre_bw_hook=lambda fqn, t: log.append(("pre_bw", fqn)),
                post_bw_hook=lambda fqn, t: log.append(("post_bw", fqn)),
            )
            compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
            compiled(x).sum().backward()

        fw_count = sum(1 for k, _ in log if k.endswith("fw"))
        bw_count = sum(1 for k, _ in log if k.endswith("bw"))
        self.assertEqual(fw_count, 6)
        self.assertEqual(bw_count, 6)

    def test_root_module_skipped(self):
        model = self._make_model()
        x = torch.randn(3, 4)
        fqns = set()

        with ModTracker() as tracker:
            tracker.register_compile_safe_hooks(
                model,
                post_fw_hook=lambda fqn, inp, out: fqns.add(fqn),
            )
            torch.compile(model, backend="aot_eager", fullgraph=True)(x)

        self.assertNotIn("", fqns)
        self.assertIn("linear1", fqns)
        self.assertIn("linear2", fqns)

    def test_gradient_correctness(self):
        model = self._make_model()
        x1 = torch.randn(3, 4, requires_grad=True)
        model(x1).sum().backward()
        grad_baseline = x1.grad.clone()

        model.zero_grad()
        x2 = x1.detach().clone().requires_grad_(True)

        with ModTracker() as tracker:
            tracker.register_compile_safe_hooks(
                model,
                pre_fw_hook=lambda fqn, t: None,
                post_fw_hook=lambda fqn, inp, out: None,
                pre_bw_hook=lambda fqn, t: None,
                post_bw_hook=lambda fqn, t: None,
            )
            compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
            compiled(x2).sum().backward()

        self.assertTrue(torch.allclose(grad_baseline, x2.grad))

    def test_post_fw_receives_inputs_and_outputs(self):
        model = self._make_model()
        x = torch.randn(3, 4, requires_grad=True)
        shapes = {}

        def record_post_fw(fqn, inputs, outputs):
            shapes[fqn] = (
                [t.shape for t in inputs],
                [t.shape for t in outputs],
            )

        with ModTracker() as tracker:
            tracker.register_compile_safe_hooks(
                model,
                post_fw_hook=record_post_fw,
            )
            compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
            compiled(x).sum().backward()

        # linear1: (3,4) -> (3,8)
        self.assertEqual(shapes["linear1"][0], [torch.Size([3, 4])])
        self.assertEqual(shapes["linear1"][1], [torch.Size([3, 8])])
        # linear2: (3,8) -> (3,2)
        self.assertEqual(shapes["linear2"][0], [torch.Size([3, 8])])
        self.assertEqual(shapes["linear2"][1], [torch.Size([3, 2])])

    def test_module_filter(self):
        model = self._make_model()
        x = torch.randn(3, 4)
        fqns = []

        with ModTracker() as tracker:
            tracker.register_compile_safe_hooks(
                model,
                post_fw_hook=lambda fqn, inp, out: fqns.append(fqn),
                module_filter=lambda fqn, mod: isinstance(mod, torch.nn.Linear),
            )
            torch.compile(model, backend="aot_eager", fullgraph=True)(x)

        self.assertEqual(set(fqns), {"linear1", "linear2"})


if __name__ == "__main__":
    run_tests()
