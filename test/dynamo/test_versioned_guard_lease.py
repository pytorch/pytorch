# Owner(s): ["module: dynamo"]

import concurrent.futures

import torch
from torch._dynamo.testing import CompileCounter
from torch.testing._internal.common_utils import run_tests, TestCase


class VersionedGuardLeaseTests(TestCase):
    @torch._dynamo.config.patch(use_versioned_guard_lease=True)
    def test_stable_module_installs_lease(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weight", torch.randn(4, 4))
                self.scale = 2

            def forward(self, x):
                return x @ self.weight + self.scale

        mod = Mod()
        counter = CompileCounter()
        compiled = torch.compile(mod, backend=counter, fullgraph=True)
        x = torch.randn(4, 4)

        self.assertEqual(compiled(x), mod(x))
        self.assertEqual(compiled(x), mod(x))
        self.assertEqual(counter.frame_count, 1)

        entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(Mod.forward)
        self.assertEqual(len(entries), 1)
        self.assertIsNotNone(entries[0].guard_manager.lease_guard_root)
        self.assertGreaterEqual(entries[0].lease_guard_hits, 1)
        self.assertEqual(entries[0].lease_guard_misses, 0)

    @torch._dynamo.config.patch(use_versioned_guard_lease=True)
    def test_module_mutation_falls_back_to_full_guards(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = 2

            def forward(self, x):
                return x + self.scale

        mod = Mod()
        counter = CompileCounter()
        compiled = torch.compile(mod, backend=counter, fullgraph=True)
        x = torch.randn(4)

        self.assertEqual(compiled(x), x + 2)
        self.assertEqual(counter.frame_count, 1)

        entry = torch._dynamo.eval_frame._debug_get_cache_entry_list(Mod.forward)[0]
        self.assertEqual(entry.lease_guard_misses, 0)

        mod.scale = 3
        self.assertEqual(compiled(x), x + 3)
        self.assertEqual(counter.frame_count, 2)
        self.assertEqual(entry.lease_guard_misses, 1)

    @torch._dynamo.config.patch(use_versioned_guard_lease=True)
    def test_different_module_cannot_reuse_lease(self):
        class Mod(torch.nn.Module):
            def __init__(self, scale):
                super().__init__()
                self.scale = scale

            def forward(self, x):
                return x + self.scale

        def fn(mod, x):
            return mod(x)

        first = Mod(2)
        second = Mod(5)
        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.randn(4)

        self.assertEqual(compiled(first, x), x + 2)
        self.assertEqual(compiled(first, x), x + 2)
        first_entry = torch._dynamo.eval_frame._debug_get_cache_entry_list(fn)[0]
        self.assertEqual(compiled(second, x), x + 5)
        self.assertEqual(first_entry.lease_guard_misses, 1)

    @torch._dynamo.config.patch(use_versioned_guard_lease=True)
    def test_non_module_guard_remains_residual(self):
        class Mod(torch.nn.Module):
            def forward(self, x):
                return x.sin()

        mod = Mod()
        scale = 2

        def fn(x):
            return mod(x) + scale

        counter = CompileCounter()
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        x = torch.randn(4)

        self.assertEqual(compiled(x), mod(x) + 2)
        self.assertEqual(counter.frame_count, 1)

        scale = 3
        self.assertEqual(compiled(x), mod(x) + 3)
        self.assertEqual(counter.frame_count, 2)

    @torch._dynamo.config.patch(use_versioned_guard_lease=True)
    def test_parameter_replacement_expires_lease(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weight", torch.randn(4, 4))

            def forward(self, x):
                return x @ self.weight

        mod = Mod()
        counter = CompileCounter()
        compiled = torch.compile(mod, backend=counter, fullgraph=True)
        x = torch.randn(4, 4)

        self.assertEqual(compiled(x), mod(x))
        entry = torch._dynamo.eval_frame._debug_get_cache_entry_list(Mod.forward)[0]
        mod.weight = torch.randn(4, 4)
        self.assertEqual(compiled(x), mod(x))
        self.assertEqual(entry.lease_guard_misses, 1)

    @torch._dynamo.config.patch(use_versioned_guard_lease=True)
    def test_tensor_metadata_remains_residual(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weight", torch.randn(4, 4))

            def forward(self, x):
                return x @ self.weight

        mod = Mod()
        counter = CompileCounter()
        compiled = torch.compile(mod, backend=counter, fullgraph=True)
        x = torch.randn(4, 4)

        self.assertEqual(compiled(x), mod(x))
        entry = torch._dynamo.eval_frame._debug_get_cache_entry_list(Mod.forward)[0]
        mod.weight.t_()
        self.assertEqual(compiled(x), mod(x))
        self.assertEqual(entry.lease_guard_misses, 1)

    @torch._dynamo.config.patch(
        use_versioned_guard_lease=True,
        skip_no_tensor_aliasing_guards_on_parameters=False,
    )
    def test_parameter_input_aliasing_expires_lease(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.randn(4), requires_grad=False
                )

            def forward(self, x):
                return x + self.weight

        mod = Mod()
        counter = CompileCounter()
        compiled = torch.compile(mod, backend=counter, fullgraph=True)
        x = torch.randn(4)

        self.assertEqual(compiled(x), mod(x))
        entry = torch._dynamo.eval_frame._debug_get_cache_entry_list(Mod.forward)[0]
        self.assertEqual(compiled(mod.weight), mod(mod.weight))
        self.assertEqual(entry.lease_guard_misses, 1)

    @torch._dynamo.config.patch(use_versioned_guard_lease=True)
    def test_no_safe_module_root_does_not_install_lease(self):
        def fn(x):
            return x.sin()

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(compiled(x), fn(x))

        entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(fn)
        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0].guard_manager.lease_guard_root)

    @torch._dynamo.config.patch(use_versioned_guard_lease=True)
    def test_stable_module_from_multiple_threads(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = 2

            def forward(self, x):
                return x + self.scale

        mod = Mod()
        counter = CompileCounter()
        compiled = torch.compile(mod, backend=counter, fullgraph=True)
        x = torch.randn(4)
        expected = mod(x)
        self.assertEqual(compiled(x), expected)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            outputs = list(executor.map(lambda _: compiled(x), range(20)))

        for output in outputs:
            self.assertEqual(output, expected)
        self.assertEqual(counter.frame_count, 1)

    @torch._dynamo.config.patch(use_versioned_guard_lease=False)
    def test_disabled_does_not_install_lease(self):
        class Mod(torch.nn.Module):
            def forward(self, x):
                return x.sin()

        mod = Mod()
        compiled = torch.compile(mod, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(compiled(x), mod(x))

        entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(Mod.forward)
        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0].guard_manager.lease_guard_root)


if __name__ == "__main__":
    run_tests()
