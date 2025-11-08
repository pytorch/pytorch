# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case


class MonkeypatchForwardTests(torch._dynamo.test_case.TestCase):
    def test_patch_forward_inside_compiled_region_eager_backend(self):
        """Test patching nn.Module.forward inside a compiled region."""
        
        class SimpleModule(torch.nn.Module):
            def forward(self, x):
                return x - 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn(mod, x, y):
            # Patch instance forward inside compiled region
            def patch(z):
                return z + y

            mod.forward = patch
            return mod(x)

        x = torch.ones(3)
        y = torch.ones(3)
        mod = SimpleModule()

        out = fn(mod, x, y)
        # Compiled path should respect the patched forward
        self.assertTrue(torch.allclose(out, x + y))
        # And the mutation should persist for eager
        self.assertTrue(torch.allclose(mod(x), x + y))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests
    run_tests()