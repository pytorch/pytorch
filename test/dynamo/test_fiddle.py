# Owner(s): ["module: dynamo"]

import unittest

import torch
import torch._dynamo.test_case
import torch._dynamo.testing


try:
    import fiddle as fdl

    HAS_FIDDLE = True
except ImportError:
    HAS_FIDDLE = False


def linear(x, weight, bias, scale=1.0):
    return (x @ weight + bias) * scale


class MLP:
    def __init__(self, hidden_size, num_layers, dropout=0.0):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout


class Nested:
    def __init__(self, inner, num_heads):
        self.inner = inner
        self.num_heads = num_heads


@unittest.skipIf(not HAS_FIDDLE, "requires fiddle")
class FiddleTests(torch._dynamo.test_case.TestCase):
    def test_config_attr_access(self):
        cfg = fdl.Config(MLP, hidden_size=256, num_layers=4)

        def fn(x, cfg):
            return x * cfg.hidden_size + cfg.num_layers

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(10)
        self.assertEqual(opt_fn(x, cfg), fn(x, cfg))

    def test_nested_config_attr_access(self):
        inner_cfg = fdl.Config(MLP, hidden_size=128, num_layers=2)
        outer_cfg = fdl.Config(Nested, inner=inner_cfg, num_heads=8)

        def fn(x, cfg):
            return x * cfg.inner.hidden_size + cfg.num_heads

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(10)
        self.assertEqual(opt_fn(x, outer_cfg), fn(x, outer_cfg))

    def test_config_attr_branch(self):
        cfg = fdl.Config(MLP, hidden_size=256, num_layers=4, dropout=0.0)

        def fn(x, cfg):
            if cfg.dropout > 0:
                return x * (1 - cfg.dropout)
            return x * cfg.hidden_size

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(10)
        self.assertEqual(opt_fn(x, cfg), fn(x, cfg))

    def test_config_fn_with_defaults(self):
        cfg = fdl.Config(linear, weight=torch.randn(4, 4), bias=torch.randn(4))

        def fn(x, cfg):
            return x * cfg.scale

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(opt_fn(x, cfg), fn(x, cfg))

    def test_config_multiple_attrs(self):
        """Access multiple attributes from the same config."""
        cfg = fdl.Config(MLP, hidden_size=128, num_layers=3, dropout=0.5)

        def fn(x, cfg):
            return x * cfg.hidden_size * cfg.num_layers * cfg.dropout

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(10)
        self.assertEqual(opt_fn(x, cfg), fn(x, cfg))

    def test_config_mutation_and_access(self):
        """Mutating a config attr causes a graph break; the subsequent access uses the new value."""
        cfg = fdl.Config(MLP, hidden_size=256, num_layers=4)

        def fn(x, cfg):
            cfg.hidden_size = 128
            return x * cfg.hidden_size

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        x = torch.randn(10)
        self.assertEqual(opt_fn(x, cfg), x * 128)
        # Mutation causes a graph break, then x * cfg.hidden_size compiles
        self.assertEqual(cnt.frame_count, 1)

    def test_config_subclass_getattr(self):
        """Config subclass that overrides __getattr__ with super() call."""

        class CustomConfig(fdl.Config):
            def __getattr__(self, name):
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    raise AttributeError(
                        f"Field '{name}' not found."
                    ) from None

        cfg = CustomConfig(MLP, hidden_size=256, num_layers=4)

        def fn(x, cfg):
            return x * cfg.hidden_size + cfg.num_layers

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(10)
        self.assertEqual(opt_fn(x, cfg), fn(x, cfg))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
