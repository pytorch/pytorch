# Owner(s): ["module: dsl-native-ops"]
#
# Drift guard for overrides registered on namespaces other than `aten`.
#
# An `aten` op exists by construction: libtorch defines it before any Python
# runs, so a registration naming a symbol that does not exist dies on every
# `import torch`. An op defined by executing Python -- `torch_nn::` ops, for
# one -- exists only after its defining module has run, which turns the
# binding between a registration's string and its referent into a runtime
# property. Worse, the failure surfaces only where the DSL is installed, so
# ordinary CPU and GPU CI would never see it.
#
# This restores the build-grade invariant: a declaring module names its
# namespace, the module that defines its ops, and the overrides it registers,
# and every symbol must resolve. `import torch` has already imported these
# modules and their DSL imports are lazy, so this test needs neither a GPU
# nor any DSL runtime and runs on every shard.

import importlib
import sys

import torch  # noqa: F401  # imports the registrars that carry the declarations
from torch._native.registry import _resolve_overload
from torch.testing._internal.common_utils import run_tests, TestCase


def _declaring_modules():
    for name, mod in sorted(sys.modules.items()):
        if name.startswith("torch._native.ops.") and getattr(mod, "_NAMESPACE", None):
            yield name, mod


class TestOverrideDeclarations(TestCase):
    def test_declared_overrides_resolve(self):
        declaring = list(_declaring_modules())
        self.assertTrue(
            declaring,
            "no module under torch._native.ops declares `_NAMESPACE`; the first "
            "non-aten registrar must declare one so drift is caught here",
        )
        for name, mod in declaring:
            importlib.import_module(mod._DEFINING_MODULE)
            for op_symbol, *_ in mod._OVERRIDES:
                op = _resolve_overload(mod._NAMESPACE, op_symbol)
                self.assertIsNotNone(
                    op,
                    f"{name} registers {mod._NAMESPACE}::{op_symbol}, which does "
                    f"not resolve after importing {mod._DEFINING_MODULE}",
                )

    def test_declared_variants_are_callable_and_defaults_exist(self):
        """A module offering several implementations of an op declares them in
        `_VARIANTS`; selecting one by name is a public surface, so the table
        has to agree with `_OVERRIDES` and with its own defaults. DSL-free,
        like the symbol check above."""
        for name, mod in _declaring_modules():
            table = getattr(mod, "_VARIANTS", None)
            if table is None:
                continue
            defaults = mod._DEFAULT_VARIANTS
            declared_ops = {op_symbol for op_symbol, *_ in mod._OVERRIDES}
            self.assertEqual(
                set(table),
                declared_ops,
                f"{name}: _VARIANTS covers {sorted(table)} but the module "
                f"registers {sorted(declared_ops)}",
            )
            self.assertEqual(set(defaults), declared_ops, f"{name}: defaults")
            for op_symbol, impls in table.items():
                for variant_name, declared in impls.items():
                    # Each variant declares the eligibility `cond` evaluates
                    # for it and the implementation that then runs.
                    for part in ("eligible", "impl"):
                        self.assertTrue(
                            callable(getattr(declared, part, None)),
                            f"{name}: {op_symbol} variant {variant_name!r} has "
                            f"no callable {part}",
                        )
                self.assertIn(
                    defaults[op_symbol],
                    impls,
                    f"{name}: default variant {defaults[op_symbol]!r} for "
                    f"{op_symbol} is not declared; declared {sorted(impls)}",
                )


if __name__ == "__main__":
    run_tests()
