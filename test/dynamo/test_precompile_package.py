# Owner(s): ["module: dynamo"]

import itertools
import os
import sys
import sysconfig
import types
from unittest import mock

import torch._dynamo.precompile_package as dynamo_package_lint
import torch._inductor.test_case
from torch.compiler._precompile_types import PrecompileSummary


class TestPrecompilePackage(torch._inductor.test_case.TestCase):
    def test_library_module_requires_the_name_to_resolve_to_the_stdlib(self):
        # The risky-drop waiver keys on the OWNER's module name, and a name is
        # not an identity: graphlib, queue, code and distutils are all stdlib
        # names a third party ships, and purelib NESTS inside stdlib (conda) or
        # platstdlib (venv), so a __file__ prefix check waived every shadow.
        stdlib_root = sysconfig.get_paths()["stdlib"]
        shadowed = types.ModuleType("graphlib")
        shadowed.__file__ = os.path.join(
            stdlib_root, "site-packages", "graphlib", "__init__.py"
        )
        unlocated = types.ModuleType("graphlib")  # no __file__, no __spec__

        for module in (shadowed, unlocated):
            with mock.patch.dict(sys.modules, {"graphlib": module}):
                self.assertFalse(dynamo_package_lint._is_library_module("graphlib"))

        # Real stdlib and real torch, including the shapes with no file at all
        # (torch._C._nn owns F.gelu; pyexpat.errors is a stdlib submodule with
        # no location evidence of its own), must keep their waiver, so
        # descendants only have to not be located somewhere else.
        import xml.parsers.expat  # noqa: F401

        for name in (
            "torch",
            "torch._C",
            "torch._C._nn",
            "torch.ops",
            "os.path",
            "collections.abc",
            "sys",
            "zipimport",
            "pyexpat.errors",
            "xml.parsers.expat.model",
        ):
            self.assertTrue(
                dynamo_package_lint._is_library_module(name), f"{name} lost its waiver"
            )
        self.assertFalse(dynamo_package_lint._is_library_module("numpy"))
        self.assertFalse(dynamo_package_lint._is_library_module(None))

    def test_code_fingerprint_recurses_into_container_and_nested_consts(self):
        # _code_fingerprint names a callable by its body so an ACT2FN-style table
        # can be told apart. Two lambdas can differ ONLY inside a constant the
        # outer co_code does not distinguish: a tuple, a frozenset, or a nested
        # code object. Filtering those out whole -- rather than recursing -- gives
        # both the same digest, _object_identity names them identically, and the
        # guard that split the two compilations is reported as an invariant of
        # each.
        from torch._dynamo.precompile_package import _code_fingerprint

        pairs = {
            "tuple const": (lambda x: x * (1, 2), lambda x: x * (1, 3)),
            "frozenset const": (lambda x: x in {1, 2}, lambda x: x in {1, 3}),
            # Not called: what matters is the nested code object in co_consts.
            "nested code": (lambda x: (lambda y: y + 1), lambda x: (lambda y: y + 2)),
        }
        for label, (left, right) in pairs.items():
            self.assertEqual(
                left.__code__.co_code,
                right.__code__.co_code,
                f"{label}: the pair must differ only in co_consts",
            )
            self.assertNotEqual(
                _code_fingerprint(left.__code__),
                _code_fingerprint(right.__code__),
                f"{label}: two different bodies share a fingerprint",
            )

    def test_guard_policy_classification_is_total(self):
        # A guard type in no set is KEPT, so a drop policy can only ever
        # drop what _INVARIANT_DROPPABLE_GUARD_TYPES names. This test is
        # what makes the never-drop claim enforceable:
        # a guard type added to GuardBuilder fails here until someone triages
        # it into exactly one of the four sets.
        from torch._dynamo.guards import GuardBuilder
        from torch._dynamo.precompile_package import (
            _INVARIANT_DROPPABLE_GUARD_TYPES,
            _NOOP_GUARD_TYPES,
            _SHAPE_BEARING_GUARD_TYPES,
            _UNMODELLED_GUARD_TYPES,
        )

        guard_types = {
            name
            for name, value in vars(GuardBuilder).items()
            if name.isupper() and callable(value)
        }
        self.assertGreater(len(guard_types), 40)  # the enumeration itself works
        sets = {
            "_SHAPE_BEARING_GUARD_TYPES": _SHAPE_BEARING_GUARD_TYPES,
            "_UNMODELLED_GUARD_TYPES": _UNMODELLED_GUARD_TYPES,
            "_INVARIANT_DROPPABLE_GUARD_TYPES": _INVARIANT_DROPPABLE_GUARD_TYPES,
            "_NOOP_GUARD_TYPES": _NOOP_GUARD_TYPES,
        }
        classified: frozenset[str] = frozenset().union(*sets.values())
        self.assertEqual(
            sorted(guard_types - classified),
            [],
            "unclassified GuardBuilder guard type(s): add each to exactly one "
            "policy set in torch/_dynamo/precompile_package.py (KEPT until then)",
        )
        self.assertEqual(
            sorted(classified - guard_types),
            [],
            "phantom entries: no GuardBuilder method by these names",
        )
        for (a_name, a), (b_name, b) in itertools.combinations(sets.items(), 2):
            self.assertEqual(sorted(a & b), [], f"{a_name} overlaps {b_name}")

    def test_summary_is_incomplete_without_a_backend_graph(self):
        def summary(**kw):
            base = dict(
                frames=1,
                resume_functions=0,
                guarded_codes=1,
                backend_graphs=1,
                bypassed=(),
            )
            base.update(kw)
            return PrecompileSummary(**base)

        self.assertTrue(summary().complete)
        self.assertFalse(summary(backend_graphs=0).complete)
        self.assertFalse(summary(guarded_codes=0).complete)
        self.assertFalse(summary(capture_errors=("boom",)).complete)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
