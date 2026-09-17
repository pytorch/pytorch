# Owner(s): ["module: dynamo"]

import os
import site
import sys
import sysconfig
import types
from unittest import mock

import torch
import torch._dynamo.precompile_package as precompile_package
import torch._inductor.test_case
from torch._dynamo.guards import CheckFunctionManager, GuardBuilder, strip_local_scope
from torch._dynamo.source import GlobalSource
from torch._dynamo.types import GuardFilterEntry
from torch._guards import Guard


def _entry(source, value, guard_type="ID_MATCH", derived=()):
    guard = Guard(source, getattr(GuardBuilder, guard_type))
    return GuardFilterEntry(
        name=strip_local_scope(source.name),
        has_value=True,
        value=value,
        guard_type=guard_type,
        derived_guard_types=tuple(derived),
        is_global=isinstance(source, GlobalSource),
        orig_guard=guard,
    )


class TestPrecompilePackage(torch._inductor.test_case.TestCase):
    def test_default_guard_filter_drops_the_unserializable_types(self):
        unsupported = CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
        g = GlobalSource("g")
        refused = [_entry(g, None, guard_type=t) for t in unsupported]
        kept = precompile_package.default_guard_filter_fn(refused)
        self.assertEqual([t for t, keep in zip(unsupported, kept) if keep], [])
        # A CONSTANT_MATCH on a code object runs through ID_MATCH; the
        # serializer refuses the derived type, so the filter drops it too.
        derived = _entry(g, None, "CONSTANT_MATCH", derived=("ID_MATCH",))
        self.assertEqual(precompile_package.default_guard_filter_fn([derived]), [False])

    def test_default_guard_filter_keeps_what_the_serializer_accepts(self):
        class Local:
            pass

        # The one divergence: TYPE_MATCH marks a class whose __qualname__ is not
        # its __name__ here and serialize_guards refuses it through this
        # attribute. The filter keeps it so the refusal stays loud rather than
        # shipping an artifact that never checks the type.
        g = GlobalSource("g")
        local_type = _entry(g, None, "TYPE_MATCH")
        local_type.orig_guard._unserializable = Local
        entries = [
            _entry(g, None, "TENSOR_MATCH"),
            _entry(g, None, "TYPE_MATCH"),
            local_type,
            # An id_match_unchecked on a builtin records ID_MATCH as its derived
            # type; serialize_guards takes its TYPE_MATCH/BUILTIN_MATCH branch
            # first and never reaches the derived-type refusal, so neither does
            # the filter.
            _entry(g, None, "BUILTIN_MATCH", derived=("ID_MATCH",)),
        ]
        keep = precompile_package.default_guard_filter_fn(entries)
        self.assertEqual(keep, [True] * 4)

    def test_roots_locate_the_stdlib_install_and_torch_dirs(self):
        stdlib = precompile_package._stdlib_roots()
        install = precompile_package._install_roots()
        torch_roots = precompile_package._torch_roots()
        self.assertTrue(stdlib and install and torch_roots)
        norm, within = precompile_package._norm, precompile_package._within
        # The sets nest one way only: purelib sits inside a stdlib root (conda)
        # or platstdlib (venv) and must survive the exclusion, while on Windows
        # getsitepackages() names the prefix the stdlib sits under, which must
        # not; the install-root-wins rule only works if no stdlib root is under
        # an install root.
        self.assertIn(norm(sysconfig.get_paths()["purelib"]), install)
        for root in stdlib:
            self.assertFalse(within(root, install), root)
        self.assertTrue(within(norm(os.__file__), stdlib))
        self.assertIn(norm(os.path.dirname(torch.__file__)), torch_roots)
        root = os.path.join(os.sep, "a", "b")
        self.assertTrue(within(root, (root,)))
        self.assertTrue(within(os.path.join(root, "c"), (root,)))
        self.assertFalse(within(root + "c", (root,)))

    def test_install_roots_drop_a_directory_the_stdlib_sits_under(self):
        # On Windows getsitepackages() lists the bare prefix; replay that shape
        # here so the exclusion is pinned on every platform, not just there.
        listed = [sys.prefix, *site.getsitepackages()]
        self.addCleanup(precompile_package._install_roots.cache_clear)
        with mock.patch.object(site, "getsitepackages", return_value=listed):
            precompile_package._install_roots.cache_clear()
            install = precompile_package._install_roots()
        norm, within = precompile_package._norm, precompile_package._within
        self.assertNotIn(norm(sys.prefix), install)
        for root in precompile_package._stdlib_roots():
            self.assertFalse(within(root, install), root)
        self.assertIn(norm(sysconfig.get_paths()["purelib"]), install)

    def test_torch_roots_trust_torch_path_only_when_this_file_is_in_it(self):
        torch_roots, norm = precompile_package._torch_roots, precompile_package._norm
        own = norm(os.path.dirname(torch.__file__))
        bogus = os.path.join(os.sep, "elsewhere", "torch")
        stub = types.SimpleNamespace(__path__=[bogus])
        self.addCleanup(torch_roots.cache_clear)
        with mock.patch.dict(sys.modules, {"torch": stub}):
            # A sys.modules['torch'] that is not us cannot nominate its own
            # roots until its __path__ lists the directory this file runs from.
            torch_roots.cache_clear()
            self.assertEqual(torch_roots(), (own,))
            stub.__path__.append(os.path.dirname(torch.__file__))
            torch_roots.cache_clear()
            self.assertEqual(set(torch_roots()), {own, norm(bogus)})
        with mock.patch.object(precompile_package, "__file__", None):
            torch_roots.cache_clear()
            self.assertEqual(torch_roots(), ())  # frozen: no directory to anchor to


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
