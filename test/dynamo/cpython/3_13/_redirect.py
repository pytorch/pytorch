# Owner(s): ["module: dynamo"]

"""Shared `test.*` import redirect for the vendored CPython tests.

The vendored test files in this directory expect to import siblings as
`test.mapping_tests`, `test.support`, etc. — names that come from CPython's
own `test` stdlib package. PyTorch's repo also has a top-level `test/`
namespace package, so without intervention `test.mapping_tests` would
resolve there (where it doesn't exist).

This module installs a `MetaPathFinder` that redirects a known set of
`test.*` names to standalone modules sitting next to the test files. Each
vendored file just calls `install_redirect_finder()` once during its module
body instead of inlining the same finder class.

Notes:
- The finder removes itself from `sys.meta_path` on a failed redirect so a
  missing dep (e.g. CPython's `test.support` not being available in the
  current Python distribution) produces a clean `ImportError` instead of an
  exponential cascade of leaked finders that hangs the import system.
- Installation is idempotent — calling `install_redirect_finder()` multiple
  times (or from a module loaded *through* the redirect) is a no-op after
  the first.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys


_REDIRECT_IMPORTS = (
    "test.mapping_tests",
    "test.typinganndata",
    "test.test_grammar",
    "test.test_math",
    "test.test_iter",
    "test.typinganndata.ann_module",
)


class _RedirectImportFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _REDIRECT_IMPORTS:
            return None
        try:
            name = fullname.removeprefix("test.")
            r = importlib.import_module(name)
            sys.modules[fullname] = r
            return importlib.util.find_spec(name)
        except ImportError:
            # Drop ourselves so the failed redirect isn't retried via a
            # leaked finder; without this, every retry inserts another
            # finder and the import system loops forever.
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            return None


def install_redirect_finder() -> None:
    """Install the redirect finder on `sys.meta_path` if not already present."""
    if not any(isinstance(f, _RedirectImportFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _RedirectImportFinder())
