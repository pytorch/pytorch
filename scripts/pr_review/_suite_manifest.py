#!/usr/bin/env python3
"""The suite's own completeness guard, imported by every test module.

`unittest discover` runs what it FINDS, so deleting a test module leaves a green
run and silently retires whatever that module asserted — up to and including the
workflow invariants the CI wiring exists to hold.

Each test module imports `TestTheSuiteIsWhole` from here, and `unittest` collects
imported `TestCase` classes, so every check below runs once per surviving module.
That redundancy is the mechanism: suppressing a check in one module leaves the
copies its siblings carry, and those report it.

**Sharing this is safe, and that is measured rather than assumed.** An earlier
revision copied the guard into each module, reasoning that a shared helper "could
itself be deleted". It cannot be deleted quietly: every test module imports it at
module scope, so removing it is an import error in all of them.

## What this does NOT cover, and why the line is here

**Drift and accident, not a hostile committer.** Everything below is checked from
inside the run it is checking, so anything that controls how the run is BUILT
gets to act first. Adversarial review found several: a `load_tests` hook that
deletes itself from `globals()` after suppressing its module, one gated on
`__package__` or `sys.argv` so it is inert whenever a checker looks, one that
mutates the shared loader's `testMethodPrefix` before the siblings are collected,
and one on the package `__init__.py`, which runs before any test exists. They are
all real. None of them is drift — each takes a deliberately self-concealing edit
by someone who can already change these files, and someone who can do that can
equally change the workflow that runs them. That is what branch protection and
required review are for, and chasing it with more in-suite checks buys the
appearance of a guarantee rather than the guarantee.

So the honest claim is narrow, and it holds only while every listed module
imports successfully and at least one of them survives: a module deleted, added,
renamed, un-wired, or emptied by ORDINARY means is caught, and caught by its
siblings rather than by itself.

The two cases that break that precondition are checked from OUTSIDE the subtree
instead, in the workflow's "Refuse a suppressed or emptied suite" step — a
package-level hook, and a suite with no modules left. Neither can be caught from
in here, since both stop the suite running at all; and the second is not even
reliably red on its own, because a zero-test run exits 5 on 3.12 and 0 on older
interpreters.

Deliberately not named `test_*.py`: it must not be discovered as a module of the
suite it describes.
"""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent

EXPECTED_MODULES = {
    "test_emit_row.py",
    "test_extract_verdict.py",
    "test_validate_findings.py",
    "test_workflow_contract.py",
}


def modules_from(path: Path) -> list:
    """The modules currently in `sys.modules` whose file is `path`.

    Reads what this run actually imported rather than re-executing the file: a
    `load_tests` hook can be gated on `__package__` or any other import-context
    value, and a privately loaded copy does not reproduce that context, so the
    hook would be invisible here while suppressing the real module. An earlier
    revision did re-execute, and that blindness was the defect.

    Empty means the file was never imported, or was dropped again — which
    `unittest` does to a module that raises `SkipTest` while importing.
    `loaded_modules` turns that into a failure rather than guessing.
    """
    target = path.resolve()
    found = []
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        try:
            if Path(f).resolve() == target:
                found.append(mod)
        except OSError:
            continue
    return found


def descendant_packages(root: Path):
    """Sub-packages and directory symlinks — the only routes below a flat suite.

    `discover` reaches a nested module only through a directory that is a
    package, so refusing packages at the top level refuses every deeper one too,
    without walking a `node_modules` or a large data tree to find out. It also
    covers a nested `__init__.py`, which discovery imports whatever the filename
    pattern is, and which the `test*.py` manifest check therefore cannot see.

    Plain data directories are untouched: `fixtures/` and `data/` carry no
    `__init__.py`, so discovery ignores them and so does this.
    """
    packages, links = [], []
    for p in sorted(root.iterdir()):
        if p.is_symlink() and p.is_dir():
            links.append(p.name)  # could point discovery at a tree outside this one
        elif p.is_dir() and (p / "__init__.py").is_file():
            packages.append(p.name)
    return packages, links


class TestTheSuiteIsWhole(unittest.TestCase):
    """Runs once per surviving test module, because each one imports it."""

    def loaded_modules(self):
        """Manifest module -> the module objects this run imported from it.

        A file that is present but NOT imported is a failure, never a skip.
        `raise unittest.SkipTest` at import time makes `unittest` drop the module
        from `sys.modules` and report a skip, so the run stays green with that
        module's assertions retired — measured, and an ordinary pattern rather
        than a hostile one. Skipping here would make the guard agree with it.

        All of them are collected before failing, so one unimported module does
        not hide the next.
        """
        loaded, unloaded = {}, []
        for name in sorted(EXPECTED_MODULES):
            path = HERE / name
            if not path.is_file():
                continue  # test_every_expected_module_is_present owns that report
            found = modules_from(path)
            if found:
                loaded[name] = found
            else:
                unloaded.append(name)
        self.assertEqual(
            unloaded,
            [],
            f"module(s) present but never imported by this run: {unloaded} — an import-time "
            "unittest.SkipTest retires a module's assertions while the run stays green. To "
            "run one module's tests, run the whole suite: "
            "python3 -m unittest discover -s scripts/pr_review -t .",
        )
        return loaded

    def test_every_expected_module_is_present(self):
        missing = sorted(n for n in EXPECTED_MODULES if not (HERE / n).is_file())
        self.assertEqual(
            missing, [], f"test module(s) deleted from the suite: {missing}"
        )

    def test_the_manifest_lists_every_discoverable_module(self):
        """A module absent from the manifest is an unguarded survivor.

        The guard is only wired into the modules named above, so a new
        `test_*.py` added without updating the manifest need not carry it — and
        once such a module exists, deleting all the named ones can be green
        again, since it keeps the run non-empty. Discovery reality, not the
        manifest, is what this compares against.
        """
        self.assertEqual(
            {p.name for p in HERE.glob("test*.py")},
            EXPECTED_MODULES,
            "a test module was added or removed without updating EXPECTED_MODULES",
        )

    def test_the_suite_has_no_descendant_packages(self):
        """The glob above sees this directory only; `discover` also walks packages.

        A test module in a subpackage is discovered and run while staying
        invisible to the manifest — the same unguarded survivor by another route
        — and a nested `__init__.py` is imported whatever it is called. The suite
        is flat, so refusing sub-packages outright closes both. `__pycache__` is
        not exempted: a directory of that name can hold a package as easily as a
        cache.
        """
        packages, links = descendant_packages(HERE)
        self.assertEqual(
            packages,
            [],
            f"sub-package(s) extend discovery past the manifest: {packages}",
        )
        self.assertEqual(
            links, [], f"symlinked director(ies) can extend discovery off-tree: {links}"
        )

    def test_every_manifest_module_is_wired_to_this_guard(self):
        """A listed module that stopped importing the guard is an unguarded survivor.

        The manifest check stops an UNLISTED module appearing; this stops a listed
        one drifting out of the wiring, or shadowing the shared class with a
        private lookalike that can disagree with the manifest its siblings hold.
        """
        for name, mods in self.loaded_modules().items():
            for mod in mods:
                with self.subTest(module=name, loaded_as=mod.__name__):
                    guard = getattr(mod, "TestTheSuiteIsWhole", None)
                    self.assertTrue(
                        isinstance(guard, type)
                        and issubclass(guard, unittest.TestCase),
                        f"{name} has no TestTheSuiteIsWhole TestCase bound at module scope, "
                        "so deleting the modules that do would stay green",
                    )
                    # Compared by defining FILE, resolved: imported bare and as a
                    # package submodule, this file yields two equivalent class
                    # objects and both are correct wiring, while a same-named
                    # file elsewhere on sys.path is not.
                    self.assertEqual(
                        Path(inspect.getfile(guard)).resolve(),
                        (HERE / "_suite_manifest.py").resolve(),
                        f"{name} binds its own guard instead of the shared one; a private "
                        "copy can drift from the manifest the others enforce",
                    )

    def test_no_module_hides_its_tests_behind_load_tests(self):
        """A `load_tests` hook drops a module's collected tests from the suite.

        It would retire that module's assertions without deleting the file, so
        the presence check above cannot see it. This lives in the SHARED guard on
        purpose: held by one module alone, a hook on that same file would remove
        this check together with everything it protects. Carried by every module,
        a surviving sibling still reports it — and it reads the module DISCOVERY
        imported, so a hook defined only under the real import context is still
        visible.
        """
        for name, mods in self.loaded_modules().items():
            for mod in mods:
                with self.subTest(module=name, loaded_as=mod.__name__):
                    self.assertFalse(
                        hasattr(mod, "load_tests"),
                        f"{name} defines load_tests, which can silently drop its tests "
                        "from the suite unittest builds",
                    )
