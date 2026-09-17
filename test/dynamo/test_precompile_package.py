# Owner(s): ["module: dynamo"]

import functools
import importlib.machinery
import os
import site
import sys
import sysconfig
import traceback
import types
import xml.parsers.expat  # noqa: F401  # imported for the _LIBRARY_NAMES rows
from unittest import mock

import torch
import torch._dynamo.precompile_package as precompile_package
import torch._inductor.test_case
import torch.nn.functional as F
from torch._dynamo.guards import CheckFunctionManager, GuardBuilder, strip_local_scope
from torch._dynamo.source import (
    AttrSource,
    DictGetItemSource,
    GetItemSource,
    GlobalSource,
    LocalSource,
)
from torch._dynamo.types import GuardFilterEntry
from torch._guards import Guard
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


def _user_op(x):
    return x + 1


_BUILTINS_DICT = GlobalSource("__builtins_dict___0")
_HERE = traceback.StackSummary.from_list([(__file__, 1, "forward", "")])
_ELSEWHERE = traceback.StackSummary.from_list([(F.__file__, 1, "forward", "")])


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


# Names torch or the stdlib own, including the submodules with no location
# evidence of their own: torch.ops is a ModuleType subclass whose __file__ is a
# class attribute (its module dict has none), pyexpat.errors has no __file__ at
# all. A top-level name keeps its waiver only while it is in sys.modules, which
# is what the header's `import xml.parsers.expat` is for.
_LIBRARY_NAMES = (
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

    def test_library_module_requires_the_name_to_resolve_to_the_library(self):
        # The risky-drop waiver keys on the OWNER's module name, and a name is
        # not an identity: graphlib, queue, code and distutils are all stdlib
        # names a third party ships, and purelib NESTS inside stdlib (conda) or
        # platstdlib (venv), so a __file__ prefix check waived every shadow.
        self.addCleanup(precompile_package._classify_file.cache_clear)
        stdlib_root = sysconfig.get_paths()["stdlib"]

        def fake(name, **attrs):
            module = types.ModuleType(name)
            module.__dict__.update(attrs)
            return module

        def is_library(name, module):
            with mock.patch.dict(sys.modules, {name: module}):
                return precompile_package._is_library_module(name)

        site_packages = os.path.join(stdlib_root, "site-packages")
        frozen = importlib.machinery.FrozenImporter
        shadows = {
            "under an install dir": fake("graphlib", __file__=os.path.join(site_packages, "graphlib", "__init__.py")),
            "no __file__, no __spec__": fake("graphlib"),
            "relative __file__": fake("graphlib", __file__="graphlib.py"),
            "namespace package": fake("graphlib", __spec__=importlib.machinery.ModuleSpec("graphlib", None, is_package=True)),
            # The torch branch refuses a torch name resolving outside the torch roots;
            # the stdlib dir itself is never a pip target, so no torch root holds it
            # (site-packages/torch IS one wherever purelib nests under stdlib).
            "torch name outside the torch roots": fake("torch.foo", __file__=os.path.join(stdlib_root, "torch", "foo.py")),
            # The inittab is keyed on the full dotted name, not its top component.
            "dotted name under a built-in top": fake("sys.sub", __spec__=importlib.machinery.ModuleSpec("sys.sub", importlib.machinery.BuiltinImporter, origin="built-in")),
            # A frozen spec vouches only for a name the frozen table has.
            "frozen spec under a non-frozen name": fake("graphlib", __spec__=importlib.machinery.ModuleSpec("graphlib", frozen, origin="frozen")),
        }  # fmt: skip
        for label, module in shadows.items():
            self.assertFalse(is_library(module.__name__, module), label)
        # On 3.11+ a frozen stdlib module also carries an absolute __file__, so
        # the real zipimport never reaches the frozen table; a module dict
        # without one (3.10, or no sys._stdlib_dir) does.
        spec = importlib.machinery.ModuleSpec("zipimport", frozen, origin="frozen")
        self.assertTrue(is_library("zipimport", fake("zipimport", __spec__=spec)))
        # An install root nested inside the stdlib root with no site-packages
        # component in the path: only the _install_roots exclusion catches it.
        vendored = os.path.join(stdlib_root, "vendored")
        nested = fake(
            "graphlib", __file__=os.path.join(vendored, "graphlib", "__init__.py")
        )
        for install_roots, expected in (
            ((precompile_package._norm(vendored),), False),
            ((), True),
        ):
            precompile_package._classify_file.cache_clear()
            with mock.patch.object(
                precompile_package, "_install_roots", return_value=install_roots
            ):
                self.assertEqual(
                    is_library("graphlib", nested), expected, install_roots
                )
        # A located parent does not vouch for a descendant located elsewhere.
        shadowed_sub = fake(
            "collections.abc", __file__=os.path.join(site_packages, "abc.py")
        )
        self.assertFalse(is_library("collections.abc", shadowed_sub))
        self.assertFalse(precompile_package._is_library_module("not_a_stdlib_name"))
        self.assertFalse(precompile_package._is_library_module(None))

    @parametrize("name", _LIBRARY_NAMES)
    def test_library_module_keeps_the_waiver_for_the_real_library(self, name):
        self.assertTrue(
            precompile_package._is_library_module(name), f"{name} lost its waiver"
        )

    def test_reads_a_builtin_keys_on_where_the_read_comes_from(self):
        reads_a_builtin = precompile_package._reads_a_builtin
        self.assertTrue(reads_a_builtin(DictGetItemSource(_BUILTINS_DICT, "len"), len))
        # A builtin parked in a slot, a user table keyed by a builtin's name and
        # user code injected into builtins are all reads from a slot.
        self.assertFalse(reads_a_builtin(AttrSource(LocalSource("self"), "act"), abs))
        self.assertFalse(
            reads_a_builtin(DictGetItemSource(GlobalSource("_OPS"), "len"), len)
        )
        self.assertFalse(
            reads_a_builtin(DictGetItemSource(_BUILTINS_DICT, "op"), _user_op)
        )

    def test_dynamo_synthesized_covers_only_the_resume_function_list(self):
        synthesized = precompile_package._is_dynamo_synthesized
        resume_fns = LocalSource("__nested_resume_fns")
        self.assertTrue(synthesized(resume_fns))
        self.assertTrue(synthesized(GetItemSource(resume_fns, 0)))
        # The frame values are the enclosing frames' live locals, so a guard
        # rooted there is judged like the local it stands for.
        self.assertFalse(
            synthesized(GetItemSource(LocalSource("__nested_frame_values"), 0))
        )
        # A global spelled like one is a user binding.
        self.assertFalse(synthesized(GlobalSource("__nested_resume_fns")))
        self.assertFalse(synthesized(LocalSource("x")))

    def test_alias_module_and_owning_module(self):
        alias_module = precompile_package._dynamo_alias_module
        self.assertIs(alias_module("__import_torch_dot_nn_dot_functional"), F)
        self.assertIsNone(alias_module("F"))
        owning_module = precompile_package._owning_module
        self.assertEqual(owning_module(F), "torch.nn.functional")
        self.assertEqual(owning_module(F.gelu), "torch._C._nn")
        self.assertIsNone(owning_module(3))

    def test_defined_where_read_needs_the_name_and_the_file(self):
        defined_where_read = precompile_package._defined_where_read
        self.assertTrue(defined_where_read(_user_op, "_user_op", _HERE))
        # Paths are compared normalized, so another spelling of the file matches.
        unnormalized = os.path.join(
            os.path.dirname(__file__), os.curdir, os.path.basename(__file__)
        )
        stack = traceback.StackSummary.from_list([(unnormalized, 1, "forward", "")])
        self.assertTrue(defined_where_read(_user_op, "_user_op", stack))
        # A same-file def bound under another name is a slot, however near.
        self.assertFalse(defined_where_read(_user_op, "act", _HERE))
        self.assertFalse(defined_where_read(_user_op, "_user_op", _ELSEWHERE))
        self.assertFalse(defined_where_read(_user_op, "_user_op", None))
        self.assertFalse(defined_where_read(F.silu, "silu", _HERE))
        self.assertFalse(defined_where_read(3, "3", _HERE))
        # The guard carries the inlining stack at first use, so its innermost
        # frame can be a helper from another file; the reading file is the root
        # frame's, the one whose globals a bare GlobalSource denotes.
        two_frame = traceback.StackSummary.from_list(
            [(__file__, 1, "forward", ""), (F.__file__, 2, "helper", "")]
        )
        self.assertTrue(defined_where_read(_user_op, "_user_op", two_frame))
        self.assertFalse(defined_where_read(F.silu, "silu", two_frame))
        # functools.wraps copies __module__ along with __name__ and __qualname__,
        # so a wrapper minted in another file claims this one; its code object
        # does not. A C-implemented wrapper has no code object and is not
        # waived either, and neither is an unconditional cross-file decorator on
        # a same-file def: the object does not tell it from the flag shape.
        wrapped = torch.compile(_user_op, backend="eager")
        self.assertEqual(
            (wrapped.__name__, wrapped.__qualname__, wrapped.__module__),
            ("_user_op", "_user_op", __name__),
        )
        self.assertFalse(defined_where_read(wrapped, "_user_op", _HERE))
        cached = functools.lru_cache(_user_op)
        self.assertFalse(hasattr(cached, "__code__"))
        self.assertFalse(defined_where_read(cached, "_user_op", _HERE))
        decorated = torch.no_grad()(_user_op)
        self.assertFalse(defined_where_read(decorated, "_user_op", _HERE))
        # A class has no code object, so its __module__'s file decides.
        cls = type(self)
        self.assertTrue(defined_where_read(cls, cls.__name__, _HERE))
        self.assertFalse(defined_where_read(cls, cls.__name__, _ELSEWHERE))
        self.assertFalse(defined_where_read(torch.nn.Linear, "Linear", _HERE))

        # A method extracted under its own name and a def returned by a factory
        # are assignments, not a def under its own name: __qualname__ tells.
        class Ops:
            @staticmethod
            def op(x):
                return x

        def availability_fork():
            def _user_op(x):
                return x + 2

            return _user_op

        self.assertFalse(defined_where_read(Ops.op, "op", _HERE))
        self.assertFalse(defined_where_read(availability_fork(), "_user_op", _HERE))
        # A module-level same-name fork inside this file binds a different def
        # per machine under one checksum and cannot be told from the real one:
        # the conditional-bind KNOWN GAP of _is_risky_drop, pinned as such.
        forked = {}
        exec(compile("def _user_op(x):\n    return x + 2\n", __file__, "exec"), forked)
        self.assertTrue(defined_where_read(forked["_user_op"], "_user_op", _HERE))

    def test_minted_global_names_match_dynamo(self):
        # The predicates restate names Dynamo mints inline, in
        # install_builtins_dict_in_fglobals, import_source and the nested
        # resume prologue; a rename there must fail here rather than silently
        # turn the lint off.
        seen = []

        def record(entries):
            seen.extend(entries)
            return [True] * len(entries)

        def roots():
            return {
                precompile_package._source_root(e.orig_guard.originating_source)
                for e in seen
            }

        lin = torch.nn.Linear(2, 2)

        def fn(x):
            return lin(x) + len(x.shape)

        compiled = torch.compile(
            fn, backend="eager", options={"guard_filter_fn": record}
        )
        compiled(torch.ones(2))
        reads_a_builtin = precompile_package._reads_a_builtin
        self.assertTrue(
            any(reads_a_builtin(e.orig_guard.originating_source, e.value) for e in seen)
        )
        aliases = {
            r.global_name
            for r in roots()
            if isinstance(r, GlobalSource) and r.global_name.startswith("__import_")
        }
        alias = "__import_torch_dot_nn_dot_modules_dot_linear"
        self.assertIn(alias, aliases)
        self.assertIs(
            precompile_package._dynamo_alias_module(alias), torch.nn.modules.linear
        )

        def callee(y):
            torch._dynamo.graph_break()
            return y + 1

        def nested(x):
            z = x * 2
            return callee(z) + z

        seen.clear()
        # The harness runs every test under nested_graph_breaks=True.
        compiled = torch.compile(
            nested, backend="eager", options={"guard_filter_fn": record}
        )
        compiled(torch.ones(2))
        synthesized = {
            r.local_name: precompile_package._is_dynamo_synthesized(r)
            for r in roots()
            if isinstance(r, LocalSource) and r.local_name.startswith("__nested")
        }
        self.assertEqual(
            synthesized,
            {"__nested_resume_fns": True, "__nested_frame_values": False},
        )

    def test_module_namespaces_trust_only_bindings_config_cannot_repoint(self):
        mypkg = types.ModuleType("mypkg")
        layers = types.ModuleType("mypkg.layers")
        impl_b = types.ModuleType("mypkg.impl_b")
        entries = [
            _entry(GlobalSource("mypkg"), mypkg),  # import mypkg
            _entry(AttrSource(GlobalSource("mypkg"), "layers"), layers),  # import mypkg.layers
            _entry(AttrSource(GlobalSource("mypkg"), "impl"), impl_b),  # from . import impl_b as impl
            _entry(GlobalSource("impl"), impl_b),  # import mypkg.impl_b as impl
            _entry(AttrSource(GlobalSource("other"), "sub"), layers),  # parent never guarded
            _entry(GlobalSource("F"), F),  # import torch.nn.functional as F
            _entry(AttrSource(GlobalSource("torch"), "_dynamo"), torch._dynamo),  # library, parent or not
            _entry(GlobalSource("__import_mypkg_dot_layers"), layers),  # Dynamo's alias for an inlined function's globals
            _entry(AttrSource(GlobalSource("__import_torch"), "Tensor"), torch.Tensor),  # alias without a module-valued guard
            _entry(GlobalSource("config"), torch._dynamo.config),
        ]  # fmt: skip
        with mock.patch.dict(sys.modules, {"mypkg.layers": layers}):
            namespaces = precompile_package._module_namespaces(entries)
        self.assertEqual(
            set(namespaces),
            {
                "G['mypkg']",
                "G['mypkg'].layers",
                "G['F']",
                "G['torch']._dynamo",
                "G['__import_mypkg_dot_layers']",
                "G['__import_torch']",
            },
        )
        self.assertIs(namespaces["G['__import_torch']"], torch)
        self.assertIs(namespaces["G['mypkg'].layers"], layers)


instantiate_parametrized_tests(TestPrecompilePackage)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
