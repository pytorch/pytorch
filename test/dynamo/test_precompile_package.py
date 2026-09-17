# Owner(s): ["module: dynamo"]

import builtins
import collections
import dataclasses
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
from torch._dynamo.aot_compile import AOTCompiledFunction
from torch._dynamo.exc import PackageError
from torch._dynamo.guards import CheckFunctionManager, GuardBuilder, strip_local_scope
from torch._dynamo.package import load_guards_state
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


# Names torch or the stdlib own, including four imported submodules whose
# module dict carries neither __file__ nor __spec__, so _located has no evidence
# either way and the waiver rests on the located package: torch._C._nn (an
# extension submodule torch/__init__.py setdefaults into sys.modules), torch.ops
# (a ModuleType subclass; the _Ops.__file__ class attribute is the relative
# "_ops.py", which the module dict never sees), pyexpat.errors (registered by
# pyexpat's C init) and xml.parsers.expat.model (registered by expat.py). A
# top-level name keeps its waiver only while it is in sys.modules, which is
# what the header's `import xml.parsers.expat` is for.
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

_STDLIB_ROOT = sysconfig.get_paths()["stdlib"]

# Rows: module name, its module dict, the _install_roots to judge under (None
# keeps the real ones). Every shape a name can take without resolving to the
# library, each refused by the check its label names. graphlib, queue, code and
# distutils are all stdlib names a third party ships, and purelib NESTS inside
# stdlib (conda) or platstdlib (venv), so a __file__ prefix check waived every
# shadow; the stdlib dir itself is never a pip target, so no torch root holds
# the torch rows (site-packages/torch IS one wherever purelib nests under stdlib).
_NOT_LIBRARY_MODULES = {
    # With no install root known, only _INSTALL_DIR_NAMES catches this one.
    "under_an_install_dir": ("graphlib", {"__file__": os.path.join(_STDLIB_ROOT, "site-packages", "graphlib", "__init__.py")}, ()),
    # No site-packages component: only the _install_roots exclusion catches this one.
    "under_a_nested_install_root": ("graphlib", {"__file__": os.path.join(_STDLIB_ROOT, "vendored", "graphlib", "__init__.py")}, (precompile_package._norm(os.path.join(_STDLIB_ROOT, "vendored")),)),
    "no_file_no_spec": ("graphlib", {}, None),
    # Evidence in neither direction, and a top-level name needs some.
    "relative_file": ("graphlib", {"__file__": "graphlib.py"}, None),
    "namespace_package": ("graphlib", {"__spec__": importlib.machinery.ModuleSpec("graphlib", None, is_package=True)}, None),
    "torch_outside_the_torch_roots": ("torch", {"__file__": os.path.join(_STDLIB_ROOT, "torch", "__init__.py")}, None),
    # The ancestor loop: a located torch does not vouch for this one.
    "torch_submodule_outside_the_torch_roots": ("torch.foo", {"__file__": os.path.join(_STDLIB_ROOT, "torch", "foo.py")}, None),
    # The inittab is keyed on the full dotted name, not its top component.
    "dotted_name_under_a_built_in_top": ("sys.sub", {"__spec__": importlib.machinery.ModuleSpec("sys.sub", importlib.machinery.BuiltinImporter, origin="built-in")}, None),
    "built_in_loader_without_a_spec": ("sys.sub", {"__loader__": importlib.machinery.BuiltinImporter}, None),
    # A frozen spec vouches only for a name the frozen table has.
    "frozen_spec_under_a_non_frozen_name": ("graphlib", {"__spec__": importlib.machinery.ModuleSpec("graphlib", importlib.machinery.FrozenImporter, origin="frozen")}, None),
    "shadowed_descendant_of_a_located_parent": ("collections.abc", {"__file__": os.path.join(_STDLIB_ROOT, "site-packages", "abc.py")}, None),
    # posixpath.realpath hands the NUL to os.lstat, which raises ValueError, and
    # _classify_file must make that no evidence rather than an exception out of a
    # lint. From 3.11.5/3.12 on (gh-106242) ntpath.realpath swallows the
    # ValueError itself and returns normpath(path), which sits under the stdlib
    # root; 3.10 lets it out and would refuse as on posix, but the gate is kept
    # platform-wide, so there is nothing to pin on Windows.
    **({"embedded_nul_in_the_file": ("graphlib", {"__file__": os.path.join(_STDLIB_ROOT, "graph\x00lib.py")}, None)} if sys.platform != "win32" else {}),
}  # fmt: skip


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
        g = GlobalSource("g")
        entries = [
            _entry(g, None, "TENSOR_MATCH"),
            _entry(g, None, "TYPE_MATCH"),
            # An id_match_unchecked on a builtin records ID_MATCH as its derived
            # type; serialize_guards takes its TYPE_MATCH/BUILTIN_MATCH branch
            # first and never reaches the derived-type refusal, so neither does
            # the filter.
            _entry(g, None, "BUILTIN_MATCH", derived=("ID_MATCH",)),
        ]
        keep = precompile_package.default_guard_filter_fn(entries)
        self.assertEqual(keep, [True] * 3)

    def test_default_guard_filter_through_serialize_guards(self):
        def fn(x):
            return x + len(x.shape)

        x = torch.randn(3)
        options = {"guard_filter_fn": precompile_package.default_guard_filter_fn}
        compiled = torch.compile(fn, fullgraph=True, backend="eager", options=options)
        compiled = compiled.aot_compile(((x,), {}))
        state = load_guards_state(compiled._artifacts.guards_state)
        kept = {guard.create_fn_name() for guard in state.output_graph._guards}
        self.assertIn("BUILTIN_MATCH", kept)
        data = AOTCompiledFunction.serialize(compiled).serialized_data
        loaded = AOTCompiledFunction.deserialize(data)
        self.assertEqual(loaded(x), fn(x))
        # The kept guard is live in the loaded artifact: a swapped builtin trips it.
        real_len = len
        with mock.patch.object(builtins, "len", lambda *args: real_len(*args)):
            with self.assertRaisesRegex(RuntimeError, "GuardManager check failed"):
                loaded(x)

        # A local-scope class passes the filter (the TYPE_MATCH on L['obj']) and
        # serialization refuses it. The filter keeps the guard so the refusal is
        # loud rather than an artifact that never checks the type; dropping it
        # would not avoid the error anyway, since the pickler refuses the
        # instance wherever it sits in the guard tree.
        class Local:
            n = 1

        def fn2(x, obj):
            return x + obj.n

        compiled = torch.compile(fn2, fullgraph=True, backend="eager", options=options)
        with self.assertRaisesRegex(PackageError, "defined in local scope"):
            compiled.aot_compile(((x, Local()), {}))

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

    @parametrize("shape", sorted(_NOT_LIBRARY_MODULES))
    def test_library_module_requires_the_name_to_resolve_to_the_library(self, shape):
        name, attrs, install_roots = _NOT_LIBRARY_MODULES[shape]
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        if install_roots is None:
            install_roots = precompile_package._install_roots()
        # The verdict is cached per __file__, and two rows share one under
        # different install roots; the torch roots must be read off the real
        # torch before a row replaces sys.modules['torch'].
        self.addCleanup(precompile_package._classify_file.cache_clear)
        precompile_package._classify_file.cache_clear()
        precompile_package._torch_roots()
        with (
            mock.patch.dict(sys.modules, {name: module}),
            mock.patch.object(
                precompile_package, "_install_roots", return_value=install_roots
            ),
        ):
            self.assertFalse(precompile_package._is_library_module(name))

    @parametrize("name", _LIBRARY_NAMES)
    def test_library_module_keeps_the_waiver_for_the_real_library(self, name):
        self.assertTrue(
            precompile_package._is_library_module(name), f"{name} lost its waiver"
        )

    def test_library_module_needs_location_evidence_only_at_the_top(self):
        is_library = precompile_package._is_library_module
        self.addCleanup(precompile_package._classify_file.cache_clear)
        with mock.patch.dict(sys.modules):
            sys.modules.pop("graphlib", None)
            self.assertFalse(is_library("graphlib"))  # a stdlib name, not imported
        self.assertFalse(is_library("not_a_stdlib_name"))
        self.assertFalse(is_library(None))
        # An inner name only has to not be located ELSEWHERE, so a relative
        # __file__, evidence in neither direction, leaves the waiver its package
        # earned (its absolute counterpart is refused above).
        foo = types.ModuleType("torch.foo")
        foo.__file__ = "foo.py"
        with mock.patch.dict(sys.modules, {"torch.foo": foo}):
            self.assertTrue(is_library("torch.foo"))
        # The file the vendored row refuses is stdlib once nothing is installed there.
        name, attrs, _ = _NOT_LIBRARY_MODULES["under_a_nested_install_root"]
        vendored = types.ModuleType(name)
        vendored.__dict__.update(attrs)
        precompile_package._classify_file.cache_clear()
        with (
            mock.patch.dict(sys.modules, {name: vendored}),
            mock.patch.object(precompile_package, "_install_roots", return_value=()),
        ):
            self.assertTrue(is_library(name))
        # On 3.11+ a frozen stdlib module also carries an absolute __file__, so
        # the real zipimport never reaches the frozen table; a module dict
        # without one (3.10, or no sys._stdlib_dir) does.
        loader = importlib.machinery.FrozenImporter
        spec = importlib.machinery.ModuleSpec("zipimport", loader, origin="frozen")
        frozen = types.ModuleType("zipimport")
        frozen.__spec__ = spec
        with mock.patch.dict(sys.modules, {"zipimport": frozen}):
            self.assertTrue(is_library("zipimport"))
        # A frozen torch leaves _torch_roots nothing to anchor to, and then a
        # torch name is waived on its name alone, imported or not: the one
        # waiver with no location evidence behind it, recorded here as chosen.
        with mock.patch.object(precompile_package, "_torch_roots", return_value=()):
            self.assertTrue(is_library("torch.never_imported"))

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
        # functools.wraps copies __module__ and __name__, so a shim installed as
        # builtins.sum passes both; only a value CPython built is waived.
        shim = functools.wraps(sum)(lambda *args: 0)
        self.assertEqual((shim.__module__, shim.__name__), ("builtins", "sum"))
        sum_read = DictGetItemSource(_BUILTINS_DICT, "sum")
        self.assertFalse(reads_a_builtin(sum_read, shim))
        self.assertTrue(reads_a_builtin(sum_read, sum))
        self.assertTrue(reads_a_builtin(DictGetItemSource(_BUILTINS_DICT, "int"), int))

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
        # A class has no code object, and namedtuple, make_dataclass and type()
        # all stamp __module__ from the calling frame under a BARE __qualname__,
        # so a class a library mints for this file claims it exactly like a
        # class statement written here. The methods tell: a class statement
        # compiled its defs in this file, and a class with no def of its own
        # compiled here fails closed. A factory fed a same-file method passes,
        # the class analogue of the conditional-bind gap pinned below.
        cls = type(self)
        self.assertTrue(defined_where_read(cls, cls.__name__, _HERE))
        self.assertFalse(defined_where_read(cls, cls.__name__, _ELSEWHERE))
        self.assertFalse(defined_where_read(torch.nn.Linear, "Linear", _HERE))
        point = collections.namedtuple("Point", "x")
        self.assertEqual((point.__module__, point.__qualname__), (__name__, "Point"))
        self.assertFalse(defined_where_read(point, "Point", _HERE))
        point = dataclasses.make_dataclass("Point", [("x", int)])
        self.assertEqual((point.__module__, point.__qualname__), (__name__, "Point"))
        self.assertFalse(defined_where_read(point, "Point", _HERE))
        self.assertFalse(defined_where_read(type("Point", (), {}), "Point", _HERE))
        point = type("Point", (), {"area": _user_op})
        self.assertTrue(defined_where_read(point, "Point", _HERE))

        # A method extracted under its own name and a def returned by a factory
        # are assignments, not a def under its own name: __qualname__ tells.
        # Ops itself, a same-file class statement with a method, is waived; a
        # class with no method of its own is not, nor is one whose only methods
        # are generated (a fields-only dataclass compiles them in <string>).
        class Ops:
            @staticmethod
            def op(x):
                return x

        class Marker:
            pass

        @dataclasses.dataclass
        class Cfg:
            x: int

        def availability_fork():
            def _user_op(x):
                return x + 2

            return _user_op

        self.assertTrue(defined_where_read(Ops, Ops.__qualname__, _HERE))
        self.assertFalse(defined_where_read(Marker, Marker.__qualname__, _HERE))
        self.assertFalse(defined_where_read(Cfg, Cfg.__qualname__, _HERE))
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
        # The resume function list is always passed; whether the frame values
        # are guarded depends on which locals stay live across the break.
        self.assertTrue(synthesized["__nested_resume_fns"])
        self.assertFalse(synthesized.get("__nested_frame_values", False))

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
