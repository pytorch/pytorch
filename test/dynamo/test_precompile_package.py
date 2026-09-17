# Owner(s): ["module: dynamo"]

import builtins
import collections
import dataclasses
import functools
import importlib.machinery
import itertools
import math
import os
import site
import subprocess
import sys
import sysconfig
import threading
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


_OWN = GlobalSource(__name__)
_BUILTINS_DICT = GlobalSource("__builtins_dict___0")
_HERE = traceback.StackSummary.from_list([(__file__, 1, "forward", "")])
_ELSEWHERE = traceback.StackSummary.from_list([(F.__file__, 1, "forward", "")])


def _entry(
    source, value, guard_type="ID_MATCH", derived=(), has_value=True, user_stack=None
):
    guard = Guard(source, getattr(GuardBuilder, guard_type))
    guard.user_stack = user_stack
    return GuardFilterEntry(
        name=strip_local_scope(source.name),
        has_value=has_value,
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

# Rows: risky?, source, value, _entry keywords. The trusted namespaces are the
# torch, stdlib and own-module globals the test builds through _module_namespaces;
# G['impl'] is an aliased user module and G['config'] a config module, neither
# trusted. Every risky row was once a silent wrong answer on a serving machine.
_RISKY_DROP_CASES = {
    "torch_namespace_read": (False, AttrSource(GlobalSource("F"), "gelu"), F.gelu, {}),
    "stdlib_namespace_read": (False, AttrSource(GlobalSource("math"), "sqrt"), math.sqrt, {}),
    "dynamo_import_alias_read": (False, AttrSource(GlobalSource("__import_torch"), "relu"), torch.relu, {}),
    "trusted_module_itself": (False, GlobalSource("F"), F, {}),
    "own_module_def_read_as_namespace": (False, AttrSource(_OWN, "_user_op"), _user_op, {}),
    "own_module_def_under_another_name": (True, AttrSource(_OWN, "act"), _user_op, {}),
    "aliased_user_module": (True, AttrSource(GlobalSource("impl"), "op"), _user_op, {}),
    "config_module_attribute": (True, AttrSource(GlobalSource("config"), "attn_impl"), _user_op, {}),
    "module_in_attribute": (True, AttrSource(AttrSource(LocalSource("self"), "ns"), "gelu"), F.gelu, {}),
    "instance_attribute": (True, AttrSource(LocalSource("self"), "act"), F.gelu, {}),
    "builtin_in_a_slot": (True, AttrSource(LocalSource("self"), "act"), abs, {}),
    "builtin_read_ordinary": (False, DictGetItemSource(_BUILTINS_DICT, "len"), len, {}),
    "user_code_injected_into_builtins": (True, DictGetItemSource(_BUILTINS_DICT, "op"), _user_op, {}),
    "registry_keyed_by_builtin_name": (True, DictGetItemSource(GlobalSource("_OPS"), "len"), len, {}),
    "dict_lookup": (True, DictGetItemSource(GlobalSource("DISPATCH"), "act"), _user_op, {}),
    "global_bound_to_own_def": (False, GlobalSource("_user_op"), _user_op, {"user_stack": _HERE}),
    "global_bound_to_torch_def": (False, GlobalSource("silu"), F.silu, {}),
    "global_alias_of_a_def": (True, GlobalSource("act"), _user_op, {"user_stack": _HERE}),
    "cross_module_from_import": (True, GlobalSource("_user_op"), _user_op, {"user_stack": _ELSEWHERE}),
    "closure_cell": (True, LocalSource("fn"), _user_op, {}),
    "value_unreadable": (True, AttrSource(_OWN, "x"), None, {"has_value": False}),
    "nested_resume_function": (False, GetItemSource(LocalSource("__nested_resume_fns"), 0), _user_op, {}),
    "nested_frame_value": (True, GetItemSource(GetItemSource(LocalSource("__nested_frame_values"), 0), 1), _user_op, {}),
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

    def test_normalize_scrubs_addresses_and_counters_but_not_user_constants(self):
        # Both directions matter: anything run-varying that survives makes the
        # committed report churn, and anything meaningful that is erased makes
        # two variants guarding different values render one fact.
        from torch._dynamo.precompile_package import _normalize

        cases = {
            "___check_obj_id(G['fn'], 140311678493200), type=<class 'function'>": "___check_obj_id(G['fn'], <id>), type=<class 'function'>",
            "G['__builtins_dict___6']['len']": "G['__builtins_dict___<n>']['len']",
            "G['__import_mod_140311678493200_c1']": "G['__import_mod_<id>_c<n>']",
            "G['___unnamed_scope_140311678493200_c1']": "G['___unnamed_scope_<id>_c<n>']",
            "G['_140311678493200_c3'] is not None": "G['_<id>_c<n>'] is not None",
            "top_saved_tensors_hooks ids == (139, 140)": "top_saved_tensors_hooks ids == (<ids>)",
            # User constants and identifiers are not addresses.
            "L['dims'][0] == 140311678493200": "L['dims'][0] == 140311678493200",
            "L['w_1_c2'] == 3": "L['w_1_c2'] == 3",
            "len(L['xs']) == 6": "len(L['xs']) == 6",
        }
        for text, expected in cases.items():
            self.assertEqual(_normalize(text), expected, text)

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

    @parametrize("shape", sorted(_RISKY_DROP_CASES))
    def test_risky_drop_decision_table(self, shape):
        risky, source, value, kw = _RISKY_DROP_CASES[shape]
        entry = _entry(source, value, **kw)
        modules = [
            (GlobalSource("F"), F),
            (GlobalSource("math"), math),
            (_OWN, sys.modules[__name__]),
            (GlobalSource("impl"), types.ModuleType("mypkg.impl_b")),
            (GlobalSource("config"), torch._dynamo.config),
        ]
        entries = [_entry(s, m) for s, m in modules] + [entry]
        namespaces = precompile_package._module_namespaces(entries)
        self.assertEqual(precompile_package._is_risky_drop(entry, namespaces), risky)

    def test_risky_drop_sees_the_slot_behind_a_nested_resume(self):
        # With nested_graph_breaks the callee's locals reach its resume frame
        # as L['__nested_frame_values'][0][k] rather than as L['act']; a slot
        # filled by a call config could repoint must be flagged either way,
        # and the def read inside pick() waived either way.
        def pick():
            return _user_op

        def flat(x):
            act = pick()
            torch._dynamo.graph_break()
            return act(x * 2)

        def callee(y):
            act = pick()
            torch._dynamo.graph_break()
            return act(y)

        def nested(x):
            z = x * 2
            return callee(z) + z

        for fn, nested_graph_breaks, root in (
            (flat, False, "L['act']"),
            (nested, True, "L['__nested_frame_values']["),
        ):
            seen = []

            def record(entries):
                seen.extend(entries)
                return [True] * len(entries)

            with torch._dynamo.config.patch(nested_graph_breaks=nested_graph_breaks):
                compiled = torch.compile(
                    fn, backend="eager", options={"guard_filter_fn": record}
                )
                compiled(torch.ones(2))
            namespaces = precompile_package._module_namespaces(seen)
            is_risky = precompile_package._is_risky_drop
            verdicts = {
                e.orig_guard.originating_source.name: is_risky(e, namespaces)
                for e in seen
                if e.value is _user_op
            }
            risky = [name for name, flagged in verdicts.items() if flagged]
            self.assertEqual(verdicts["G['_user_op']"], False, verdicts)
            self.assertEqual(len(risky), 1, verdicts)
            self.assertTrue(risky[0].startswith(root), verdicts)

    def test_code_fingerprint_recurses_into_container_and_nested_consts(self):
        # _code_fingerprint names a callable by its body so an ACT2FN-style table
        # can be told apart. Two lambdas can differ ONLY inside a constant the
        # outer co_code does not distinguish: a tuple, a frozenset, or a nested
        # code object. Filtering those out whole -- rather than recursing -- gives
        # both the same digest, _object_identity names them identically, and the
        # guard that split the two compilations is reported as an invariant of
        # each.
        from torch._dynamo.precompile_package import _code_fingerprint, _stable_consts

        pairs = {
            "tuple const": (lambda x: x * (1, 2), lambda x: x * (1, 3)),
            "frozenset const": (lambda x: x in {1, 2}, lambda x: x in {1, 3}),
            # Not called: what matters is the nested code object in co_consts.
            "nested code": (lambda x: (lambda y: y + 1), lambda x: (lambda y: y + 2)),
            # A subscript with Ellipsis folds to ONE const tuple at one index, so
            # a const type outside the stable set must keep its slot.
            "ellipsis const": (lambda x: x[..., 0], lambda x: x[0, ...]),
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
        # An unrenderable const keeps its position as a type marker.
        self.assertEqual(_stable_consts((object(), 1)), ("<object>", 1))
        # And the digest is a function of the body, not of the code object:
        # the same source compiled twice must agree.
        src = "lambda x: (x * 2, 'a', (lambda y: y + 1))"
        self.assertEqual(
            _code_fingerprint(compile(src, "<a>", "eval")),
            _code_fingerprint(compile(src, "<b>", "eval")),
        )

    def test_code_fingerprint_is_stable_across_processes(self):
        # The digest goes into a file meant to be committed and diffed, so it
        # has to agree in a fresh interpreter; within one process any two calls
        # trivially agree, which is why the positive case above cannot catch a
        # regression that puts an address back (repr of an arbitrary const).
        from torch._dynamo.precompile_package import _code_fingerprint

        src = "lambda x: (x * 2, 'a', (lambda y: y + 1), x in {1, 2})"
        probe = (
            "from torch._dynamo.precompile_package import _code_fingerprint;"
            f"print(_code_fingerprint(compile({src!r}, '<p>', 'eval')))"
        )
        out = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        self.assertEqual(
            out.stdout.strip(), _code_fingerprint(compile(src, "<p>", "eval"))
        )

    def test_object_identity_puts_the_digest_before_the_truncation_point(self):
        from torch._dynamo.precompile_package import _code_fingerprint, _object_identity

        self.assertEqual(
            _object_identity(torch.nn.functional), "is module torch.nn.functional"
        )
        self.assertEqual(_object_identity(object()), "is a builtins.object")

        def fn():
            pass

        # A qualname that alone exceeds the 160-character bound: the site and
        # digest must survive the cut and the qualname tail is what goes.
        fn.__qualname__ = "Outer." * 40 + "fn"
        rendered = _object_identity(fn)
        code = fn.__code__
        prefix = f"is @{os.path.basename(code.co_filename)}:{code.co_firstlineno}#{_code_fingerprint(code)} "
        self.assertEqual(len(rendered), 160)
        self.assertTrue(rendered.startswith(prefix), rendered)
        self.assertNotIn("#", rendered[len(prefix) :])

    def test_guard_policy_classification_is_total(self):
        # A guard type in no set is KEPT, so a drop policy can only ever
        # drop what _INVARIANT_DROPPABLE_GUARD_TYPES names. This test is
        # what makes the never-drop claim enforceable:
        # a guard type added to GuardBuilder fails here until someone triages
        # it into exactly one of the four sets.
        from torch._dynamo.guards import GuardBuilder
        from torch._dynamo.precompile_package import (
            _IDENTITY_GUARD_TYPES,
            _INVARIANT_DROPPABLE_GUARD_TYPES,
            _NOOP_GUARD_TYPES,
            _SHAPE_BEARING_GUARD_TYPES,
            _UNMODELLED_GUARD_TYPES,
        )

        # dir() rather than vars(): a guard method added on GuardBuilderBase or
        # a future mixin is a GuardBuilder guard type too.
        guard_types = {
            name
            for name in dir(GuardBuilder)
            if name.isupper() and callable(getattr(GuardBuilder, name))
        }
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
        # The identity guards the default filter drops are droppable by
        # construction; a literal rewrite of the set must not lose that.
        self.assertTrue(_IDENTITY_GUARD_TYPES <= _INVARIANT_DROPPABLE_GUARD_TYPES)

    def test_noop_guard_type_follows_the_hook_guard_config(self):
        # EMPTY_NN_MODULE_HOOKS_DICT emits nothing under the default config and
        # a SEQUENCE_LENGTH on the hook dicts otherwise, so whether a report may
        # treat it as a marker depends on the config the frame compiled under.
        from torch._dynamo.precompile_package import _is_noop_guard_type

        self.assertTrue(_is_noop_guard_type("GRAD_MODE"))
        self.assertFalse(_is_noop_guard_type("TENSOR_MATCH"))
        self.assertTrue(_is_noop_guard_type("EMPTY_NN_MODULE_HOOKS_DICT"))
        with torch._dynamo.config.patch(skip_nnmodule_hook_guards=False):
            self.assertFalse(_is_noop_guard_type("EMPTY_NN_MODULE_HOOKS_DICT"))
            self.assertTrue(_is_noop_guard_type("GRAD_MODE"))

    def test_value_fingerprint_dispatches_on_the_guard_type(self):
        from torch.compiler._precompile_types import GuardFact

        fingerprint = precompile_package._value_fingerprint
        src = LocalSource("x")
        x = torch.zeros(2, 3)
        with torch.inference_mode():
            inference = torch.zeros(2, 3)
        variants = (x, x.double(), x[:, :2], torch.nn.Parameter(x), inference)
        rendered = [fingerprint(_entry(src, v, "TENSOR_MATCH")) for v in variants]
        self.assertEqual(len(set(rendered)), len(variants), rendered)
        for line in rendered:
            self.assertTrue(line.startswith("check_tensor(<value>, "), line)
        # The guard type decides, not the value's type: NOT_NONE_MATCH is what
        # Dynamo installs on an optimizer's .grad, and it checks only presence.
        for guard_type in ("NOT_NONE_MATCH", "TYPE_MATCH", "COW_TENSOR_MATCH"):
            self.assertEqual(fingerprint(_entry(src, x, guard_type)), "")
        self.assertEqual(fingerprint(_entry(LocalSource("n"), 1, "TYPE_MATCH")), "")
        self.assertEqual(
            fingerprint(_entry(_OWN, _user_op, "ID_MATCH")),
            precompile_package._object_identity(_user_op),
        )
        grad_mode = _entry(src, None, "GRAD_MODE", has_value=False)
        with torch.no_grad():
            self.assertEqual(fingerprint(grad_mode), "grad_enabled=False")

        class Opaque(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                raise RuntimeError("no attribute reads")

        opaque = torch.zeros(2).as_subclass(Opaque)
        unrenderable = fingerprint(_entry(src, opaque, "TENSOR_MATCH"))
        self.assertEqual(unrenderable, "type=Opaque, <unrenderable>")
        # Once the boilerplate parts are filtered a TENSOR_MATCH renders no code,
        # so the value is what keeps two shape specializations in a fixed order.
        facts = [
            GuardFact(
                guard_type="TENSOR_MATCH",
                source="L['x']",
                code=(),
                value=v,
                enforced=True,
            )
            for v in rendered
        ]
        ordered = sorted(facts, key=precompile_package._fact_order)
        self.assertEqual([f.value for f in ordered], sorted(rendered))

    def test_saved_hooks_fingerprint_mirrors_what_the_guard_stores(self):
        fingerprint = precompile_package._saved_hooks_fingerprint
        self.assertEqual(fingerprint(), "hooks=None")
        # The guard stores None for hooks it cannot inline, so plain-Python
        # hooks are one value to it and must be one value here.
        with torch.autograd.graph.saved_tensors_hooks(_user_op, _user_op):
            self.assertEqual(fingerprint(), "hooks=None")

        def identity(x):
            return x

        pack = torch.fx.symbolic_trace(identity)
        unpack = torch.fx.symbolic_trace(identity)
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            rendered = fingerprint()
        # Named by rendered graph, never by address: two GraphModules with one
        # code read the same here although the guard compares their ids.
        digest = precompile_package._hash_text(pack.code)
        self.assertEqual(rendered, f"hooks=({digest}, {digest})")

    def test_wont_generalize_cancels_pins_only_within_a_frame(self):
        from torch._dynamo.precompile_package import (
            _pins_a_value,
            _SHAPE_BEARING_GUARD_TYPES,
            _VALUE_EQUALITY_GUARD_TYPES,
            _wont_generalize,
        )
        from torch.compiler._precompile_types import GuardFact

        # A value pin is never policy-droppable, so a new value-pinning guard
        # type has to be triaged into the shape-bearing set to land here.
        self.assertTrue(_VALUE_EQUALITY_GUARD_TYPES <= _SHAPE_BEARING_GUARD_TYPES)
        self.assertTrue(_pins_a_value("EQUALS_MATCH", "scale"))
        self.assertTrue(_pins_a_value("CONSTANT_MATCH", "___stack0"))
        self.assertTrue(_pins_a_value("CONSTANT_SUBCLASS_MATCH", "n"))
        # Reached THROUGH an argument, or a container element: not counted.
        self.assertFalse(_pins_a_value("CONSTANT_MATCH", "self.eps"))
        self.assertFalse(_pins_a_value("EQUALS_MATCH", "dims[0]"))
        self.assertFalse(_pins_a_value("EQUALS_MATCH", "G['CFG'].width"))
        self.assertFalse(_pins_a_value("TENSOR_MATCH", "x"))
        self.assertFalse(_pins_a_value("SEQUENCE_LENGTH", "xs"))

        def fact(guard_type, source):
            return GuardFact(
                guard_type=guard_type, source=source, code=(), value="", enforced=True
            )

        entry = ("step", "m.py", 1)
        resume_a = ("torch_dynamo_resume_in_step_at_7", "m.py", 7)
        resume_b = ("torch_dynamo_resume_in_step_at_9", "m.py", 9)
        kept = {
            ("EQUALS_MATCH", "scale"),
            ("EQUALS_MATCH", "mode"),
            ("CONSTANT_MATCH", "___stack0"),
            ("TENSOR_MATCH", "x"),
        }
        pinned_scale = fact("EQUALS_MATCH", "scale")
        pinned_mode = fact("EQUALS_MATCH", "mode")
        generic_scale = fact("TYPE_MATCH", "scale")
        x = fact("TENSOR_MATCH", "x")
        guard_sets = {
            # Two variants of the entry: one pins scale and mode, the other
            # serves scale generically -- the ordinary shape once two examples
            # are captured -- so only mode stays pinned.
            entry: [
                frozenset({pinned_scale, pinned_mode, x}),
                frozenset({generic_scale, pinned_mode, x}),
            ],
            # ___stack0 is a tensor in this resume frame ...
            resume_a: [frozenset({fact("TENSOR_MATCH", "___stack0")})],
            # ... and the .item() int in this one. The tensor elsewhere is a
            # different local under the same bare name and must not cancel it.
            resume_b: [frozenset({fact("CONSTANT_MATCH", "___stack0")})],
        }
        self.assertEqual(_wont_generalize(kept, guard_sets), ("___stack0", "mode"))
        # Nothing pinned: nothing to report, whatever the frames say.
        self.assertEqual(_wont_generalize({("TENSOR_MATCH", "x")}, guard_sets), ())

    def test_varying_guard_slots_are_the_differing_and_present_in_some_ones(self):
        from torch._dynamo.precompile_package import _varying_guard_slots
        from torch.compiler._precompile_types import GuardFact

        def fact(guard_type, source, code=(), value="", enforced=True):
            return GuardFact(
                guard_type=guard_type,
                source=source,
                code=code,
                value=value,
                enforced=enforced,
            )

        x_f32 = fact("TENSOR_MATCH", "L['x']", value="dtype=float32")
        x_f16 = fact("TENSOR_MATCH", "L['x']", value="dtype=float16")
        flag = fact("CONSTANT_MATCH", "L['flag']", code=("L['flag'] == 1",))
        fn_id = fact("ID_MATCH", "G['fn']", value="is mod.fn", enforced=False)
        # Same check as fn_id, only the filter's verdict differs.
        fn_id_kept = fact("ID_MATCH", "G['fn']", value="is mod.fn")
        frame = ("forward", "m.py", 12)

        self.assertEqual(_varying_guard_slots({}), frozenset())
        # One variant discriminates nothing.
        self.assertEqual(
            _varying_guard_slots({frame: [frozenset({x_f32, flag, fn_id})]}),
            frozenset(),
        )
        varying = _varying_guard_slots(
            {frame: [frozenset({x_f32, fn_id}), frozenset({x_f16, flag, fn_id_kept})]}
        )
        self.assertEqual(
            varying,
            frozenset({("TENSOR_MATCH", "L['x']"), ("CONSTANT_MATCH", "L['flag']")}),
        )
        # Frames are never compared with each other: the same slot pinned to
        # different values in two frames is invariant within each.
        other = ("torch_dynamo_resume_in_forward_at_14", "m.py", 14)
        self.assertEqual(
            _varying_guard_slots(
                {frame: [frozenset({x_f32})], other: [frozenset({x_f16})]}
            ),
            frozenset(),
        )

    def test_capture_config_is_scoped_per_entry_and_per_thread(self):
        import torch._functorch.config as functorch_config
        from torch._dynamo.precompile_package import _capture_config

        def flags():
            return (
                functorch_config.bundled_autograd_cache,
                functorch_config.bypass_autograd_cache_key,
                functorch_config.force_non_lazy_backward_lowering,
                torch._dynamo.config.allow_empty_graphs,
            )

        ambient = (False, False, False, False)
        with (
            functorch_config.patch(
                bundled_autograd_cache=False,
                bypass_autograd_cache_key=False,
                force_non_lazy_backward_lowering=False,
            ),
            torch._dynamo.config.patch(allow_empty_graphs=False),
        ):
            self.assertEqual(flags(), ambient)
            with _capture_config(training=False):
                self.assertEqual(flags(), (True, True, False, True))
                # The inner scope's training wins while it is open, and the
                # outer scope's setting comes back when it closes.
                with _capture_config(training=True):
                    self.assertEqual(flags(), (True, True, True, True))
                self.assertEqual(flags(), (True, True, False, True))
            self.assertEqual(flags(), ambient)

            with self.assertRaisesRegex(RuntimeError, "boom"):
                with _capture_config(training=True):
                    raise RuntimeError("boom")
            self.assertEqual(flags(), ambient)

            # Config values are per thread, so a worker entering the scope
            # patches only itself and the main thread stays ambient.
            entered, release = threading.Event(), threading.Event()
            seen = []

            def hold():
                seen.append(flags())
                with _capture_config(training=False):
                    seen.append(flags())
                    entered.set()
                    release.wait(10)
                seen.append(flags())

            worker = threading.Thread(target=hold)
            worker.start()
            self.assertTrue(entered.wait(10))
            self.assertEqual(flags(), ambient)
            release.set()
            worker.join(10)
            self.assertFalse(worker.is_alive())
            self.assertEqual(seen[1], (True, True, False, True))
            self.assertEqual(seen[0], seen[2])
            self.assertNotEqual(seen[0], seen[1])

    def test_capture_config_refuses_disabled_caches_and_honours_strict(self):
        import torch._functorch.config as functorch_config
        from torch._dynamo.exc import PackageError
        from torch._dynamo.precompile_package import _capture_config

        # Backends reach the artifact through the bundled AOTAutograd cache, so
        # a capture with caches forced off would record nothing; say so up front.
        with torch.compiler.config.patch(force_disable_caches=True):
            with self.assertRaisesRegex(PackageError, "force_disable_caches"):
                with _capture_config(training=False):
                    pass

        with functorch_config.patch(strict_autograd_cache=False):
            with torch._dynamo.config.patch(strict_precompile=False):
                with _capture_config(training=False):
                    self.assertFalse(functorch_config.strict_autograd_cache)
            with torch._dynamo.config.patch(strict_precompile=True):
                with _capture_config(training=False):
                    self.assertTrue(functorch_config.strict_autograd_cache)
            self.assertFalse(functorch_config.strict_autograd_cache)

    def test_allow_empty_graphs_convert_frame_patches_around_the_compile(self):
        from torch._dynamo.convert_frame import ConvertFrame
        from torch._dynamo.hooks import Hooks
        from torch._dynamo.precompile_package import _AllowEmptyGraphsConvertFrame

        seen = []

        def fake_convert(self, frame, cache_entry, hooks, frame_state, skip=0):
            seen.append((torch._dynamo.config.allow_empty_graphs, skip))
            raise RuntimeError("boom")

        converter = _AllowEmptyGraphsConvertFrame(lambda gm, inputs: gm, Hooks())
        with (
            torch._dynamo.config.patch(allow_empty_graphs=False),
            mock.patch.object(ConvertFrame, "__call__", fake_convert),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                converter(sys._getframe(), None, Hooks(), {}, skip=1)
            self.assertFalse(torch._dynamo.config.allow_empty_graphs)
        # The flag was on for the compile and the extra frame is accounted for
        # in the traceback skip count.
        self.assertEqual(seen, [(True, 2)])


instantiate_parametrized_tests(TestPrecompilePackage)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
