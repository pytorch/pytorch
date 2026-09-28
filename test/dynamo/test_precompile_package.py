# Owner(s): ["module: dynamo"]

import builtins
import collections
import dataclasses
import enum
import functools
import importlib.machinery
import importlib.util
import itertools
import math
import os
import re
import site
import subprocess
import sys
import sysconfig
import tempfile
import threading
import traceback
import types
import typing
import xml.parsers.expat  # noqa: F401  # imported for the _LIBRARY_NAMES rows
import zipfile
from unittest import mock

import torch
import torch._dynamo.precompile_package as precompile_package
import torch._inductor.test_case
import torch.nn.functional as F
import torch.utils._pytree as pytree
from torch._dynamo.aot_compile import AOTCompiledFunction
from torch._dynamo.exc import PackageError
from torch._dynamo.guards import CheckFunctionManager, GuardBuilder, strip_local_scope
from torch._dynamo.package import load_guards_state
from torch._dynamo.source import (
    AttrSource,
    CellContentsSource,
    ClosureSource,
    DictGetItemSource,
    get_global_source_name,
    GetItemSource,
    GlobalSource,
    LocalSource,
    TypeSource,
)
from torch._dynamo.types import GuardFilterEntry
from torch._guards import ChainedSource, Guard
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


def _user_op(x):
    return x + 1


def _stack(*filenames):
    """A guard's user_stack, outermost frame first."""
    return traceback.StackSummary.from_list([(f, 1, "forward", "") for f in filenames])


def _aot_compile(fn, *args, guard_filter_fn=None, seen=None):
    """aot_compile fn on args through guard_filter_fn (the default filter when
    None), appending every (entry, verdict) pair to seen."""

    def recording(entries):
        keep = (guard_filter_fn or precompile_package.default_guard_filter_fn)(entries)
        if seen is not None:
            seen.extend(zip(entries, keep))
        return keep

    opts = {"guard_filter_fn": recording}
    fn_c = torch.compile(fn, fullgraph=True, backend="eager", options=opts)
    return fn_c.aot_compile((args, {}))


def _kept_types(compiled):
    state = load_guards_state(compiled._artifacts.guards_state)
    return state, {g.create_fn_name() for g in state.output_graph.guards}


def _pre_check_accepts(entry):
    # The type tests of serialize_guards' pre-check, over the entry's own derived
    # types: a second implementation of them, so a control written on it does not
    # call the filter under test. Not the whole pre-check, which raises rather
    # than returning a verdict and which also refuses a TYPE_MATCH or
    # BUILTIN_MATCH whose guard carries _unserializable (a local-scope type).
    # Harmless in the three controls here: the pytree test's TYPE_MATCHes are on
    # global types, drop_type_match drops them all, and drop_local_type_match
    # itself removes the local-scope ones, the only ones that carry it.
    unsupported = CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
    return entry.guard_type in ("TYPE_MATCH", "BUILTIN_MATCH") or (
        entry.guard_type not in unsupported
        and not any(d in unsupported for d in entry.derived_guard_types)
    )


_OWN = GlobalSource(__name__)
_BUILTINS_DICT = GlobalSource("__builtins_dict___0")
_HERE = _stack(__file__)
_ELSEWHERE = _stack(F.__file__)


def _entry(
    source, value, guard_type="ID_MATCH", derived=(), has_value=True, user_stack=None
):
    guard = Guard(source, getattr(GuardBuilder, guard_type))
    guard.user_stack = user_stack
    guard.guard_types = list(derived) or None
    return GuardFilterEntry(
        name=strip_local_scope(source.name),
        has_value=has_value,
        value=value,
        guard_type=guard_type,
        derived_guard_types=tuple(derived),
        is_global=get_global_source_name(source) is not None,
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
# keeps the real ones). Every shape a name can take while resolving somewhere
# other than the library, each refused by the check its label names; the
# refusals that need no location (not a stdlib name, not imported, None) sit
# with the waiver rows in the next commit's tests. graphlib, queue, code and
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
    # Evidence in neither direction, and a top-level name needs some. The test
    # judges this row from _STDLIB_ROOT, where the real graphlib.py sits, so an
    # ungated resolve against the cwd would land on stdlib and waive the row.
    "relative_file": ("graphlib", {"__file__": "graphlib.py"}, None),
    "torch_outside_the_torch_roots": ("torch", {"__file__": os.path.join(_STDLIB_ROOT, "torch", "__init__.py")}, None),
    # The ancestor loop: a located torch does not vouch for this one.
    "torch_submodule_outside_the_torch_roots": ("torch.foo", {"__file__": os.path.join(_STDLIB_ROOT, "torch", "foo.py")}, None),
    # The inittab is keyed on the full dotted name, not its top component.
    "dotted_name_under_a_built_in_top": ("sys.sub", {"__spec__": importlib.machinery.ModuleSpec("sys.sub", importlib.machinery.BuiltinImporter, origin="built-in")}, None),
    # A frozen spec vouches only for a name the frozen table has.
    "frozen_spec_under_a_non_frozen_name": ("graphlib", {"__spec__": importlib.machinery.ModuleSpec("graphlib", importlib.machinery.FrozenImporter, origin="frozen")}, None),
    "shadowed_descendant_of_a_located_parent": ("collections.abc", {"__file__": os.path.join(_STDLIB_ROOT, "site-packages", "abc.py")}, None),
}  # fmt: skip


class _Ops:
    @staticmethod
    def op(x):
        return x


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
    "own_module_def_lifted_off_a_class": (True, AttrSource(_OWN, "op"), _Ops.op, {}),
    # mypkg/__init__.py did `from .impl_b import _user_op`: the same def, owned by mypkg.impl_b.
    "reexport_from_another_module": (True, AttrSource(GlobalSource("mypkg"), "_user_op"), types.FunctionType(_user_op.__code__, {"__name__": "mypkg.impl_b"}), {}),
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
        filter_fn = precompile_package.default_guard_filter_fn
        unsupported = CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
        # Spelled out: the filter reads the same constant, so on a shrunk one the
        # two would agree on less. The entries are bare; the builder gives most
        # of these a derived ID_MATCH, which the next table drops too, so the
        # verdicts are the same either way.
        refused_types = {
            "ID_MATCH",
            "FUNCTION_MATCH",
            "CLOSURE_MATCH",
            "MODULE_MATCH",
            "NN_MODULE",
            "CLASS_MATCH",
            "DICT_VERSION",
            "WEAKREF_ALIVE",
        }
        self.assertTrue(refused_types <= set(unsupported), unsupported)
        g = GlobalSource("g")
        refused = [_entry(g, None, guard_type=t) for t in unsupported]
        verdicts = dict(zip(unsupported, filter_fn(refused)))
        self.assertEqual(verdicts, dict.fromkeys(unsupported, False))
        # A refused derived type drops the guard too (a CONSTANT_MATCH on a code
        # object and a TENSOR_MATCH under match_on_id_for_tensor both run through
        # ID_MATCH); the TENSOR_MATCH row also pins the accepted-by-type pair as
        # exactly TYPE_MATCH and BUILTIN_MATCH. The DICT_VERSION exemption is for a
        # DICT_KEYS_MATCH only, and only for that derived type, so both halves of
        # its condition have a row here. Mixed verdicts in one call: the filter
        # returns them positionally, and a table of one verdict cannot tell a
        # misordered list from a right one.
        rows = [
            ("CONSTANT_MATCH", ("ID_MATCH",), False),
            ("TENSOR_MATCH", ("ID_MATCH",), False),
            ("DICT_KEYS_MATCH", ("ID_MATCH",), False),
            ("DICT_CONTAINS", ("DICT_VERSION",), False),
            ("DICT_KEYS_MATCH", ("DICT_VERSION",), True),
            ("CONSTANT_MATCH", (), True),
        ]
        derived = [_entry(g, None, t, derived=d) for t, d, _ in rows]
        keep = filter_fn(derived)
        self.assertEqual([(t, d, v) for (t, d, _), v in zip(rows, keep)], rows)

    def test_default_guard_filter_keeps_what_the_serializer_accepts(self):
        rows = [
            ("TENSOR_MATCH", ()),
            ("TYPE_MATCH", ()),
            # BUILTIN_MATCH is an id_match_unchecked deriving ID_MATCH; the
            # pre-check takes TYPE_MATCH and BUILTIN_MATCH on their own type,
            # before it looks at derived types, so the filter keeps it (no
            # GuardBuilder path gives a TYPE_MATCH a refused derived type). That
            # branch is not an unconditional accept: it refuses these two for a
            # local-scope type, which is what
            # test_default_guard_filter_keeps_local_type_guards_for_a_loud_refusal
            # covers; the rows are on a local source since that is where the kept
            # TYPE_MATCH the refusal needs sits, and the filter reads no scope.
            ("BUILTIN_MATCH", ("ID_MATCH",)),
        ]
        entries = [_entry(LocalSource("obj"), None, t, derived=d) for t, d in rows]
        keep = precompile_package.default_guard_filter_fn(entries)
        self.assertEqual(list(zip(rows, keep)), [(row, True) for row in rows])

    def test_default_guard_filter_through_serialize_guards(self):
        def fn(x):
            return _user_op(x) + len(x.shape)

        seen = []
        x = torch.randn(3)
        compiled = _aot_compile(fn, x, seen=seen)
        state, kept = _kept_types(compiled)
        builtins_key = state.output_graph.name_of_builtins_dict_key_in_fglobals
        # The refused guard on the global function is dropped, the builtin's
        # BUILTIN_MATCH is kept by slot, and the serialized types are exactly the
        # types of the entries the filter kept: that is the wiring under test.
        # "CLOSURE_MATCH is absent from the serialized set" could not fail on its
        # own -- serialize_guards raises on a refused type in the guards it
        # pickles, so a kept CLOSURE_MATCH would have thrown above.
        len_slot = f"G['{builtins_key}']['len']"
        verdicts = {(e.guard_type, e.name): keep for e, keep in seen}
        self.assertIs(verdicts.get(("CLOSURE_MATCH", "G['_user_op']")), False)
        self.assertIs(verdicts.get(("BUILTIN_MATCH", len_slot)), True)
        self.assertEqual(kept, {t for (t, _), keep in verdicts.items() if keep})
        data = AOTCompiledFunction.serialize(compiled).serialized_data
        loaded = AOTCompiledFunction.deserialize(data)
        self.assertEqual(loaded(x), fn(x))
        # What the dropped guard would have noticed: with _user_op rebound the
        # loaded artifact still passes its guards and serves the old graph.
        with mock.patch.object(sys.modules[__name__], "_user_op", lambda x: x - 1):
            self.assertEqual(fn(x), x)
            self.assertTrue(loaded.guard_check(x))
            self.assertEqual(loaded(x), x + 2)
        # The kept guard is live in the loaded artifact: a swapped builtin trips
        # that guard, by name, and the original passes again once restored.
        failed_on_len = rf"___check_obj_id\({re.escape(len_slot)}"
        real_len = len
        with mock.patch.object(builtins, "len", lambda *args: real_len(*args)):
            with self.assertRaisesRegex(RuntimeError, failed_on_len):
                loaded(x)
        self.assertTrue(loaded.guard_check(x))

    def test_default_guard_filter_keeps_the_pytree_registry_keys_match(self):
        # The DICT_KEYS_MATCH on SUPPORTED_NODES reaches the filter as the
        # unsaved build's DICT_VERSION and is kept; the save build serializes it
        # as a keys-match. A load rebuilds the guards against the live registry,
        # so no filter notices a change made before it; after load, the
        # DictGuardManager the registry's other kept guards build notices a
        # registration by length with or without the keys-match, and a
        # same-count change of keys is what only the keys-match notices.
        def fn_tree(x):
            return pytree.tree_flatten({"a": x, "b": x * 2})[0][1]

        def dropping(entries):
            # The control: the pre-check's type tests, which drop on the derived
            # DICT_VERSION.
            return [_pre_check_accepts(e) for e in entries]

        def load(data):
            # Module globals the kept guards read (G['pytree']) resolve against
            # the live scope, as in test_aot_compile.py's f_globals=globals() loads.
            return AOTCompiledFunction.deserialize(data, f_globals=globals())

        def deregister(cls):
            if cls in pytree.SUPPORTED_NODES:
                pytree._deregister_pytree_node(cls)

        def register(cls):
            # The Python registry only, as _deregister_pytree_node is, so the
            # class does not stay in the optree registry register_pytree_node
            # also writes. The name is only a registry key here, so the <locals>
            # in it is harmless; it must be unique because nameless registrations
            # share one slot of SERIALIZED_TYPE_TO_PYTHON_TYPE, so the first
            # deregistration deletes it and the next raises KeyError after it has
            # already dropped its class from SUPPORTED_NODES.
            name = f"{__name__}.{cls.__qualname__}"
            pytree._private_register_pytree_node(
                cls, lambda n: ([], None), lambda c, _: cls(), serialized_type_name=name
            )
            self.addCleanup(deregister, cls)

        class Node:
            pass

        class Other:
            pass

        class Extra:
            pass

        register(Node)
        seen = []
        x = torch.randn(3)
        compiled = _aot_compile(fn_tree, x, seen=seen)
        promoted = {
            (e.guard_type, e.name.rpartition(".")[2], keep)
            for e, keep in seen
            if "DICT_VERSION" in e.derived_guard_types
        }
        self.assertEqual(promoted, {("DICT_KEYS_MATCH", "SUPPORTED_NODES", True)})
        state, kept = _kept_types(compiled)
        data = AOTCompiledFunction.serialize(compiled).serialized_data
        control = _aot_compile(fn_tree, x, guard_filter_fn=dropping)
        control_state, control_kept = _kept_types(control)
        # The kept keys-match is all that tells the two artifacts' guards apart:
        # one type name and one guard more, the count because a set of type names
        # hides a second divergence in a type that keeps another instance.
        self.assertIn("DICT_KEYS_MATCH", kept)
        self.assertEqual(kept ^ control_kept, {"DICT_KEYS_MATCH"})
        n_control = len(control_state.output_graph.guards)
        self.assertEqual(len(state.output_graph.guards), n_control + 1)
        control_data = AOTCompiledFunction.serialize(control).serialized_data
        # A node registered before load is baked into either artifact's guards.
        register(Extra)
        self.assertTrue(load(data).guard_check(x))
        self.assertTrue(load(control_data).guard_check(x))
        deregister(Extra)
        loaded, loaded_control = load(data), load(control_data)
        self.assertEqual(loaded(x), fn_tree(x))
        register(Extra)
        self.assertFalse(loaded.guard_check(x))
        self.assertFalse(loaded_control.guard_check(x))
        deregister(Extra)
        # A failed check does not latch, so the failures below are the registry
        # change and not a stuck guard manager.
        self.assertTrue(loaded.guard_check(x))
        deregister(Node)
        register(Other)
        self.assertFalse(loaded.guard_check(x))
        self.assertTrue(loaded_control.guard_check(x))

    def test_default_guard_filter_keeps_local_type_guards_for_a_loud_refusal(self):
        # A local-scope class passes the filter (the TYPE_MATCH on L['obj'] is
        # kept) and serialization refuses it, naming the class. The filter keeps
        # the guard so the refusal is loud rather than an artifact that never
        # checks the type. For a plain instance the pickler refuses anyway, so
        # dropping the TYPE_MATCH only moves the refusal from serialize_guards'
        # pre-check to reducer_override; for an nn.Module of a local class it
        # does not (the instance is rebuilt as a plain torch.nn.Module), so
        # there the kept TYPE_MATCH is the only refusal: drop it and the
        # artifact ships, and serves the local class's graph to a module of
        # another class.
        def drop_type_match(entries):
            # The control: the pre-check's type tests minus every TYPE_MATCH.
            return [
                e.guard_type != "TYPE_MATCH" and _pre_check_accepts(e) for e in entries
            ]

        def drop_local_type_match(entries):
            # The narrow control: minus only the TYPE_MATCHes the pre-check
            # refuses, on a type whose qualname is not its name (guards.py's own
            # test, in TYPE_MATCH's _unserializable and in reducer_override). It
            # is narrow so that the only guard the module case below loses is the
            # one on the module's own type.
            def local_type_match(e):
                t = type(e.value)
                return e.guard_type == "TYPE_MATCH" and t.__qualname__ != t.__name__

            return [not local_type_match(e) and _pre_check_accepts(e) for e in entries]

        # Not assertRaises: it stores the exception with its traceback cleared.
        # Both paths raise through guards.py's raise_local_type_error with one
        # message; the frame that called it, serialize_guards' pre-check or
        # GuardsStatePickler.reducer_override, is what tells them apart.
        def refusal_frames(regex, guard_filter_fn, fn, *args):
            try:
                _aot_compile(fn, *args, guard_filter_fn=guard_filter_fn)
            except PackageError as e:
                self.assertRegex(str(e), regex)
                return {f.name for f in traceback.extract_tb(e.__traceback__)}
            self.fail(f"{fn.__name__} serialized, expected {regex}")

        class Local:
            n = 1

        def fn2(x, obj):
            return x + obj.n

        x = torch.randn(3)
        refused = "Local'> cannot be saved.*defined in local scope"
        frames = refusal_frames(refused, None, fn2, x, Local())
        self.assertIn("raise_local_type_error", frames)
        self.assertNotIn("reducer_override", frames)
        frames = refusal_frames(refused, drop_type_match, fn2, x, Local())
        self.assertIn("reducer_override", frames)

        class LocalModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        def fn3(x, mod):
            return mod(x)

        refused_module = "LocalModule'> cannot be saved.*defined in local scope"
        frames = refusal_frames(refused_module, None, fn3, x, LocalModule())
        self.assertIn("raise_local_type_error", frames)
        self.assertNotIn("reducer_override", frames)

        class Other(torch.nn.Module):
            def forward(self, x):
                return x - 1

        compiled = _aot_compile(
            fn3, x, LocalModule(), guard_filter_fn=drop_local_type_match
        )
        data = AOTCompiledFunction.serialize(compiled).serialized_data
        loaded = AOTCompiledFunction.deserialize(data)
        self.assertTrue(loaded.guard_check(x, Other()))
        self.assertEqual(loaded(x, Other()), x + 1)

    def _clear_root_caches(self):
        # The roots are cached for the process: clear them now, so a test sees
        # its own interpreter or patches rather than what a sibling left, and
        # again at exit, so a patched answer outlives no test.
        for name in ("_stdlib_roots", "_install_roots", "_torch_roots"):
            roots = getattr(precompile_package, name)
            roots.cache_clear()
            self.addCleanup(roots.cache_clear)

    def test_roots_locate_the_stdlib_install_and_torch_dirs(self):
        self._clear_root_caches()
        stdlib = precompile_package._stdlib_roots()
        install = precompile_package._install_roots()
        torch_roots = precompile_package._torch_roots()
        self.assertTrue(stdlib and install and torch_roots)
        norm, within = precompile_package._norm, precompile_package._within
        # The sets nest one way only: purelib sits inside a stdlib root (conda)
        # or platstdlib (venv; platlib on a lib64 build) and must survive the
        # exclusion, while on Windows getsitepackages() names the prefix the
        # stdlib sits under, which must not, or the install-root-wins rule would
        # read the whole stdlib as third party. No layout sysconfig and site can
        # produce puts an install root ON a stdlib root either, which is why the
        # loop below can test containment; the code permits that case on purpose
        # (test_install_roots_keep_a_directory_that_is_a_stdlib_root).
        self.assertIn(norm(sysconfig.get_paths()["purelib"]), install)
        for root in stdlib:
            self.assertFalse(within(root, install), root)
        self.assertTrue(within(norm(os.__file__), stdlib))
        self.assertIn(norm(os.path.dirname(torch.__file__)), torch_roots)
        root = os.path.join(os.sep, "a", "b")
        self.assertTrue(within(root, (root,)))
        self.assertTrue(within(os.path.join(root, "c"), (root,)))
        self.assertFalse(within(root + "c", (root,)))

    def test_norm_resolves_a_symlinked_prefix_into_its_root(self):
        # CPython absolutizes a __file__ but never resolves symlinks, so a
        # module imported through a linked prefix lies under the real root only
        # after realpath; abspath alone would put it outside every root.
        norm, within = precompile_package._norm, precompile_package._within
        with tempfile.TemporaryDirectory() as tmp:
            real, link = os.path.join(tmp, "real"), os.path.join(tmp, "link")
            os.mkdir(real)
            try:
                os.symlink(real, link)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable")
            file = os.path.join(link, "mod.py")
            self.assertEqual(norm(file), os.path.join(norm(real), "mod.py"))
            self.assertTrue(within(norm(file), (norm(real),)))

    def test_stdlib_roots_follow_a_symlink_farm_into_the_store(self):
        # The other symlink shape: the stdlib directory is real and each file in
        # it is a link into a per-package store, which is what python -m venv
        # makes of a Nix, Guix or Spack profile (venv records the unresolved
        # sys._base_executable as home, so the child's base_prefix, sysconfig's
        # stdlib and sys._stdlib_dir all stay on the farm). realpath of any file
        # lands in the store, which none of those sources name.
        norm, within = precompile_package._norm, precompile_package._within
        with tempfile.TemporaryDirectory() as tmp:
            store = os.path.join(tmp, "store", "lib", "python3.12")
            farm = os.path.join(tmp, "profile", "lib", "python3.12")
            venv = os.path.join(tmp, "venv", "lib", "python3.12")
            os.makedirs(store)
            os.makedirs(farm)
            open(os.path.join(store, "os.py"), "w").close()
            farm_os = os.path.join(farm, "os.py")
            try:
                os.symlink(os.path.join(store, "os.py"), farm_os)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable")
            paths = dict(sysconfig.get_paths())
            paths.update(stdlib=farm, platstdlib=venv)
            with (
                mock.patch.object(os, "__file__", farm_os),
                mock.patch.object(sys, "_stdlib_dir", farm, create=True),
                mock.patch.object(sysconfig, "get_paths", return_value=paths),
            ):
                self._clear_root_caches()
                stdlib = precompile_package._stdlib_roots()
            self.assertIn(norm(farm), stdlib)
            self.assertIn(norm(store), stdlib)
            self.assertTrue(within(norm(farm_os), stdlib))

    def test_stdlib_roots_take_the_archive_a_zipped_os_comes_from(self):
        # With the stdlib in a zip (py2exe, cx_Freeze, the Windows embeddable
        # build) os.__file__ is <archive>/os.py and the root is the archive
        # itself, a file rather than a directory: a third party bundled into it
        # is waived with the stdlib, which the docstring records, and an isdir
        # gate on the root would silently drop it.
        norm, within = precompile_package._norm, precompile_package._within
        with tempfile.TemporaryDirectory() as tmp:
            archive = os.path.join(tmp, "library.zip")
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("os.py", "")
                zf.writestr("queue.py", "")
            with mock.patch.object(os, "__file__", os.path.join(archive, "os.py")):
                self._clear_root_caches()
                stdlib = precompile_package._stdlib_roots()
            self.assertIn(norm(archive), stdlib)
            self.assertTrue(within(norm(os.path.join(archive, "queue.py")), stdlib))

    def test_stdlib_roots_take_the_windows_dlls_dir(self):
        # Reachable only on a Windows runner otherwise. sysconfig's first init
        # imports _sysconfigdata_*_{sys.platform}_*, so prime it unpatched. The
        # root is taken off base_prefix rather than prefix, and base_prefix is
        # fabricated here so that reading prefix instead fails: a Windows venv
        # has no DLLs directory of its own, so under prefix the stdlib's C
        # extensions (_socket, select, _ssl) would sit under no stdlib root at
        # all and every guard they own would stop being library-owned. Only
        # base_prefix is patched, never prefix: sysconfig re-initializes its
        # config vars whenever sys.prefix moves away from the value it cached
        # (GH-126789), which would import a _sysconfigdata for the patched
        # platform however early this test primed it.
        stdlib_roots, norm = precompile_package._stdlib_roots, precompile_package._norm
        sysconfig.get_paths()
        # abspath here and in the fabricated roots below: drive-qualified on
        # Windows, where 3.13's isabs rejects a bare \x and the code drops it.
        base = os.path.abspath(os.path.join(os.sep, "winbase"))
        with (
            mock.patch.object(sys, "platform", "win32"),
            mock.patch.object(sys, "base_prefix", base),
        ):
            self._clear_root_caches()
            roots = stdlib_roots()
        self.assertIn(norm(os.path.join(base, "DLLs")), roots)
        self.assertNotIn(norm(os.path.join(sys.prefix, "DLLs")), roots)

    def test_stdlib_roots_take_each_source_in_a_venv_layout(self):
        # Here every source names the one <prefix>/lib/pythonX.Y, so dropping
        # any of them leaves the roots unchanged. Replay a venv, where stdlib
        # stays at the base prefix while platstdlib is the venv's own lib
        # directory with purelib nested under it, and point the frozen-stdlib
        # hint elsewhere still, so each source is the only one for its root.
        # sys.platform is patched off win32 as well: the DLLs root is the one
        # input that is not a path, and a Windows runner would add it to the
        # exact tuple below.
        base = os.path.abspath(os.path.join(os.sep, "base", "lib", "python3.12"))
        venv = os.path.abspath(os.path.join(os.sep, "venv", "lib", "python3.12"))
        frozen = os.path.abspath(os.path.join(os.sep, "frozen", "stdlib"))
        purelib = os.path.join(venv, "site-packages")
        paths = dict(sysconfig.get_paths())
        paths.update(stdlib=base, platstdlib=venv, purelib=purelib, platlib=purelib)
        stdlib_roots = precompile_package._stdlib_roots
        install_roots = precompile_package._install_roots
        norm, within = precompile_package._norm, precompile_package._within
        with (
            mock.patch.object(sysconfig, "get_paths", return_value=paths),
            mock.patch.object(sys, "_stdlib_dir", frozen, create=True),
            mock.patch.object(sys, "platform", "linux"),
        ):
            self._clear_root_caches()
            stdlib, install = stdlib_roots(), install_roots()
            # A frozen os has no __file__ (None stands in for the missing
            # attribute): the other three sources are then all there is, in
            # sorted order, which _classify_file reads as outermost first.
            with mock.patch.object(os, "__file__", None):
                stdlib_roots.cache_clear()
                no_os = stdlib_roots()
            # A relative os.__file__ is dropped the same way rather than
            # resolved against the process cwd, which would make the caller's
            # own project a stdlib root and waive a local module whose name
            # shadows a stdlib one.
            with mock.patch.object(os, "__file__", os.path.join("rel", "os.py")):
                stdlib_roots.cache_clear()
                relative_os = stdlib_roots()
        for root in (base, venv, frozen, os.path.dirname(norm(os.__file__))):
            self.assertIn(norm(root), stdlib)
        self.assertEqual(no_os, tuple(sorted(norm(r) for r in (base, venv, frozen))))
        self.assertEqual(relative_os, no_os)
        # The nesting the exclusion exists for: purelib is under a stdlib root
        # and is an install root all the same.
        self.assertTrue(within(norm(purelib), stdlib))
        self.assertIn(norm(purelib), install)

    def test_install_roots_take_purelib_and_platlib_on_a_lib64_build(self):
        # sysconfig's prefix and venv schemes hard-code lib in purelib but
        # interpolate platlibdir into platlib and the stdlib keys, so on a system
        # interpreter built --with-platlibdir=lib64 (Fedora, RHEL, openSUSE) it
        # is platlib that nests inside the stdlib root and purelib that does
        # not; each key is a root of its own. A venv links lib64 -> lib. Every
        # stdlib source is patched onto a made-up prefix: a Debian os.py lives
        # at /usr/lib/python3.12, the parent of a /usr/lib purelib, and a
        # merged-lib host resolves /usr/lib64 onto /usr/lib; sys.platform is
        # patched off win32, or a Windows runner's DLLs root joins the stdlib.
        prefix = os.path.abspath(os.path.join(os.sep, "lib64build"))
        stdlib = os.path.join(prefix, "lib64", "python3.12")
        purelib = os.path.join(prefix, "lib", "python3.12", "site-packages")
        platlib = os.path.join(stdlib, "site-packages")
        # site lists a string prefix of the stdlib directory that is not its
        # parent, which the exclusion must keep.
        sibling = os.path.join(prefix, "lib64", "python3")
        paths = dict(sysconfig.get_paths())
        paths.update(stdlib=stdlib, platstdlib=stdlib, purelib=purelib, platlib=platlib)
        norm, within = precompile_package._norm, precompile_package._within
        with (
            mock.patch.object(sysconfig, "get_paths", return_value=paths),
            mock.patch.object(os, "__file__", os.path.join(stdlib, "os.py")),
            mock.patch.object(sys, "_stdlib_dir", stdlib, create=True),
            mock.patch.object(sys, "platform", "linux"),
            mock.patch.object(site, "getsitepackages", return_value=[sibling]),
        ):
            self._clear_root_caches()
            stdlib_found = precompile_package._stdlib_roots()
            install = precompile_package._install_roots()
        self.assertEqual(stdlib_found, (norm(stdlib),))
        self.assertTrue(within(norm(platlib), stdlib_found))
        self.assertFalse(within(norm(purelib), stdlib_found))
        for root in (platlib, purelib, sibling):
            self.assertIn(norm(root), install)

    def test_install_roots_drop_a_directory_the_stdlib_sits_under(self):
        # On Windows getsitepackages() lists the bare prefix, which the whole
        # stdlib sits under; replay that shape on every platform. The listed
        # directory is the parent of a stdlib root, which is a strict ancestor of
        # it whatever the layout, rather than sys.prefix, whose shape is the
        # host's: a --prefix=/ build or a Windows embeddable unpacked at a drive
        # root leaves the prefix at a filesystem root, which this exclusion
        # cannot drop because `r + os.sep` is then a doubled separator no root
        # starts with. The code degrades harmlessly there, a root that swallows
        # everything being one that _within matches nothing against, so only an
        # assertion on sys.prefix would fail. site.getsitepackages is replaced
        # rather than extended for the same reason it is read through getattr in
        # a try: the old-virtualenv site.py this survives does not define it.
        self._clear_root_caches()
        above = os.path.dirname(precompile_package._stdlib_roots()[0])
        with mock.patch.object(site, "getsitepackages", return_value=[above]):
            self._clear_root_caches()
            stdlib = precompile_package._stdlib_roots()
            install = precompile_package._install_roots()
        norm, within = precompile_package._norm, precompile_package._within
        self.assertNotIn(norm(above), install)
        # Containment, not equality: no layout sysconfig and site can produce
        # puts an install root ON a stdlib root, so the stronger property holds
        # here; the equal shape the code allows on purpose is the next test.
        for root in stdlib:
            self.assertFalse(within(root, install), root)
        self.assertIn(norm(sysconfig.get_paths()["purelib"]), install)

    def test_install_roots_keep_a_directory_that_is_a_stdlib_root(self):
        # The boundary of that exclusion, and the one asymmetry in it: only a
        # strict ancestor of a stdlib root is dropped, so a directory that IS
        # one stays an install root. Dropping it instead would leave a third
        # party installed straight into the stdlib directory under no install
        # root at all and waived with the stdlib, while keeping it only reads
        # the stdlib as third party, which over-refuses. No scheme
        # sysconfig.get_paths() returns has this shape (posix_home is the only
        # one with purelib at stdlib, and no preferred scheme is home), so the
        # layout is fabricated: every stdlib source and every install source
        # names the one directory.
        eq = os.path.abspath(os.path.join(os.sep, "eqbuild", "lib", "python3.12"))
        paths = dict(sysconfig.get_paths())
        paths.update(stdlib=eq, platstdlib=eq, purelib=eq, platlib=eq)
        norm, within = precompile_package._norm, precompile_package._within
        with (
            mock.patch.object(sysconfig, "get_paths", return_value=paths),
            mock.patch.object(os, "__file__", os.path.join(eq, "os.py")),
            mock.patch.object(sys, "_stdlib_dir", eq, create=True),
            mock.patch.object(sys, "platform", "linux"),
            mock.patch.object(site, "getsitepackages", return_value=[]),
            mock.patch.object(site, "getusersitepackages", return_value=None),
        ):
            self._clear_root_caches()
            stdlib = precompile_package._stdlib_roots()
            install = precompile_package._install_roots()
        self.assertEqual(stdlib, (norm(eq),))
        self.assertEqual(install, (norm(eq),))
        self.assertTrue(within(norm(os.path.join(eq, "third_party.py")), install))

    def test_torch_roots_follow_a_symlink_farm_into_the_store(self):
        # The same farm for torch: torch.__path__ and this file's directory
        # both name the farm, whose subdirectories are real while every module
        # in them is a link into the store, so a root resolved as a directory
        # stays on the farm and matches no torch file a consumer resolves.
        torch_roots, norm = precompile_package._torch_roots, precompile_package._norm
        with tempfile.TemporaryDirectory() as tmp:
            store = os.path.join(tmp, "store", "torch")
            farm = os.path.join(tmp, "farm", "torch")
            for rel in (("_dynamo", "precompile_package.py"), ("nn", "functional.py")):
                os.makedirs(os.path.join(store, rel[0]), exist_ok=True)
                os.makedirs(os.path.join(farm, rel[0]), exist_ok=True)
                open(os.path.join(store, *rel), "w").close()
                try:
                    os.symlink(os.path.join(store, *rel), os.path.join(farm, *rel))
                except (OSError, NotImplementedError):
                    self.skipTest("symlinks unavailable")
            own = os.path.join(farm, "_dynamo", "precompile_package.py")
            stub = types.SimpleNamespace(__path__=[farm])
            with (
                mock.patch.object(precompile_package, "__file__", own),
                mock.patch.dict(sys.modules, {"torch": stub}),
            ):
                self._clear_root_caches()
                roots = torch_roots()
            self.assertEqual(roots, tuple(sorted((norm(farm), norm(store)))))
            functional = norm(os.path.join(farm, "nn", "functional.py"))
            self.assertTrue(precompile_package._within(functional, roots))

    def test_torch_roots_drop_a_link_that_flattens_this_files_depth(self):
        # The resolved spelling is two levels up from where this file resolves,
        # which is the torch directory only while the link keeps the file at
        # <root>/_dynamo/<file>. A link to a flat patch directory would make its
        # grandparent, which holds unrelated code, a torch root and waive every
        # dropped guard over that code; the spelling is skipped instead, so the
        # patched file lies under no torch root and its guards are kept.
        torch_roots, norm = precompile_package._torch_roots, precompile_package._norm
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "exp")
            farm = os.path.join(base, "site-packages", "torch")
            for sub in ((farm, "_dynamo"), (base, "patchdir"), (base, "my_project")):
                os.makedirs(os.path.join(*sub))
            target = os.path.join(base, "patchdir", "precompile_package.py")
            open(target, "w").close()
            own = os.path.join(farm, "_dynamo", "precompile_package.py")
            try:
                os.symlink(target, own)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable")
            mymod = os.path.join(base, "my_project", "mymod.py")
            open(mymod, "w").close()
            stub = types.SimpleNamespace(__path__=[farm])
            with (
                mock.patch.object(precompile_package, "__file__", own),
                mock.patch.dict(sys.modules, {"torch": stub}),
            ):
                self._clear_root_caches()
                roots = torch_roots()
            self.assertEqual(roots, (norm(farm),))
            self.assertFalse(precompile_package._within(norm(mymod), roots))
            self.assertFalse(precompile_package._within(norm(own), roots))

    def test_roots_drop_a_relative_interpreter_path(self):
        # A venv whose pyvenv.cfg home is relative, or a relative PYTHONHOME,
        # leaves sys.base_prefix, sys._stdlib_dir and every sysconfig path
        # relative while os.__file__ alone is absolute (site.abs_paths() at
        # startup), and a relative PYTHONUSERBASE leaves the user site so; all
        # measured on 3.12 with nothing patched. Resolved, each would sit at the
        # process cwd of the first call and stay cached there, so every one is
        # dropped like a relative os.__file__: the directory os resolves into is
        # the only stdlib root left, and there is no install root at all.
        # sys.platform is patched onto win32 so the DLLs join is exercised too.
        home = os.path.join("relhome", "lib", "python3.12")
        purelib = os.path.join(home, "site-packages")
        user_site = os.path.join("reluser", "lib", "python3.12", "site-packages")
        paths = dict(sysconfig.get_paths())
        paths.update(stdlib=home, platstdlib=home, purelib=purelib, platlib=purelib)
        with (
            mock.patch.object(sysconfig, "get_paths", return_value=paths),
            mock.patch.object(sys, "_stdlib_dir", home, create=True),
            mock.patch.object(sys, "base_prefix", "relhome"),
            mock.patch.object(sys, "platform", "win32"),
            mock.patch.object(site, "getsitepackages", return_value=[purelib]),
            mock.patch.object(site, "getusersitepackages", return_value=user_site),
        ):
            self._clear_root_caches()
            stdlib = precompile_package._stdlib_roots()
            install = precompile_package._install_roots()
        norm = precompile_package._norm
        self.assertEqual(stdlib, (os.path.dirname(norm(os.__file__)),))
        self.assertEqual(install, ())

    def test_install_roots_skip_a_site_accessor_that_raises(self):
        # A site.py that cannot answer is skipped, not propagated: this runs
        # inside save()'s lint, which must not abort the capture. sysconfig's
        # purelib is read outside the try and the other accessor is still
        # consulted, so what a failing accessor loses is only its own extras.
        user_site = os.path.abspath(os.path.join(os.sep, "elsewhere", "user-site"))
        with (
            mock.patch.object(site, "getsitepackages", side_effect=RuntimeError),
            mock.patch.object(site, "getusersitepackages", return_value=user_site),
        ):
            self._clear_root_caches()
            install = precompile_package._install_roots()
        norm = precompile_package._norm
        self.assertIn(norm(sysconfig.get_paths()["purelib"]), install)
        self.assertIn(norm(user_site), install)

    def test_install_roots_take_only_the_strings_a_site_accessor_lists(self):
        # A non-string entry is dropped rather than handed to realpath outside
        # the try, and the None getusersitepackages returns where there is no
        # home directory (WASI) is skipped like a raise.
        extra = os.path.abspath(os.path.join(os.sep, "elsewhere", "site"))
        with (
            mock.patch.object(site, "getsitepackages", return_value=[None, extra]),
            mock.patch.object(site, "getusersitepackages", return_value=None),
        ):
            self._clear_root_caches()
            install = precompile_package._install_roots()
        norm = precompile_package._norm
        self.assertIn(norm(extra), install)
        self.assertIn(norm(sysconfig.get_paths()["purelib"]), install)

    def test_torch_roots_trust_torch_path_only_when_this_file_is_in_it(self):
        torch_roots, norm = precompile_package._torch_roots, precompile_package._norm
        own = norm(os.path.dirname(torch.__file__))
        # abspath: drive-qualified on Windows, where 3.13's isabs rejects \x
        bogus = os.path.abspath(os.path.join(os.sep, "elsewhere", "torch"))
        stub = types.SimpleNamespace(__path__=[None, bogus])
        # The package directory has two spellings, both roots: the one torch's
        # __path__ names and two levels up from where this file resolves. They
        # are one directory on a plain checkout and two in a per-file symlink
        # farm (a Buck link-tree), so the expectation carries both.
        resolved = norm(precompile_package.__file__)
        anchored = {own, os.path.dirname(os.path.dirname(resolved))}
        self._clear_root_caches()
        with mock.patch.dict(sys.modules, {"torch": stub}):
            # A substituted torch's __path__ is ignored until it lists the torch
            # package directory this file sits under; then every absolute string
            # entry is adopted, the bogus one included, which an editable build
            # relies on, and a relative one, which realpath would resolve into
            # the process cwd, is not. A __path__ that is no sequence at all is
            # ignored as well.
            self.assertEqual(set(torch_roots()), anchored)
            stub.__path__ += [os.path.dirname(torch.__file__), "", "relative"]
            torch_roots.cache_clear()
            self.assertEqual(set(torch_roots()), anchored | {norm(bogus)})
            stub.__path__ = None
            torch_roots.cache_clear()
            self.assertEqual(set(torch_roots()), anchored)
        with mock.patch.object(precompile_package, "__file__", None):
            torch_roots.cache_clear()
            self.assertEqual(torch_roots(), ())  # frozen: no directory to anchor to

    def test_classify_file_places_a_path_by_the_roots_it_lies_under(self):
        classify, norm = precompile_package._classify_file, precompile_package._norm
        patch_module = functools.partial(mock.patch.object, precompile_package)
        paths = sysconfig.get_paths()
        stdlib_root = paths["stdlib"]
        stdlib_graphlib = os.path.join(stdlib_root, "graphlib.py")
        torch_in_stdlib = os.path.join(stdlib_root, "torch", "__init__.py")
        # The verdict is cached per (file, flag), and files below are judged
        # under more than one root set.
        self._clear_root_caches()
        self.addCleanup(classify.cache_clear)
        classify.cache_clear()
        self.assertIs(classify(stdlib_graphlib, True), True)
        # The torch arm reads the torch roots, not the stdlib ones, and the
        # stdlib dir itself is never a pip target.
        self.assertIs(classify(torch.__file__, False), True, "torch arm")
        self.assertIs(classify(torch.__file__, True), False, "stdlib arm")
        self.assertIs(classify(torch_in_stdlib, False), False)
        # The install-root exclusion is the stdlib arm's alone: torch ships
        # platform wheels (Root-Is-Purelib: false), so a pip-installed torch
        # lies under platlib. In a wheel install that file IS torch.__file__,
        # judged already by the torch-arm row, so the cache is cleared first.
        platlib_torch = os.path.join(paths["platlib"], "torch")
        installed_torch = norm(os.path.join(platlib_torch, "__init__.py"))
        install_roots = precompile_package._install_roots()
        self.assertTrue(precompile_package._within(installed_torch, install_roots))
        classify.cache_clear()
        with patch_module("_torch_roots", return_value=(norm(platlib_torch),)):
            self.assertIs(classify(installed_torch, False), True)
        # A frozen app bundles the stdlib and every third party under one root,
        # which PyInstaller also names in sys._stdlib_dir and in the __file__ it
        # gives the CPython-frozen modules, so no path is evidence for the
        # stdlib arm; the torch arm keeps its root, torch's own package
        # directory, and still tells torch from the rest of the bundle. The flag
        # is True under PyInstaller and cx_Freeze and a string under py2exe.
        for flag in (True, "console_exe"):
            classify.cache_clear()
            with mock.patch.object(sys, "frozen", flag, create=True):
                self.assertIsNone(classify(stdlib_graphlib, True), flag)
                self.assertIs(classify(torch.__file__, False), True, flag)
                self.assertIs(classify(stdlib_graphlib, False), False, flag)
        classify.cache_clear()
        self.assertIs(classify(stdlib_graphlib, True), True, "after the clear")
        # Evidence in neither direction: a relative path would resolve against
        # a cwd it was not recorded under, and no file has a NUL in its name
        # (posixpath.realpath raises ValueError on one; from 3.11.5/3.12 on,
        # gh-106242, ntpath.realpath returns the path unresolved instead, and
        # the Windows block below replays what _within would make of that).
        self.assertIsNone(classify("graphlib.py", True))
        self.assertIsNone(classify("graphlib.py", False))
        self.assertIsNone(classify(os.path.join(stdlib_root, "graph\x00lib.py"), True))
        # Before 3.13 ntpath.isabs accepts a driveless path (its LEGACY BUG),
        # which realpath resolves against the current drive; replay the Windows
        # path module here so the gates are pinned on every platform. The root
        # finders do not work under this patch, so they are patched to a Windows
        # stdlib. The first two rows return at a gate (without the NUL gate the
        # unresolved path lies under that root and is waived); the last one gets
        # as far as _INSTALL_DIR_NAMES, which is what needs the Windows
        # separator: _norm yields backslashes, so splitting the part below the
        # root on "/" would leave one component that names nothing.
        import ntpath

        nul_under_lib = "C:\\Python312\\Lib\\graphlib.py\x00C:\\evil\\evil.py"
        win_installed = r"c:\python312\lib\vendored\site-packages\graphlib.py"
        with (
            mock.patch.object(sys, "platform", "win32"),
            mock.patch.object(os, "path", ntpath),
            mock.patch.object(os, "sep", "\\"),
            patch_module("_install_roots", return_value=()),
            patch_module("_stdlib_roots", return_value=(r"c:\python312\lib",)),
        ):
            self.assertIsNone(classify(r"\Lib\graphlib.py", True))
            self.assertIsNone(classify(nul_under_lib, True))
            self.assertIs(classify(win_installed, True), False)
        # purelib nests inside stdlib (conda) or platstdlib (venv), so the
        # install-root exclusion is what refuses an installed file; with no
        # install root known, _INSTALL_DIR_NAMES still does, at any depth below
        # the root and case folded since normcase is the identity on posix and a
        # macOS filesystem is not case-sensitive. Under no stdlib root at all
        # the file is elsewhere. These rows are spelled through a directory that
        # cannot pre-exist, so that _norm cannot resolve them out of the stdlib
        # root and into the terminal False instead.
        nested = os.path.join(stdlib_root, "vendored", "graphlib", "__init__.py")
        with patch_module("_install_roots", return_value=()):
            for dir_name in ("site-packages", "dist-packages", "Site-Packages"):
                installed = os.path.join(stdlib_root, "vendored", dir_name, "g.py")
                self.assertIs(classify(installed, True), False, dir_name)
            self.assertIs(classify(nested, True), True, "nothing installed there")
            with patch_module("_stdlib_roots", return_value=()):
                self.assertIs(classify(os.path.join(stdlib_root, "os.py"), True), False)
        classify.cache_clear()
        roots = (norm(os.path.join(stdlib_root, "vendored")),)
        with patch_module("_install_roots", return_value=roots):
            self.assertIs(classify(nested, True), False, "vendored is an install root")
        # _INSTALL_DIR_NAMES is matched below the stdlib root the file is under,
        # so an interpreter bundled inside another environment's site-packages
        # keeps its own stdlib. Spelled on a tree of its own: a real
        # <stdlib>/site-packages that is a symlink out of the stdlib (a
        # relocated site-packages, a Windows junction) would resolve the bundled
        # root out from under the outer one and stop the two nesting.
        with tempfile.TemporaryDirectory() as tmp:
            bundled = os.path.join(tmp, "site-packages", "runtime", "lib")
            in_bundled = os.path.join(bundled, "graphlib.py")
            below = os.path.join(bundled, "site-packages", "graphlib.py")
            with (
                patch_module("_install_roots", return_value=()),
                patch_module("_stdlib_roots", return_value=(norm(bundled),)),
            ):
                self.assertIs(classify(in_bundled, True), True)
                self.assertIs(classify(below, True), False)
            # Of nested stdlib roots the outermost is matched, so the part below
            # it is the longest and the check the strictest.
            classify.cache_clear()
            with (
                patch_module("_install_roots", return_value=()),
                patch_module("_stdlib_roots", return_value=(norm(tmp), norm(bundled))),
            ):
                self.assertIs(classify(in_bundled, True), False)
            # The path is resolved before it is judged, which is what the root
            # finders' own docstrings rely on: unresolved, this one names an
            # install directory, and every venv or symlink-farm __file__ would
            # be judged by where its link sits rather than where the file is.
            classify.cache_clear()
            with (
                patch_module("_install_roots", return_value=()),
                patch_module("_stdlib_roots", return_value=(norm(tmp),)),
            ):
                unresolved = os.path.join(tmp, "site-packages", os.pardir, "g.py")
                self.assertIs(classify(unresolved, True), True)

    def test_located_reads_the_file_from_the_module_dict(self):
        located, machinery = precompile_package._located, importlib.machinery
        builtin, frozen = machinery.BuiltinImporter, machinery.FrozenImporter
        stdlib_root = sysconfig.get_paths()["stdlib"]
        installed = os.path.join(stdlib_root, "site-packages", "graphlib.py")
        self._clear_root_caches()
        self.addCleanup(precompile_package._classify_file.cache_clear)
        precompile_package._classify_file.cache_clear()
        graphlib = types.ModuleType("graphlib")
        graphlib.__file__ = installed
        self.assertIs(located(graphlib, "graphlib", True), False)
        graphlib.__file__ = os.path.join(stdlib_root, "graphlib.py")
        self.assertIs(located(graphlib, "graphlib", True), True)
        # The flag reaches the file arm: the torch shape, a torch module on disk.
        on_disk = types.ModuleType("torch")
        on_disk.__file__ = torch.__file__
        self.assertIs(located(on_disk, "torch", False), True)
        self.assertIs(located(on_disk, "torch", True), False)

        # Only the module dict is read. A class attribute is not in it (torch.ops
        # is a ModuleType subclass whose __file__ is the class's "_ops.py"), and
        # getattr would run a PEP 562 module __getattr__, user code a lint must
        # not run: ModuleType seeds __spec__ and __loader__ into the dict, so
        # both are deleted for that read to be reachable at all.
        class Shadow(types.ModuleType):
            __file__ = installed

        self.assertIsNone(located(Shadow("graphlib"), "graphlib", True))
        self.assertNotIn("__file__", vars(torch.ops))
        self.assertIsNone(located(torch.ops, "torch.ops", False))
        raising = types.ModuleType("graphlib")
        raising.__getattr__ = mock.Mock(side_effect=RuntimeError("no such attribute"))
        del raising.__spec__, raising.__loader__
        self.assertIsNone(located(raising, "graphlib", True))
        raising.__getattr__.assert_not_called()
        # importlib.util.LazyLoader leaves a _LazyModule whose __getattribute__
        # executes the module body on any attribute read, __dict__ included;
        # the dict is read through object, so the body stays unrun.
        eager, util = mock.Mock(**{"create_module.return_value": None}), importlib.util
        wrapped = util.LazyLoader(eager)
        lazy_spec = util.spec_from_file_location("graphlib", installed, loader=wrapped)
        lazy = util.module_from_spec(lazy_spec)
        wrapped.exec_module(lazy)
        self.assertIs(located(lazy, "graphlib", True), False)
        eager.exec_module.assert_not_called()
        self.assertIsNone(located(types.ModuleType("graphlib"), "graphlib", True))
        # sys.modules can hold any object: object's __dict__ read raises
        # AttributeError on a slotted proxy (its __getattr__ is not consulted),
        # spec.loader is user code on a hand-rolled spec, and a __file__ that is
        # not a string would raise from isabs.
        odd = types.ModuleType("graphlib")
        odd.__file__ = 42
        self.assertIsNone(located(odd, "graphlib", True))

        class Proxy:
            __slots__ = ()

            def __getattr__(self, attr):
                raise RuntimeError(attr)

        self.assertIsNone(located(Proxy(), "graphlib", True))
        odd.__spec__ = Proxy()
        self.assertIsNone(located(odd, "graphlib", True))
        odd.__file__ = os.path.join(stdlib_root, "graphlib.py")
        self.assertIs(located(odd, "graphlib", True), True)  # the loader is not needed
        # A __loader__ in the dict is read first, so the spec is never touched.
        spec_poisoned = types.ModuleType("sys")
        spec_poisoned.__spec__ = Proxy()
        spec_poisoned.__loader__ = builtin
        self.assertIs(located(spec_poisoned, "sys", True), True)
        # A placed __file__ decides before the loader is read. The real
        # importlib._bootstrap has both: importlib/__init__.py gives it a stdlib
        # __file__, its __loader__ is FrozenImporter, and the frozen table knows
        # it only as _frozen_importlib. The converse is an installed file under
        # a built-in name.
        bootstrap = types.ModuleType("importlib._bootstrap")
        bootstrap.__file__ = os.path.join(stdlib_root, "importlib", "_bootstrap.py")
        bootstrap.__loader__ = frozen
        self.assertIsNone(frozen.find_spec("importlib._bootstrap"))
        self.assertIs(located(bootstrap, "importlib._bootstrap", True), True)
        shadow_sys = types.ModuleType("sys")
        shadow_sys.__file__ = installed
        shadow_sys.__loader__ = builtin
        self.assertIs(located(shadow_sys, "sys", True), False)
        # A __file__ that cannot be placed (not a string, or relative) is no
        # evidence, so the loader is still read.
        unplaced = (("sys", 42, builtin), ("zipimport", "zipimport.py", frozen))
        for name, file, loader in unplaced:
            with_loader = types.ModuleType(name)
            with_loader.__file__ = file
            with_loader.__loader__ = loader
            self.assertIs(located(with_loader, name, True), True, name)
        # In a frozen app the file arm is no evidence (see _classify_file), so
        # a module CPython itself freezes keeps its waiver through the loader.
        # zipimport, not os: os is only frozen from 3.11 on (gh-45020).
        frozen_zip = types.ModuleType("zipimport")
        frozen_zip.__file__ = os.path.join(stdlib_root, "zipimport.py")
        frozen_zip.__loader__ = frozen
        with mock.patch.object(sys, "frozen", True, create=True):
            self.assertIs(located(frozen_zip, "zipimport", True), True)
        precompile_package._classify_file.cache_clear()
        # Built in or frozen, the table is keyed on the full dotted name, and
        # looked up under the caller's name, not the module's __name__ (the
        # sys.modules entry for os.path is posixpath).
        self.assertIs(located(sys, "sys", True), True)
        self.assertIs(located(sys, "sys.sub", True), False)
        zipimport = types.ModuleType("zipimport")
        zipimport.__loader__ = frozen
        self.assertIs(located(zipimport, "zipimport.sub", True), False)
        sub = types.ModuleType("sys.sub")
        sub.__spec__ = machinery.ModuleSpec("sys.sub", builtin, origin="built-in")
        self.assertIs(located(sub, "sys.sub", True), False)
        frozen_rows = {"zipimport": True, "zipimport.sub": False, "graphlib": False}
        for name, expected in frozen_rows.items():
            by_spec = types.ModuleType(name)
            by_spec.__spec__ = machinery.ModuleSpec(name, frozen, origin="frozen")
            self.assertIs(located(by_spec, name, True), expected, name)
        # find_spec raises ImportError on an excluded or invalid frozen table
        # entry; a raise anywhere in _located is None, never a lint's error.
        with mock.patch.object(frozen, "find_spec", side_effect=ImportError):
            self.assertIsNone(located(zipimport, "zipimport", True))
        # Both arms compare the loader by identity, so a __loader__ whose __eq__
        # answers true for anything (mock.ANY is a stdlib object of exactly that
        # shape) takes neither waiver, and its __eq__ never runs at all.
        lying_eq = types.ModuleType("sys")
        lying_eq.__loader__ = mock.ANY
        self.assertIsNone(located(lying_eq, "sys", True))
        # __loader__ alone in the dict, with no spec, is the same evidence, and
        # both arms answer under either flag: the torch shape is an embedding
        # that registers torch._C through PyImport_AppendInittab.
        for name, loader in (("sys", builtin), ("zipimport", frozen)):
            by_loader = types.ModuleType(name)
            by_loader.__loader__ = loader
            for stdlib in (True, False):
                self.assertIs(located(by_loader, name, stdlib), True, (name, stdlib))
        self.assertNotIn("torch._C", sys.builtin_module_names)
        inittab = (*sys.builtin_module_names, "torch._C")
        with mock.patch.object(sys, "builtin_module_names", inittab):
            embedded = types.ModuleType("torch._C")
            embedded.__loader__ = builtin
            self.assertIs(located(embedded, "torch._C", False), True)

    @parametrize("shape", sorted(_NOT_LIBRARY_MODULES))
    def test_library_module_requires_the_name_to_resolve_to_the_library(self, shape):
        name, attrs, install_roots = _NOT_LIBRARY_MODULES[shape]
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        if install_roots is None:
            install_roots = precompile_package._install_roots()
        if shape == "relative_file":
            self.addCleanup(os.chdir, os.getcwd())
            os.chdir(_STDLIB_ROOT)
        # The verdict is cached per __file__ while _install_roots is patched per
        # row, so a verdict another test left for a row's file would be read back
        # under the wrong roots; the torch roots must be read off the real torch
        # before a row replaces sys.modules['torch'].
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
        precompile_package._classify_file.cache_clear()
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
        # Not imported at all, an inner name has nothing to check: deliberate.
        self.assertTrue(is_library("collections.never_imported"))
        # A file under a nested install root is stdlib once nothing is installed
        # there: the very file the refusal table refuses with that root installed.
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
        # user code injected into builtins are all reads from a slot; the lint
        # hands every dropped guard's source over unnarrowed, so the table read
        # off a slot is refused rather than read as a global.
        self.assertFalse(reads_a_builtin(AttrSource(LocalSource("self"), "act"), abs))
        self.assertFalse(
            reads_a_builtin(DictGetItemSource(GlobalSource("_OPS"), "len"), len)
        )
        table = DictGetItemSource(AttrSource(LocalSource("self"), "act_fns"), "len")
        self.assertFalse(reads_a_builtin(table, len))
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

        # Only under the builtin's own name (IOError and EnvironmentError are
        # CPython's aliases of OSError), only a value CPython built and only one
        # builtins owns. The flag is what refuses a user class installed into
        # builtins under its own name, whether its __module__ names the test
        # module or, exec'd with the builtins namespace as its globals, claims
        # "builtins" outright; it is read through type's descriptor, so a
        # metaclass property cannot shadow it or raise into the predicate.
        # Among the real builtin types it costs the heap type ExceptionGroup
        # (3.11+) alone (__loader__ is a heap type too, but its __name__ is not
        # the key). The __module__ test is what refuses open, the one builtin
        # function builtins does not own: io before 3.12, _io since.
        class UserError(Exception):
            pass

        class Shadowing(type):
            @property
            def __flags__(cls):
                return 1 << 8

        class Shadow(metaclass=Shadowing):
            __module__ = "builtins"

        class Raising(type):
            @property
            def __flags__(cls):
                raise RuntimeError("shadowed")

        class Loud(metaclass=Raising):
            pass

        for name in ("IOError", "EnvironmentError"):
            self.assertIs(vars(builtins)[name], OSError)
            alias_read = DictGetItemSource(_BUILTINS_DICT, name)
            self.assertFalse(reads_a_builtin(alias_read, OSError), name)
        self.assertEqual(open.__name__, "open")
        self.assertIn(open.__module__, ("io", "_io"))
        open_read = DictGetItemSource(_BUILTINS_DICT, "open")
        self.assertFalse(reads_a_builtin(open_read, open))
        loader = vars(builtins)["__loader__"]
        self.assertNotEqual(getattr(loader, "__name__", None), "__loader__")
        user_read = DictGetItemSource(_BUILTINS_DICT, "UserError")
        self.assertFalse(reads_a_builtin(user_read, UserError))
        ns = dict(vars(builtins))
        exec("class unicode(str): pass", ns)
        self.assertEqual(ns["unicode"].__module__, "builtins")
        shim_read = DictGetItemSource(_BUILTINS_DICT, "unicode")
        self.assertFalse(reads_a_builtin(shim_read, ns["unicode"]))
        self.assertEqual(Shadow.__flags__ & (1 << 8), 1 << 8)
        self.assertEqual(Shadow.__module__, "builtins")
        shadow_read = DictGetItemSource(_BUILTINS_DICT, "Shadow")
        self.assertFalse(reads_a_builtin(shadow_read, Shadow))
        with self.assertRaises(RuntimeError):
            Loud.__flags__
        loud_read = DictGetItemSource(_BUILTINS_DICT, "Loud")
        self.assertFalse(reads_a_builtin(loud_read, Loud))
        refused = [
            name
            for name, value in vars(builtins).items()
            if isinstance(value, type)
            and value.__name__ == name
            and not reads_a_builtin(DictGetItemSource(_BUILTINS_DICT, name), value)
        ]
        heap_types = ["ExceptionGroup"] if sys.version_info >= (3, 11) else []
        self.assertEqual(refused, heap_types)

    def test_dynamo_synthesized_covers_only_the_resume_function_list(self):
        synthesized = precompile_package._is_dynamo_synthesized
        resume_fns = LocalSource("__nested_resume_fns")
        entry = GetItemSource(resume_fns, 0)
        self.assertTrue(synthesized(resume_fns))
        self.assertTrue(synthesized(entry))
        # Past an entry the value is the user's: a resume function's closure
        # cells carry the resumed frame's cell variables, and Dynamo guards a
        # callable an inner def captured as type(act).__call__ through one.
        closure = ClosureSource(entry)
        cell = CellContentsSource(
            GetItemSource(closure, 0), "cell_contents", freevar_name="act"
        )
        self.assertFalse(synthesized(closure))
        self.assertFalse(synthesized(cell))
        self.assertFalse(synthesized(AttrSource(TypeSource(cell), "__call__")))
        # The frame values are the live stack and locals of the frames nested
        # inside the one resuming, so a guard rooted there is judged like the
        # value it stands for.
        self.assertFalse(
            synthesized(GetItemSource(LocalSource("__nested_frame_values"), 0))
        )
        # A global spelled like one is a user binding.
        self.assertFalse(synthesized(GlobalSource("__nested_resume_fns")))
        self.assertFalse(synthesized(LocalSource("x")))

    def test_alias_module_and_owning_module(self):
        alias_module = precompile_package._dynamo_alias_module
        self.assertIs(alias_module("__import_torch_dot_nn_dot_functional"), F)
        # A user global is no alias even when its tail past the prefix length
        # names a module; an unknown tail and a torch_package module, which
        # import_source aliases without the prefix, come back None.
        self.assertIsNone(alias_module("imported_torch"))
        self.assertIsNone(alias_module("__import_not_a_module"))
        self.assertIsNone(alias_module("_torch_package_0__dot_mypkg_dot_impl"))
        # The unmangling collides for a module whose name contains _dot_:
        # mypkg.sub_dot_mod is aliased exactly like mypkg.sub.mod, which is
        # what comes back when it is loaded (documented, fails open).
        collided = types.ModuleType("mypkg.sub.mod")
        alias = "__import_" + "mypkg.sub_dot_mod".replace(".", "_dot_")
        self.assertEqual(alias, "__import_mypkg_dot_sub_dot_mod")
        with mock.patch.dict(sys.modules, {"mypkg.sub.mod": collided}):
            self.assertIs(alias_module(alias), collided)
        owning_module = precompile_package._owning_module
        self.assertEqual(owning_module(F), "torch.nn.functional")
        self.assertEqual(owning_module(F.gelu), "torch._C._nn")
        self.assertIsNone(owning_module(3))
        self.assertIsNone(owning_module(types.SimpleNamespace(__module__=3)))

    def test_defined_where_read_needs_the_name_and_the_file(self):
        defined_where_read = precompile_package._defined_where_read
        self.assertTrue(defined_where_read(_user_op, "_user_op", _HERE))
        # Paths are compared normalized, so another spelling of the file matches.
        unnormalized = os.path.join(
            os.path.dirname(__file__), os.curdir, os.path.basename(__file__)
        )
        stack = _stack(unnormalized)
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
        two_frame = _stack(__file__, F.__file__)
        self.assertTrue(defined_where_read(_user_op, "_user_op", two_frame))
        self.assertFalse(defined_where_read(F.silu, "silu", two_frame))

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

    def test_defined_where_read_takes_the_file_off_the_code_object(self):
        defined_where_read = precompile_package._defined_where_read
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

        # A same-file wraps decorator a flag turns on forges __qualname__ and
        # compiles in this file, so the code object's own name is what tells the
        # two arms apart: off is the def itself, on is a slot.
        def maybe_log(fn, on):
            if not on:
                return fn

            @functools.wraps(fn)
            def wrapper(*args):
                return fn(*args)

            return wrapper

        plain, logged = maybe_log(_user_op, False), maybe_log(_user_op, True)
        self.assertIs(plain, _user_op)
        self.assertEqual(logged.__qualname__, "_user_op")
        self.assertEqual(logged.__code__.co_name, "wrapper")
        self.assertTrue(defined_where_read(plain, "_user_op", _HERE))
        self.assertFalse(defined_where_read(logged, "_user_op", _HERE))
        # Only a plain function or a class is judged: a bound method forwards
        # __qualname__ and __code__ to its function, and a namespace carrying
        # both is refused before either is read.
        bound = types.MethodType(_user_op, object())
        self.assertEqual(bound.__qualname__, "_user_op")
        self.assertIs(bound.__code__, _user_op.__code__)
        self.assertFalse(defined_where_read(bound, "_user_op", _HERE))
        fake = types.SimpleNamespace(__qualname__="_user_op", __code__=bound.__code__)
        self.assertFalse(defined_where_read(fake, "_user_op", _HERE))

    def test_defined_where_read_refuses_a_pseudo_filename(self):
        defined_where_read = precompile_package._defined_where_read

        # A co_filename is not always a path: an exec-generated frame records
        # <string>, and so does a def exec'd under it, so realpath would resolve
        # both against the cwd and waive a def whose bind no checksum covers
        # (on 3.10 the <string> a fields-only dataclass compiles __init__ in
        # collides the same way; from 3.11 co_qualname refuses it first). Only
        # absolute filenames compare; a relative one, spelled so that it does
        # resolve to this file from the cwd, and an embedded NUL on either side
        # fail closed instead of raising.
        pseudo = _stack("<string>")
        exec_ns = {}
        exec(compile("def op(x):\n    return x\n", "<string>", "exec"), exec_ns)
        self.assertEqual(exec_ns["op"].__code__.co_filename, "<string>")
        self.assertFalse(defined_where_read(exec_ns["op"], "op", pseudo))
        relative = os.path.relpath(__file__)
        self.assertTrue(os.path.samefile(relative, __file__))
        stack = _stack(relative)
        self.assertFalse(defined_where_read(_user_op, "_user_op", stack))
        code = _user_op.__code__.replace(co_filename=relative)
        relative_op = types.FunctionType(code, globals(), "_user_op")
        self.assertFalse(defined_where_read(relative_op, "_user_op", _HERE))
        stack = _stack(__file__ + "\x00")
        self.assertFalse(defined_where_read(_user_op, "_user_op", stack))
        code = _user_op.__code__.replace(co_filename=__file__ + "\x00")
        nul_op = types.FunctionType(code, globals(), "_user_op")
        self.assertFalse(defined_where_read(nul_op, "_user_op", _HERE))

    def test_defined_where_read_judges_a_class_by_its_own_methods(self):
        defined_where_read = precompile_package._defined_where_read
        # A class has no code object; its methods tell. A class statement
        # compiled its defs in this file under its own qualname prefix, and a
        # class with no such def fails closed.
        cls = type(self)
        self.assertTrue(defined_where_read(cls, cls.__name__, _HERE))
        self.assertFalse(defined_where_read(cls, cls.__name__, _ELSEWHERE))
        self.assertFalse(defined_where_read(torch.nn.Linear, "Linear", _HERE))

        # Ops and Cm, same-file class statements whose only def is a
        # staticmethod or a classmethod (the __func__ arm), are waived, so is
        # one whose only def is a property (the fget arm); a cached_property
        # keeps its function under .func and is not unwrapped, so a class with
        # nothing else fails closed, as does a class with no method of its own
        # and one whose only methods are generated: a fields-only dataclass's
        # and a NamedTuple's are defs of a factory (__create_fn__, namedtuple),
        # so their code objects' own qualname carries <locals>. (on 3.10, where
        # only co_name exists, the <string> or stdlib file they compile in
        # refuses them), and an Enum's arrive under Enum. qualnames the key
        # rule refuses. Members are unwrapped by type, never probed with
        # getattr: a torch.classes proxy answers any attribute read by raising
        # RuntimeError, and a class holding one is still judged.
        class Ops:
            @staticmethod
            def op(x):
                return x

        class Cm:
            @classmethod
            def make(cls):
                return cls()

        class Prop:
            @property
            def x(self):
                return 1

        class Cached:
            @functools.cached_property
            def x(self):
                return 1

        class Marker:
            pass

        @dataclasses.dataclass
        class Cfg:
            x: int

        class Color(enum.Enum):
            RED = 1

        class Pt(typing.NamedTuple):
            x: int

        # _Classes.__getattr__ installs the namespace it fabricates on the
        # global torch.classes module; take it back off after the test.
        self.addCleanup(delattr, torch.classes, "precompile_package_test")

        class Model(torch.nn.Module):
            ns = torch.classes.precompile_package_test

            def forward(self, x):
                return x

        self.assertTrue(defined_where_read(Ops, Ops.__qualname__, _HERE))
        self.assertTrue(defined_where_read(Cm, Cm.__qualname__, _HERE))
        self.assertTrue(defined_where_read(Prop, Prop.__qualname__, _HERE))
        self.assertFalse(defined_where_read(Cached, Cached.__qualname__, _HERE))
        self.assertFalse(defined_where_read(Marker, Marker.__qualname__, _HERE))
        self.assertFalse(defined_where_read(Cfg, Cfg.__qualname__, _HERE))
        self.assertFalse(defined_where_read(Color, Color.__qualname__, _HERE))
        self.assertFalse(defined_where_read(Pt, Pt.__qualname__, _HERE))
        with self.assertRaises(RuntimeError):
            Model.ns.__func__
        self.assertTrue(defined_where_read(Model, Model.__qualname__, _HERE))
        self.assertFalse(defined_where_read(Model, Model.__qualname__, _ELSEWHERE))
        # The proxy itself as the value is refused by type before any read.
        self.assertFalse(defined_where_read(Model.ns, "precompile_package_test", _HERE))

    def test_defined_where_read_refuses_a_class_minted_for_the_file(self):
        defined_where_read = precompile_package._defined_where_read
        # namedtuple and type() (make_dataclass from 3.12) stamp __module__
        # from the calling frame under a BARE __qualname__, so a class a
        # library mints for this file claims it exactly like a class statement
        # written here. None of its functions compiled under its qualname
        # prefix, so it fails closed, including a factory fed a same-file def
        # (bare qualname) and an imported class the reader attaches one to.
        point = collections.namedtuple("Point", "x")
        self.assertEqual((point.__module__, point.__qualname__), (__name__, "Point"))
        self.assertFalse(defined_where_read(point, "Point", _HERE))
        point = dataclasses.make_dataclass("Point", [("x", int)])
        # 3.12+ stamps the caller's module on the class; 3.10/3.11 leave "types".
        # Either way the bare qualname and library-compiled methods refuse it.
        self.assertEqual(point.__qualname__, "Point")
        self.assertFalse(defined_where_read(point, "Point", _HERE))
        self.assertFalse(defined_where_read(type("Point", (), {}), "Point", _HERE))
        point = type("Point", (), {"area": _user_op})
        self.assertFalse(defined_where_read(point, "Point", _HERE))
        # Under its own name as the key too: the qualname is _user_op, not
        # Point._user_op, so the class half of the rule is what refuses it.
        point = type("Point", (), {"_user_op": _user_op})
        self.assertFalse(defined_where_read(point, "Point", _HERE))

        # The key half: a same-file class statement's method carries its
        # Named. prefix, so handed to a type() class of the same qualname it
        # waives that class under its own key and not borrowed under another.
        class Named:
            def m(self):
                return 0

        def under(key):
            qn = Named.__qualname__
            return type("Named", (), {key: Named.m, "__qualname__": qn})

        self.assertTrue(defined_where_read(under("m"), Named.__qualname__, _HERE))
        self.assertFalse(defined_where_read(under("other"), Named.__qualname__, _HERE))
        # A class imported from another file is not written here however many
        # same-file functions the reader attaches to it, nor when it patches a
        # method through functools.wraps, which forges the Point.norm qualname
        # but not the code object's own name.
        imported = {"__name__": "mypkg.impl"}
        source = "class Point:\n    def norm(self):\n        return 0\n"
        exec(compile(source, F.__file__, "exec"), imported)
        imported["Point"].extra = _user_op
        self.assertFalse(defined_where_read(imported["Point"], "Point", _HERE))

        @functools.wraps(imported["Point"].norm)
        def _norm(self):
            return 1

        self.assertEqual(_norm.__qualname__, "Point.norm")
        self.assertEqual(_norm.__code__.co_name, "_norm")
        imported["Point"].norm = _norm
        self.assertFalse(defined_where_read(imported["Point"], "Point", _HERE))

    def test_defined_where_read_skips_the_compiler_annotate_function(self):
        defined_where_read = precompile_package._defined_where_read

        # On 3.14 the compiler stores the PEP 649 annotate function of an
        # annotated class body in its __dict__, compiled in this file, under
        # key __annotate_func__ with qualname Cfg.__annotate__; the key rule
        # refuses it, so a fields-only dataclass is reported on every version.
        @dataclasses.dataclass
        class Cfg:
            x: int

        if sys.version_info >= (3, 14):
            annotate = vars(Cfg)["__annotate_func__"]
            code = annotate.__code__
            qualname = f"{Cfg.__qualname__}.__annotate__"
            self.assertEqual(
                (annotate.__qualname__, code.co_qualname, code.co_filename),
                (qualname, qualname, __file__),
            )
        self.assertFalse(defined_where_read(Cfg, Cfg.__qualname__, _HERE))

        # Replayed on every version with a type() class handed a same-file
        # function under that key, carrying the qualname on the function and on
        # its code object as a class statement's def does. The control row, the
        # same function under an ordinary key, shows the replay is what a class
        # statement produces; the skip of both keys, against a version that
        # stores the function under its own name, is pinned last.
        def minted(key, qualname):
            code = _user_op.__code__.replace(co_name=qualname.rpartition(".")[2])
            if sys.version_info >= (3, 11):
                code = code.replace(co_qualname=qualname)
            fn = types.FunctionType(code, globals(), "__annotate__")
            fn.__qualname__ = qualname
            return type("Cfg", (), {key: fn})

        self.assertTrue(defined_where_read(minted("op", "Cfg.op"), "Cfg", _HERE))
        real = minted("__annotate_func__", "Cfg.__annotate__")
        self.assertFalse(defined_where_read(real, "Cfg", _HERE))
        for key in ("__annotate__", "__annotate_func__"):
            own_key = minted(key, f"Cfg.{key}")
            self.assertFalse(defined_where_read(own_key, "Cfg", _HERE), key)

    def test_minted_global_names_match_dynamo(self):
        # The predicates lean on names Dynamo mints inline (the two prefixes
        # through aot_compile.py), in install_builtins_dict_in_fglobals,
        # import_source and the nested resume prologue; a rename there must
        # fail here rather than silently turn the lint off.
        seen = []

        def record(entries):
            seen.extend(entries)
            return [True] * len(entries)

        def root(entry):
            source = entry.orig_guard.originating_source
            return source.get_base() if isinstance(source, ChainedSource) else source

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
            for r in map(root, seen)
            if isinstance(r, GlobalSource) and r.global_name.startswith("__import_")
        }
        alias = "__import_torch_dot_nn_dot_modules_dot_linear"
        self.assertIn(alias, aliases)
        self.assertIs(
            precompile_package._dynamo_alias_module(alias), torch.nn.modules.linear
        )

        class Act:
            def __call__(self, v):
                return v + 1

        def mid(x, act):
            def inner(v):
                return act(v)

            torch._dynamo.graph_break()
            return inner(x)

        def outer(x, act):
            return mid(x, act) + 1

        seen.clear()
        # The harness runs every test under nested_graph_breaks=True (the config
        # default is off).
        compiled = torch.compile(
            outer, backend="eager", options={"guard_filter_fn": record}
        )
        compiled(torch.ones(2), Act())
        synthesized = precompile_package._is_dynamo_synthesized
        verdicts: dict[str, dict[str, bool]] = collections.defaultdict(dict)
        for e in seen:
            r = root(e)
            if isinstance(r, LocalSource) and r.local_name.startswith("__nested"):
                source = e.orig_guard.originating_source
                verdicts[r.local_name][source.name] = synthesized(source)
        # Both lists are passed into every resume function and both are guard
        # roots; only the first is synthesized, and of what hangs off it only
        # the list and its entry: mid's resume function closes over act, and
        # the guard on its __call__ Dynamo mints through that cell is the
        # user's.
        waived = {name for name, ok in verdicts["__nested_resume_fns"].items() if ok}
        self.assertEqual(
            waived, {"L['__nested_resume_fns']", "L['__nested_resume_fns'][0]"}
        )
        past = "type(L['__nested_resume_fns'][0].__closure__[0].cell_contents).__call__"
        self.assertIn(past, verdicts["__nested_resume_fns"])
        self.assertFalse(verdicts["__nested_resume_fns"][past])
        self.assertTrue(verdicts["__nested_frame_values"])
        self.assertFalse(any(verdicts["__nested_frame_values"].values()))

    def test_module_namespaces_trust_only_bindings_config_cannot_repoint(self):
        mypkg = types.ModuleType("mypkg")
        layers = types.ModuleType("mypkg.layers")
        impl_b = types.ModuleType("mypkg.impl_b")
        deep = types.ModuleType("mypkg.impl_b.deep")
        entries = [
            _entry(GlobalSource("mypkg"), mypkg),  # import mypkg
            _entry(AttrSource(GlobalSource("mypkg"), "layers"), layers),  # import mypkg.layers
            _entry(AttrSource(GlobalSource("mypkg"), "impl"), impl_b),  # from . import impl_b as impl
            _entry(GlobalSource("impl"), impl_b),  # import mypkg.impl_b as impl
            _entry(AttrSource(GlobalSource("impl"), "deep"), deep),  # name owned by the parent, but the parent is untrusted
            _entry(AttrSource(GlobalSource("other"), "sub"), layers),  # parent never guarded
            _entry(GlobalSource("F"), F),  # import torch.nn.functional as F
            _entry(AttrSource(GlobalSource("torch"), "_dynamo"), torch._dynamo),  # library, parent or not
            _entry(GlobalSource("__import_mypkg_dot_layers"), layers),  # Dynamo's alias for an inlined function's globals
            _entry(AttrSource(GlobalSource("__import_torch"), "Tensor"), torch.Tensor),  # alias without a module-valued guard
            _entry(GlobalSource("config"), torch._dynamo.config),
            _entry(AttrSource(GlobalSource("__import_torch_dot__dynamo_dot_config"), "verbose"), False),  # from torch._dynamo.config import verbose, inlined: recovered from the alias, still a config module
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
            (GlobalSource("mypkg"), types.ModuleType("mypkg")),
            (GlobalSource("impl"), types.ModuleType("mypkg.impl_b")),
            (GlobalSource("config"), torch._dynamo.config),
        ]
        entries = [_entry(s, m) for s, m in modules] + [entry]
        namespaces = precompile_package._module_namespaces(entries)
        self.assertEqual(precompile_package._is_risky_drop(entry, namespaces), risky)

    def test_risky_drop_sees_the_slot_behind_a_nested_resume(self):
        # With nested_graph_breaks the callee's locals reach the caller's resume
        # frame as positional entries of L['__nested_frame_values'][0] rather
        # than as L['act']; a slot
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
        self.assertTrue(_pins_a_value("RANGE_ITERATOR_MATCH", "it"))
        self.assertTrue(_pins_a_value("COUNT_ITERATOR_MATCH", "it"))
        # The empty source of a sourceless guard is not a bare name.
        self.assertFalse(_pins_a_value("EQUALS_MATCH", ""))
        # Reached THROUGH an argument, or a container element: not counted.
        self.assertFalse(_pins_a_value("CONSTANT_MATCH", "self.eps"))
        self.assertFalse(_pins_a_value("EQUALS_MATCH", "dims[0]"))
        self.assertFalse(_pins_a_value("EQUALS_MATCH", "G['CFG'].width"))
        self.assertFalse(_pins_a_value("TENSOR_MATCH", "x"))
        self.assertFalse(_pins_a_value("SEQUENCE_LENGTH", "xs"))

        def fact(guard_type, source, live=True):
            return GuardFact(
                guard_type=guard_type, source=source, code=(), value="", enforced=live
            )

        entry = ("step", "m.py", 1)
        resume_a = ("torch_dynamo_resume_in_step_at_7", "m.py", 7)
        resume_b = ("torch_dynamo_resume_in_step_at_9", "m.py", 9)
        kept = {
            ("EQUALS_MATCH", "scale"),
            ("EQUALS_MATCH", "mode"),
            ("CONSTANT_MATCH", "___stack0"),
            ("CONSTANT_MATCH", "fn"),
            ("EQUALS_MATCH", "keys"),
            ("TENSOR_MATCH", "x"),
        }
        pinned_scale = fact("EQUALS_MATCH", "scale")
        pinned_mode = fact("EQUALS_MATCH", "mode")
        generic_scale = fact("TYPE_MATCH", "scale")
        pinned_keys = fact("EQUALS_MATCH", "keys")
        it = fact("COUNT_ITERATOR_MATCH", "it")
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
            # A variant whose value guard the filter dropped checks nothing and
            # serves any fn, so it cancels the sibling's pin like a generic one.
            ("gate", "m.py", 12): [
                frozenset({fact("CONSTANT_MATCH", "fn")}),
                frozenset({fact("CONSTANT_MATCH", "fn", live=False)}),
            ],
            # A dict_keys argument gets EQUALS_MATCH and SEQUENCE_LENGTH on one
            # source in ONE variant (variables/builder.py): a variant's own
            # companion guard must not cancel its pin. The iterator pin on `it`
            # has no kept slot, so it is never reported.
            ("lookup", "m.py", 15): [
                frozenset({pinned_keys, fact("SEQUENCE_LENGTH", "keys"), it})
            ],
        }
        self.assertEqual(
            _wont_generalize(kept, guard_sets), ("___stack0", "keys", "mode")
        )
        # A frame that pins scale in its only variant is a real pin; the entry
        # frame's generic variant cancels the entry's pin, not this one.
        guard_sets[("helper", "m.py", 20)] = [frozenset({pinned_scale})]
        self.assertEqual(
            _wont_generalize(kept, guard_sets), ("___stack0", "keys", "mode", "scale")
        )
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
        # One HASATTR per attribute, all on the parent source: two facts on one
        # slot inside a variant are that variant's rendering of it, and the slot
        # varies only when the variants' renderings differ (here by code alone).
        has_w = fact("HASATTR", "L['mod']", code=("hasattr(L['mod'], 'weight')",))
        has_b = fact("HASATTR", "L['mod']", code=("hasattr(L['mod'], 'bias')",))
        both = frozenset({has_w, has_b})
        self.assertEqual(_varying_guard_slots({frame: [both]}), frozenset())
        self.assertEqual(_varying_guard_slots({frame: [both, both]}), frozenset())
        self.assertEqual(
            _varying_guard_slots({frame: [frozenset({has_w}), frozenset({has_b})]}),
            frozenset({("HASATTR", "L['mod']")}),
        )

    def test_summarize_reads_the_frame_lists_off_the_entry(self):
        from torch._dynamo.package import (
            _DynamoCacheEntry,
            _DynamoCodeCacheEntry,
            _GuardedCodeCacheEntry,
            SerializedCode,
            SourceInfo,
        )
        from torch._dynamo.precompile_package import _summarize
        from torch.compiler._precompile_types import GuardFact

        def entry(
            code,
            guarded=0,
            backend_ids=(),
            bypassed=False,
            entered=True,
            install_to_global=False,
        ):
            serialized = SerializedCode.from_code_object(code)
            return _DynamoCodeCacheEntry(
                python_code=serialized,
                python_module=__name__,
                function_names=[],
                guarded_codes=[
                    _GuardedCodeCacheEntry(guards_state=b"", dynamo_code=serialized)
                    for _ in range(guarded)
                ],
                import_sources={},
                backend_ids=list(backend_ids),
                code_source=None,
                install_to_global=install_to_global,
                has_compile_id=entered,
                bypassed=bypassed,
            )

        # Three distinct code objects that share the name every nn.Module has.
        class A:
            def forward(self):
                pass

        class B:
            def forward(self):
                pass

        class C:
            def forward(self):
                pass

        def helper():
            pass

        def resume():
            pass

        # A save-time bypass (from_cache_entry) leaves the entry's backend ids
        # in place; install() loads none of them, so they are not counted.
        codes = [
            entry(A.forward.__code__, guarded=1, backend_ids=["__compiled_fn_1"]),
            entry(B.forward.__code__),
            entry(C.forward.__code__),
            entry(helper.__code__, bypassed=True, backend_ids=["__compiled_fn_2"]),
            # Generated but never executed: no compile id, so not a gap.
            entry(resume.__code__, entered=False, install_to_global=True),
        ]
        info = SourceInfo(inlined_sources=set())
        cache = _DynamoCacheEntry(codes=codes, source_info=info, device_type="cpu")
        fn_id, flag = ("ID_MATCH", "G['fn']"), ("HASATTR", "mod")
        mode, torch_mod = ("EQUALS_MATCH", "mode"), ("MODULE_MATCH", "G['torch']")
        has_bias, is_torch = "hasattr(L['mod'], 'bias')", "G['torch'] is torch"
        pinned_mode = GuardFact(
            guard_type="EQUALS_MATCH",
            source="mode",
            code=("L['mode'] == 1",),
            value="",
            enforced=True,
        )
        summary = _summarize(
            cache,
            dropped={fn_id, flag},
            kept={mode, ("TENSOR_MATCH", "x")},
            policy_dropped={torch_mod},
            risky={fn_id},
            truncated=frozenset({"forward (m.py:3)"}),
            capture_errors=("boom",),
            guard_sets={("forward", "m.py", 3): [frozenset({pinned_mode})]},
            # One rendering per dropped slot, from either drop list; fn_id has none.
            dropped_code={flag: has_bias, torch_mod: is_torch},
        )
        # One bare co_name per frame: two uncovered forwards stay two, and the
        # frame lists are drawn from the frames the count covers.
        self.assertEqual(summary.frames, 5)
        self.assertEqual(summary.resume_functions, 1)
        self.assertEqual(summary.guarded_codes, 1)
        self.assertEqual(summary.backend_graphs, 1)
        self.assertEqual(summary.bypassed, ("helper",))
        self.assertEqual(summary.uncovered_frames, ("forward", "forward"))
        self.assertFalse(summary.complete)
        self.assertEqual(summary.dropped_guards, (flag, fn_id))
        self.assertEqual(summary.kept_guards, (mode, ("TENSOR_MATCH", "x")))
        self.assertEqual(summary.policy_dropped_guards, (torch_mod,))
        self.assertEqual(summary.risky_dropped_guards, (fn_id,))
        self.assertEqual(
            summary.dropped_guard_code, ((*flag, has_bias), (*torch_mod, is_torch))
        )
        self.assertEqual(summary.wont_generalize, ("mode",))
        self.assertEqual(summary.capture_errors, ("boom",))

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
            with _capture_config():
                self.assertEqual(flags(), (True, True, True, True))
            self.assertEqual(flags(), ambient)

            with self.assertRaisesRegex(RuntimeError, "boom"):
                with _capture_config():
                    raise RuntimeError("boom")
            self.assertEqual(flags(), ambient)

            # Config values are per thread, so a worker entering the scope
            # patches only itself and the main thread stays ambient.
            entered, release = threading.Event(), threading.Event()
            seen = []

            def hold():
                seen.append(flags())
                with _capture_config():
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
            self.assertEqual(seen[1], (True, True, True, True))
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
                with _capture_config():
                    pass

        with functorch_config.patch(strict_autograd_cache=False):
            with torch._dynamo.config.patch(strict_precompile=False):
                with _capture_config():
                    self.assertFalse(functorch_config.strict_autograd_cache)
            with torch._dynamo.config.patch(strict_precompile=True):
                with _capture_config():
                    self.assertTrue(functorch_config.strict_autograd_cache)
            self.assertFalse(functorch_config.strict_autograd_cache)

    def test_allow_empty_graphs_convert_frame_keeps_a_no_op_frame_compilable(self):
        from torch._dynamo.convert_frame import ConvertFrame
        from torch._dynamo.precompile_package import _AllowEmptyGraphsConvertFrame
        from torch._dynamo.testing import CompileCounter

        def fn(x, flag):
            if flag:
                return x.sin()
            return x

        # Install a converter of the given class beneath the CatchErrorsWrapper
        # torch._dynamo.optimize built, the way a capture session will, and
        # count the variants of fn the backend compiles after a no-op call.
        def compiled_variants(cls):
            torch._dynamo.reset()
            counter = CompileCounter()
            optimize_ctx = torch._dynamo.optimize(counter)
            wrapper = optimize_ctx.callback
            built = wrapper._torchdynamo_orig_backend
            wrapper._torchdynamo_orig_backend = cls(
                built._torchdynamo_orig_backend,
                wrapper.hooks,
                recompile_limit=built._recompile_limit,
            )
            x = torch.ones(2)
            optimize_ctx(fn)(x, False)
            optimize_ctx(fn)(x, True)
            return counter.frame_count

        # The plain converter skips the code object on the empty graph, so the
        # later sin variant is never compiled; the subclass compiles both.
        self.assertEqual(compiled_variants(ConvertFrame), 0)
        self.assertEqual(compiled_variants(_AllowEmptyGraphsConvertFrame), 2)
        self.assertFalse(torch._dynamo.config.allow_empty_graphs)

    def test_allow_empty_graphs_convert_frame_refuses_a_ddp_optimizer_frame(self):
        from torch._dynamo.hooks import Hooks
        from torch._dynamo.package import CompilePackage
        from torch._dynamo.precompile_package import _AllowEmptyGraphsConvertFrame
        from torch._dynamo.testing import CompileCounter
        from torch.nn.parallel import DistributedDataParallel

        def fn(x):
            return x.sin()

        # A stub active DDP module takes CatchErrorsWrapper down its DDPOptimizer
        # branch, the one that asks the converter for a clone.
        ddp_module = types.SimpleNamespace(bucket_bytes_cap=25 * 1024 * 1024)
        for optimize_ddp in (True, "ddp_optimizer", "no_optimization"):
            torch._dynamo.reset()
            counter = CompileCounter()
            optimize_ctx = torch._dynamo.optimize(counter)
            wrapper = optimize_ctx.callback
            built = wrapper._torchdynamo_orig_backend
            conv = _AllowEmptyGraphsConvertFrame(
                built._torchdynamo_orig_backend,
                wrapper.hooks,
                package=CompilePackage(fn),
                recompile_limit=built._recompile_limit,
            )
            wrapper._torchdynamo_orig_backend = conv
            # The wrapper's capability probe must still say yes with a package.
            self.assertTrue(hasattr(conv, "_clone_with_backend"))
            with (
                torch._dynamo.config.patch(optimize_ddp=optimize_ddp),
                mock.patch.object(
                    DistributedDataParallel, "_active_ddp_module", ddp_module
                ),
            ):
                if optimize_ddp == "no_optimization":
                    optimize_ctx(fn)(torch.ones(2))
                    self.assertEqual(counter.frame_count, 1)
                else:
                    msg = r'DistributedDataParallel forward.*optimize_ddp=.*optimize_ddp="no_optimization"'
                    with self.assertRaisesRegex(PackageError, msg):
                        optimize_ctx(fn)(torch.ones(2))
                    self.assertEqual(counter.frame_count, 0)
            self.assertFalse(torch._dynamo.config.allow_empty_graphs)
        # Without a package the DDP clone keeps the subclass, hooks and limit.
        backend, hooks = CompileCounter(), Hooks()
        plain = _AllowEmptyGraphsConvertFrame(backend, hooks, recompile_limit=3)
        clone = plain._clone_with_backend(backend)
        self.assertIs(type(clone), _AllowEmptyGraphsConvertFrame)
        self.assertIs(clone._hooks, hooks)
        self.assertEqual(clone._recompile_limit, 3)

    def test_allow_empty_graphs_convert_frame_reverts_the_flag_when_the_compile_raises(
        self,
    ):
        from torch._dynamo.convert_frame import ConvertFrame
        from torch._dynamo.hooks import Hooks
        from torch._dynamo.precompile_package import _AllowEmptyGraphsConvertFrame

        seen = []

        def failing_compile(self, frame, cache_entry, hooks, frame_state, skip=0):
            seen.append((torch._dynamo.config.allow_empty_graphs, skip))
            raise RuntimeError("compile failed")

        converter = _AllowEmptyGraphsConvertFrame(lambda gm, inputs: gm, Hooks())
        with mock.patch.object(ConvertFrame, "__call__", failing_compile):
            with self.assertRaisesRegex(RuntimeError, "compile failed"):
                converter(mock.Mock(), None, Hooks(), {}, skip=1)
        # The flag was on for the compile and the converter's own frame is
        # accounted for in the traceback skip count.
        self.assertEqual(seen, [(True, 2)])
        self.assertFalse(torch._dynamo.config.allow_empty_graphs)


instantiate_parametrized_tests(TestPrecompilePackage)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
