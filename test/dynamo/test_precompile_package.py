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
import tempfile
import traceback
import types
import xml.parsers.expat  # noqa: F401  # imported for the _LIBRARY_NAMES rows
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
    DictGetItemSource,
    get_global_source_name,
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
    # Evidence in neither direction, and a top-level name needs some. The test
    # judges this row from _STDLIB_ROOT, where the real graphlib.py sits, so an
    # ungated resolve against the cwd would land on stdlib and waive the row.
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
        all_dropped = dict.fromkeys(unsupported, False)
        self.assertEqual(dict(zip(unsupported, kept)), all_dropped)
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
            _entry(g, None, "TYPE_MATCH", derived=("ID_MATCH",)),
            # The entry carries the unsaved build's derived types: there a
            # DICT_KEYS_MATCH on SUPPORTED_NODES is a DICT_VERSION, in the save
            # build the keys-match the serializer accepts.
            _entry(g, None, "DICT_KEYS_MATCH", derived=("DICT_VERSION",)),
        ]
        keep = precompile_package.default_guard_filter_fn(entries)
        self.assertEqual(keep, [True] * 5)

    def test_default_guard_filter_through_serialize_guards(self):
        seen = []

        def recording(entries):
            keep = precompile_package.default_guard_filter_fn(entries)
            seen.extend(zip(entries, keep))
            return keep

        def aot_compile(fn, *args, guard_filter_fn=recording):
            seen.clear()
            opts = {"guard_filter_fn": guard_filter_fn}
            fn_c = torch.compile(fn, fullgraph=True, backend="eager", options=opts)
            return fn_c.aot_compile((args, {}))

        def kept_types(compiled):
            state = load_guards_state(compiled._artifacts.guards_state)
            return state, {g.create_fn_name() for g in state.output_graph.guards}

        def fn(x):
            return _user_op(x) + len(x.shape)

        x = torch.randn(3)
        compiled = aot_compile(fn, x)
        state, kept = kept_types(compiled)
        # The refused guard on the global function is dropped and gone from the
        # serialized set; the builtin's BUILTIN_MATCH is kept.
        dropped = {(e.guard_type, e.name) for e, keep in seen if not keep}
        self.assertIn(("CLOSURE_MATCH", "G['_user_op']"), dropped)
        self.assertNotIn("CLOSURE_MATCH", kept)
        self.assertIn("BUILTIN_MATCH", kept)
        data = AOTCompiledFunction.serialize(compiled).serialized_data
        loaded = AOTCompiledFunction.deserialize(data)
        self.assertEqual(loaded(x), fn(x))
        # The kept guard is live in the loaded artifact: a swapped builtin trips
        # that guard, by name, and the original passes again once restored.
        builtins_key = state.output_graph.name_of_builtins_dict_key_in_fglobals
        failed_on_len = rf"___check_obj_id\(G\['{builtins_key}'\]\['len'\]"
        real_len = len
        with mock.patch.object(builtins, "len", lambda *args: real_len(*args)):
            with self.assertRaisesRegex(RuntimeError, failed_on_len):
                loaded(x)
        self.assertTrue(loaded.guard_check(x))

        # The DICT_KEYS_MATCH on SUPPORTED_NODES reaches the filter as the
        # unsaved build's DICT_VERSION and is kept; the save build serializes it
        # as a keys-match, which trips on a node registered after capture.
        def fn_tree(x):
            return pytree.tree_flatten({"a": x, "b": x * 2})[0][1]

        compiled = aot_compile(fn_tree, x)
        promoted = [keep for e, keep in seen if "DICT_VERSION" in e.derived_guard_types]
        self.assertEqual(promoted, [True])
        self.assertIn("DICT_KEYS_MATCH", kept_types(compiled)[1])
        data = AOTCompiledFunction.serialize(compiled).serialized_data
        # Module globals the kept guards read (G['pytree']) resolve against the
        # live scope, as in test_aot_compile.py's f_globals=globals() loads.
        loaded = AOTCompiledFunction.deserialize(data, f_globals=globals())
        self.assertEqual(loaded(x), fn_tree(x))

        class Node:
            pass

        pytree.register_pytree_node(
            Node,
            lambda n: ([], None),
            lambda c, _: Node(),
            serialized_type_name=f"{__name__}.{Node.__qualname__}",
        )
        self.addCleanup(pytree._deregister_pytree_node, Node)
        self.assertFalse(loaded.guard_check(x))

        # A local-scope class passes the filter (the TYPE_MATCH on L['obj'] is
        # kept) and serialization refuses it, naming the class. The filter keeps
        # the guard so the refusal is loud rather than an artifact that never
        # checks the type. For a plain instance the pickler refuses anyway,
        # wherever it sits in the guard tree, so dropping the TYPE_MATCH changes
        # nothing; for an nn.Module of a local class it does not (the instance
        # is rebuilt as a plain torch.nn.Module), so there the kept TYPE_MATCH
        # is the only refusal: drop it and the artifact ships, and serves the
        # local class's graph to a module of another class.
        class Local:
            n = 1

        def fn2(x, obj):
            return x + obj.n

        def drop_type_match(entries):
            keep = precompile_package.default_guard_filter_fn(entries)
            return [k and e.guard_type != "TYPE_MATCH" for k, e in zip(keep, entries)]

        refused_local = "Local'> cannot be saved.*defined in local scope"
        with self.assertRaisesRegex(PackageError, refused_local):
            aot_compile(fn2, x, Local())
        with self.assertRaisesRegex(PackageError, refused_local):
            aot_compile(fn2, x, Local(), guard_filter_fn=drop_type_match)

        class LocalModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        def fn3(x, mod):
            return mod(x)

        refused_module = "LocalModule'> cannot be saved.*defined in local scope"
        with self.assertRaisesRegex(PackageError, refused_module):
            aot_compile(fn3, x, LocalModule())

        class Other(torch.nn.Module):
            def forward(self, x):
                return x - 1

        compiled = aot_compile(fn3, x, LocalModule(), guard_filter_fn=drop_type_match)
        data = AOTCompiledFunction.serialize(compiled).serialized_data
        loaded = AOTCompiledFunction.deserialize(data)
        self.assertTrue(loaded.guard_check(x, Other()))
        self.assertEqual(loaded(x, Other()), x + 1)

    def test_roots_locate_the_stdlib_install_and_torch_dirs(self):
        # The real interpreter's roots, not what a sibling test left cached.
        precompile_package._stdlib_roots.cache_clear()
        precompile_package._install_roots.cache_clear()
        precompile_package._torch_roots.cache_clear()
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

    def test_stdlib_roots_take_the_windows_dlls_dir(self):
        # Reachable only on a Windows runner otherwise. sysconfig's first init
        # imports _sysconfigdata_*_{sys.platform}_*, so prime it unpatched.
        stdlib_roots, norm = precompile_package._stdlib_roots, precompile_package._norm
        sysconfig.get_paths()
        self.addCleanup(stdlib_roots.cache_clear)
        with mock.patch.object(sys, "platform", "win32"):
            stdlib_roots.cache_clear()
            self.assertIn(norm(os.path.join(sys.base_prefix, "DLLs")), stdlib_roots())

    def test_stdlib_roots_take_each_source_in_a_venv_layout(self):
        # Here every source names the one <prefix>/lib/pythonX.Y, so dropping
        # any of them leaves the roots unchanged. Replay a venv, where stdlib
        # stays at the base prefix while platstdlib is the venv's own lib
        # directory with purelib nested under it, and point the frozen-stdlib
        # hint elsewhere still, so each source is the only one for its root.
        base = os.path.join(os.sep, "base", "lib", "python3.12")
        venv = os.path.join(os.sep, "venv", "lib", "python3.12")
        frozen = os.path.join(os.sep, "frozen", "stdlib")
        purelib = os.path.join(venv, "site-packages")
        paths = dict(sysconfig.get_paths())
        paths.update(stdlib=base, platstdlib=venv, purelib=purelib, platlib=purelib)
        stdlib_roots = precompile_package._stdlib_roots
        install_roots = precompile_package._install_roots
        norm, within = precompile_package._norm, precompile_package._within
        self.addCleanup(stdlib_roots.cache_clear)
        self.addCleanup(install_roots.cache_clear)
        with (
            mock.patch.object(sysconfig, "get_paths", return_value=paths),
            mock.patch.object(sys, "_stdlib_dir", frozen, create=True),
        ):
            stdlib_roots.cache_clear()
            install_roots.cache_clear()
            stdlib, install = stdlib_roots(), install_roots()
        for root in (base, venv, frozen, os.path.dirname(os.__file__)):
            self.assertIn(norm(root), stdlib)
        # The nesting the exclusion exists for: purelib is under a stdlib root
        # and is an install root all the same.
        self.assertTrue(within(norm(purelib), stdlib))
        self.assertIn(norm(purelib), install)

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

    def test_install_roots_skip_a_site_accessor_that_raises(self):
        # A site.py that cannot answer is skipped, not propagated: this runs
        # inside save()'s lint, which must not abort the capture. sysconfig's
        # purelib is read outside the try and the other accessor is still
        # consulted, so what a failing accessor loses is only its own extras.
        user_site = os.path.join(os.sep, "elsewhere", "user-site")
        self.addCleanup(precompile_package._install_roots.cache_clear)
        with (
            mock.patch.object(site, "getsitepackages", side_effect=RuntimeError),
            mock.patch.object(site, "getusersitepackages", return_value=user_site),
        ):
            precompile_package._install_roots.cache_clear()
            install = precompile_package._install_roots()
        norm = precompile_package._norm
        self.assertIn(norm(sysconfig.get_paths()["purelib"]), install)
        self.assertIn(norm(user_site), install)

    def test_torch_roots_trust_torch_path_only_when_this_file_is_in_it(self):
        torch_roots, norm = precompile_package._torch_roots, precompile_package._norm
        own = norm(os.path.dirname(torch.__file__))
        bogus = os.path.join(os.sep, "elsewhere", "torch")
        stub = types.SimpleNamespace(__path__=[bogus])
        self.addCleanup(torch_roots.cache_clear)
        with mock.patch.dict(sys.modules, {"torch": stub}):
            # A substituted torch's __path__ is ignored until it lists the torch
            # package directory this file sits under; then every entry is
            # adopted, the bogus one included, which an editable build relies on.
            torch_roots.cache_clear()
            self.assertEqual(torch_roots(), (own,))
            stub.__path__.append(os.path.dirname(torch.__file__))
            torch_roots.cache_clear()
            self.assertEqual(set(torch_roots()), {own, norm(bogus)})
        with mock.patch.object(precompile_package, "__file__", None):
            torch_roots.cache_clear()
            self.assertEqual(torch_roots(), ())  # frozen: no directory to anchor to

    def test_classify_file_places_a_path_by_the_roots_it_lies_under(self):
        classify, norm = precompile_package._classify_file, precompile_package._norm
        stdlib_root = sysconfig.get_paths()["stdlib"]
        torch_in_stdlib = os.path.join(stdlib_root, "torch", "__init__.py")
        # The verdict is cached per __file__, and one file below is judged
        # under two install-root sets.
        self.addCleanup(classify.cache_clear)
        classify.cache_clear()
        self.assertIs(classify(os.path.join(stdlib_root, "graphlib.py"), True), True)
        # The torch arm reads the torch roots, not the stdlib ones, and the
        # stdlib dir itself is never a pip target.
        self.assertIs(classify(torch.__file__, False), True)
        self.assertIs(classify(torch.__file__, True), False)
        self.assertIs(classify(torch_in_stdlib, False), False)
        # Evidence in neither direction: a relative path would resolve against
        # a cwd it was not recorded under, and posixpath.realpath hands an
        # embedded NUL to os.lstat, which raises ValueError (from 3.11.5/3.12
        # on, gh-106242, ntpath.realpath swallows it, so nothing to pin there).
        self.assertIsNone(classify("graphlib.py", True))
        self.assertIsNone(classify("graphlib.py", False))
        if sys.platform != "win32":
            nul = os.path.join(stdlib_root, "graph\x00lib.py")
            self.assertIsNone(classify(nul, True))
        # Before 3.13 ntpath.isabs accepts a driveless path (its LEGACY BUG),
        # which realpath resolves against the current drive; replay the Windows
        # path module here so the gate is pinned on every platform.
        import ntpath

        patched = mock.patch.object
        with patched(sys, "platform", "win32"), patched(os, "path", ntpath):
            self.assertIsNone(classify(r"\Lib\graphlib.py", True))
        # purelib nests inside stdlib (conda) or platstdlib (venv), so the
        # install-root exclusion is what refuses an installed file; with no
        # install root known, _INSTALL_DIR_NAMES still does.
        patch = functools.partial(mock.patch.object, precompile_package)
        nested = os.path.join(stdlib_root, "vendored", "graphlib", "__init__.py")
        with patch("_install_roots", return_value=()):
            for dir_name in ("site-packages", "dist-packages"):
                installed = os.path.join(stdlib_root, dir_name, "graphlib.py")
                self.assertIs(classify(installed, True), False, dir_name)
            self.assertIs(classify(nested, True), True)
        classify.cache_clear()
        roots = (norm(os.path.join(stdlib_root, "vendored")),)
        with patch("_install_roots", return_value=roots):
            self.assertIs(classify(nested, True), False)
        # _INSTALL_DIR_NAMES is matched below the stdlib root the file is under,
        # so an interpreter bundled inside another environment's site-packages
        # keeps its own stdlib.
        bundled = os.path.join(stdlib_root, "site-packages", "runtime", "lib")
        below = os.path.join(bundled, "site-packages", "graphlib.py")
        with (
            patch("_install_roots", return_value=()),
            patch("_stdlib_roots", return_value=(norm(bundled),)),
        ):
            self.assertIs(classify(os.path.join(bundled, "graphlib.py"), True), True)
            self.assertIs(classify(below, True), False)

    def test_located_reads_the_file_from_the_module_dict(self):
        located, machinery = precompile_package._located, importlib.machinery
        builtin, frozen = machinery.BuiltinImporter, machinery.FrozenImporter
        stdlib_root = sysconfig.get_paths()["stdlib"]
        installed = os.path.join(stdlib_root, "site-packages", "graphlib.py")
        module = types.ModuleType("graphlib")
        module.__file__ = installed
        self.assertIs(located(module, "graphlib", True), False)
        module.__file__ = os.path.join(stdlib_root, "graphlib.py")
        self.assertIs(located(module, "graphlib", True), True)

        # Only the module dict is read. A class attribute is not in it (torch.ops
        # is a ModuleType subclass whose __file__ is the class's "_ops.py"), and
        # getattr would run a PEP 562 module __getattr__, user code that may
        # raise from inside a lint.
        class Shadow(types.ModuleType):
            __file__ = installed

        self.assertIsNone(located(Shadow("graphlib"), "graphlib", True))
        self.assertNotIn("__file__", vars(torch.ops))
        self.assertIsNone(located(torch.ops, "torch.ops", False))
        raising = types.ModuleType("graphlib")
        raising.__getattr__ = mock.Mock(side_effect=RuntimeError("no __file__"))
        self.assertIsNone(located(raising, "graphlib", True))
        raising.__getattr__.assert_not_called()
        self.assertIsNone(located(types.ModuleType("graphlib"), "graphlib", True))
        # sys.modules can hold any object: the __dict__ and spec.loader reads
        # are user code on a slotted proxy or a hand-rolled spec, and a __file__
        # that is not a string would raise from isabs.
        odd = types.ModuleType("graphlib")
        odd.__file__ = 42
        self.assertIsNone(located(odd, "graphlib", True))

        class Proxy:
            __slots__ = ()

            def __getattr__(self, attr):
                raise RuntimeError(attr)

        self.assertIsNone(located(Proxy(), "graphlib", True))  # type: ignore[arg-type]
        odd.__spec__ = Proxy()
        self.assertIsNone(located(odd, "graphlib", True))
        odd.__file__ = os.path.join(stdlib_root, "graphlib.py")
        self.assertIs(located(odd, "graphlib", True), True)  # the loader is not needed
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
        # Built in or frozen, the table is keyed on the full dotted name.
        self.assertIs(located(sys, "sys", True), True)
        sub = types.ModuleType("sys.sub")
        sub.__spec__ = machinery.ModuleSpec("sys.sub", builtin, origin="built-in")
        self.assertIs(located(sub, "sys.sub", True), False)
        for name, expected in (("zipimport", True), ("graphlib", False)):
            module = types.ModuleType(name)
            module.__spec__ = machinery.ModuleSpec(name, frozen, origin="frozen")
            self.assertIs(located(module, name, True), expected, name)
        # __loader__ alone in the dict, with no spec, is the same evidence.
        for name, loader in (("sys", builtin), ("zipimport", frozen)):
            module = types.ModuleType(name)
            module.__loader__ = loader
            self.assertIs(located(module, name, True), True, name)

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
        # A class has no code object, and namedtuple and type() (make_dataclass
        # from 3.12) stamp __module__ from the calling frame under a BARE
        # __qualname__, so a class a library mints for this file claims it
        # exactly like a class statement written here. The methods tell: a
        # class statement compiled its defs in this file under its own qualname
        # prefix, and a class with no such def fails closed, including a
        # factory fed a same-file def (bare qualname) and, below, an imported
        # class the reader attaches one to.
        cls = type(self)
        self.assertTrue(defined_where_read(cls, cls.__name__, _HERE))
        self.assertFalse(defined_where_read(cls, cls.__name__, _ELSEWHERE))
        self.assertFalse(defined_where_read(torch.nn.Linear, "Linear", _HERE))
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
        # A class imported from another file is not written here however many
        # same-file functions the reader attaches to it.
        imported = {"__name__": "mypkg.impl"}
        source = "class Point:\n    def norm(self):\n        return 0\n"
        exec(compile(source, F.__file__, "exec"), imported)
        imported["Point"].extra = _user_op
        self.assertFalse(defined_where_read(imported["Point"], "Point", _HERE))

        # A method extracted under its own name and a def returned by a factory
        # are assignments, not a def under its own name: __qualname__ tells.
        # Ops itself, a same-file class statement with a method, is waived, so
        # is one whose only def is a property (the fget arm); a cached_property
        # keeps its function under .func and is not unwrapped, so a class with
        # nothing else fails closed, as does a class with no method of its own
        # and one whose only methods are generated (a fields-only dataclass
        # compiles them in <string>). On 3.14 the compiler also stores the PEP
        # 649 annotate function of an annotated class body in its __dict__,
        # compiled in this file under the class's prefix; the predicate skips
        # it, replayed under both keys since only 3.14 mints it.
        class Ops:
            @staticmethod
            def op(x):
                return x

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

        def availability_fork():
            def _user_op(x):
                return x + 2

            return _user_op

        self.assertTrue(defined_where_read(Ops, Ops.__qualname__, _HERE))
        self.assertTrue(defined_where_read(Prop, Prop.__qualname__, _HERE))
        self.assertFalse(defined_where_read(Cached, Cached.__qualname__, _HERE))
        self.assertFalse(defined_where_read(Marker, Marker.__qualname__, _HERE))
        self.assertFalse(defined_where_read(Cfg, Cfg.__qualname__, _HERE))
        for key in ("__annotate__", "__annotate_func__"):
            annotate = types.FunctionType(_user_op.__code__, globals(), "__annotate__")
            annotate.__qualname__ = f"Cfg.{key}"
            annotated = type("Cfg", (), {key: annotate})
            self.assertFalse(defined_where_read(annotated, "Cfg", _HERE), key)
        self.assertFalse(defined_where_read(Ops.op, "op", _HERE))
        self.assertFalse(defined_where_read(availability_fork(), "_user_op", _HERE))
        # A co_filename is not always a path: an exec-generated frame records
        # <string>, which realpath would resolve against the cwd, so it would
        # compare equal to the <string> a fields-only dataclass compiles its
        # methods in and waive the one class the lint must report. Only
        # absolute filenames compare; a relative one and an embedded NUL on
        # either side fail closed instead of raising.
        pseudo = traceback.StackSummary.from_list([("<string>", 1, "forward", "")])
        self.assertEqual(Cfg.__init__.__code__.co_filename, "<string>")
        self.assertFalse(defined_where_read(Cfg, Cfg.__qualname__, pseudo))
        exec_ns = {}
        exec(compile("def op(x):\n    return x\n", "<string>", "exec"), exec_ns)
        self.assertFalse(defined_where_read(exec_ns["op"], "op", pseudo))
        relative = os.path.basename(__file__)
        stack = traceback.StackSummary.from_list([(relative, 1, "forward", "")])
        self.assertFalse(defined_where_read(_user_op, "_user_op", stack))
        stack = traceback.StackSummary.from_list(
            [(__file__ + "\x00", 1, "forward", "")]
        )
        self.assertFalse(defined_where_read(_user_op, "_user_op", stack))
        code = _user_op.__code__.replace(co_filename=__file__ + "\x00")
        nul_op = types.FunctionType(code, globals(), "_user_op")
        self.assertFalse(defined_where_read(nul_op, "_user_op", _HERE))
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


instantiate_parametrized_tests(TestPrecompilePackage)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
