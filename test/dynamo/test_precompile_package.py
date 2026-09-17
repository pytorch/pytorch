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
import tempfile
import traceback
import types
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


def _user_op(x):
    return x + 1


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


_BUILTINS_DICT = GlobalSource("__builtins_dict___0")
_HERE = traceback.StackSummary.from_list([(__file__, 1, "forward", "")])
_ELSEWHERE = traceback.StackSummary.from_list([(F.__file__, 1, "forward", "")])


def _entry(source, value, guard_type="ID_MATCH", derived=()):
    guard = Guard(source, getattr(GuardBuilder, guard_type))
    guard.guard_types = list(derived) or None
    return GuardFilterEntry(
        name=strip_local_scope(source.name),
        has_value=True,
        value=value,
        guard_type=guard_type,
        derived_guard_types=tuple(derived),
        is_global=get_global_source_name(source) is not None,
        orig_guard=guard,
    )


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
        def fn(x):
            return _user_op(x) + len(x.shape)

        seen = []
        x = torch.randn(3)
        compiled = _aot_compile(fn, x, seen=seen)
        state, kept = _kept_types(compiled)
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

    def test_default_guard_filter_keeps_the_pytree_registry_keys_match(self):
        # The DICT_KEYS_MATCH on SUPPORTED_NODES reaches the filter as the
        # unsaved build's DICT_VERSION and is kept; the save build serializes it
        # as a keys-match. The registry's other kept guards put a
        # DictGuardManager over it whose length check notices a node registered
        # after capture with or without the keys-match; a same-count change of
        # keys is what only the keys-match notices.
        def fn_tree(x):
            return pytree.tree_flatten({"a": x, "b": x * 2})[0][1]

        def deregister(cls):
            if cls in pytree.SUPPORTED_NODES:
                pytree._deregister_pytree_node(cls)

        def register(cls):
            # Only a registry key here, so the <locals> in it is harmless; it
            # must be unique because nameless registrations share one slot of
            # SERIALIZED_TYPE_TO_PYTHON_TYPE and deregistering the first fails.
            name = f"{__name__}.{cls.__qualname__}"
            pytree.register_pytree_node(
                cls, lambda n: ([], None), lambda c, _: cls(), serialized_type_name=name
            )
            self.addCleanup(deregister, cls)

        class Node:
            pass

        class Other:
            pass

        register(Node)
        seen = []
        x = torch.randn(3)
        compiled = _aot_compile(fn_tree, x, seen=seen)
        promoted = [keep for e, keep in seen if "DICT_VERSION" in e.derived_guard_types]
        self.assertEqual(promoted, [True])
        self.assertIn("DICT_KEYS_MATCH", _kept_types(compiled)[1])
        data = AOTCompiledFunction.serialize(compiled).serialized_data
        # Module globals the kept guards read (G['pytree']) resolve against the
        # live scope, as in test_aot_compile.py's f_globals=globals() loads.
        loaded = AOTCompiledFunction.deserialize(data, f_globals=globals())
        self.assertEqual(loaded(x), fn_tree(x))
        deregister(Node)
        register(Other)
        self.assertFalse(loaded.guard_check(x))

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
            keep = precompile_package.default_guard_filter_fn(entries)
            return [k and e.guard_type != "TYPE_MATCH" for k, e in zip(keep, entries)]

        # Not assertRaises: it stores the exception with its traceback cleared.
        def refusal_frames(regex, guard_filter_fn, fn, *args):
            try:
                _aot_compile(fn, *args, guard_filter_fn=guard_filter_fn)
            except PackageError as e:
                self.assertRegex(str(e), regex)
                return {f.name for f in traceback.extract_tb(e.__traceback__)}
            self.fail("serialized")

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
        refusal_frames(refused_module, None, fn3, x, LocalModule())

        class Other(torch.nn.Module):
            def forward(self, x):
                return x - 1

        compiled = _aot_compile(fn3, x, LocalModule(), guard_filter_fn=drop_type_match)
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
        # or platstdlib (venv; platlib on a lib64 build) and must survive the
        # exclusion, while on Windows
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

    def test_stdlib_roots_follow_a_symlink_farm_into_the_store(self):
        # The other symlink shape: the stdlib directory is real and each file in
        # it is a link into a per-package store (a venv over a Nix, Guix or Spack
        # profile). realpath of the directory stays on the farm while realpath
        # of any file lands in the store, so the roots need both.
        stdlib_roots = precompile_package._stdlib_roots
        norm, within = precompile_package._norm, precompile_package._within
        self.addCleanup(stdlib_roots.cache_clear)
        with tempfile.TemporaryDirectory() as tmp:
            store = os.path.join(tmp, "store", "lib", "python3.12")
            farm = os.path.join(tmp, "profile", "lib", "python3.12")
            os.makedirs(store)
            os.makedirs(farm)
            open(os.path.join(store, "os.py"), "w").close()
            farm_os = os.path.join(farm, "os.py")
            try:
                os.symlink(os.path.join(store, "os.py"), farm_os)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable")
            with mock.patch.object(os, "__file__", farm_os):
                stdlib_roots.cache_clear()
                stdlib = stdlib_roots()
            self.assertIn(norm(farm), stdlib)
            self.assertIn(norm(store), stdlib)
            self.assertTrue(within(norm(farm_os), stdlib))

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

    def test_install_roots_take_purelib_and_platlib_on_a_lib64_build(self):
        # sysconfig's posix schemes hard-code lib in purelib but interpolate
        # platlibdir into platlib and the stdlib keys, so on a system
        # interpreter built --with-platlibdir=lib64 (Fedora, RHEL, openSUSE) it
        # is platlib that nests inside the stdlib root and purelib that does
        # not; each key is a root of its own. A venv links lib64 -> lib.
        stdlib = os.path.join(os.sep, "usr", "lib64", "python3.12")
        purelib = os.path.join(os.sep, "usr", "lib", "python3.12", "site-packages")
        platlib = os.path.join(stdlib, "site-packages")
        paths = dict(sysconfig.get_paths())
        paths.update(stdlib=stdlib, platstdlib=stdlib, purelib=purelib, platlib=platlib)
        stdlib_roots = precompile_package._stdlib_roots
        install_roots = precompile_package._install_roots
        norm, within = precompile_package._norm, precompile_package._within
        self.addCleanup(stdlib_roots.cache_clear)
        self.addCleanup(install_roots.cache_clear)
        with mock.patch.object(sysconfig, "get_paths", return_value=paths):
            stdlib_roots.cache_clear()
            install_roots.cache_clear()
            stdlib_found, install = stdlib_roots(), install_roots()
        self.assertTrue(within(norm(platlib), stdlib_found))
        self.assertFalse(within(norm(purelib), stdlib_found))
        self.assertIn(norm(platlib), install)
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
        # posix_home puts purelib AT the stdlib directory rather than under it;
        # a candidate a stdlib root equals is not an install root either, or the
        # install-wins rule would read the whole stdlib as third party.
        home = os.path.join(os.sep, "home", "lib", "python")
        paths = dict(sysconfig.get_paths())
        paths.update(stdlib=home, platstdlib=home, purelib=home, platlib=home)
        self.addCleanup(precompile_package._stdlib_roots.cache_clear)
        with mock.patch.object(sysconfig, "get_paths", return_value=paths):
            precompile_package._stdlib_roots.cache_clear()
            precompile_package._install_roots.cache_clear()
            install = precompile_package._install_roots()
        self.assertNotIn(norm(home), install)

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

    def test_install_roots_take_only_the_strings_a_site_accessor_lists(self):
        # A non-string entry is dropped rather than handed to realpath outside
        # the try, and the None getusersitepackages returns where there is no
        # home directory (WASI) is skipped like a raise.
        extra = os.path.join(os.sep, "elsewhere", "site")
        self.addCleanup(precompile_package._install_roots.cache_clear)
        with (
            mock.patch.object(site, "getsitepackages", return_value=[None, extra]),
            mock.patch.object(site, "getusersitepackages", return_value=None),
        ):
            precompile_package._install_roots.cache_clear()
            install = precompile_package._install_roots()
        norm = precompile_package._norm
        self.assertIn(norm(extra), install)
        self.assertIn(norm(sysconfig.get_paths()["purelib"]), install)

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
        # Of nested stdlib roots the outermost is matched, so the part below it
        # is the longest and the check the strictest.
        classify.cache_clear()
        with (
            patch("_install_roots", return_value=()),
            patch("_stdlib_roots", return_value=(norm(stdlib_root), norm(bundled))),
        ):
            self.assertIs(classify(os.path.join(bundled, "graphlib.py"), True), False)

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

        self.assertIsNone(located(Proxy(), "graphlib", True))
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
        # A __file__ that cannot be placed (not a string, or relative) is no
        # evidence, so the loader is still read.
        unplaced = (("sys", 42, builtin), ("zipimport", "zipimport.py", frozen))
        for name, file, loader in unplaced:
            module = types.ModuleType(name)
            module.__file__ = file
            module.__loader__ = loader
            self.assertIs(located(module, name, True), True, name)
        # Built in or frozen, the table is keyed on the full dotted name.
        self.assertIs(located(sys, "sys", True), True)
        sub = types.ModuleType("sys.sub")
        sub.__spec__ = machinery.ModuleSpec("sys.sub", builtin, origin="built-in")
        self.assertIs(located(sub, "sys.sub", True), False)
        for name, expected in (("zipimport", True), ("graphlib", False)):
            module = types.ModuleType(name)
            module.__spec__ = machinery.ModuleSpec(name, frozen, origin="frozen")
            self.assertIs(located(module, name, True), expected, name)
        # __loader__ alone in the dict, with no spec, is the same evidence, and
        # both arms answer under either flag: the torch shape is an embedding
        # that registers torch._C through PyImport_AppendInittab.
        for name, loader in (("sys", builtin), ("zipimport", frozen)):
            module = types.ModuleType(name)
            module.__loader__ = loader
            for stdlib in (True, False):
                self.assertIs(located(module, name, stdlib), True, (name, stdlib))
        self.assertNotIn("torch._C", sys.builtin_module_names)
        inittab = (*sys.builtin_module_names, "torch._C")
        with mock.patch.object(sys, "builtin_module_names", inittab):
            embedded = types.ModuleType("torch._C")
            embedded.__loader__ = builtin
            self.assertIs(located(embedded, "torch._C", False), True)

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

    def test_defined_where_read_refuses_a_pseudo_filename(self):
        defined_where_read = precompile_package._defined_where_read

        # A co_filename is not always a path: an exec-generated frame records
        # <string>, which realpath would resolve against the cwd, so it would
        # compare equal to the <string> a fields-only dataclass compiles its
        # methods in and waive the one class the lint must report. Only
        # absolute filenames compare; a relative one and an embedded NUL on
        # either side fail closed instead of raising.
        @dataclasses.dataclass
        class Cfg:
            x: int

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

    def test_defined_where_read_judges_a_class_by_its_own_methods(self):
        defined_where_read = precompile_package._defined_where_read
        # A class has no code object; its methods tell. A class statement
        # compiled its defs in this file under its own qualname prefix, and a
        # class with no such def fails closed.
        cls = type(self)
        self.assertTrue(defined_where_read(cls, cls.__name__, _HERE))
        self.assertFalse(defined_where_read(cls, cls.__name__, _ELSEWHERE))
        self.assertFalse(defined_where_read(torch.nn.Linear, "Linear", _HERE))

        # Ops, a same-file class statement with a method, is waived, so is one
        # whose only def is a property (the fget arm); a cached_property keeps
        # its function under .func and is not unwrapped, so a class with
        # nothing else fails closed, as does a class with no method of its own
        # and one whose only methods are generated (a fields-only dataclass
        # compiles them in <string>). Members are unwrapped by type, never
        # probed with getattr: a torch.classes proxy answers any attribute
        # read by raising RuntimeError, and a class holding one is still judged.
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

        class Model(torch.nn.Module):
            ns = torch.classes.precompile_package_test

            def forward(self, x):
                return x

        self.assertTrue(defined_where_read(Ops, Ops.__qualname__, _HERE))
        self.assertTrue(defined_where_read(Prop, Prop.__qualname__, _HERE))
        self.assertFalse(defined_where_read(Cached, Cached.__qualname__, _HERE))
        self.assertFalse(defined_where_read(Marker, Marker.__qualname__, _HERE))
        self.assertFalse(defined_where_read(Cfg, Cfg.__qualname__, _HERE))
        with self.assertRaises(RuntimeError):
            Model.ns.__func__
        self.assertTrue(defined_where_read(Model, Model.__qualname__, _HERE))
        self.assertFalse(defined_where_read(Model, Model.__qualname__, _ELSEWHERE))

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
        # A class imported from another file is not written here however many
        # same-file functions the reader attaches to it.
        imported = {"__name__": "mypkg.impl"}
        source = "class Point:\n    def norm(self):\n        return 0\n"
        exec(compile(source, F.__file__, "exec"), imported)
        imported["Point"].extra = _user_op
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
            self.assertEqual(
                (annotate.__qualname__, annotate.__code__.co_filename),
                (f"{Cfg.__qualname__}.__annotate__", __file__),
            )
        self.assertFalse(defined_where_read(Cfg, Cfg.__qualname__, _HERE))

        # Replayed on every version with a type() class handed a same-file
        # function under that key and qualname. The control row, the same
        # function under an ordinary key, shows the replay is what a class
        # statement produces; the skip of both keys, against a version that
        # stores the function under its own name, is pinned last.
        def minted(key, qualname):
            fn = types.FunctionType(_user_op.__code__, globals(), "__annotate__")
            fn.__qualname__ = qualname
            return type("Cfg", (), {key: fn})

        self.assertTrue(defined_where_read(minted("op", "Cfg.op"), "Cfg", _HERE))
        real = minted("__annotate_func__", "Cfg.__annotate__")
        self.assertFalse(defined_where_read(real, "Cfg", _HERE))
        for key in ("__annotate__", "__annotate_func__"):
            own_key = minted(key, f"Cfg.{key}")
            self.assertFalse(defined_where_read(own_key, "Cfg", _HERE), key)

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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
