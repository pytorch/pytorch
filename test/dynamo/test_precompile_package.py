# Owner(s): ["module: dynamo"]

import builtins
import os
import re
import site
import sys
import sysconfig
import tempfile
import traceback
import types
import zipfile
from unittest import mock

import torch
import torch._dynamo.precompile_package as precompile_package
import torch._inductor.test_case
import torch.utils._pytree as pytree
from torch._dynamo.aot_compile import AOTCompiledFunction
from torch._dynamo.exc import PackageError
from torch._dynamo.guards import CheckFunctionManager, GuardBuilder, strip_local_scope
from torch._dynamo.package import load_guards_state
from torch._dynamo.source import get_global_source_name, GlobalSource, LocalSource
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
        filter_fn = precompile_package.default_guard_filter_fn
        unsupported = CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
        # Spelled out: the filter reads the same constant, so on a shrunk one the
        # two would agree on less.
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
            # before it looks at derived types, so the filter keeps them whatever
            # they derive. That branch is not an unconditional accept: it refuses
            # these two for a local-scope type, which is what
            # test_default_guard_filter_keeps_local_type_guards_for_a_loud_refusal
            # covers; the rows are on a local source since that is where the kept
            # TYPE_MATCH the refusal needs sits, and the filter reads no scope.
            ("BUILTIN_MATCH", ("ID_MATCH",)),
            ("TYPE_MATCH", ("ID_MATCH",)),
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
        self._clear_root_caches()
        with mock.patch.dict(sys.modules, {"torch": stub}):
            # A substituted torch's __path__ is ignored until it lists the torch
            # package directory this file sits under; then every absolute string
            # entry is adopted, the bogus one included, which an editable build
            # relies on, and a relative one, which realpath would resolve into
            # the process cwd, is not. A __path__ that is no sequence at all is
            # ignored as well.
            self.assertEqual(torch_roots(), (own,))
            stub.__path__ += [os.path.dirname(torch.__file__), "", "relative"]
            torch_roots.cache_clear()
            self.assertEqual(set(torch_roots()), {own, norm(bogus)})
            stub.__path__ = None
            torch_roots.cache_clear()
            self.assertEqual(torch_roots(), (own,))
        with mock.patch.object(precompile_package, "__file__", None):
            torch_roots.cache_clear()
            self.assertEqual(torch_roots(), ())  # frozen: no directory to anchor to


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
