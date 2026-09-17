# Owner(s): ["module: dynamo"]

import builtins
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
import torch.utils._pytree as pytree
from torch._dynamo.aot_compile import AOTCompiledFunction
from torch._dynamo.exc import PackageError
from torch._dynamo.guards import CheckFunctionManager, GuardBuilder, strip_local_scope
from torch._dynamo.package import load_guards_state
from torch._dynamo.source import get_global_source_name, GlobalSource
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
        # as a keys-match. A load rebuilds the guards against the live registry,
        # so no filter notices a change made before it; after load, the
        # DictGuardManager the registry's other kept guards build notices a
        # registration by length with or without the keys-match, and a
        # same-count change of keys is what only the keys-match notices.
        def fn_tree(x):
            return pytree.tree_flatten({"a": x, "b": x * 2})[0][1]

        def dropping(entries):
            # The predicate before the exemption: drops on the derived DICT_VERSION.
            keep = precompile_package.default_guard_filter_fn(entries)
            promoted = ["DICT_VERSION" in e.derived_guard_types for e in entries]
            return [k and not p for k, p in zip(keep, promoted)]

        def load(data):
            # Module globals the kept guards read (G['pytree']) resolve against
            # the live scope, as in test_aot_compile.py's f_globals=globals() loads.
            return AOTCompiledFunction.deserialize(data, f_globals=globals())

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

        class Extra:
            pass

        register(Node)
        seen = []
        x = torch.randn(3)
        compiled = _aot_compile(fn_tree, x, seen=seen)
        promoted = [keep for e, keep in seen if "DICT_VERSION" in e.derived_guard_types]
        self.assertEqual(promoted, [True])
        self.assertIn("DICT_KEYS_MATCH", _kept_types(compiled)[1])
        data = AOTCompiledFunction.serialize(compiled).serialized_data
        control = _aot_compile(fn_tree, x, guard_filter_fn=dropping)
        self.assertNotIn("DICT_KEYS_MATCH", _kept_types(control)[1])
        control_data = AOTCompiledFunction.serialize(control).serialized_data
        # A node registered before load is baked into the rebuilt guards.
        register(Extra)
        self.assertTrue(load(data).guard_check(x))
        deregister(Extra)
        loaded, loaded_control = load(data), load(control_data)
        self.assertEqual(loaded(x), fn_tree(x))
        register(Extra)
        self.assertFalse(loaded.guard_check(x))
        self.assertFalse(loaded_control.guard_check(x))
        deregister(Extra)
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

    def test_stdlib_roots_take_the_windows_dlls_dir(self):
        # Reachable only on a Windows runner otherwise. sysconfig's first init
        # imports _sysconfigdata_*_{sys.platform}_*, so prime it unpatched.
        stdlib_roots, norm = precompile_package._stdlib_roots, precompile_package._norm
        sysconfig.get_paths()
        with mock.patch.object(sys, "platform", "win32"):
            self._clear_root_caches()
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
        with (
            mock.patch.object(sysconfig, "get_paths", return_value=paths),
            mock.patch.object(sys, "_stdlib_dir", frozen, create=True),
        ):
            self._clear_root_caches()
            stdlib, install = stdlib_roots(), install_roots()
            # A frozen os has no __file__ (None stands in for the missing
            # attribute): the other three sources are then all there is, in
            # sorted order, which _classify_file reads as outermost first.
            with mock.patch.object(os, "__file__", None):
                stdlib_roots.cache_clear()
                no_os = stdlib_roots()
        for root in (base, venv, frozen, os.path.dirname(norm(os.__file__))):
            self.assertIn(norm(root), stdlib)
        self.assertEqual(no_os, tuple(sorted(norm(r) for r in (base, venv, frozen))))
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
        # merged-lib host resolves /usr/lib64 onto /usr/lib.
        prefix = os.path.join(os.sep, "lib64build")
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
        # On Windows getsitepackages() lists the bare prefix; replay that shape
        # here so the exclusion is pinned on every platform, not just there.
        listed = [sys.prefix, *site.getsitepackages()]
        with mock.patch.object(site, "getsitepackages", return_value=listed):
            self._clear_root_caches()
            stdlib = precompile_package._stdlib_roots()
            install = precompile_package._install_roots()
        norm, within = precompile_package._norm, precompile_package._within
        self.assertNotIn(norm(sys.prefix), install)
        for root in stdlib:
            self.assertFalse(within(root, install), root)
        self.assertIn(norm(sysconfig.get_paths()["purelib"]), install)
        # posix_home puts purelib AT the stdlib directory rather than under it;
        # a candidate a stdlib root equals is not an install root either, or the
        # install-wins rule would read the whole stdlib as third party.
        home = os.path.join(os.sep, "home", "lib", "python")
        paths = dict(sysconfig.get_paths())
        paths.update(stdlib=home, platstdlib=home, purelib=home, platlib=home)
        with mock.patch.object(sysconfig, "get_paths", return_value=paths):
            self._clear_root_caches()
            install = precompile_package._install_roots()
        self.assertNotIn(norm(home), install)

    def test_install_roots_skip_a_site_accessor_that_raises(self):
        # A site.py that cannot answer is skipped, not propagated: this runs
        # inside save()'s lint, which must not abort the capture. sysconfig's
        # purelib is read outside the try and the other accessor is still
        # consulted, so what a failing accessor loses is only its own extras.
        user_site = os.path.join(os.sep, "elsewhere", "user-site")
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
        extra = os.path.join(os.sep, "elsewhere", "site")
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
        bogus = os.path.join(os.sep, "elsewhere", "torch")
        stub = types.SimpleNamespace(__path__=[None, bogus])
        self._clear_root_caches()
        with mock.patch.dict(sys.modules, {"torch": stub}):
            # A substituted torch's __path__ is ignored until it lists the torch
            # package directory this file sits under; then every string entry is
            # adopted, the bogus one included, which an editable build relies
            # on. A __path__ that is no sequence at all is ignored as well.
            self.assertEqual(torch_roots(), (own,))
            stub.__path__.append(os.path.dirname(torch.__file__))
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
