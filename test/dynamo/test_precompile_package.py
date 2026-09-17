# Owner(s): ["module: dynamo"]

import builtins
import traceback
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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
