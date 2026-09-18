# Owner(s): ["module: dynamo"]

import builtins
import re
import sys
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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
