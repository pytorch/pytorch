# Owner(s): ["module: dynamo"]

import builtins
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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
