# Owner(s): ["module: dynamo"]

import builtins
from unittest import mock

import torch
import torch._dynamo.precompile_package as precompile_package
import torch._inductor.test_case
from torch._dynamo.aot_compile import AOTCompiledFunction
from torch._dynamo.exc import PackageError
from torch._dynamo.guards import CheckFunctionManager, GuardBuilder, strip_local_scope
from torch._dynamo.package import load_guards_state
from torch._dynamo.source import get_global_source_name, GlobalSource
from torch._dynamo.types import GuardFilterEntry
from torch._guards import Guard


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
        # The kept guard is live in the loaded artifact: a swapped builtin trips
        # that guard, by name, and the original passes again once restored.
        builtins_key = state.output_graph.name_of_builtins_dict_key_in_fglobals
        failed_on_len = rf"___check_obj_id\(G\['{builtins_key}'\]\['len'\]"
        real_len = len
        with mock.patch.object(builtins, "len", lambda *args: real_len(*args)):
            with self.assertRaisesRegex(RuntimeError, failed_on_len):
                loaded(x)
        self.assertTrue(loaded.guard_check(x))

        # A local-scope class passes the filter (the TYPE_MATCH on L['obj'] is
        # kept) and serialization refuses it. The filter keeps the guard so the
        # refusal is loud rather than an artifact that never checks the type.
        # For a plain instance the pickler refuses anyway, wherever it sits in
        # the guard tree; for an nn.Module of a local class it does not (the
        # instance is rebuilt as a plain torch.nn.Module), so there the kept
        # TYPE_MATCH is the only refusal: drop it and the artifact ships, and
        # serves a module of another class.
        class Local:
            n = 1

        def fn2(x, obj):
            return x + obj.n

        compiled = torch.compile(fn2, fullgraph=True, backend="eager", options=options)
        with self.assertRaisesRegex(PackageError, "defined in local scope"):
            compiled.aot_compile(((x, Local()), {}))

        class LocalModule(torch.nn.Module):
            def forward(self, x):
                return x + 1

        def fn3(x, mod):
            return mod(x)

        compiled = torch.compile(fn3, fullgraph=True, backend="eager", options=options)
        with self.assertRaisesRegex(PackageError, "defined in local scope"):
            compiled.aot_compile(((x, LocalModule()), {}))

        def drop_type_match(entries):
            keep = precompile_package.default_guard_filter_fn(entries)
            return [k and e.guard_type != "TYPE_MATCH" for k, e in zip(keep, entries)]

        class Other(torch.nn.Module):
            def forward(self, x):
                return x - 1

        options = {"guard_filter_fn": drop_type_match}
        compiled = torch.compile(fn3, fullgraph=True, backend="eager", options=options)
        compiled = compiled.aot_compile(((x, LocalModule()), {}))
        data = AOTCompiledFunction.serialize(compiled).serialized_data
        loaded = AOTCompiledFunction.deserialize(data)
        self.assertTrue(loaded.guard_check(x, Other()))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
