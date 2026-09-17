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
from torch._dynamo.source import GlobalSource
from torch._dynamo.types import GuardFilterEntry
from torch._guards import Guard


def _entry(source, value, guard_type="ID_MATCH", derived=()):
    guard = Guard(source, getattr(GuardBuilder, guard_type))
    return GuardFilterEntry(
        name=strip_local_scope(source.name),
        has_value=True,
        value=value,
        guard_type=guard_type,
        derived_guard_types=tuple(derived),
        is_global=isinstance(source, GlobalSource),
        orig_guard=guard,
    )


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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
