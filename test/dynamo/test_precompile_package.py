# Owner(s): ["module: dynamo"]


import torch
import torch._dynamo.precompile_package as precompile_package
import torch._inductor.test_case
from torch._dynamo.guards import CheckFunctionManager, GuardBuilder, strip_local_scope
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
        class Local:
            pass

        # The one divergence: TYPE_MATCH marks a class whose __qualname__ is not
        # its __name__ here and serialize_guards refuses it through this
        # attribute. The filter keeps it so the refusal stays loud rather than
        # shipping an artifact that never checks the type.
        g = GlobalSource("g")
        local_type = _entry(g, None, "TYPE_MATCH")
        local_type.orig_guard._unserializable = Local
        entries = [
            _entry(g, None, "TENSOR_MATCH"),
            _entry(g, None, "TYPE_MATCH"),
            local_type,
            # An id_match_unchecked on a builtin records ID_MATCH as its derived
            # type; serialize_guards takes its TYPE_MATCH/BUILTIN_MATCH branch
            # first and never reaches the derived-type refusal, so neither does
            # the filter.
            _entry(g, None, "BUILTIN_MATCH", derived=("ID_MATCH",)),
        ]
        keep = precompile_package.default_guard_filter_fn(entries)
        self.assertEqual(keep, [True] * 4)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
