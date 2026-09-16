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
        refused = [_entry(GlobalSource("g"), None, guard_type=t) for t in unsupported]
        self.assertEqual(
            precompile_package.default_guard_filter_fn(refused),
            [False] * len(unsupported),
        )
        entries = [
            _entry(GlobalSource("g"), None, "TENSOR_MATCH"),
            # Looser than serialize_guards, which refuses a TYPE_MATCH on a
            # local-scope class. orig_guard._unserializable would tell, but a
            # dropped guard ships an artifact that never checks the type; kept,
            # the serializer refuses it loudly.
            _entry(GlobalSource("g"), None, "TYPE_MATCH"),
            _entry(GlobalSource("g"), None, "TYPE_MATCH", derived=("NN_MODULE",)),
            # Every BUILTIN_MATCH is an id_match_unchecked that records ID_MATCH
            # as its derived type, so none survives although serialize_guards
            # would accept them: a builtin rebound between capture and load goes
            # unnoticed, like every other identity drop.
            _entry(GlobalSource("g"), None, "BUILTIN_MATCH", derived=("ID_MATCH",)),
        ]
        self.assertEqual(
            precompile_package.default_guard_filter_fn(entries),
            [True, True, False, False],
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
