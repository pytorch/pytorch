# Owner(s): ["oncall: pt2"]
import importlib
import pickle
import sys
import types
import typing

import torch
from torch._precompile import PrecompileError
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)


_PRECOMPILE_PUBLIC_METHODS = [
    name
    for name in dir(torch.compiler.precompile)
    if not name.startswith("_") and callable(getattr(torch.compiler.precompile, name))
]


# precompile drives make_fx internally, which cannot symbolically trace a
# dynamo-optimized function; the whole suite is therefore incompatible with
# PYTORCH_TEST_WITH_DYNAMO (dynamo_wrapped CI), so skip it there.
@skipIfTorchDynamo("precompile's make_fx capture is incompatible with dynamo wrapping")
@instantiate_parametrized_tests
class TestPrecompile(TestCase):
    def test_summary_types_pickle(self):
        # A capture summary or invariants report is the kind of value users
        # stash next to an artifact (torch.save of a diagnostics record, a
        # multiprocessing capture farm). A previous revision pointed these
        # classes' __module__ at torch.compiler, which does not export them,
        # so pickle could not resolve the class and every instance raised.
        from torch.compiler._precompile_types import (
            FrameInvariants,
            GuardFact,
            PrecompileSummary,
        )

        fact = GuardFact("TYPE_MATCH", "L['x']", ("code",), "is int", True)
        inv = FrameInvariants("f", "f.py", 1, 2, (fact,), (), ())
        summary = PrecompileSummary(1, 0, 1, 1, ())
        for obj in (fact, inv, summary):
            self.assertEqual(pickle.loads(pickle.dumps(obj)), obj)

    @parametrize("name", _PRECOMPILE_PUBLIC_METHODS)
    def test_precompile_public_members_resolve(self, name):
        typing.get_type_hints(getattr(torch.compiler.precompile, name))

    def test_no_dispatchable_graph_names_the_cause(self):
        # An entry frame with no variants has two very different causes. If
        # Dynamo BYPASSED the frame it recorded why, and saying so beats the
        # thin-wrapper advice, which in that case is simply wrong. Only the
        # ENTRY's own bypassed codes count: an unrelated bypassed helper frame
        # must not relabel a thin-wrapper entry as a bypass.
        from torch._dynamo.package import SerializedCode
        from torch._precompile import _reject_uninstallable_entry

        def fwd_loss_bwd():
            pass

        def helper():
            pass

        def bypassed_code(fn):
            return types.SimpleNamespace(
                bypassed=True,
                bypass_reason="cannot pickle 'generator' object",
                install_to_global=False,
                python_code=SerializedCode.from_code_object(fn.__code__),
            )

        entry = types.SimpleNamespace(
            fn_name="fwd_loss_bwd", codes=[bypassed_code(fwd_loss_bwd)]
        )
        # The state a bypassed ENTRY actually arrives in: _multigraph_frames
        # DROPS bypassed codes, so there is no entry frame at all -- the
        # diagnostic must fire from the empty list, not from a variant-less
        # entry frame it would never see.
        with self.assertRaisesRegex(PrecompileError, "were BYPASSED during capture"):
            _reject_uninstallable_entry([], entry)
        with self.assertRaisesRegex(PrecompileError, "cannot pickle 'generator'"):
            _reject_uninstallable_entry([], entry)
        # An entry frame that compiled but produced no variants, with a
        # bypassed sibling code of the same name, reports the bypass too.
        frames = [{"is_entry": True, "variants": []}]
        with self.assertRaisesRegex(PrecompileError, "were BYPASSED during capture"):
            _reject_uninstallable_entry(frames, entry)
        foreign = types.SimpleNamespace(
            fn_name="fwd_loss_bwd", codes=[bypassed_code(helper)]
        )
        with self.assertRaisesRegex(PrecompileError, "thin wrapper"):
            _reject_uninstallable_entry(frames, foreign)
        # No entry frame and only a FOREIGN bypassed code: neither diagnostic
        # applies, so neither may fire as a guess.
        _reject_uninstallable_entry([], foreign)
        with self.assertRaisesRegex(PrecompileError, "thin wrapper"):
            _reject_uninstallable_entry(
                frames, types.SimpleNamespace(fn_name="step", codes=[])
            )

    def test_precompile_module_identity(self):
        # torch.compiler.precompile is a submodule: re-importing it resolves to the
        # SAME module object, and its name is the stable public path.
        p = torch.compiler.precompile
        self.assertIs(importlib.import_module("torch.compiler.precompile"), p)
        self.assertIs(sys.modules["torch.compiler.precompile"], p)
        self.assertEqual(p.__name__, "torch.compiler.precompile")

    @parametrize("name", _PRECOMPILE_PUBLIC_METHODS)
    def test_precompile_member_module_and_qualname_resolve_to_it(self, name):
        # Nothing hung off the singleton rewrites __module__/__qualname__: the
        # docs place these under torch.compiler.precompile.<name>, but only a
        # name torch.compiler.__all__ exports may claim torch.compiler, or
        # pickle cannot resolve the class and inspect cannot find its source.
        member = getattr(torch.compiler.precompile, name)
        target = sys.modules[member.__module__]
        for part in member.__qualname__.split("."):
            target = getattr(target, part)
        self.assertIs(target, getattr(member, "__func__", member))


if __name__ == "__main__":
    run_tests()
