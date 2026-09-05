# Owner(s): ["oncall: pt2"]
import importlib
import io
import linecache
import pickle
import sys
import types
import typing

import torch
import torch.utils._pytree as _pytree
from torch._precompile import PrecompileError
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)


# A module-level (global) model + a function referencing it, to exercise the
# constant-tensor guard against a baked global.
_GLOBAL_TENSOR = torch.randn(3)

# A plain scalar global folded into the output must be baked by the dynamo
# tracer, not left dangling as an uncovered external reference.
_GLOBAL_SCALE = 10


_PRECOMPILE_FIXED_INPUT = torch.randn(4)


_PRECOMPILE_PUBLIC_METHODS = [
    name
    for name in dir(torch.compiler.precompile)
    if not name.startswith("_") and callable(getattr(torch.compiler.precompile, name))
]

_PRECOMPILE_X4 = torch.randn(4)

_PRECOMPILE_X28 = torch.randn(2, 8)


_PRECOMPILE_GRAD_MODES_SEEN: list[bool] = []


# A custom pytree node whose context (a set) is not JSON-dumpable and which has no
# to_dumpable_context serializer, so treespec_dumps raises TypeError (distinct from the
# unregistered-namedtuple NotImplementedError path). Registered once at module load and
# used by test_unserializable_context_in_spec_still_compiles.
class _UnserializableCtxInput:
    def __init__(self, a, b):
        self.a = a
        self.b = b


_pytree.register_pytree_node(
    _UnserializableCtxInput,
    lambda n: ([n.a, n.b], {"ctx"}),
    lambda children, _ctx: _UnserializableCtxInput(children[0], children[1]),
    serialized_type_name="test_precompile._UnserializableCtxInput",
)


def _strip_artifact(cache: bytes) -> bytes:
    """Return the cache envelope with its compiled artifact removed, forcing load()
    onto the inlined (no-cache) path that JIT-compiles from python_code. Many tests
    reload the same artifact both cache-primed and stripped to check they agree."""
    blob = torch.load(io.BytesIO(cache), weights_only=True)
    blob["artifact"] = None
    buf = io.BytesIO()
    torch.save(blob, buf)
    return buf.getvalue()


_LUT_MODULE = None


_precompile_reads_shadowed = {
    "pytype": lambda x: x * x.pytype,
    "fake_mode": lambda x: x * x.fake_mode,
    "dispatch_keys": lambda x: x * x.dispatch_keys,
    "_fake_device": lambda x: x * x._fake_device,
}


# A user global the rendered inductor source shadows with a name of its own.
BACKEND = 5.0


_DRIFT_MODULE = None


_PRECOMPILE_ACCUM_RAN: list[str] = []


_PRECOMPILE_RECURSIVE_CAPTURE: list = []


_PRECOMPILE_CLOSING_CAPTURE: list = []


# precompile drives make_fx internally, which cannot symbolically trace a
# dynamo-optimized function; the whole suite is therefore incompatible with
# PYTORCH_TEST_WITH_DYNAMO (dynamo_wrapped CI), so skip it there.
def _multigraph_step(m, x, scale=2.0):
    y = m(x)
    torch._dynamo.graph_break()
    return y * scale


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

    @staticmethod
    def _module_with(src: str, name: str):
        """A real module whose globals are exactly what the source binds."""
        mod = types.ModuleType(name)
        mod.__file__ = f"{name}.py"
        linecache.cache[mod.__file__] = (
            len(src),
            None,
            src.splitlines(True),
            mod.__file__,
        )
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
        sys.modules[name] = mod
        return mod

    def _multigraph_frames(self, code):
        from torch._precompile import _parse_artifact_metadata

        return _parse_artifact_metadata(code)["FRAMES"]

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
