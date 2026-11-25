# mypy: allow-untyped-defs
"""torch.vm: runs Python on the torch._rust CFG executor and optimizes it
with one ordered list of passes::

    passes = (
        *vm.DEFAULT_PASSES,      # the VM's own cfg passes
        fuse_swiglu,             # block pass: the forward graph
        vm.passes.Autograd(),    # cfg: block passes after it see the joint graph
        offload_activations,     # block pass: the joint graph
        vm.passes.Partition(),   # cfg: joint -> forward + backward graphs
        vm.passes.Inductor(),    # block pass: compiles each
    )
    step = torch.vm(step, passes=passes)

Each pass is a plain function whose signature says what kind it is:
``(cfg: vm.Cfg) -> None`` is a cfg pass, which changes what a region is;
``(gm: fx.GraphModule, input_types: list[vm.Type]) -> fx.GraphModule`` is a
block pass, which changes what a region computes. ``input_types`` has one
``vm.TensorType`` per input of the region's graph: the join of every value
the VM saw there (``f32[16..32, 64]``), and for a weight, the tensor itself
(``t.const``).

A block pass specializes with guards, which are graph nodes
(``vm.guard(gm, node, *checks)``). Versions are inferred: a pass whose guards
hold for every input the region runs on edits the current version; one whose
guards vary between calls (``vm.Aligned``, ``vm.Is``, ``vm.NotOverlapping``)
stacks a new version, and a call where they don't hold runs the version
below. ``vm.explain(fn)`` shows what each pass did to each region, and
``vm.versions(fn, block)`` how often each version ran.

Without ``passes``, ``torch.vm(fn)`` runs the VM's own passes and compiles
regions with Inductor. Regions and bypassing cold code are process-wide once
a function's schedule turns them on: run one function on the VM per process.
"""

import dataclasses
import functools
import inspect
import sys
import types
import warnings
from collections.abc import Callable, Iterable
from typing import Any

import torch.fx

from torch.vm import _pipeline, passes
from torch.vm._cfg import Block, Branch, Call, CannotGuard, Carried, Cfg, CfgValue, Edit, Loop, Merge, Region, Slice, Span
from torch.vm._graph import (
    async_ops,
    effect,
    first_backward_user,
    forward_values_used_in_backward,
    last_forward_user,
    lower_to,
    mark_recompute,
    match,
    Match,
    Mismatch,
    placeholder,
    replace_backward_uses,
    replace_segment,
    run_until,
    Segment,
    segments,
    specialize_floats,
    swap,
)
from torch.vm._guards import (
    Aligned,
    Check,
    Contiguous,
    Device,
    Dim,
    Divisible,
    Dtype,
    Exact,
    GradEnabled,
    guard,
    Ints,
    Is,
    NoForwardAD,
    NoSavedTensorHooks,
    NotOverlapping,
    NoTorchFunctionMode,
    PlainTensor,
    RequiresGrad,
    Stride,
    StrideOrder,
    TensorMeta,
    Value,
)
from torch.vm._pipeline import VersionInfo
from torch.vm._types import AnyType, ConstType, DimSet, FloatType, IntType, Interval, materialize, nbytes, ObjectType, SeqType, TensorType, Type, type_of, UnionType, Varies


__all__ = [
    "Aligned",
    "AnyType",
    "Block",
    "Branch",
    "Call",
    "CannotGuard",
    "Carried",
    "Cfg",
    "CfgValue",
    "Check",
    "ConstType",
    "Contiguous",
    "DEFAULT_PASSES",
    "Device",
    "Dim",
    "DimSet",
    "Divisible",
    "Dtype",
    "Edit",
    "Exact",
    "FAST",
    "FloatType",
    "GradEnabled",
    "IntType",
    "Interval",
    "Ints",
    "Is",
    "Loop",
    "Match",
    "Merge",
    "Mismatch",
    "NoForwardAD",
    "NoSavedTensorHooks",
    "NoTorchFunctionMode",
    "NotOverlapping",
    "ObjectType",
    "PlainTensor",
    "Region",
    "RequiresGrad",
    "Segment",
    "SeqType",
    "Slice",
    "Span",
    "specialize_floats",
    "Stride",
    "StrideOrder",
    "TensorMeta",
    "TensorType",
    "Type",
    "UnionType",
    "Value",
    "Varies",
    "VersionInfo",
    "async_ops",
    "compile_now",
    "configure",
    "effect",
    "explain",
    "first_backward_user",
    "forward_values_used_in_backward",
    "guard",
    "kind",
    "last_forward_user",
    "lower_to",
    "mark_recompute",
    "match",
    "materialize",
    "nbytes",
    "passes",
    "placeholder",
    "replace_backward_uses",
    "replace_segment",
    "run_until",
    "segments",
    "swap",
    "type_of",
    "versions",
    "vm",
]

# The VM's own passes: its optimizer and region formation (vm.passes.Elide
# is opt-in).
DEFAULT_PASSES: tuple[Any, ...] = (
    passes.Split(),
    passes.PinCalls(),
    passes.UseAten(),
    passes.LiftGuard(),
    passes.PureCalls(),
    passes.Spans(),
    passes.Regions(),
    passes.BypassCold(),
)
# Inductor with CUDA graphs that read the weights in place.
FAST: tuple[Any, ...] = (*DEFAULT_PASSES, passes.static_parameters, passes.Inductor({"triton.cudagraphs": True}))


def _runtime() -> Any:
    """``torch._rust``, the executor that runs a function on the VM; a
    RuntimeError that says why when this build of PyTorch has none."""
    rust = torch._rust
    if rust is None:
        raise RuntimeError(
            "torch.vm needs the torch._rust extension, which this build of PyTorch does not have: the build makes it "
            "with cargo for CPython 3.12 on Linux and skips it elsewhere "
            f"(this is Python {sys.version_info.major}.{sys.version_info.minor} on {sys.platform})"
        )
    return rust


def kind(p: Any) -> str:
    """``"cfg"`` or ``"block"``: what pass ``p`` is, from its first
    parameter's annotation (``vm.Cfg`` or ``fx.GraphModule``), else from how
    many arguments it takes."""
    try:
        sig = inspect.signature(p)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{p!r} isn't a pass: {e}") from e
    positional = [q for q in sig.parameters.values() if q.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    first = positional[0].annotation if positional else inspect.Parameter.empty
    if first is Cfg or first in ("Cfg", "vm.Cfg"):
        return "cfg"
    if first is torch.fx.GraphModule or first in ("GraphModule", "fx.GraphModule", "torch.fx.GraphModule"):
        return "block"
    n = sum(1 for q in positional if q.default is q.empty)
    if n == 1:
        return "cfg"
    if n == 2:
        return "block"
    raise TypeError(f"{p!r} isn't a pass: a cfg pass takes (cfg), a block pass (gm, input_types); it takes {sig}")


def _pass_name(p: Any) -> str:
    return getattr(p, "name", None) or getattr(p, "__name__", None) or type(p).__name__


def configure(base: Iterable[Any] = DEFAULT_PASSES, **changes: Any) -> tuple[Any, ...]:
    """``base`` with passes changed by name (``regions``, ``bypass_cold``, a
    function's name): ``False`` drops one, a dict sets its parameters, and
    another value its first (``regions=8`` is ``Regions(variants=8)``)."""
    out = list(base)
    for name, value in changes.items():
        idx = [i for i, p in enumerate(out) if _pass_name(p) == name]
        if not idx:
            raise ValueError(f"no pass named {name!r} in {', '.join(map(_pass_name, out))}")
        if value is False:
            out = [p for i, p in enumerate(out) if i not in idx]
            continue
        for i in idx:
            p = out[i]
            if not dataclasses.is_dataclass(p):
                raise ValueError(f"pass {name!r} has no parameters")
            params = value if isinstance(value, dict) else {dataclasses.fields(p)[0].name: value}
            out[i] = dataclasses.replace(p, **params)
    return tuple(out)


class _Program:
    """A function on the VM with its passes."""

    def __init__(self, fn: Callable, pass_list: Iterable[Any], verify: bool) -> None:
        cfg = Cfg(fn)
        blocks: list[Any] = []
        positions: dict[int, int] = {}
        markers = 0
        # Cfg passes other than the VM's own: run on each function's CFG.
        self.cfg_passes: list[Any] = []
        self.cfg_notes: list[str] = []
        for p in pass_list:
            if kind(p) == "block":
                positions[len(blocks)] = markers
                blocks.append(p)
                continue
            if isinstance(p, (passes.Autograd, passes.Partition)):
                if isinstance(p, passes.Partition) != (markers == 1):
                    raise ValueError("vm.passes.Autograd() comes once, then vm.passes.Partition()")
                markers += 1
            elif blocks:
                raise ValueError(f"cfg pass {_pass_name(p)} comes after block pass {_pass_name(blocks[-1])}: cfg passes change what a region is, so they come first")
            if isinstance(p, (passes._Builtin, passes.Autograd, passes.Partition, passes.ThreadRng)):
                p(cfg)
            else:
                self.cfg_passes.append(p)
        if markers == 1:
            raise ValueError("vm.passes.Autograd() needs vm.passes.Partition() after it")
        builtin = cfg.builtin
        optimizer = [n in builtin for n in passes.OPTIMIZER]
        if any(optimizer) and not all(optimizer):
            raise NotImplementedError(f"the VM runs {', '.join(passes.OPTIMIZER)} as one optimizer: list all of them or none")
        self.optimize = all(optimizer)
        self.regions = builtin.get("regions")
        if self.regions is not None and not self.optimize:
            raise ValueError("Regions needs the optimizer's passes (Split, PinCalls, UseAten, LiftGuard)")
        if blocks and self.regions is None:
            raise ValueError("block passes run on regions: add vm.passes.Regions()")
        self.cfg = cfg
        self.pipeline = _pipeline.Pipeline(blocks, cfg.autograd, cfg.partition, positions, verify)
        self.fn = fn
        self.calls = 0

    def run(self, *args, **kwargs):
        import torch._rust_inductor as ri

        rust = _runtime()
        builtin = self.cfg.builtin
        regions = self.regions
        # Process-wide settings: set on each call, so each function's passes
        # apply to its calls.
        rust._set_cfg_optimize(self.optimize)
        rust._set_cfg_elide("elide" in builtin)
        rust._set_cfg_pure_calls("pure_calls" in builtin)
        rust._set_cfg_spans("spans" in builtin)
        rust._set_cfg_pass_hook(self._cfg_hook if self.cfg_passes else None)
        rust._set_cfg_scalar_reads(True)
        ri.block_pipeline = self.pipeline
        for begin in self.pipeline.step_begins:
            begin()
        if regions is not None:
            rounds = builtin["pure_calls"].rounds if "pure_calls" in builtin else 0
            observe_at = regions.at - regions.observe
            if observe_at - rounds <= self.calls < observe_at:
                rust._reoptimize_all()
            if self.calls == observe_at:
                rust._set_cfg_region_observe(True)
            if self.calls == regions.at:
                rust._set_cfg_regions(True, 1, regions.variants)
        if "bypass_cold" in builtin and self.calls == builtin["bypass_cold"].at:
            rust._cfg_bypass_cold()
        self.calls += 1
        return rust.call_with_cfg_execution(self.fn, *args, **kwargs)


    def _cfg_hook(self, view: dict[str, Any]) -> dict[str, list[Any]]:
        """Runs the cfg passes on one function's CFG (see torch._rust's
        cfg_view) and returns their edits. A pass that raises is dropped for
        that function, its edits with it."""
        rust = _runtime()
        edits: dict[str, list[Any]] = {"movable": [], "spans": [], "split": [], "opaque": []}
        for p in self.cfg_passes:
            cfg = Cfg(self.fn, view)
            try:
                p(cfg)
            except Exception as e:
                self.cfg_notes.append(f"{cfg.qualname}: {_pass_name(p)} raised {type(e).__name__}: {e}; dropped")
                warnings.warn(f"torch.vm: cfg pass {_pass_name(p)} raised on {cfg.qualname}, so it's skipped there: {type(e).__name__}: {e}", stacklevel=2)
                continue
            for k in edits:
                edits[k].extend(cfg._edits[k])
            self.cfg_notes.extend(f"{cfg.qualname}: {_pass_name(p)}: {n}" for n in cfg.notes)
        for code, origin in edits["opaque"]:
            rust._cfg_opaque_site(code, origin, True)
        return edits


def vm(fn: Callable | None = None, /, *, passes: Iterable[Any] | None = None, verify: bool = False) -> Callable:
    """Returns ``fn`` run on the VM with ``passes`` (a decorator without
    ``fn``). The default is the VM's own passes, which keep eager's numerics
bit for bit; a lowering such as ``vm.passes.Inductor()`` is opt-in. ``verify``
    compares each new version with the one below on the run it was traced
    from, within its replacements' tolerances, and drops it if they differ."""
    if fn is None:
        return functools.partial(vm, passes=passes, verify=verify)
    rust = _runtime()
    program = _Program(fn, DEFAULT_PASSES if passes is None else passes, verify)
    rust._set_cfg_reoptimize_after(2**31)
    for f in program.cfg.skipped:
        rust._cfg_skip(f)

    @functools.wraps(fn)
    def run(*args, **kwargs):
        return program.run(*args, **kwargs)

    run.__vm_program__ = program  # type: ignore[attr-defined]
    return run


def _program(fn: Callable) -> _Program:
    program = getattr(fn, "__vm_program__", None)
    if program is None:
        raise TypeError(f"{fn!r} isn't a function from torch.vm")
    return program


def explain(fn: Callable, *, block: int | None = None, view: str = "cfg") -> str:
    """What the passes did to each region of ``fn`` (a function from
    ``torch.vm``) so far: its input types, versions with their guards and
    hits, and passes that were dropped and why. ``block`` selects one block's
    regions; ``view="fx"`` adds each version's graph."""
    program = _program(fn)
    text = _pipeline.explain(program.pipeline.reports, block, view)
    if program.cfg_notes and block is None:
        text = "cfg passes:\n" + "".join(f"  {n}\n" for n in program.cfg_notes) + text
    return text


def versions(fn: Callable, block: int | None = None) -> list[VersionInfo]:
    """The versions of ``fn``'s regions (of one block's, with ``block``)."""
    return _pipeline.versions(_program(fn).pipeline.reports, block)


def compile_now(fn: Callable, *, types: dict[str, Type]) -> None:
    """Optimizes with declared input types instead of waiting for recorded
    ones. Not implemented yet."""
    raise NotImplementedError("vm.compile_now isn't implemented yet")


class _Module(types.ModuleType):
    def __call__(self, fn: Callable | None = None, /, **kwargs: Any) -> Callable:
        return vm(fn, **kwargs)


sys.modules[__name__].__class__ = _Module
