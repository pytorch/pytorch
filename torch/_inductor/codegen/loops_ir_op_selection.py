"""Isolated measured selection for LoopsIR per-op candidates.

Unfused-only adapter over the frozen SDK
(``loops_ir_op_options`` + ``loops_ir_op_autotune``). No backend / IR /
wrapper / GraphLowering edits; the training worker owns the backend.

Scope (explicit):
* Only variant 0 (unfused) candidates are measured. Every op's fused
  variants are counted and explicitly deferred with
  :data:`FUSION_DEFERRED_REASON` -- never silently treated as selected.
* Fair whole-region comparison is therefore trivially satisfied: all
  measured choices share one signature / output layout, consume only the
  op node, and run from the same external root inputs. Fused choices that
  would consume different producer/consumer nodes and produce different
  final outputs are NOT timed against the naked op here.
* Python factories exist only at compile-time validation/autotune time
  (``autotune_op``). No Python IR interpreter runs in the timed hot path:
  native runners call the SDK ``NativeCall``; generated runners call the
  already-compiled Triton launcher.
* Timing defaults to the vetted serial service
  (``benchmarker.benchmark_gpu(..., is_vetted_benchmarking=True)`` via
  ``autotune_op``'s default). This module never implements manual CUDA
  event timers; pass ``benchmark_fn=None`` for genuine measurement.

Frozen facts recorded per op (for the future FrozenPlan/wrapper owner):
consumed nodes, disjoint shared-claim verification, placement
(execution index), ``produces`` / ``eliminated_buffers``,
selected option/config/cache key/source hash, measurements, and the
native fallback target.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
from typing import Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

log = logging.getLogger(__name__)

__all__ = [
    "FUSION_DEFERRED_REASON",
    "SELECTION_VERSION",
    "SelectedOp",
    "SelectionResult",
    "bind_tensor",
    "select_op_unfused",
    "select_plan",
    "spec_from_layout",
]

SELECTION_VERSION = 1

FUSION_DEFERRED_REASON = (
    "fusion-deferred: unfused-only adapter; fused variants require whole-region "
    "factories (unconsumed prologue/epilogue work + remaining region execution) "
    "owned by the backend worker, so they are counted here but never measured "
    "or selected"
)


def _dtype_str(layout_dtype: Any) -> str:
    name = getattr(layout_dtype, "name", layout_dtype)
    if not isinstance(name, str):
        name = str(name)
    short = name.split(".")[-1].upper()
    mapping = {
        "F16": "float16",
        "BF16": "bfloat16",
        "F32": "float32",
        "F64": "float64",
        "I32": "int32",
        "I64": "int64",
        "BOOL": "bool",
    }
    if short not in mapping:
        raise ValueError(f"unsupported layout dtype {name!r}")
    return mapping[short]


def _kind_str(kind: Any) -> str:
    """Normalize CandidateKind (enum or string) to native/generated/original."""
    name = getattr(kind, "name", kind)
    if not isinstance(name, str):
        name = str(name)
    upper = name.upper()
    if "GENERATED" in upper:
        return "generated"
    if "NATIVE" in upper:
        return "native"
    if "ORIGINAL" in upper or "EXTERN" in upper:
        return "original"
    low = name.lower()
    if low in ("native", "generated", "original", "original_extern"):
        return "original" if low.startswith("original") else low
    raise ValueError(f"unknown candidate kind {kind!r}")


def _canonical_config(config: Any) -> str:
    if hasattr(config, "canonical") and callable(config.canonical):
        return str(config.canonical())
    return str(config)


def bind_tensor(root: torch.Tensor, binding: Any, layout: Any) -> torch.Tensor:
    """Bind ``TensorBinding(root, root_offset)`` to a tensor.

    ``as_strided(root, layout.sizes, layout.strides,
    root.storage_offset() + binding.root_offset)``. Extras are already flat
    root buffers; the generated output pointer is the output root (the
    epilogue ``store_offset`` lives inside the kernel).
    """
    return torch.as_strided(
        root,
        list(layout.sizes),
        list(layout.strides),
        root.storage_offset() + int(binding.root_offset),
    )


def spec_from_layout(layout: Any, offset: int, device_str: str) -> Any:
    """Build an autotune ``TensorSpec`` from a signature layout + binding offset."""
    from torch._inductor.codegen import loops_ir_op_autotune as at

    return at.TensorSpec(
        dtype=_dtype_str(layout.dtype),
        sizes=tuple(int(v) for v in layout.sizes),
        strides=tuple(int(v) for v in layout.strides),
        offset=int(offset),
        device=device_str,
    )


def _numerics_fence(native_ctx: Any) -> Any:
    from torch._inductor.codegen import loops_ir_op_autotune as at
    from torch._inductor.codegen import loops_ir_op_options as oo

    ext = oo.load_op_options_native()
    policy = (
        "strict_eager"
        if native_ctx.numerics == ext.NumericsPolicy.STRICT_EAGER
        else "inductor"
    )
    precision = (
        "tf32"
        if native_ctx.stock_fp32_matmul_precision == ext.MatmulPrecision.TF32
        else "ieee"
    )
    return at.NumericsFence(
        policy=policy,
        enable_fp_fusion=bool(native_ctx.enable_fp_fusion),
        disable_ftz=bool(native_ctx.disable_ftz),
        division_rounding=bool(native_ctx.division_rounding),
        matmul_precision=precision,
    )


def _inductor_only(option_id: str, kind: str) -> bool:
    try:
        from torch._inductor.codegen import loops_ir_op_options as oo

        ext = oo.load_op_options_native()
        fence = ext.default_registry().get(option_id).fence
        return fence == ext.NumericsFence.INDUCTOR_ONLY
    except Exception:
        # Mock/test fallback: generated options are inductor-only by
        # construction (native_* are strict-eager).
        return kind == "generated"


def _source_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class SelectedOp:
    """Frozen per-op selection. JSON-only; never holds callables or tensors."""

    op_pos: int
    decoded_index: int
    node_index: int
    variant: int
    consumed_nodes: tuple[int, ...]
    placement: int
    produces: str
    eliminated_buffers: tuple[str, ...]
    shared_nodes: tuple[int, ...]
    shared_claims_disjoint: bool
    option_id: str
    option_version: int
    kind: str
    config: str
    cache_key: str
    source_hash: str
    measured_ms: float
    origin: str  # cold | cache | singleton | fallback
    measured: bool
    autotuned: bool
    cache_hit: bool
    native_fallback: bool
    fallback_target: str
    numerics: str
    device: str
    fused_variants_deferred: int
    fusion_deferred_reason: str
    measurements_ms: tuple[tuple[str, float], ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "version": SELECTION_VERSION,
            "op_pos": int(self.op_pos),
            "decoded_index": int(self.decoded_index),
            "node_index": int(self.node_index),
            "variant": int(self.variant),
            "consumed_nodes": [int(v) for v in self.consumed_nodes],
            "placement": int(self.placement),
            "produces": self.produces,
            "eliminated_buffers": list(self.eliminated_buffers),
            "shared_nodes": [int(v) for v in self.shared_nodes],
            "shared_claims_disjoint": bool(self.shared_claims_disjoint),
            "option_id": self.option_id,
            "option_version": int(self.option_version),
            "kind": self.kind,
            "config": self.config,
            "cache_key": self.cache_key,
            "source_hash": self.source_hash,
            "measured_ms": float(self.measured_ms),
            "origin": self.origin,
            "measured": bool(self.measured),
            "autotuned": bool(self.autotuned),
            "cache_hit": bool(self.cache_hit),
            "native_fallback": bool(self.native_fallback),
            "fallback_target": self.fallback_target,
            "numerics": self.numerics,
            "device": self.device,
            "fused_variants_deferred": int(self.fused_variants_deferred),
            "fusion_deferred_reason": self.fusion_deferred_reason,
            "measurements_ms": [[k, float(v)] for k, v in self.measurements_ms],
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> SelectedOp:
        if not isinstance(d, dict) or d.get("version") != SELECTION_VERSION:
            raise ValueError("bad SelectedOp version")
        return SelectedOp(
            op_pos=int(d["op_pos"]),
            decoded_index=int(d["decoded_index"]),
            node_index=int(d["node_index"]),
            variant=int(d["variant"]),
            consumed_nodes=tuple(int(v) for v in d["consumed_nodes"]),
            placement=int(d["placement"]),
            produces=str(d["produces"]),
            eliminated_buffers=tuple(str(v) for v in d["eliminated_buffers"]),
            shared_nodes=tuple(int(v) for v in d["shared_nodes"]),
            shared_claims_disjoint=bool(d["shared_claims_disjoint"]),
            option_id=str(d["option_id"]),
            option_version=int(d["option_version"]),
            kind=str(d["kind"]),
            config=str(d["config"]),
            cache_key=str(d["cache_key"]),
            source_hash=str(d["source_hash"]),
            measured_ms=float(d["measured_ms"]),
            origin=str(d["origin"]),
            measured=bool(d["measured"]),
            autotuned=bool(d["autotuned"]),
            cache_hit=bool(d["cache_hit"]),
            native_fallback=bool(d["native_fallback"]),
            fallback_target=str(d["fallback_target"]),
            numerics=str(d["numerics"]),
            device=str(d["device"]),
            fused_variants_deferred=int(d["fused_variants_deferred"]),
            fusion_deferred_reason=str(d["fusion_deferred_reason"]),
            measurements_ms=tuple(
                (str(k), float(v)) for k, v in d.get("measurements_ms", [])
            ),
        )


@dataclasses.dataclass(frozen=True)
class SelectionResult:
    """Frozen plan-level selection (unfused-only; fusion deferred)."""

    selections: tuple[SelectedOp, ...]
    shared_nodes: tuple[int, ...]
    all_disjoint: bool
    numerics: str

    def to_json(self) -> dict[str, Any]:
        return {
            "version": SELECTION_VERSION,
            "selections": [s.to_json() for s in self.selections],
            "shared_nodes": [int(v) for v in self.shared_nodes],
            "all_disjoint": bool(self.all_disjoint),
            "numerics": self.numerics,
        }


def _fallback_selection(
    *,
    op_pos: int,
    op: Any,
    ctx: Any,
    device_str: str,
    numerics_policy: str,
    reason: str,
) -> SelectedOp:
    variant = op.variants[0] if getattr(op, "variants", None) else None
    consumed = (
        tuple(int(v) for v in variant.consumed_nodes)
        if variant is not None
        else (int(op.node_index),)
    )
    placement = int(variant.placement) if variant is not None else int(op.node_index)
    produces = str(getattr(variant, "produces", "")) if variant is not None else ""
    eliminated = (
        tuple(str(v) for v in variant.eliminated_buffers) if variant is not None else ()
    )
    fallback_target = str(getattr(getattr(op, "fallback", None), "target", ""))
    fused_deferred = max(len(getattr(op, "variants", []) or []) - 1, 0)
    log.warning("loops_ir_op_selection: op %d fallback (%s)", op_pos, reason)
    return SelectedOp(
        op_pos=int(op_pos),
        decoded_index=int(getattr(op, "decoded_index", -1)),
        node_index=int(op.node_index),
        variant=0,
        consumed_nodes=consumed,
        placement=placement,
        produces=produces,
        eliminated_buffers=eliminated,
        shared_nodes=(),
        shared_claims_disjoint=True,
        option_id="",
        option_version=0,
        kind="original",
        config="",
        cache_key="",
        source_hash="",
        measured_ms=float("nan"),
        origin="fallback",
        measured=False,
        autotuned=False,
        cache_hit=False,
        native_fallback=True,
        fallback_target=fallback_target,
        numerics=numerics_policy,
        device=device_str,
        fused_variants_deferred=int(fused_deferred),
        fusion_deferred_reason=FUSION_DEFERRED_REASON,
        measurements_ms=(),
    )


def select_op_unfused(
    op: Any,
    ctx: Any,
    *,
    op_pos: int = 0,
    shared_nodes: Sequence[int] = (),
    codegen_fn: Callable[..., Any] | None = None,
    compile_fn: Callable[..., Any] | None = None,
    load_native_fn: Callable[..., Any] | None = None,
    autotune_fn: Callable[..., Any] | None = None,
    benchmark_fn: Callable[[Callable[[], Any]], float] | None = None,
    cache: Any | None = None,
    use_cache: bool = True,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    device_str: str | None = None,
) -> SelectedOp:
    """Measure the unfused (variant 0) candidates of one op and freeze the winner.

    * ``op`` is one ``OpCandidates`` from ``enumerate_op_candidates``.
    * ``ctx`` is a ``FrozenCodegenContext``.
    * Only ``variant == 0`` native/generated candidates are considered;
      fused variants are counted in ``fused_variants_deferred`` and never
      timed (see :data:`FUSION_DEFERRED_REASON`).
    * ``benchmark_fn=None`` selects the vetted serial service
      (``benchmark_gpu`` with ``is_vetted_benchmarking=True``); tests may
      inject a mock. No manual CUDA-event timers are implemented here.
    * Compile/codegen failures fall back to the next valid candidate
      (``autotune_op`` skips them); total failure records a native
      fallback ``SelectedOp`` (``origin="fallback"``) rather than raising.
    """
    from torch._inductor.codegen import loops_ir_op_autotune as at
    from torch._inductor.codegen import loops_ir_op_options as oo

    if codegen_fn is None:
        codegen_fn = oo.codegen
    if compile_fn is None:
        compile_fn = oo.compile_generated_kernel
    if load_native_fn is None:
        load_native_fn = oo.load_native_call
    if autotune_fn is None:
        autotune_fn = at.autotune_op

    dev = device_str or f"cuda:{ctx.device_index}"
    fence = _numerics_fence(ctx.native)
    numerics_policy = fence.policy
    fused_deferred = max(len(getattr(op, "variants", []) or []) - 1, 0)
    fallback_target = str(getattr(getattr(op, "fallback", None), "target", ""))

    if not getattr(op, "variants", None):
        return _fallback_selection(
            op_pos=op_pos,
            op=op,
            ctx=ctx,
            device_str=dev,
            numerics_policy=numerics_policy,
            reason="no variants",
        )
    variant0 = op.variants[0]

    # Unfused variant must bind every input slot (None == prologue-computed,
    # i.e. fused) and carry no extras. Anything else fails closed to fallback:
    # this adapter never fakes a fused measurement.
    try:
        bindings = list(variant0.input_bindings)
    except Exception as exc:
        return _fallback_selection(
            op_pos=op_pos,
            op=op,
            ctx=ctx,
            device_str=dev,
            numerics_policy=numerics_policy,
            reason=f"bindings unreadable: {exc}",
        )
    if any(b is None for b in bindings):
        return _fallback_selection(
            op_pos=op_pos,
            op=op,
            ctx=ctx,
            device_str=dev,
            numerics_policy=numerics_policy,
            reason="variant 0 has prologue-computed (None) input; refusing fused shape",
        )
    if list(getattr(variant0, "extra_bindings", []) or []):
        return _fallback_selection(
            op_pos=op_pos,
            op=op,
            ctx=ctx,
            device_str=dev,
            numerics_policy=numerics_policy,
            reason="variant 0 carries extra inputs; refusing fused shape",
        )

    unfused = [
        (i, c)
        for i, c in enumerate(getattr(op, "candidates", []) or [])
        if int(getattr(c, "variant", -1)) == 0
        and _kind_str(c.kind) in ("native", "generated")
    ]
    if not unfused:
        return _fallback_selection(
            op_pos=op_pos,
            op=op,
            ctx=ctx,
            device_str=dev,
            numerics_policy=numerics_policy,
            reason="no unfused native/generated candidates (OriginalExtern baseline?)",
        )

    sig = variant0.signature
    try:
        input_specs = tuple(
            spec_from_layout(sig.inputs[k].layout, bindings[k].root_offset, dev)
            for k in range(len(bindings))
        )
        output_spec = spec_from_layout(
            sig.output.layout, variant0.output_binding.root_offset, dev
        )
    except Exception as exc:
        return _fallback_selection(
            op_pos=op_pos,
            op=op,
            ctx=ctx,
            device_str=dev,
            numerics_policy=numerics_policy,
            reason=f"spec build failed: {exc}",
        )

    # Codegen each unfused candidate for its source hash (failures skip the
    # candidate; never substitute another option).
    rendered: list[tuple[Any, Any]] = []
    for _i, cand in unfused:
        try:
            rendered_obj = codegen_fn(
                cand.option_id, cand.config, sig, variant0.fused_loops, ctx
            )
        except Exception as exc:
            log.warning(
                "loops_ir_op_selection: op %d candidate %s codegen skipped (%s)",
                op_pos,
                getattr(cand, "option_id", "?"),
                exc,
            )
            continue
        rendered.append((cand, rendered_obj))
    if not rendered:
        return _fallback_selection(
            op_pos=op_pos,
            op=op,
            ctx=ctx,
            device_str=dev,
            numerics_policy=numerics_policy,
            reason="all unfused codegens rejected",
        )

    descriptors: list[Any] = []
    factories: dict[str, Any] = {}
    for pos, (cand, obj) in enumerate(rendered):
        kind = _kind_str(cand.kind)
        kind_at = "native" if kind == "native" else "generated"
        cid = f"v0-{cand.option_id}-{pos:03d}"
        source = getattr(obj, "source", "")
        desc = at.CandidateDescriptor(
            candidate_id=cid,
            option_id=str(cand.option_id),
            option_version=int(cand.option_version),
            kind=kind_at,
            config=_canonical_config(cand.config),
            cache_key=str(cand.cache_key),
            source_hash=_source_hash(str(source)),
            inductor_only=_inductor_only(str(cand.option_id), kind_at),
            requires_zeroed_output=False,
        )
        descriptors.append(desc)

        if kind_at == "native":
            loaded = load_native_fn(obj)

            def _native_factory(_loaded: Any = loaded) -> Any:
                def runner(inputs: Sequence[torch.Tensor], out: torch.Tensor) -> Any:
                    return _loaded(*inputs)

                return runner

            factories[cid] = _native_factory
        else:

            def _gen_factory(_obj: Any = obj) -> Any:
                compiled = compile_fn(_obj, ctx)

                def runner(inputs: Sequence[torch.Tensor], out: torch.Tensor) -> Any:
                    compiled.launch(*inputs, out)
                    return None

                return runner

            factories[cid] = _gen_factory

    baseline_obj = None
    for cand, obj in rendered:
        if bool(getattr(cand, "baseline", False)) and _kind_str(cand.kind) == "native":
            baseline_obj = obj
            break
    if baseline_obj is None:
        for cand, obj in rendered:
            if _kind_str(cand.kind) == "native":
                baseline_obj = obj
                break
    if baseline_obj is None:
        return _fallback_selection(
            op_pos=op_pos,
            op=op,
            ctx=ctx,
            device_str=dev,
            numerics_policy=numerics_policy,
            reason="manifest has no native baseline for the reference runner",
        )
    baseline_loaded = load_native_fn(baseline_obj)

    def reference_runner(inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        return baseline_loaded(*inputs)

    request = at.OpTuneRequest(
        op_name=f"loops_ir_op{op_pos}_{fallback_target or 'op'}",
        numerics=fence,
        inputs=input_specs,
        output=output_spec,
        rtol=float(rtol),
        atol=float(atol),
        equal_nan=True,
    )

    try:
        result = autotune_fn(
            request,
            descriptors,
            factories,
            reference_runner,
            benchmark_fn=benchmark_fn,
            cache=cache,
            use_cache=use_cache,
        )
    except Exception as exc:
        return _fallback_selection(
            op_pos=op_pos,
            op=op,
            ctx=ctx,
            device_str=dev,
            numerics_policy=numerics_policy,
            reason=f"autotune found no valid candidate: {exc}",
        )

    by_id = {d.candidate_id: (c, o) for d, (c, o) in zip(descriptors, rendered)}
    won_cand, _won_obj = by_id[result.winner.candidate_id]
    kind = _kind_str(won_cand.kind)
    consumed = tuple(int(v) for v in variant0.consumed_nodes)
    shared = tuple(int(v) for v in (shared_nodes or ()))
    # Global overlap across ops is enforced by select_plan (fail closed);
    # per-op we freeze the shared set; local disjointness is trivially True
    # for a single claim and is re-verified at plan level.
    disjoint = True
    return SelectedOp(
        op_pos=int(op_pos),
        decoded_index=int(getattr(op, "decoded_index", -1)),
        node_index=int(op.node_index),
        variant=0,
        consumed_nodes=consumed,
        placement=int(variant0.placement),
        produces=str(variant0.produces),
        eliminated_buffers=tuple(str(v) for v in variant0.eliminated_buffers),
        shared_nodes=shared,
        shared_claims_disjoint=bool(disjoint),
        option_id=str(result.winner.option_id),
        option_version=int(result.winner.option_version),
        kind=kind,
        config=str(result.winner.config),
        cache_key=str(result.winner.cache_key),
        source_hash=str(result.winner.source_hash),
        measured_ms=float(result.winner.measured_ms),
        origin=str(result.winner.origin),
        measured=bool(result.measured),
        autotuned=bool(result.autotuned),
        cache_hit=bool(result.cache_hit),
        native_fallback=bool(kind == "native"),
        fallback_target=fallback_target,
        numerics=numerics_policy,
        device=dev,
        fused_variants_deferred=int(fused_deferred),
        fusion_deferred_reason=FUSION_DEFERRED_REASON,
        measurements_ms=tuple(
            (str(k), float(v)) for k, v in dict(result.measurements_ms).items()
        ),
    )


def select_plan(
    plan: Any,
    ctx: Any,
    *,
    codegen_fn: Callable[..., Any] | None = None,
    compile_fn: Callable[..., Any] | None = None,
    load_native_fn: Callable[..., Any] | None = None,
    autotune_fn: Callable[..., Any] | None = None,
    benchmark_fn: Callable[[Callable[[], Any]], float] | None = None,
    cache: Any | None = None,
    use_cache: bool = True,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    device_str: str | None = None,
) -> SelectionResult:
    """Select every op in a ``CandidatePlan`` (unfused-only) and freeze disjointness.

    Raises ``ValueError`` if two selected ops consume overlapping nodes
    (shared-claim violation): selection fails closed rather than double-
    counting shared producer/consumer work.
    """
    shared = tuple(int(v) for v in (getattr(plan, "shared_nodes", None) or ()))
    selections: list[SelectedOp] = []
    seen: set[int] = set()
    for pos, op in enumerate(getattr(plan, "ops", []) or []):
        sel = select_op_unfused(
            op,
            ctx,
            op_pos=pos,
            shared_nodes=shared,
            codegen_fn=codegen_fn,
            compile_fn=compile_fn,
            load_native_fn=load_native_fn,
            autotune_fn=autotune_fn,
            benchmark_fn=benchmark_fn,
            cache=cache,
            use_cache=use_cache,
            rtol=rtol,
            atol=atol,
            device_str=device_str,
        )
        overlap = seen.intersection(set(sel.consumed_nodes))
        if overlap:
            raise ValueError(
                f"loops_ir_op_selection: overlapping consumed nodes {sorted(overlap)} "
                f"(op {pos} claims {list(sel.consumed_nodes)}); shared={list(shared)}"
            )
        seen.update(sel.consumed_nodes)
        selections.append(dataclasses.replace(sel, shared_nodes=shared))
    numerics = str(getattr(plan, "numerics", "")) or str(
        getattr(getattr(ctx, "native", None), "numerics", "")
    )
    return SelectionResult(
        selections=tuple(selections),
        shared_nodes=shared,
        all_disjoint=True,
        numerics=numerics,
    )
