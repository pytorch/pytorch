# Copyright (c) 2026, Tri Dao.
"""The single torch custom op for all epilogue-GEMM objects (HANDOFF Tier 4).

``quack::gemm_epi(str digest, Tensor[] ins, Tensor(a!)[] outs, str meta)``:
one mutating op + one no-op fake covers every epilogue — including
user-defined ``@gemm_epilogue`` fns, which otherwise have no torch.compile
story (precedent: torch.compile's own triton_kernel_wrapper_mutation).
``digest`` resolves the epilogue through an in-process registry, falling back
to import by the module locator carried in ``meta`` (so compiled artifacts
survive process boundaries when the epilogue is bound to an importable name —
the same constraint the async-compile pool imposes).

``ins`` is a packed list of the non-None input tensors, named positionally by
``meta['ins_names']``; ``outs`` carries D + declared outputs + reduce-partial
buffers (all graph-owned, since the op only mutates). Host constants (config,
flags, scalar operands) ride ``meta`` as a repr'd dict — Dynamo guards on the
string, which is exactly right since they select compiled behavior.

Reduce sinks under torch.compile: the wrapper pins the config (the partial
buffers must be graph-allocated at exact shapes BEFORE the op runs, so
runtime autotuning inside the op cannot pick a different tiling) and
finalizes the partials with traced torch ops.

A-operand transforms ride the same op: ``meta['transform']`` carries the
handle's semantic digest (+ import locator when the mod is bound to a
module global — @a_transform fns get this automatically), runtime operand
tensors ride ``ins`` as ``ta__<name>``, a layout-owning transform's SF strip
as ``transform_sf`` (its blob is already B, with D's N coming from the blob
rows via n_override). The op body resolves the handle through _TA_REGISTRY
and re-enters ``EpiMod.__call__`` eagerly, which rebuilds the
TransformAOperand bundle from the resolved config.
"""

from __future__ import annotations

import ast
from typing import Optional

import torch

from torch._vendor.quack.blockscaled.operand import BlockScaledFormat
from torch._vendor.quack.gemm_config import GemmConfig, blockscaled_default_config, cta_tile_shape_m
from torch._vendor.quack.gemm_runtime.autotune import _select_mod_config, mod_selection_args
from torch._vendor.quack.gemm_runtime.identity import (
    TORCH_OP_EPI_MODS as _EPI_REGISTRY,
    TORCH_OP_TRANSFORM_MODS as _TA_REGISTRY,
)
from torch._vendor.quack.rounding import RoundingMode


def _sf_encode(SF: torch.Tensor) -> torch.Tensor:
    """e8m0 -> uint8 view across the mutable custom-op boundary (the Inductor
    decompose_auto_functionalized workaround; same seam as
    gemm_interface._sf_encode). Decoded format-driven in the op body."""
    return SF.view(torch.uint8) if SF.dtype == torch.float8_e8m0fnu else SF


def _sf_decode(SF, bs_format):
    if SF is not None and SF.dtype == torch.uint8:
        SF = SF.view(BlockScaledFormat.from_name(bs_format).scale_dtype)
    return SF


def _resolve(digest: str, locator, registry=_EPI_REGISTRY, what="epilogue"):
    mod = registry.get(digest)
    if mod is None and locator:
        import importlib

        module = importlib.import_module(locator[0])
        mod = getattr(module, locator[1])
        if mod.semantic_digest != digest:
            raise RuntimeError(
                f"{what} {locator[0]}.{locator[1]} changed since this graph was compiled"
            )
        registry[digest] = mod
    if mod is None:
        raise RuntimeError(
            f"{what} digest not resolvable in this process; bind the object "
            "to a module-global name in an importable module"
        )
    return mod


@torch.library.custom_op("torch_vendor_quack::gemm_epi", mutates_args={"outs"}, device_types="cuda")
def _gemm_epi(digest: str, ins: list[torch.Tensor], outs: list[torch.Tensor], meta: str) -> None:
    m = ast.literal_eval(meta)
    mod = _resolve(digest, m["locator"])
    named = dict(zip(m["ins_names"], ins))
    operands = {k[4:]: v for k, v in named.items() if k.startswith("op__")}
    operands.update(m["scalar_ops"])
    i = 0
    out = {}
    if m["store_d"]:
        out["D"] = outs[0]
        i = 1
    for name in m["out_names"]:
        out[name] = outs[i]
        i += 1
    for name in m["sink_names"]:  # exact-shape partials: finalized by the wrapper
        operands[name] = outs[i]
        i += 1
    cfg = GemmConfig(**m["config"]) if m["config"] is not None else None
    t = m.get("transform")
    transform_a = None
    transform_operands = None
    if t is not None:
        transform_a = _resolve(t["digest"], t["locator"], _TA_REGISTRY, "transform")
        transform_operands = {k[4:]: v for k, v in named.items() if k.startswith("ta__")} or None
    mod(
        named["A"],
        named["B"],
        named.get("C"),
        out=out,
        store_d=m["store_d"],
        config_constraints=m.get("config_constraints", ()),
        config=cfg,
        tuned=m["tuned"],
        cu_seqlens_m=named.get("cu_seqlens_m"),
        A_idx=named.get("A_idx"),
        SFA=_sf_decode(named.get("SFA"), m.get("bs_format_a")),
        SFB=_sf_decode(named.get("SFB"), m.get("bs_format_b")),
        bs_format_a=m.get("bs_format_a"),
        bs_format_b=m.get("bs_format_b"),
        rounding_mode=m["rounding_mode"],
        transform_a=transform_a,
        transform_sf=named.get("transform_sf"),
        transform_operands=transform_operands,
        add_to_output=m.get("add_to_output", False),
        **operands,
    )


@_gemm_epi.register_fake
def _gemm_epi_fake(digest, ins, outs, meta) -> None:
    # Pure no-op: the op only mutates ``outs``; compilation is owned by
    # jit_cache + the async pool at real execution time.
    return


def _sink_config_from_meta(mod, named, outputs, meta, transform_a):
    """Resolve the exact sink config outside Dynamo for fake and real allocation."""
    if meta["config"] is not None:
        return GemmConfig(**meta["config"])
    A, B = named["A"], named["B"]
    b_kn = (
        A.device.type == "cuda"
        and torch.cuda.get_device_capability(A.device)[0] >= 9
        and not meta.get("concat_layout")
    )
    B_d = B if b_kn or (transform_a is not None and transform_a.owned_fmt is not None) else B.mT
    selection_args = mod_selection_args(
        {
            **meta["scalar_ops"],
            **{key[4:]: value for key, value in named.items() if key.startswith("op__")},
        },
        outputs,
        A=A,
        B=B_d,
        b_kn=b_kn,
        cu_seqlens_m=named.get("cu_seqlens_m"),
        A_idx=named.get("A_idx"),
        SFA=named.get("SFA"),
        concat_layout=meta.get("concat_layout"),
    )
    if named.get("SFA") is not None:
        n = transform_a.padded_n(B) if transform_a is not None else B.shape[-1]
        preferred_config = blockscaled_default_config(
            A.shape[-2],
            n,
            device_capacity=torch.cuda.get_device_capability(A.device)[0],
        )
    else:
        preferred_config = mod._default_config(A, B, transform_a)
    return _select_mod_config(
        mod,
        A.device,
        meta.get("config_constraints", ()),
        selection_args,
        preferred_config=preferred_config,
        transform_a=transform_a,
    )


def _alloc_outs_from_meta(digest: str, ins: list, meta: str) -> list:
    """Allocate the graph-owned outs list ([D?] + declared outputs + reduce
    partials) from meta + ins alone. Shared by the functional op body and its
    fake: under FakeTensorMode the same torch.empty calls yield fakes, so the
    two sides cannot drift."""
    m = ast.literal_eval(meta)
    mod = _resolve(digest, m["locator"])
    named = dict(zip(m["ins_names"], ins))
    A, B, C = named["A"], named["B"], named.get("C")
    cu, A_idx = named.get("cu_seqlens_m"), named.get("A_idx")
    dt = getattr(torch, m["out_dtype"]) if m.get("out_dtype") else None
    # layout-owning transform: B is the repacked blob, N comes from its rows
    # (the padded-N rule lives on the transform handle)
    t = m.get("transform")
    transform_a = None
    n_ov = None
    if t is not None:
        transform_a = _resolve(t["digest"], t["locator"], _TA_REGISTRY, "transform")
        if t["owned"]:
            n_ov = transform_a.padded_n(B)
    out = mod._alloc_outputs(None, A, B, C, m["store_d"], dt, cu, A_idx, n_override=n_ov)
    outs = ([out["D"]] if m["store_d"] else []) + [out[name] for name in m["out_names"]]
    if m["sink_names"]:
        cfg = _sink_config_from_meta(
            mod, named, {name: out[name] for name in m["out_names"]}, m, transform_a
        )
        lead = mod._lead_shape(A, cu, A_idx)
        n = B.shape[-1] if n_ov is None else n_ov
        cta_tile_m = cta_tile_shape_m(
            cfg.tile_m,
            cfg.cluster_m,
            cfg.device_capacity,
            named.get("SFA") is not None,
        )
        for name in m["sink_names"]:
            op = mod.sinks[name]
            shape = op.sink_alloc_shape(lead, n, cta_tile_m, cfg.tile_n)
            outs.append(torch.empty(shape, dtype=op.sink_alloc_dtype(), device=A.device))
    return outs


@torch.library.custom_op("torch_vendor_quack::gemm_epi_f", mutates_args=(), device_types="cuda")
def _gemm_epi_f(digest: str, ins: list[torch.Tensor], meta: str) -> list[torch.Tensor]:
    """Functional form of ``quack::gemm_epi``: outputs allocated inside, real
    fake — one graph-insertable node per epilogue-GEMM call. Graph-owned
    buffers only; caller-provided out=/partial buffers take the mutating op."""
    outs = _alloc_outs_from_meta(digest, ins, meta)
    torch.ops.torch_vendor_quack.gemm_epi(digest, ins, outs, meta)
    return outs


@_gemm_epi_f.register_fake
def _gemm_epi_f_fake(digest, ins, meta):
    return _alloc_outs_from_meta(digest, ins, meta)


def compile_call(
    mod,
    A,
    B,
    C,
    *,
    out,
    out_dtype,
    store_d,
    config_constraints,
    config,
    tuned,
    cu_seqlens_m,
    A_idx,
    SFA,
    SFB,
    bs_format_a,
    bs_format_b,
    rounding_mode,
    operands,
    transform_a=None,
    transform_sf=None,
    transform_operands=None,
    concat_layout=None,
    add_to_output=False,
):
    """torch.compile-path body of ``EpiMod.__call__``: record one functional
    ``quack::gemm_epi_f`` call (allocation inside the op, so the graph gets a
    single node) and finalize reduces with traced ops. Caller-provided out=/
    partial buffers cannot be graph-owned, so that case keeps the mutating
    ``quack::gemm_epi`` form. Returns the same dict as eager.

    transform_a crosses by semantic digest (registry + optional import
    locator, exactly like the epilogue itself); runtime operand tensors ride
    ``ins`` under ``ta__<name>`` and the op body hands them back to
    ``__call__`` as ``transform_operands`` (the bundle is rebuilt there from
    the config the op resolves — same deterministic path as this trace)."""
    constraints = tuple(config_constraints)
    cfg: Optional[GemmConfig] = config
    caller_owned = bool(out) or any(operands.get(name) is not None for name in mod.sinks)
    sink_names = tuple(name for name in mod.sinks if operands.get(name) is None)
    if sink_names and caller_owned and cfg is None:
        raise NotImplementedError(
            "compiled calls with caller-owned outputs and graph-allocated sinks "
            "require an exact config"
        )
    if mod.sinks and cfg is None:
        # The functional custom op resolves the same deterministic config in
        # fake and real execution before allocating exact-shaped partials.
        tuned = False

    transform_meta = None
    n_override = None
    if transform_a is not None:
        n_override = transform_a.padded_n(B)  # None unless B is a repacked blob
        for k, v in (transform_operands or {}).items():
            assert isinstance(v, torch.Tensor), f"transform operand {k!r} must be a tensor"
        transform_meta = dict(
            digest=transform_a.semantic_digest,  # registered at construction
            locator=transform_a._module_locator(),
            owned=transform_a.owned_fmt is not None,
        )

    ins_names, ins = [], []
    for name, t in (
        ("A", A),
        ("B", B),
        ("C", C),
        ("cu_seqlens_m", cu_seqlens_m),
        ("A_idx", A_idx),
        ("SFA", _sf_encode(SFA) if SFA is not None else None),
        ("SFB", _sf_encode(SFB) if SFB is not None else None),
        ("transform_sf", transform_sf),
        *((f"ta__{k}", v) for k, v in (transform_operands or {}).items()),
        *((f"op__{k}", v) for k, v in operands.items() if isinstance(v, torch.Tensor)),
    ):
        if t is not None:
            ins_names.append(name)
            ins.append(t)
    scalar_ops = {k: v for k, v in operands.items() if not isinstance(v, torch.Tensor)}
    meta = repr(
        dict(
            ins_names=tuple(ins_names),
            out_names=tuple(mod.outputs),
            sink_names=sink_names,
            store_d=bool(store_d),
            tuned=bool(tuned),
            config_constraints=constraints,
            config=None if cfg is None else cfg.__dict__,
            rounding_mode=int(rounding_mode),
            bs_format_a=bs_format_a,
            bs_format_b=bs_format_b,
            scalar_ops=scalar_ops,
            out_dtype=None if out_dtype is None else str(out_dtype).split(".")[-1],
            locator=mod._module_locator(),
            transform=transform_meta,
            concat_layout=concat_layout,
            add_to_output=bool(add_to_output),
        )
    )
    if caller_owned:
        out = mod._alloc_outputs(
            out, A, B, C, store_d, out_dtype, cu_seqlens_m, A_idx, n_override=n_override
        )
        lead = mod._lead_shape(A, cu_seqlens_m, A_idx)
        n = B.shape[-1] if n_override is None else n_override
        partials = {}
        cta_tile_m = cta_tile_shape_m(
            cfg.tile_m, cfg.cluster_m, cfg.device_capacity, SFA is not None
        )
        num_seqs = None if cu_seqlens_m is None else cu_seqlens_m.shape[0] - 1
        for name in sink_names:
            op = mod.sinks[name]
            shape = op.sink_alloc_shape(
                lead,
                n,
                cta_tile_m,
                cfg.tile_n,
                num_seqs=num_seqs if getattr(op, "dim", 0) == 1 else None,
            )
            partials[name] = torch.empty(shape, dtype=op.sink_alloc_dtype(), device=A.device)
        outs = []
        if store_d:
            outs.append(out["D"])
        outs.extend(out[name] for name in mod.outputs)
        outs.extend(partials.values())
        torch.ops.torch_vendor_quack.gemm_epi(mod.semantic_digest, ins, outs, meta)
    else:
        if cu_seqlens_m is not None and any(
            getattr(mod.sinks[name], "dim", 0) == 1 for name in sink_names
        ):
            # The functional op picks its config internally, so the wrapper
            # can't know the per-CTA tile the cu_tiles finalize needs.
            raise NotImplementedError(
                "varlen_m M-fold sinks require the caller-owned op path "
                "(pass out= buffers) — the functional path can't finalize "
                "per-sequence partials"
            )
        cta_tile_m = None
        outs = torch.ops.torch_vendor_quack.gemm_epi_f(mod.semantic_digest, ins, meta)
        i = 1 if store_d else 0
        out = {"D": outs[0]} if store_d else {}
        for j, name in enumerate(mod.outputs):
            out[name] = outs[i + j]
        i += len(mod.outputs)
        partials = {name: outs[i + j] for j, name in enumerate(sink_names)}
    result = dict(out) if store_d else {k: v for k, v in out.items() if k != "D"}
    for name, buf in partials.items():
        op = mod.sinks[name]
        finalize_varlen = getattr(op, "host_finalize_varlen", None)
        if cu_seqlens_m is not None and cta_tile_m is not None and finalize_varlen is not None:
            result[name] = finalize_varlen(buf, cu_seqlens_m, cta_tile_m)
        else:
            finalize = getattr(op, "host_finalize", None)
            result[name] = finalize(buf) if finalize is not None else buf
    return result


_DEFAULT_RN = RoundingMode.RN  # re-export convenience for the __call__ branch
