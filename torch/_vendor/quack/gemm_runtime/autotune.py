# Copyright (c) 2026, Tri Dao.
"""Generic autotuning for @gemm_epilogue mods.

``tuned_mod_gemm(mod, A, B, D, C, epi_args=...)`` sweeps the arch's GemmConfig
space directly through ``mod.gemm()`` with the existing Autotuner machinery
(CUDA-graph L2-cold bench, async-compile pool overlap, disk cache under
``$QUACK_CACHE_DIR``) — no per-variant torch interface layer involved. Any mod
gets tuning for free; ``mod.gemm_tuned(...)`` is the method form.

Mechanics that make the generic path work with the Autotuner:

* One ``Autotuner`` per (mod semantic digest, epi-arg name set, has-C): the
  Autotuner derives its cache key and its L2-rotate clone sets from TOP-LEVEL
  named tensor kwargs, so ``epi_args`` is flattened into explicit kwargs and
  the wrapper carries a synthetic ``__signature__`` naming them (a dict value
  would neither key nor clone — every bench replay would share one buffer).
* ``mod_digest`` rides ``key=`` so editing the epilogue fn body invalidates
  in-memory AND disk tuning caches; the wrapper ``__name__`` embeds it so the
  ``<fn>.autotune.json`` files stay human-attributable.
* Reduce-sink buffers are tile-shaped ((l, m, n_tiles) / (l, m_tiles, n)):
  callers allocate them at the sweep's worst case (``sink_arg_shapes``) and
  the wrapper slices per config, so one buffer serves every tile size. The
  winning slice is returned in ``TunedModGemm.sinks``.
* mod.gemm validation errors (ValueError/TypeError) are rewrapped into
  RuntimeError: the bench loop only converts RuntimeError/MemoryError into an
  inf timing, and a config a prune rule missed must not abort the sweep.
* Scalar epi args do not enter the tuning key (only tensor metadata does):
  tile choice is insensitive to scalar VALUES, and mod.gemm re-plans per
  metadata on the real call anyway.

varlen_m/gather_A tune through this path (cu_seqlens_m/A_idx are top-level
tensor kwargs, so they key and clone like any operand); swap_ab rides
swap-at-trace for element-mode sink-less mods; ``dynamic_scheduler=True``
forces dynamic-persistent scheduling on every candidate (matching the old
per-variant tuned wrappers); blockscaled SFA/SFB sweeps the
_blockscaled_ok-pruned space; ``concat_layout`` enters the tuner key (the
old per-variant tuners aliased concat/non-concat winners); A-operand
transforms tune through it too (the handle's semantic digest keys the tuner,
``transform_a.config_ok`` + geometry validation prune the space, and
runtime-operand bundles are rebuilt per config so their strip views bake the
candidate tiles). Not supported (yet): split_k.
"""

from __future__ import annotations

import inspect
from functools import partial
from typing import NamedTuple

import torch

from torch._vendor.quack.autotuner import AutotuneConfig, Autotuner
from torch._vendor.quack.cute_dsl_utils import get_device_capacity
from torch._vendor.quack.gemm_config import (
    blockscaled_config_ok,
    canonicalize_config_constraints,
    config_supports,
    cta_tile_shape_m,
    get_all_configs,
)

__all__ = ["tuned_mod_gemm", "sink_arg_shapes", "TunedModGemm"]


class TunedModGemm(NamedTuple):
    plan: object  # GemmEpiPlan of the winning call (already executed)
    config: object  # winning GemmConfig
    sinks: dict  # name -> the winning config's slice of the caller's buffer


def _cdiv(a, b):
    return (a + b - 1) // b


def _config_space(mod, device, config_constraints=()):
    """Coarse per-arch config list for this mod, filtered by requested fields."""
    constraints = canonicalize_config_constraints(config_constraints)
    cap = get_device_capacity(device)[0]
    hint = "gated" if mod.mode in ("acc_pair", "packed_cd_b16x2") else None
    cfgs = [
        c
        for c in get_all_configs(epilogue=hint)
        if c.device_capacity == cap
        # Swap-at-trace is admitted only when every orientation-sensitive
        # EpiOp owns its transposed physical geometry.
        and not (c.swap_ab and not mod.supports_swap_ab())
        and not c.use_tma_gather  # gather_A untested through the fn frontend
        and (c.split_k is None or c.split_k == 1)  # split-K is default-epilogue-only
        and all(getattr(c, name) == value for name, value in constraints)
    ]
    if not cfgs:
        detail = f" matching config_constraints={dict(constraints)!r}" if constraints else ""
        raise ValueError(f"no GemmConfigs{detail} for device capacity {cap}")
    return cfgs


def _gemm_mn(A, B, b_kn):
    n = B.shape[-1] if b_kn else B.shape[-2]
    m = A.shape[-2] if A.ndim == 3 else A.shape[0]
    return m, n


def _lead(A, A_idx, m_gemm):
    """(batch?, m) lead shape matching EpiMod._lead_shape (m_gemm already
    accounts for gather)."""
    return (A.shape[0], m_gemm) if A.ndim == 3 else (m_gemm,)


def _sink_slice(buf, shape):
    """The leading `shape` view of a worst-case sink buffer."""
    if tuple(buf.shape) == tuple(shape):
        return buf
    return buf[tuple(slice(0, s) for s in shape)]


def sink_arg_shapes(
    mod,
    m,
    n_gemm,
    l=None,
    device="cuda",
    num_seqs=None,
    config_constraints=None,
):
    """Return worst-case sink-buffer shapes over the matching config sweep."""
    cfgs = _config_space(
        mod, torch.device(device), canonicalize_config_constraints(config_constraints)
    )
    min_tile_n = min(c.tile_n for c in cfgs)
    # blockscaled=False halves whenever the config could run 2-CTA — the
    # smallest possible per-CTA tile, so the buffer upper-bounds both modes.
    min_tile_m = min(cta_tile_shape_m(c.tile_m, c.cluster_m, c.device_capacity) for c in cfgs)
    lead = (m,) if l is None else (l, m)
    shapes = {}
    for name, op in mod.sinks.items():
        alloc = getattr(op, "sink_alloc_shape", None)
        if alloc is None:
            continue
        # cdiv is monotone in the tile size, so the min tiles give the sweep's
        # worst case (config-independent sinks ignore the tiles).
        shapes[name] = alloc(
            lead,
            n_gemm,
            min_tile_m,
            min_tile_n,
            num_seqs=num_seqs if getattr(op, "dim", 0) == 1 else None,
        )
    return shapes


def _slice_sinks(mod, epi_args, config, lead, n_gemm, blockscaled=False, num_seqs=None):
    views = {}
    for name, op in mod.sinks.items():
        alloc = getattr(op, "sink_alloc_shape", None)
        if alloc is None or name not in epi_args:
            continue
        cta_tile_m = cta_tile_shape_m(
            config.tile_m, config.cluster_m, config.device_capacity, blockscaled
        )
        views[name] = _sink_slice(
            epi_args[name],
            alloc(
                lead,
                n_gemm,
                cta_tile_m,
                config.tile_n,
                num_seqs=num_seqs if getattr(op, "dim", 0) == 1 else None,
            ),
        )
    return views


def _prune_for_mod(mod, transform_a, configs, named_args, *, config_constraints=(), **kwargs):
    kwargs = named_args | kwargs
    A, B = kwargs["A"], kwargs["B"]
    n_full = transform_a.padded_n(B) if transform_a is not None else None
    cap = get_device_capacity(A.device)[0]
    A_idx = kwargs.get("A_idx")
    m_gemm, n_gemm = _gemm_mn(A, B, kwargs.get("b_kn", False))
    if A_idx is not None:
        m_gemm = A_idx.shape[0]
    has_out = bool(mod.outputs)
    survivors = []
    b_kn_call = kwargs.get("b_kn", False)
    varlen_m = kwargs.get("cu_seqlens_m") is not None
    varlen_or_gather = varlen_m or A_idx is not None
    blockscaled = kwargs.get("SFA") is not None
    has_concat = bool(kwargs.get("concat_layout"))
    epi_ops = tuple(
        dict.fromkeys(
            (
                *mod.ops.values(),
                *mod.sinks.values(),
                *mod.output_ops.values(),
                *mod.extra_ops,
            )
        )
    )
    for conf in configs:
        c = conf.kwargs["config"]
        if c.device_capacity != cap:
            continue
        if not config_supports(c, gather_A=A_idx is not None, varlen_m=varlen_m):
            continue
        if transform_a is not None:
            if not transform_a.config_ok(c):
                continue
            if n_full is not None and n_full % c.tile_m:
                continue  # blob tiles kernel-M in whole CTA tiles
        if any(
            not supports(c)
            for op in epi_ops
            if (supports := getattr(op, "supports_config", None)) is not None
        ):
            continue
        if any(
            not supports(c, m_gemm, n_gemm)
            for op in epi_ops
            if (supports := getattr(op, "supports_problem", None)) is not None
        ):
            continue
        if blockscaled and not blockscaled_config_ok(c):
            continue
        if c.swap_ab and (
            not b_kn_call
            or n_gemm % 8
            or varlen_or_gather
            or has_concat
            or not mod.supports_swap_ab()
        ):
            continue
        if mod.mode == "acc_pair":
            if c.tile_n % 2:
                continue
            if cap == 9 and has_out and c.tile_n % 32:
                continue
        ok = True
        cta_tile_m = cta_tile_shape_m(c.tile_m, c.cluster_m, c.device_capacity, blockscaled)
        for name, op in mod.sinks.items():
            if getattr(op, "check_oob", True) is False:
                ragged = n_gemm % c.tile_n if getattr(op, "dim", 0) == 0 else m_gemm % cta_tile_m
                if ragged:
                    ok = False
            buf = kwargs.get(name)
            alloc = getattr(op, "sink_alloc_shape", None)
            if buf is not None and alloc is not None:
                need = alloc(_lead(A, A_idx, m_gemm), n_gemm, cta_tile_m, c.tile_n)
                if any(b < s for b, s in zip(buf.shape, need)):
                    ok = False  # caller's partial buffer too small for this tiling
        if ok:
            survivors.append(conf)
    if not survivors:
        raw_configs = tuple(conf.kwargs["config"] for conf in configs)
        diagnostic_configs = (
            tuple(_config_space(mod, A.device)) if config_constraints else raw_configs
        )
        unsupported_ops = [
            op
            for op in epi_ops
            if (supports := getattr(op, "supports_config", None)) is not None
            and not any(supports(config) for config in diagnostic_configs)
        ]
        if len(unsupported_ops) == 1 and hasattr(unsupported_ops[0], "config_support_error"):
            raise ValueError(
                "no supported GemmConfig: "
                f"{unsupported_ops[0].config_support_error(diagnostic_configs)}"
            )
        if config_constraints:
            raise ValueError(
                "no supported GemmConfig matches "
                f"config_constraints={dict(config_constraints)!r} for this call"
            )
        raise ValueError("no supported GemmConfig for this epilogue and call")
    return survivors


def _make_tuned_fn(mod, epi_names, transform_a=None, ta_names=()):
    sink_allocs = {
        n: op.sink_alloc_shape for n, op in mod.sinks.items() if hasattr(op, "sink_alloc_shape")
    }

    def fn(
        A=None,
        B=None,
        D=None,
        C=None,
        mod_digest=None,
        b_kn=False,
        cu_seqlens_m=None,
        A_idx=None,
        dynamic_scheduler=False,
        SFA=None,
        SFB=None,
        bs_format_a=None,
        bs_format_b=None,
        concat_layout=None,
        config_constraints=(),
        transform_digest=None,  # keyed; the mod itself is a closure capture
        transform_sf=None,
        config=None,
        **epi_flat,  # epi args by name + transform operands as ta__<name>
    ):
        c = config
        m_gemm, n_gemm = _gemm_mn(A, B, b_kn)
        if transform_a is not None and transform_a.padded_n(B) is not None:
            n_gemm = transform_a.padded_n(B)  # B is the repacked blob
        if A_idx is not None:
            m_gemm = A_idx.shape[0]
        lead = _lead(A, A_idx, m_gemm)
        num_seqs = None if cu_seqlens_m is None else cu_seqlens_m.shape[0] - 1
        epi_args = {}
        for name in epi_names:
            v = epi_flat[name]
            alloc = sink_allocs.get(name)
            if alloc is not None and isinstance(v, torch.Tensor):
                op_dim = getattr(mod.sinks.get(name), "dim", 0)
                v = _sink_slice(
                    v,
                    alloc(
                        lead,
                        n_gemm,
                        c.tile_m,
                        c.tile_n,
                        num_seqs=num_seqs if op_dim == 1 else None,
                    ),
                )
            epi_args[name] = v
        dyn = c.is_dynamic_persistent or dynamic_scheduler
        # SM90 dynamic-persistent scheduling consumes a semaphore; a fresh
        # zeros(1) per call is the gemm_interface pattern (under the CUDA-graph
        # bench the captured memset re-zeros it on every replay).
        sem = None
        if dyn and get_device_capacity(A.device)[0] == 9:
            sem = torch.zeros(1, dtype=torch.int32, device=A.device)
        B_pass, bkn_pass = B, b_kn
        if c.swap_ab and not b_kn:
            B_pass, bkn_pass = B.mT, True  # swap_ab requires B given (k, n)
        try:
            A_pass = A
            if transform_a is not None and transform_a.needs_operands:
                # per-config bundle: the strip views bake this config's tiles
                # (a geometry mismatch with the caller's strips raises here —
                # pre-compile — and benches as inf)
                A_pass = transform_a.bundle(
                    A, {n: epi_flat[f"ta__{n}"] for n in ta_names}, c.tile_m, c.tile_k
                )
            return mod.gemm(
                A_pass,
                B_pass,
                D,
                C,
                epi_args=epi_args,
                tile_M=c.tile_m,
                tile_N=c.tile_n,
                tile_K=None if SFA is not None else c.tile_k,
                cluster_M=c.cluster_m,
                cluster_N=c.cluster_n,
                pingpong=c.pingpong,
                is_dynamic_persistent=dyn,
                max_swizzle_size=c.max_swizzle_size,
                tile_count_semaphore=sem,
                cu_seqlens_m=cu_seqlens_m,
                A_idx=A_idx,
                SFA=SFA,
                SFB=SFB,
                bs_format_a=bs_format_a,
                bs_format_b=bs_format_b,
                concat_layout=concat_layout,
                b_kn=b_kn,
                swap_ab=c.swap_ab,
                transform_a=transform_a,
                transform_sf=transform_sf,
            )
        except (ValueError, TypeError, AssertionError) as e:
            # The bench loop only maps RuntimeError/MemoryError to an inf
            # timing; a config the prune missed must not abort the sweep.
            raise RuntimeError(f"config {c} rejected: {e}") from e

    fn.__name__ = f"mod_{mod._ident}"
    kw = inspect.Parameter.KEYWORD_ONLY
    params = [
        inspect.Parameter(n, kw, default=None)
        for n in (
            "A",
            "B",
            "D",
            "C",
            "mod_digest",
            "cu_seqlens_m",
            "A_idx",
            "SFA",
            "SFB",
            "bs_format_a",
            "bs_format_b",
            "config_constraints",
            "config",
        )
    ]
    params.append(inspect.Parameter("b_kn", kw, default=False))
    params.append(inspect.Parameter("dynamic_scheduler", kw, default=False))
    params.append(inspect.Parameter("concat_layout", kw, default=None))
    params.append(inspect.Parameter("transform_digest", kw, default=None))
    params.append(inspect.Parameter("transform_sf", kw, default=None))
    params.extend(inspect.Parameter(f"ta__{n}", kw, default=None) for n in ta_names)
    params.extend(inspect.Parameter(n, kw, default=None) for n in epi_names)
    fn.__signature__ = inspect.Signature(params)
    return fn


_MOD_TUNERS: dict = {}


def _get_tuner(
    mod, epi_names, has_c, device, transform_a=None, ta_names=(), config_constraints=None
):
    constraints = canonicalize_config_constraints(config_constraints)
    key = (
        mod.semantic_digest,
        epi_names,
        has_c,
        get_device_capacity(device)[0],
        getattr(transform_a, "semantic_digest", None),
        ta_names,
        constraints,
    )
    tuner = _MOD_TUNERS.get(key)
    if tuner is None:
        tuner = Autotuner(
            _make_tuned_fn(mod, epi_names, transform_a, ta_names),
            key=[
                "mod_digest",
                "config_constraints",
                "b_kn",
                "dynamic_scheduler",
                "concat_layout",
                "bs_format_a",
                "bs_format_b",
                "transform_digest",
            ],
            configs=[AutotuneConfig(config=c) for c in _config_space(mod, device, constraints)],
            prune_configs_by={
                "early_config_prune": partial(
                    _prune_for_mod, mod, transform_a, config_constraints=constraints
                )
            },
            cache_results=True,
        )
        _MOD_TUNERS[key] = tuner
    return tuner


def mod_selection_args(
    operands,
    outputs,
    *,
    A,
    B,
    b_kn,
    cu_seqlens_m=None,
    A_idx=None,
    SFA=None,
    concat_layout=None,
):
    """Build the canonical argument map consumed by EpiMod config pruning."""
    return {
        **operands,
        **outputs,
        "A": A,
        "B": B,
        "b_kn": b_kn,
        "cu_seqlens_m": cu_seqlens_m,
        "A_idx": A_idx,
        "SFA": SFA,
        "concat_layout": concat_layout,
    }


def _legal_mod_configs(
    mod, device, config_constraints, named_args, *, preferred_config=None, transform_a=None
):
    """Return every supported native config, with the preferred one first when legal."""
    constraints = canonicalize_config_constraints(config_constraints)
    configs = _config_space(mod, device, constraints)
    if preferred_config in configs:
        configs = [preferred_config, *(config for config in configs if config != preferred_config)]
    candidates = [AutotuneConfig(config=config) for config in configs]
    survivors = _prune_for_mod(
        mod, transform_a, candidates, named_args, config_constraints=constraints
    )
    return [candidate.kwargs["config"] for candidate in survivors]


def _select_mod_config(
    mod, device, config_constraints, named_args, *, preferred_config=None, transform_a=None
):
    """Return the preferred native config when legal, or the first supported one."""
    return _legal_mod_configs(
        mod,
        device,
        config_constraints,
        named_args,
        preferred_config=preferred_config,
        transform_a=transform_a,
    )[0]


def tuned_mod_gemm(
    mod,
    A,
    B,
    D,
    C=None,
    *,
    epi_args,
    b_kn=False,
    cu_seqlens_m=None,
    A_idx=None,
    dynamic_scheduler=False,
    SFA=None,
    SFB=None,
    bs_format_a=None,
    bs_format_b=None,
    concat_layout=None,
    # A-operand transform: the handle keys the tuner (semantic digest);
    # layout-owning transforms pass B as the repacked blob (+ transform_sf),
    # runtime-operand transforms pass RAW operand tensors (bundles are built
    # per config inside the sweep — their strip views bake the tiles).
    transform_a=None,
    transform_sf=None,
    transform_operands=None,
    config_constraints=None,
):
    """Autotuned ``mod.gemm``: sweep the arch's config space on the first call
    per (mod, tensor metadata), then run the winner (warm calls replay through
    mod.gemm's own plan cache). Reduce-sink buffers in ``epi_args`` must be
    allocated at the sweep's worst case — see ``sink_arg_shapes``. Returns
    TunedModGemm(plan, config, sinks) with the winning config's sink views."""
    if transform_a is not None:
        from torch._vendor.quack.operand_transform.host import as_transform_mod

        transform_a = as_transform_mod(transform_a)
    constraints = canonicalize_config_constraints(config_constraints)
    epi_names = tuple(sorted(epi_args))
    ta_names = tuple(sorted(transform_operands)) if transform_operands else ()
    assert not any(f"ta__{n}" in epi_args for n in ta_names)
    tuner = _get_tuner(mod, epi_names, C is not None, A.device, transform_a, ta_names, constraints)
    call_kwargs = dict(
        b_kn=b_kn,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        dynamic_scheduler=dynamic_scheduler,
        SFA=SFA,
        SFB=SFB,
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        concat_layout=concat_layout,
    )
    if len(tuner.configs) == 1:
        _prune_for_mod(
            mod,
            transform_a,
            tuner.configs,
            {"A": A, "B": B, **epi_args},
            config_constraints=constraints,
            **call_kwargs,
        )
    plan = tuner(
        A=A,
        B=B,
        D=D,
        C=C,
        mod_digest=mod.semantic_digest,
        config_constraints=constraints,
        b_kn=b_kn,
        cu_seqlens_m=cu_seqlens_m,
        A_idx=A_idx,
        dynamic_scheduler=dynamic_scheduler,
        SFA=SFA,
        SFB=SFB,
        bs_format_a=bs_format_a,
        bs_format_b=bs_format_b,
        concat_layout=concat_layout,
        transform_digest=getattr(transform_a, "semantic_digest", None),
        transform_sf=transform_sf,
        **{f"ta__{k}": v for k, v in (transform_operands or {}).items()},
        **epi_args,
    )
    best = tuner.best_config.kwargs["config"]
    m_gemm, n_gemm = _gemm_mn(A, B, b_kn)
    if transform_a is not None and transform_a.padded_n(B) is not None:
        n_gemm = transform_a.padded_n(B)
    if A_idx is not None:
        m_gemm = A_idx.shape[0]
    return TunedModGemm(
        plan,
        best,
        _slice_sinks(
            mod,
            epi_args,
            best,
            _lead(A, A_idx, m_gemm),
            n_gemm,
            SFA is not None,
            num_seqs=None if cu_seqlens_m is None else cu_seqlens_m.shape[0] - 1,
        ),
    )
