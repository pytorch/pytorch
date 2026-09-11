# Copyright (c) 2026, Tri Dao.
"""Shared torch-facing interface machinery for epilogue-GEMM variants.

Every variant interface in ``gemm_interface.py`` (act/gated, dact/dgated,
norm_act/norm_gated, rms, symmetric) used to hand-copy the same four blocks:

1. an operand canonicalizer (``_gemm_X_execute``): the ``b_kn`` trace-time
   relabel, the dense-2D vs unsqueeze-to-batch choice, vector unsqueezes and
   the swap_ab operand relabels;
2. the tile-count-semaphore rule;
3. an interface-plan NamedTuple + module cache + warm fast path re-deriving
   the canonicalization on every call;
4. an ``@autotune`` wrapper resolving config defaults and returning the
   resolved decisions for the plan record.

This module owns those blocks once, declaratively. A variant supplies a
:class:`VariantSpec` (operand roles + epilogue-field assignment + hooks for
its genuine quirks) and keeps only: its public eager function (signature,
output allocation, empty-input semantics), its custom-op schemas, and its
reference implementation.

Canonicalization rules (shared by every variant, extracted verbatim):

* role ``'a'``/``'b'``: the GEMM operands. ``'b'`` arrives caller-shaped
  ``(k, n)`` and is relabeled to ``(n, k)`` via ``.mT`` unless ``b_kn`` (dense
  SM90+ keeps the ``(k, n)`` view and relabels at trace time). Under
  ``swap_ab`` the dispatch-level A/B are exchanged.
* role ``'mn'``: an M×N epilogue tensor (D, C, PreAct, PostAct, aux);
  transposed via ``.mT`` under ``swap_ab``.
* role ``'row'``: an ``(n,)``/``(l, n)`` vector, unsqueezed to ``(1, n)``;
  becomes a colvec under ``swap_ab`` (the output transpose relabels N to M —
  same tensor, different epilogue port).
* role ``'col'``: an ``(m,)``/``(l, m)`` vector (kept 1-D under varlen);
  becomes a rowvec under ``swap_ab``.
* dense-2D: on SM90+ with every ``a``/``b``/``mn`` operand 2-D and no varlen
  (and no variant veto), tensors pass through unbatched; otherwise every 2-D
  ``a``/``b``/``mn`` tensor gains a size-1 batch dim (varlen ``a``/``mn``
  tensors stay 2-D — total_m is not a batch).
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import torch

from torch._vendor.quack.cute_dsl_utils import get_device_capacity
from torch._vendor.quack.gemm_config import GemmConfig


class VariantSpec(NamedTuple):
    """Declarative interface recipe for one epilogue-GEMM variant."""

    name: str
    # ((operand_name, role), ...) with roles 'a' | 'b' | 'mn' | 'row' | 'col'.
    tensor_roles: tuple
    # Cold-path launcher: cold(canon, semaphore, b_kn, config, dynamic, ctx)
    # -> dispatch plan. Owns the variant's dispatch-wrapper call.
    cold: Callable
    # Warm-path launcher: warm(plan, canon, semaphore, ctx) -> None. Owns the
    # variant's run_*_plan call for variants the declarative ``warm_slots``
    # mapping can't express (custom plan layer, per-call buffers). Variants
    # with ``warm_slots`` leave this None.
    warm: Optional[Callable] = None
    # b_kn rule: (sm90_plus, varlen_m, swap_ab, ctx) -> bool. b_kn=True keeps
    # B caller-shaped (k, n) for the trace-time relabel; else B is .mT'd here.
    # varlen_m rides the relabel too since 2026-07-14 (the fake builder's
    # varlen path gained b_kn); varlen_k callers pass their own rule.
    b_kn_rule: Callable = lambda sm90_plus, varlen_m, swap_ab, ctx: sm90_plus and not swap_ab
    # Veto for the dense-2D passthrough beyond the shared conditions
    # (act vetoes it under concat_layout): ctx -> bool (True = allow).
    dense_2d_ok: Callable = lambda ctx: True
    # Semaphore rule: (dynamic_scheduler, capacity, device, warm) ->
    # Optional[Tensor]. ``warm`` lets a variant whose run_*_plan never
    # consumes the semaphore skip the allocation on cache-hit calls.
    semaphore: Callable = lambda dynamic, capacity, device, warm: (
        torch.zeros(1, dtype=torch.int32, device=device) if dynamic and capacity == 9 else None
    )
    # Declarative warm mapping for variants whose warm replay is a plain
    # ``run_gemm_epi_plan`` call. When set, make_iface_plan builds a fully
    # specialized closure (swap, vec ports, view codes and the semaphore rule
    # all resolved at record time) instead of going through the ``warm`` hook.
    # (a_operand, b_operand, d_operand_or_None, c_operand_or_None):
    warm_slots: Optional[tuple] = None
    # Epi-value entries: ("plain", epi_field, operand) or
    # ("vec", row_field, col_field, operand, base_role).
    warm_epi: tuple = ()
    # ctx keys forwarded as run_gemm_epi_plan kwargs on warm replays (per-call
    # values like SFA/SFB; metadata-static kwargs need no forwarding).
    warm_extras: tuple = ()


class Canon(NamedTuple):
    """Canonicalized operands plus the decisions the launchers need."""

    tensors: dict  # operand name -> transformed tensor (or None)
    swapped: bool
    b_kn: bool
    dense_2d: bool


def canonicalize(
    spec: VariantSpec, named: dict, *, sm90_plus: bool, varlen_m: bool, swap_ab: bool, ctx=None
) -> Canon:
    """Apply the shared canonicalization rules to one variant's operands.

    ``named`` maps operand names (as declared in ``spec.tensor_roles``) to
    tensors or None. Runs on both the cold and warm paths, so it must stay
    allocation-free apart from the tensor views it creates.
    """
    b_kn = spec.b_kn_rule(sm90_plus, varlen_m, swap_ab, ctx)
    roles = spec.tensor_roles
    tensors = named
    dense_2d = (
        sm90_plus
        and not varlen_m
        and spec.dense_2d_ok(ctx)
        and all(
            tensors[name] is None or tensors[name].ndim == 2
            for name, role in roles
            if role in ("a", "b", "mn")
        )
    )
    out = {}
    for name, role in roles:
        t = tensors[name]
        if t is None:
            out[name] = None
            continue
        if role == "b":
            if not b_kn:
                t = t.mT  # (n, k) or (l, n, k)
            if not dense_2d and t.ndim == 2 and not varlen_m:
                # varlen_m B is per-sequence indexed (3D by contract; the old
                # unsqueeze(0) silently read OOB garbage past the first
                # sequence — found 2026-07-14). 2D B passes through to
                # mod.gemm's loud validation; shared weights are the caller's
                # zero-copy expand.
                t = t.unsqueeze(0)
        elif role in ("a", "mn"):
            if not dense_2d and t.ndim == 2 and not varlen_m:
                t = t.unsqueeze(0)
        elif role == "row":
            if t.ndim == 1:
                t = t.unsqueeze(0)  # (l, n)
        elif role == "col":
            if t.ndim == 1 and not varlen_m:
                t = t.unsqueeze(0)  # (l, m)
        else:
            raise ValueError(f"unknown operand role {role!r} for {name!r}")
        if swap_ab and role == "mn":
            t = t.mT
        out[name] = t
    return Canon(tensors=out, swapped=swap_ab, b_kn=b_kn, dense_2d=dense_2d)


class IfacePlan(NamedTuple):
    """Interface-level launch plan: the metadata-derived decisions of one
    eager variant call (validation, resolved config — including the autotuned
    winner — and the output-allocation recipes), plus the captured dispatch
    plan and a warm-replay closure. Cached per metadata key by the eager
    wrapper; a warm call allocates outputs from the recipes and calls
    ``replay`` — the canonicalization is captured as per-operand view codes at
    record time, so the hot path re-derives nothing. The key must subsume
    everything the dispatch plan's key covers."""

    config: Optional[GemmConfig]
    dynamic_scheduler: bool
    # ((output_name, shape, dtype), ...) — outputs allocated when passed None.
    out_recipes: tuple
    dispatch_plan: object
    # replay(named, ctx) -> None; built by make_replay at record time.
    # None for plans recorded without a warm path.
    replay: Optional[Callable] = None


def alloc_outputs(plan: IfacePlan, provided: dict, device) -> dict:
    """Allocate the outputs the caller left as None, from the plan's recipes."""
    for name, shape, dtype in plan.out_recipes:
        if provided[name] is None:
            provided[name] = torch.empty(shape, dtype=dtype, device=device)
    return provided


# View codes captured per operand at plan-record time (bit-composable, applied
# in bit order): the warm path replays them with three branch tests per tensor
# instead of re-running the role rules.
_MT_PRE = 1  # .mT before batching (role 'b' without b_kn)
_UNSQ = 2  # unsqueeze(0)
_MT_POST = 4  # .mT after batching (role 'mn' under swap_ab)


def _canon_view_codes(spec: VariantSpec, canon: Canon, named: dict, *, varlen_m: bool) -> tuple:
    """Recover ((operand_name, code), ...) describing the views canonicalize()
    chose for this metadata. Replaying the codes on same-metadata tensors
    (the interface-plan key guarantees that) reproduces canonicalize()."""
    codes = []
    for name, role in spec.tensor_roles:
        t = named[name]
        code = 0
        if t is not None:
            if role == "b":
                if not canon.b_kn:
                    code |= _MT_PRE
                if not canon.dense_2d and t.ndim == 2:
                    code |= _UNSQ
            elif role in ("a", "mn"):
                if not canon.dense_2d and t.ndim == 2 and not varlen_m:
                    code |= _UNSQ
            elif role in ("row", "col"):
                if t.ndim == 1 and not (role == "col" and varlen_m):
                    code |= _UNSQ
            if canon.swapped and role == "mn":
                code |= _MT_POST
        codes.append((name, code))
    return tuple(codes)


def _apply_code(t, code):
    """Replay one operand's captured canonicalization views."""
    if t is None or not code:
        return t
    if code & _MT_PRE:
        t = t.mT
    if code & _UNSQ:
        t = t.unsqueeze(0)
    if code & _MT_POST:
        t = t.mT
    return t


def make_iface_plan(
    spec: VariantSpec,
    named: dict,
    *,
    config: Optional[GemmConfig],
    dynamic_scheduler: bool,
    out_recipes: tuple,
    dispatch_plan,
    ctx=None,
    varlen_m: bool = False,
) -> IfacePlan:
    """Record an IfacePlan with its warm-replay closure.

    Called once per metadata key, right after the cold call resolved the
    config and built the dispatch plan. Everything metadata-derived is
    resolved here: the per-operand view codes, the swap flag, the vec-port
    field names and whether a semaphore must be allocated — the warm path
    re-derives nothing.

    Variants with ``warm_slots`` get a specialized closure that calls
    ``run_gemm_epi_plan`` directly; the rest go through their ``warm`` hook
    with a replayed Canon."""
    device = named[spec.tensor_roles[0][0]].device
    capacity = get_device_capacity(device)[0]
    swap_ab = config.swap_ab if config is not None else False
    canon = canonicalize(
        spec, named, sm90_plus=capacity >= 9, varlen_m=varlen_m, swap_ab=swap_ab, ctx=ctx
    )
    codes = _canon_view_codes(spec, canon, named, varlen_m=varlen_m)
    sem_fn = spec.semaphore
    # The semaphore decision is metadata-static; only the allocation is
    # per-call. Probe once so replays with no semaphore skip the rule.
    sem_dynamic = sem_fn(dynamic_scheduler, capacity, device, True) is not None

    if spec.warm_slots is not None:
        from torch._vendor.quack.gemm_runtime.host import run_gemm_epi_plan

        code_of = dict(codes)
        a_n, b_n, d_n, c_n = spec.warm_slots
        if canon.swapped:
            a_n, b_n = b_n, a_n
        a_c, b_c = code_of[a_n], code_of[b_n]
        d_c = code_of[d_n] if d_n is not None else 0
        c_c = code_of[c_n] if c_n is not None else 0
        prog = []
        for entry in spec.warm_epi:
            if entry[0] == "vec":
                _, row_f, col_f, op_n, base = entry
                is_row = (base == "row") != canon.swapped
                prog.append((row_f if is_row else col_f, op_n, code_of[op_n]))
            else:
                _, field, op_n = entry
                prog.append((field, op_n, code_of[op_n]))
        prog = tuple(prog)
        extras = spec.warm_extras

        def replay(named, ctx):
            sem = sem_fn(dynamic_scheduler, capacity, device, True) if sem_dynamic else None
            run_gemm_epi_plan(
                dispatch_plan,
                _apply_code(named[a_n], a_c),
                _apply_code(named[b_n], b_c),
                _apply_code(named[d_n], d_c) if d_n is not None else None,
                _apply_code(named[c_n], c_c) if c_n is not None else None,
                {field: _apply_code(named[n], c) for field, n, c in prog},
                tile_count_semaphore=sem,
                **({k: ctx[k] for k in extras} if extras else {}),
            )

    else:
        swapped, b_kn, dense_2d = canon.swapped, canon.b_kn, canon.dense_2d
        warm = spec.warm

        def replay(named, ctx):
            tensors = {name: _apply_code(named[name], code) for name, code in codes}
            sem = sem_fn(dynamic_scheduler, capacity, device, True) if sem_dynamic else None
            warm(dispatch_plan, Canon(tensors, swapped, b_kn, dense_2d), sem, ctx)

    return IfacePlan(
        config=config,
        dynamic_scheduler=dynamic_scheduler,
        out_recipes=out_recipes,
        dispatch_plan=dispatch_plan,
        replay=replay,
    )


def run_variant(
    spec: VariantSpec,
    named: dict,
    *,
    config: Optional[GemmConfig],
    dynamic_scheduler: bool,
    varlen_m: bool = False,
    ctx=None,
    dispatch_plan=None,
):
    """Canonicalize and launch one variant call (cold or warm).

    ``ctx`` is the variant's opaque extras bag (SF tensors, colvec buffers,
    activation strings, ...) passed through to the hooks. Returns the dispatch
    plan (cold: freshly built; warm: the one passed in). Warm replays coming
    from an interface plan should prefer ``IfacePlan.replay`` (no
    re-derivation); this path serves cold calls and plan-less warm replays.
    """
    device = named[spec.tensor_roles[0][0]].device
    capacity = get_device_capacity(device)[0]
    sm90_plus = capacity >= 9
    swap_ab = config.swap_ab if config is not None else False
    canon = canonicalize(
        spec, named, sm90_plus=sm90_plus, varlen_m=varlen_m, swap_ab=swap_ab, ctx=ctx
    )
    semaphore = spec.semaphore(dynamic_scheduler, capacity, device, dispatch_plan is not None)
    if dispatch_plan is not None:
        # Warm replay: the interface plan key vouches for the metadata.
        if spec.warm is None:
            raise ValueError(
                f"variant {spec.name!r} has no warm hook; replay through IfacePlan.replay"
            )
        spec.warm(dispatch_plan, canon, semaphore, ctx)
        return dispatch_plan
    return spec.cold(canon, semaphore, config, dynamic_scheduler, ctx)


def swap_pair(canon: Canon, a: str, b: str):
    """The dispatch-level (A, B) pair under swap_ab."""
    ta, tb = canon.tensors[a], canon.tensors[b]
    return (tb, ta) if canon.swapped else (ta, tb)


def vec_ports(canon: Canon, name: str, *, base: str):
    """Map a 'row'/'col' operand onto (rowvec, colvec) dispatch ports.

    ``base`` is the unswapped port: a row vector rides the rowvec port until
    swap_ab transposes the output and it becomes a colvec (and vice versa).
    """
    t = canon.tensors[name]
    row_first = base == "row"
    if canon.swapped:
        row_first = not row_first
    return (t, None) if row_first else (None, t)
