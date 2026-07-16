# mypy: allow-untyped-defs
"""
Per-problem operator discovery + lookup for NVIDIA Universal GEMM.

Discovery: query the cutlass.operators catalog (O(10^5) operator combinations)
for only the operators compatible with a specific problem+arch, via
get_operators(args, target_sm). Results are memoized per problem in
_operator_cache_by_arg_sm.

Lookup: autotune keeps only an operator_name; get_operator_by_name() re-fetches
that operator later. Operators seen during discovery are indexed by name in
_operator_cache_by_name. On a cold-process miss (no discovery ran here, e.g. a
graph loaded from the FX/inductor cache), _ensure_full_catalog() backfills that
same index from the whole catalog, once.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    import cutlass.operators as ops

log = logging.getLogger(__name__)


def _epilogue_args_signature(epilogue_args: ops.EpilogueArguments | None) -> tuple:
    """Hashable fingerprint of an epilogue's aux tensors (dtype/shape/stride).

    One component of both operator-cache keys:
    - _operator_cache_by_arg_sm: (target_sm, A/B/out specs, this signature)
    - _efc_operator_cache: (operator_name, epilogue_source, this signature)

    It ignores the epilogue function body, so _efc_operator_cache also keys on
    epilogue_source (a hash of the body) to tell different bodies apart.
    """
    if epilogue_args is None:
        return ()
    tensors = getattr(epilogue_args, "tensors", None)
    if not tensors:
        return ()
    sig: list[tuple] = []
    for name, val in tensors.items():
        if torch.is_tensor(val):
            sig.append(
                (name, "tensor", val.dtype, tuple(val.shape), tuple(val.stride()))
            )
        else:
            sig.append((name, type(val).__name__))
    return tuple(sig)


_cache_lock = threading.Lock()

# operator_name -> ops.Operator. Populated as a side effect of discovery (only holds
# base, non-epilogue operators). On a cold-process miss, _ensure_full_catalog()
# backfills this from the whole catalog. Consumed by get_operator_by_name().
_operator_cache_by_name: dict[str, ops.Operator] = {}

# discovery_signature (args, target_sm,epilogue) -> Operators compatible with it
# Populated and read by compatible_operators().
_operator_cache_by_arg_sm: dict[Any, list[ops.Operator]] = {}

# (operator_name, epilogue_source, aux-tensor signature) -> ops.Operator bound to
# that epilogue. A bound operator is a fresh instance; epilogue_source (a hash of
# the epilogue fn body) plus the aux signature are BOTH required so two distinct
# epilogues never reuse one operator's jit (a silent miscompile) -- the aux
# signature alone does not capture the function body.
_efc_operator_cache: dict[Any, ops.Operator] = {}

# Whether _operator_cache_by_name has been backfilled from the whole catalog
# (the cold-process fallback in _ensure_full_catalog). At most once per process.
_full_catalog_loaded = False


def _discovery_signature(
    args: ops.RuntimeArguments,
    target_sm: ops.TargetSm | str | None,
) -> Any:
    """Build a hashable cache key for compatible_operators().

    Combines the target arch, the per-operand (dtype, shape, stride) of A/B/out
    (plus scale dtype/mode/swizzle for scaled operands), and the epilogue
    signature.

    Returns None on ANY failure (including an unhashable key), which disables
    caching for that call. Correctness is preferred over caching: a missing key
    only costs a re-discovery, never a wrong result.
    """
    try:

        def _operand_sig(operand: Any) -> tuple:
            sig: tuple = (
                operand.dtype,
                tuple(operand.shape),
                tuple(operand.stride()),
            )
            scale = getattr(operand, "scale", None)
            if scale is not None:
                # ScaledOperand: fold in the scale dtype plus the scale
                # mode/swizzle (str() because ScaleMode overrides __eq__ and is
                # therefore unhashable).
                sig = (
                    *sig,
                    getattr(scale, "dtype", None),
                    str(getattr(operand, "mode", None)),
                    str(getattr(operand, "swizzle", None)),
                )
            return sig

        key = (
            str(target_sm),
            _operand_sig(args.A),
            _operand_sig(args.B),
            _operand_sig(args.out),
            _epilogue_args_signature(getattr(args, "epilogue", None)),
        )
        hash(key)  # ensure the key is actually hashable before returning it
        return key
    except Exception:
        return None


def compatible_operators(
    args: ops.RuntimeArguments,
    target_sm: ops.TargetSm | str | None,
) -> list[ops.Operator]:
    """Per-problem operator discovery: returns only the operators compatible
    with args + target_sm (the library prunes internally)."""
    import cutlass.operators as ops

    key = _discovery_signature(args, target_sm)
    if key is not None and key in _operator_cache_by_arg_sm:
        return _operator_cache_by_arg_sm[key]

    # Register vendored operator wrappers (e.g. the dense block-scaled GEMM)
    # with CuTeDSLProvider via import side-effects. This must run before
    # get_operators() or those operators are undiscoverable. Idempotent: Python
    # caches the module, so the register() call only fires on first import.
    try:
        from torch._inductor.kernel.vendored_templates.cutedsl import (  # noqa: F401
            wrappers,
        )
    except ImportError:
        log.debug("Vendored operator wrappers not available")

    result = ops.get_operators(
        args,
        target_sm=target_sm,
    )

    if key is not None:
        _operator_cache_by_arg_sm[key] = result

    # Index base (no-epilogue) operators by name so get_operator_by_name() can
    # re-fetch the autotune-chosen operator later without re-discovering. Only
    # base operators: an epilogue-bound operator must be rebuilt per epilogue
    # (see _bind_epilogue), never reused by bare name.
    for op in result:
        if getattr(op.metadata, "epilogue", None) is None:
            _operator_cache_by_name[op.metadata.operator_name] = op

    return result


def materialized_operator_count() -> int:
    """Number of distinct operators PyTorch currently retains.

    Reflects the operators held for the current problem(s): the total across the
    per-problem discovery cache, falling back to the name cache when the
    discovery cache is empty.
    """
    if _operator_cache_by_arg_sm:
        return sum(len(v) for v in _operator_cache_by_arg_sm.values())
    return len(_operator_cache_by_name)


def _ensure_full_catalog() -> None:
    """Backfill _operator_cache_by_name from the whole catalog, once.

    Fallback for cold-process name lookups where discovery never ran in this
    process (e.g. a graph loaded from the FX/inductor cache). Materializes the
    full operator set a single time; the hot per-problem discovery path never
    triggers this.
    """
    global _full_catalog_loaded
    if _full_catalog_loaded:
        return

    import cutlass.operators as ops

    try:
        from torch._inductor.kernel.vendored_templates.cutedsl import (  # noqa: F401
            wrappers,
        )
    except ImportError:
        log.debug("Vendored operator wrappers not available")

    with _cache_lock:
        if not _full_catalog_loaded:
            for op in ops.get_operators(providers=[ops.CuTeDSLProvider]):
                _operator_cache_by_name.setdefault(op.metadata.operator_name, op)
            _full_catalog_loaded = True


def _bind_epilogue(
    base_operator: ops.Operator,
    name: str,
    epilogue_args: ops.EpilogueArguments,
    epilogue_source: str = "",
) -> ops.Operator:
    """Bind an epilogue to a base operator, returning a fresh bound instance.

    An EFC operator is chosen at autotune using the GEMM's PRE-epilogue output
    identity, then bound to the concrete epilogue here. Cached per
    (name, epilogue_source, aux-tensor signature). epilogue_source (a hash of the
    epilogue function body) is essential: two epilogues with the same base
    operator and same aux tensors -- e.g. relu vs x*2, both no-aux -- have equal
    aux signatures, so without epilogue_source they would collide on one key and
    the second graph would silently run the first's baked-in epilogue.
    """
    cache_key = (name, epilogue_source, _epilogue_args_signature(epilogue_args))
    with _cache_lock:
        cached = _efc_operator_cache.get(cache_key)
        if cached is not None:
            return cached
        import cutlass.operators as ops
        from cutlass.operators.metadata import EpilogueMetadata

        base_md = base_operator.metadata
        bound_md = ops.OperatorMetadata(
            operator_name=base_md.operator_name,
            operator_class=base_md.operator_class,
            supported_targets=base_md.supported_targets,
            operands=base_md.operands,
            design=base_md.design,
            epilogue=EpilogueMetadata.from_args(epilogue_args),
        )
        bound = base_md.operator_class(bound_md)
        _efc_operator_cache[cache_key] = bound
        return bound


def get_operator_by_name(
    name: str,
    args: ops.RuntimeArguments | None = None,
    target_sm: ops.TargetSm | str | None = None,
    epilogue_source: str = "",
) -> ops.Operator | None:
    """Re-fetch the ops.Operator that autotune already chose, by its operator_name.

    Autotune picks the operator from get_operators() on the BARE gemm (no
    epilogue, out = native dtype). By the time we get here an epilogue may have
    been fused on, which changes args.out (e.g. a cast to fp32). We must still
    return the operator autotune chose, so we look it up by name -- not by
    re-running discovery against the (now different) args.
    """
    epilogue = getattr(args, "epilogue", None) if args is not None else None

    if epilogue is None:
        # No fusion: identity is unchanged since autotune, so re-discovery/name
        # match is valid. Fast path: it was indexed during discovery here.
        op = _operator_cache_by_name.get(name)
        if op is not None:
            return op
        # Not indexed: re-discover this exact problem and match by name. Callers
        # that only have a name (args=None, e.g. best-effort metadata lookups)
        # simply get a miss -- we do NOT materialize the full catalog for them.
        if args is not None:
            for op in compatible_operators(args, target_sm):
                if op.metadata.operator_name == name:
                    return op
        return None

    # Fusion present. Re-discovering with THESE args is wrong: the epilogue
    # changed args.out, so get_operators() would only return operators for the
    # new out dtype -- never the one autotune picked on the native out dtype.
    # So fetch the base operator by name (indexed at autotune, else -- in a cold
    # process with no in-process discovery -- the full-catalog fallback) and
    # attach THIS epilogue to it.
    base = _operator_cache_by_name.get(name)
    if base is None:
        _ensure_full_catalog()
        base = _operator_cache_by_name.get(name)
    if base is None:
        return None
    return _bind_epilogue(base, name, epilogue, epilogue_source)


def clear_cache() -> None:
    """Clear all operator caches and reset the full-catalog flag."""
    global _full_catalog_loaded
    with _cache_lock:
        _operator_cache_by_name.clear()
        _operator_cache_by_arg_sm.clear()
        _efc_operator_cache.clear()
        _full_catalog_loaded = False


class _NVGEMMCacheWrapper:
    def cache_clear(self) -> None:
        clear_cache()


from torch._inductor.utils import clear_on_fresh_cache


clear_on_fresh_cache(_NVGEMMCacheWrapper())
