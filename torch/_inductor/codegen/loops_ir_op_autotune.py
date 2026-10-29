"""Measured per-op autotuning for the LoopsIR C++ path.

Small, explicit service: the caller supplies a typed op request (layout +
frozen numerics + tolerances), a finite manifest of typed candidate
descriptors, compile-time runner factories, and a reference runner (built
from the SDK ``GeneratedKernel`` / ``NativeCall`` helpers in
``loops_ir_op_options``).  This module never reuses stock
``TritonTemplate`` / ``GraphLowering`` machinery.

Separation of concerns:

* :class:`CandidateDescriptor` is pure data (id / version / canonical
  config / selection key / source hash / numerics fence).  It is JSON
  serializable and safe to cache.
* Runner factories / runners are live compile-time callbacks and are
  never serialized, hashed via ``repr``, or written to the cache.

Policy:

* strict ``strict_eager`` numerics veto generated ``inductor_only``
  candidates regardless of speed;
* only eligible + accuracy-passing candidates are timed; the minimum
  finite measurement (including native) wins; ties break on
  ``candidate_id``;
* singleton manifests take a fast path that is never advertised as
  measured/autotuned;
* native/reference failure raises :class:`NoValidCandidateError`;
* cache entries are local-only generic winner JSON keyed by SHA256 of
  structured canonical data; corruption/mismatch fails closed.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

log = logging.getLogger(__name__)

__all__ = [
    "CandidateDescriptor",
    "NoValidCandidateError",
    "NumericsFence",
    "OpTuneRequest",
    "TensorSpec",
    "TuneResult",
    "WinnerRecord",
    "autotune_op",
    "cache_key_hex",
    "implementation_hash_hex",
    "manifest_hash_hex",
]

STRICT_POLICY = "strict_eager"
INDUCTOR_POLICY = "inductor"
_CACHE_VERSION = 1
_CACHE_DIRNAME = "loops_ir_op_autotune"

Runner = Callable[[Sequence[torch.Tensor], torch.Tensor], "torch.Tensor | None"]
Factory = Callable[[], Runner]
ReferenceRunner = Callable[[Sequence[torch.Tensor]], torch.Tensor]
BenchmarkFn = Callable[[Callable[[], Any]], float]


class NoValidCandidateError(RuntimeError):
    """Raised when no candidate (including native) validates and times."""


class _AccuracyError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    """Layout metadata for one tensor. All fields are JSON primitives."""

    dtype: str = "float32"
    sizes: tuple[int, ...] = ()
    strides: tuple[int, ...] = ()
    offset: int = 0
    device: str = "cpu"

    def to_json(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "sizes": list(self.sizes),
            "strides": list(self.strides),
            "offset": int(self.offset),
            "device": self.device,
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> TensorSpec:
        return TensorSpec(
            dtype=str(d["dtype"]),
            sizes=tuple(int(v) for v in d["sizes"]),
            strides=tuple(int(v) for v in d["strides"]),
            offset=int(d.get("offset", 0)),
            device=str(d.get("device", "cpu")),
        )


@dataclasses.dataclass(frozen=True)
class NumericsFence:
    """Frozen numerical context. Part of eligibility and the cache key."""

    policy: str = INDUCTOR_POLICY
    enable_fp_fusion: bool = False
    disable_ftz: bool = False
    division_rounding: bool = False
    matmul_precision: str = "ieee"

    def to_json(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "enable_fp_fusion": bool(self.enable_fp_fusion),
            "disable_ftz": bool(self.disable_ftz),
            "division_rounding": bool(self.division_rounding),
            "matmul_precision": self.matmul_precision,
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> NumericsFence:
        return NumericsFence(
            policy=str(d["policy"]),
            enable_fp_fusion=bool(d.get("enable_fp_fusion", False)),
            disable_ftz=bool(d.get("disable_ftz", False)),
            division_rounding=bool(d.get("division_rounding", False)),
            matmul_precision=str(d.get("matmul_precision", "ieee")),
        )


@dataclasses.dataclass(frozen=True)
class CandidateDescriptor:
    """Typed, serializable candidate identity. No callbacks or tensors."""

    candidate_id: str
    option_id: str
    option_version: int
    kind: str  # "generated" or "native"
    config: str  # canonical config string
    cache_key: str  # selection key from the registry
    source_hash: str  # sha256 hex of the rendered source
    inductor_only: bool = False
    requires_zeroed_output: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "option_id": self.option_id,
            "option_version": int(self.option_version),
            "kind": self.kind,
            "config": self.config,
            "cache_key": self.cache_key,
            "source_hash": self.source_hash,
            "inductor_only": bool(self.inductor_only),
            "requires_zeroed_output": bool(self.requires_zeroed_output),
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> CandidateDescriptor:
        kind = str(d["kind"])
        if kind not in ("generated", "native"):
            raise ValueError(f"bad kind {kind!r}")
        return CandidateDescriptor(
            candidate_id=str(d["candidate_id"]),
            option_id=str(d["option_id"]),
            option_version=int(d["option_version"]),
            kind=kind,
            config=str(d["config"]),
            cache_key=str(d["cache_key"]),
            source_hash=str(d["source_hash"]),
            inductor_only=bool(d.get("inductor_only", False)),
            requires_zeroed_output=bool(d.get("requires_zeroed_output", False)),
        )


@dataclasses.dataclass(frozen=True)
class OpTuneRequest:
    """Caller-supplied op request: layout metadata + fence + tolerances."""

    op_name: str
    numerics: NumericsFence
    inputs: tuple[TensorSpec, ...]
    output: TensorSpec
    rtol: float = 1e-5
    atol: float = 1e-8
    equal_nan: bool = True

    def to_json(self) -> dict[str, Any]:
        return {
            "op_name": self.op_name,
            "numerics": self.numerics.to_json(),
            "inputs": [t.to_json() for t in self.inputs],
            "output": self.output.to_json(),
            "rtol": float(self.rtol),
            "atol": float(self.atol),
            "equal_nan": bool(self.equal_nan),
        }


@dataclasses.dataclass(frozen=True)
class WinnerRecord:
    """Frozen, JSON-only winner. Never holds callbacks, AST, or Tensors."""

    candidate_id: str
    option_id: str
    option_version: int
    kind: str
    config: str
    cache_key: str
    source_hash: str
    numerics: NumericsFence
    rtol: float
    atol: float
    equal_nan: bool
    measured_ms: float
    origin: str  # "cold" | "cache" | "singleton"
    manifest_hash: str
    implementation_hash: str
    device: str

    def to_json(self) -> dict[str, Any]:
        return {
            "version": _CACHE_VERSION,
            "candidate_id": self.candidate_id,
            "option_id": self.option_id,
            "option_version": int(self.option_version),
            "kind": self.kind,
            "config": self.config,
            "cache_key": self.cache_key,
            "source_hash": self.source_hash,
            "numerics": self.numerics.to_json(),
            "rtol": float(self.rtol),
            "atol": float(self.atol),
            "equal_nan": bool(self.equal_nan),
            "measured_ms": float(self.measured_ms),
            "origin": self.origin,
            "manifest_hash": self.manifest_hash,
            "implementation_hash": self.implementation_hash,
            "device": self.device,
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> WinnerRecord:
        if not isinstance(d, dict) or d.get("version") != _CACHE_VERSION:
            raise ValueError("bad winner version")
        for k in (
            "candidate_id",
            "option_id",
            "cache_key",
            "source_hash",
            "manifest_hash",
            "implementation_hash",
        ):
            if not isinstance(d[k], str):
                raise ValueError(f"bad winner field {k}")
        return WinnerRecord(
            candidate_id=d["candidate_id"],
            option_id=d["option_id"],
            option_version=int(d["option_version"]),
            kind=str(d["kind"]),
            config=str(d["config"]),
            cache_key=d["cache_key"],
            source_hash=d["source_hash"],
            numerics=NumericsFence.from_json(d["numerics"]),
            rtol=float(d["rtol"]),
            atol=float(d["atol"]),
            equal_nan=bool(d["equal_nan"]),
            measured_ms=float(d["measured_ms"]),
            origin=str(d["origin"]),
            manifest_hash=d["manifest_hash"],
            implementation_hash=d["implementation_hash"],
            device=str(d["device"]),
        )


@dataclasses.dataclass(frozen=True)
class TuneResult:
    winner: WinnerRecord
    measurements_ms: dict[str, float]
    cache_hit: bool
    autotuned: bool
    measured: bool


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def manifest_hash_hex(candidates: Sequence[CandidateDescriptor]) -> str:
    manifest = sorted(
        (c.to_json() for c in candidates), key=lambda d: d["candidate_id"]
    )
    return _sha256_hex(_canonical_json(manifest))


def implementation_hash_hex(candidates: Sequence[CandidateDescriptor]) -> str:
    rows = sorted(
        [
            [
                c.candidate_id,
                c.option_id,
                c.option_version,
                c.kind,
                c.config,
                c.cache_key,
                c.source_hash,
                c.inductor_only,
                c.requires_zeroed_output,
            ]
            for c in candidates
        ]
    )
    return _sha256_hex(_canonical_json(rows))


def _dtype_from_str(name: str) -> torch.dtype:
    # Short names ("float32") and fully qualified ("torch.float32") both work.
    short = name.split(".")[-1]
    dtype = getattr(torch, short, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown dtype {name!r}")
    return dtype


def _flat_buffer(numel: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    # Mirrors torch._dynamo.testing.rand_strided allocation semantics:
    # randn for floating point (with the fp4/fp8 workarounds), zeros
    # otherwise. torch.randn has no integer/bool kernels, so a shared
    # helper keeps the offset branch from leaking raw dtype errors.
    if dtype.is_floating_point:
        if dtype == getattr(torch, "float4_e2m1fn_x2", None):
            return torch.randint(
                0, 256, (numel,), dtype=torch.uint8, device=device
            ).view(dtype)
        if dtype.itemsize == 1:
            return torch.randn(numel, dtype=torch.float16, device=device).to(dtype)
        return torch.randn(numel, dtype=dtype, device=device)
    return torch.zeros(numel, dtype=dtype, device=device)


def _fresh_tensor(spec: TensorSpec) -> torch.Tensor:
    from torch._dynamo.testing import rand_strided

    dtype = _dtype_from_str(spec.dtype)
    device = torch.device(spec.device)
    sizes = tuple(spec.sizes)
    strides = tuple(spec.strides)
    offset = int(spec.offset)
    if offset:
        # Layout-preserving view at a nonzero storage offset: allocate a
        # larger flat buffer and stride into it (never clone: clone would
        # lose the offset/strides).
        if all(s > 0 for s in sizes):
            span = sum((sh - 1) * st for sh, st in zip(sizes, strides)) + 1
        else:
            span = 1
        flat = _flat_buffer(offset + span, dtype, device)
        return torch.as_strided(flat, sizes, strides, storage_offset=offset)
    return rand_strided(sizes, strides, dtype=dtype, device=device)


def _isolated_like(source: torch.Tensor, spec: TensorSpec) -> torch.Tensor:
    """Fresh tensor with the exact spec layout holding ``source``'s values.

    ``Tensor.clone()`` densifies strided/offset views, so every handoff to
    a reference/candidate runner copies values into a new exact-layout
    tensor instead.
    """
    dst = _fresh_tensor(spec)
    dst.copy_(source)
    return dst


def _is_eligible(descriptor: CandidateDescriptor, request: OpTuneRequest) -> bool:
    if request.numerics.policy == STRICT_POLICY:
        if descriptor.kind == "generated" and descriptor.inductor_only:
            return False
    return True


def _check_layout_match(candidate: torch.Tensor, reference: torch.Tensor) -> None:
    if candidate.dtype != reference.dtype:
        raise _AccuracyError(f"dtype mismatch {candidate.dtype} vs {reference.dtype}")
    if tuple(candidate.shape) != tuple(reference.shape):
        raise _AccuracyError(
            f"shape mismatch {tuple(candidate.shape)} vs {tuple(reference.shape)}"
        )
    for d, (n, cs, rs) in enumerate(
        zip(reference.shape, candidate.stride(), reference.stride())
    ):
        if int(n) > 1 and int(cs) != int(rs):
            raise _AccuracyError(f"stride mismatch on dim {d}: {cs} vs {rs}")
    if candidate.device != reference.device:
        raise _AccuracyError(
            f"device mismatch {candidate.device} vs {reference.device}"
        )


def _validate_output(
    candidate: torch.Tensor, reference: torch.Tensor, request: OpTuneRequest
) -> None:
    if not isinstance(candidate, torch.Tensor) or not isinstance(
        reference, torch.Tensor
    ):
        raise _AccuracyError("non-tensor output")
    if candidate.is_complex() or reference.is_complex():
        raise _AccuracyError("complex dtype unsupported (SDK is FP32-only)")
    _check_layout_match(candidate, reference)
    strict = request.numerics.policy == STRICT_POLICY
    if candidate.is_floating_point() or reference.is_floating_point():
        nan_c = torch.isnan(candidate)
        nan_r = torch.isnan(reference)
        if not bool(torch.equal(nan_c, nan_r)):
            raise _AccuracyError("NaN positions differ from reference")
        inf_c = torch.isinf(candidate)
        inf_r = torch.isinf(reference)
        if not bool(torch.equal(inf_c, inf_r)):
            raise _AccuracyError("Inf positions differ from reference")
        if bool(torch.any(inf_c)):
            # Compare signs only at Inf positions: finite lanes (including
            # near-zero +-0.0 differences) are judged by the tolerance check
            # below, never by this gate.
            if not bool(
                torch.equal(
                    torch.signbit(candidate).masked_select(inf_c),
                    torch.signbit(reference).masked_select(inf_r),
                )
            ):
                raise _AccuracyError("Inf signs differ from reference")
        if strict:
            # Bytes-exact logical equality: == everywhere outside NaNs, plus
            # signbit equality (distinguishes -0.0/+0.0 and signed Infs).
            eq = (candidate == reference) | (nan_c & nan_r)
            if not bool(torch.all(eq)):
                raise _AccuracyError("strict bytes-exact mismatch")
            if not bool(
                torch.equal(torch.signbit(candidate), torch.signbit(reference))
            ):
                raise _AccuracyError("strict signbit mismatch")
            return
        if not torch.allclose(
            candidate,
            reference,
            rtol=request.rtol,
            atol=request.atol,
            equal_nan=request.equal_nan,
        ):
            raise _AccuracyError(
                f"values differ beyond rtol={request.rtol} atol={request.atol}"
            )
    else:
        if strict:
            if not bool(torch.equal(candidate, reference)):
                raise _AccuracyError("strict exact mismatch (non-float)")
        elif not bool(torch.equal(candidate, reference)):
            raise _AccuracyError("integer/bool output differs from reference")


def _device_props_json(request: OpTuneRequest) -> dict[str, Any]:
    try:
        dev = torch.device(request.output.device)
    except Exception:
        dev = torch.device("cpu")
    if dev.type == "cuda" and torch.cuda.is_available():
        try:
            idx = dev.index if dev.index is not None else torch.cuda.current_device()
            p = torch.cuda.get_device_properties(idx)
            return {
                "type": "cuda",
                "index": int(idx),
                "major": int(p.major),
                "minor": int(p.minor),
                "multi_processor_count": int(p.multi_processor_count),
                "total_memory": int(p.total_memory),
            }
        except Exception:
            pass
    return {"type": "cpu", "index": -1}


def cache_key_hex(
    request: OpTuneRequest,
    candidates: Sequence[CandidateDescriptor],
    device_props: Mapping[str, Any] | None = None,
) -> str:
    props = (
        dict(device_props) if device_props is not None else _device_props_json(request)
    )
    payload = {
        "version": _CACHE_VERSION,
        "op": request.to_json(),
        "manifest": sorted(
            (c.to_json() for c in candidates), key=lambda d: d["candidate_id"]
        ),
        "manifest_hash": manifest_hash_hex(candidates),
        "implementation_hash": implementation_hash_hex(candidates),
        "device_props": props,
        "torch_version": torch.__version__,
    }
    return _sha256_hex(_canonical_json(payload))


def _default_local_cache() -> Any | None:
    try:
        from torch._inductor.remote_cache import LocalAutotuneCache

        return LocalAutotuneCache("loops-ir-op-autotune")
    except Exception as exc:  # fail open: autotune without cache
        log.warning("loops_ir_op_autotune: no local cache (%s)", exc)
        return None


def _cache_path(key_hex: str) -> str:
    try:
        from torch._inductor.runtime.cache_dir_utils import cache_dir
    except Exception:
        from torch._inductor.runtime.runtime_utils import cache_dir  # type: ignore[no-redef]

    return os.path.join(cache_dir(), _CACHE_DIRNAME, f"{key_hex}.json")


def _load_cached_winner(
    cache: Any,
    path: str,
    request: OpTuneRequest,
    candidates: Sequence[CandidateDescriptor],
    manifest_hash: str,
    implementation_hash: str,
) -> WinnerRecord | None:
    # Fail closed: any error, corruption, or mismatch -> remeasure.
    try:
        raw = cache.get(path)
    except Exception as exc:
        log.warning("loops_ir_op_autotune: cache get failed closed (%s)", exc)
        return None
    if raw is None:
        return None
    try:
        winner = WinnerRecord.from_json(raw)
    except Exception as exc:
        log.warning("loops_ir_op_autotune: corrupt cache entry ignored (%s)", exc)
        return None
    by_id = {c.candidate_id: c for c in candidates}
    cand = by_id.get(winner.candidate_id)
    if cand is None:
        return None
    if not _is_eligible(cand, request):
        return None
    if (
        winner.manifest_hash != manifest_hash
        or winner.implementation_hash != implementation_hash
        or cand.option_id != winner.option_id
        or int(cand.option_version) != int(winner.option_version)
        or cand.config != winner.config
        or cand.cache_key != winner.cache_key
        or cand.source_hash != winner.source_hash
        or cand.kind != winner.kind
    ):
        return None
    if winner.numerics.to_json() != request.numerics.to_json():
        return None
    if not (
        winner.measured_ms == winner.measured_ms
        and winner.measured_ms not in (float("inf"), float("-inf"))
        and winner.measured_ms >= 0
    ):
        # Cold path rejects non-finite/negative timings; a cached record
        # must satisfy the same bar or fail closed (finding 7).
        log.warning("loops_ir_op_autotune: corrupt cached timing ignored")
        return None
    cached = dataclasses.replace(winner, origin="cache")
    return cached


def _store_winner(cache: Any, path: str, winner: WinnerRecord) -> None:
    try:
        cache.put(path, winner.to_json())
    except Exception as exc:
        log.warning("loops_ir_op_autotune: cache put failed (%s)", exc)


def _check_reference_layout(reference: torch.Tensor, request: OpTuneRequest) -> None:
    """Fence the caller: reference must match the declared output spec.

    Strides follow the same size-1 latitude as the candidate matcher
    (stride on a size-1 dim carries no layout meaning); device compares by
    type and index so "cuda" vs "cuda:0" spellings of the same device pass.
    """
    try:
        ref_dtype = _dtype_from_str(request.output.dtype)
    except Exception as exc:
        raise NoValidCandidateError(f"reference check failed: {exc}") from exc
    if not isinstance(reference, torch.Tensor):
        raise NoValidCandidateError("reference runner did not return a Tensor")
    if reference.dtype != ref_dtype:
        raise NoValidCandidateError("reference output mismatches declared spec")
    if tuple(reference.shape) != tuple(request.output.sizes):
        raise NoValidCandidateError("reference output mismatches declared spec")
    for d, (n, rs, es) in enumerate(
        zip(request.output.sizes, reference.stride(), request.output.strides)
    ):
        if int(n) > 1 and int(rs) != int(es):
            raise NoValidCandidateError(
                f"reference output stride mismatch on dim {d}: {rs} vs {es}"
            )
    want = torch.device(request.output.device)
    if reference.device.type != want.type or (
        want.index is not None and reference.device.index != want.index
    ):
        raise NoValidCandidateError("reference output mismatches declared spec")


def _run_reference(
    reference_runner: ReferenceRunner,
    master_inputs: Sequence[torch.Tensor],
    request: OpTuneRequest,
) -> torch.Tensor:
    """Run the reference on isolated exact-layout copies; gate its layout."""
    try:
        expected = reference_runner(
            [_isolated_like(t, s) for t, s in zip(master_inputs, request.inputs)]
        )
    except Exception as exc:
        raise NoValidCandidateError(f"reference runner failed: {exc}") from exc
    _check_reference_layout(expected, request)
    return expected


def _run_candidate(
    runner: Runner,
    master_inputs: Sequence[torch.Tensor],
    request: OpTuneRequest,
    descriptor: CandidateDescriptor,
) -> torch.Tensor:
    trial_out = _fresh_tensor(request.output)
    if descriptor.requires_zeroed_output:
        trial_out.zero_()
    got = runner(
        [_isolated_like(t, s) for t, s in zip(master_inputs, request.inputs)],
        trial_out,
    )
    return got if isinstance(got, torch.Tensor) else trial_out


def _default_benchmark(zero_arg: Callable[[], Any]) -> float:
    from torch._inductor.runtime.benchmarking import (
        benchmarker,
        maybe_gpu_benchmark_lock,
    )

    with maybe_gpu_benchmark_lock():
        return float(benchmarker.benchmark_gpu(zero_arg, is_vetted_benchmarking=True))


def _device_guard(device_str: str) -> Any:
    try:
        dev = torch.device(device_str)
    except Exception:
        return contextlib.nullcontext()
    if dev.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.device(dev.index if dev.index is not None else 0)
    return contextlib.nullcontext()


def autotune_op(
    request: OpTuneRequest,
    candidates: Sequence[CandidateDescriptor],
    factories: Mapping[str, Factory],
    reference_runner: ReferenceRunner,
    *,
    benchmark_fn: BenchmarkFn | None = None,
    cache: Any | None = None,
    use_cache: bool = True,
    device_props: Mapping[str, Any] | None = None,
) -> TuneResult:
    """Measure candidates and return the frozen winner.

    * ``candidates`` is the full explicit manifest (finite, per-op).
    * ``factories`` maps ``candidate_id`` -> compile factory returning a
      ``runner(inputs, output)`` callable (``Tensor`` return or in-place
      ``None``).
    * ``reference_runner`` is the SDK native/reference callable used for
      the expected result and the native baseline.
    * ``benchmark_fn`` defaults to ``benchmarker.benchmark_gpu(...,
      is_vetted_benchmarking=True)``; tests inject a CPU mock.
    """
    if not candidates:
        raise ValueError("autotune_op requires at least one candidate")
    ids = [c.candidate_id for c in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate_id in manifest")
    missing = [c for c in ids if c not in factories]
    if missing:
        raise ValueError(f"missing runner factories for {missing}")
    if not any(c.kind == "native" for c in candidates):
        raise ValueError("manifest must include a native baseline candidate")
    if not (request.rtol >= 0 and request.atol >= 0):
        raise ValueError("rtol/atol must be non-negative")
    for v in (request.rtol, request.atol):
        if not (v == v and v not in (float("inf"), float("-inf"))):
            raise ValueError("rtol/atol must be finite")

    from torch._dynamo.utils import preserve_rng_state

    manifest_hash = manifest_hash_hex(candidates)
    implementation_hash = implementation_hash_hex(candidates)
    props = (
        dict(device_props) if device_props is not None else _device_props_json(request)
    )
    key_hex = cache_key_hex(request, candidates, props)
    path = _cache_path(key_hex)

    active_cache = (
        cache if cache is not None else (_default_local_cache() if use_cache else None)
    )

    ordered = sorted(candidates, key=lambda c: c.candidate_id)

    # Singleton fast path: validate once, never advertise as measured.
    if len(ordered) == 1:
        (only,) = ordered
        with (
            preserve_rng_state(),
            torch.no_grad(),
            _device_guard(request.output.device),
        ):
            master_inputs = [_fresh_tensor(s) for s in request.inputs]
            expected = _run_reference(reference_runner, master_inputs, request)
            try:
                runner = factories[only.candidate_id]()
                produced = _run_candidate(runner, master_inputs, request, only)
                _validate_output(produced, expected, request)
            except _AccuracyError as exc:
                raise NoValidCandidateError(
                    f"singleton candidate accuracy failed: {exc}"
                ) from exc
            except NoValidCandidateError:
                raise
            except Exception as exc:
                raise NoValidCandidateError(
                    f"singleton candidate failed: {exc}"
                ) from exc
        winner = WinnerRecord(
            candidate_id=only.candidate_id,
            option_id=only.option_id,
            option_version=int(only.option_version),
            kind=only.kind,
            config=only.config,
            cache_key=only.cache_key,
            source_hash=only.source_hash,
            numerics=request.numerics,
            rtol=float(request.rtol),
            atol=float(request.atol),
            equal_nan=bool(request.equal_nan),
            measured_ms=float("nan"),
            origin="singleton",
            manifest_hash=manifest_hash,
            implementation_hash=implementation_hash,
            device=request.output.device,
        )
        return TuneResult(
            winner=winner,
            measurements_ms={},
            cache_hit=False,
            autotuned=False,
            measured=False,
        )

    if active_cache is not None:
        hit = _load_cached_winner(
            active_cache, path, request, ordered, manifest_hash, implementation_hash
        )
        if hit is not None:
            return TuneResult(
                winner=hit,
                measurements_ms={hit.candidate_id: hit.measured_ms},
                cache_hit=True,
                autotuned=False,
                measured=False,
            )

    bench = benchmark_fn if benchmark_fn is not None else _default_benchmark

    with preserve_rng_state(), torch.no_grad(), _device_guard(request.output.device):
        master_inputs = [_fresh_tensor(s) for s in request.inputs]
        expected = _run_reference(reference_runner, master_inputs, request)

        measurements: dict[str, float] = {}
        for cand in ordered:
            if not _is_eligible(cand, request):
                continue
            try:
                runner = factories[cand.candidate_id]()
            except Exception as exc:
                log.warning("candidate %s compile failed: %s", cand.candidate_id, exc)
                continue
            try:
                produced = _run_candidate(runner, master_inputs, request, cand)
                _validate_output(produced, expected, request)
            except _AccuracyError as exc:
                log.warning("candidate %s accuracy failed: %s", cand.candidate_id, exc)
                continue
            except Exception as exc:
                log.warning("candidate %s run failed: %s", cand.candidate_id, exc)
                continue
            # Time only eligible + correct candidates, on isolated inputs.

            def _zero_arg(_runner: Runner = runner) -> Any:
                ins = [_fresh_tensor(s) for s in request.inputs]
                ob = _fresh_tensor(request.output)
                if cand.requires_zeroed_output:
                    ob.zero_()
                r = _runner(ins, ob)
                return r if isinstance(r, torch.Tensor) else ob

            try:
                ms = float(bench(_zero_arg))
            except Exception as exc:
                log.warning("candidate %s timing failed: %s", cand.candidate_id, exc)
                continue
            if not (ms == ms and ms not in (float("inf"), float("-inf")) and ms >= 0):
                log.warning("candidate %s non-finite timing %r", cand.candidate_id, ms)
                continue
            measurements[cand.candidate_id] = ms

    if not measurements:
        raise NoValidCandidateError("no valid candidate (all failed validation/timing)")

    best_id = min(measurements, key=lambda k: (measurements[k], k))
    best = next(c for c in ordered if c.candidate_id == best_id)
    winner = WinnerRecord(
        candidate_id=best.candidate_id,
        option_id=best.option_id,
        option_version=int(best.option_version),
        kind=best.kind,
        config=best.config,
        cache_key=best.cache_key,
        source_hash=best.source_hash,
        numerics=request.numerics,
        rtol=float(request.rtol),
        atol=float(request.atol),
        equal_nan=bool(request.equal_nan),
        measured_ms=float(measurements[best_id]),
        origin="cold",
        manifest_hash=manifest_hash,
        implementation_hash=implementation_hash,
        device=request.output.device,
    )
    if active_cache is not None:
        _store_winner(active_cache, path, winner)
    return TuneResult(
        winner=winner,
        measurements_ms=dict(measurements),
        cache_hit=False,
        autotuned=True,
        measured=True,
    )
