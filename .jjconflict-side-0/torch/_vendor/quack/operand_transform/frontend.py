# Copyright (c) 2026, Tri Dao.
"""``@a_transform``: author an A-operand transform as a plain Python function,
mirroring ``@gemm_epilogue`` (the fn is the COMPOSITION site; the kernel-side
plumbing — canonical s2r load or blob TMA, the interleaved produce/WGMMA
schedule, fences and commit groups — is written once and never by the fn).

Two families, one decorator:

* VALUE transforms (default): unpacked 16-bit A, canonically ldmatrix-loaded;
  the fn is called per lane per ``vec_size`` fragment elements as a TensorSSA
  vector in the MMA dtype and returns the transformed vector::

      @a_transform(vec_size=2)
      def halve_a(x):
          return x * 0.5

  The vector is FRAGMENT-SLOT-ORDERED, not k-contiguous (a lane's 8 elements
  per k16 block are 2 adjacent k x 2 rows x 2 k-halves); chunks are
  pair-aligned so packed 16-bit math vectorizes. ``vec_size`` in {2, 4, 8} and
  is capped at one k16 block: the schedule (produce(b+1) overlapping WGMMA(b))
  belongs to the framework, never the fn. Compile-time constants may be closed
  over — they are part of the semantic key. ``consts=callable`` is called once
  per kernel (hoisted — LUTs, packed constants); its result is the fn's LAST
  parameter.

  Runtime operands are declared per fn parameter with ``args={param: kind}``,
  a kind taxonomy over the transform's (M, K) index space — the mainloop
  mirror of EpiOps' operand kinds over (M, N). Each kind owns its indexing,
  delivery and register staging; the fn just receives a same-length TensorSSA
  vector of values aligned with x. Kinds (kinds.ARG_KINDS):

  * the strip family — one MMA-dtype value per (m-group, k-group) at 2-D
    granularity, delivered via the aux A-side TMA slot (per-stage smem under
    the AB mbarrier): ``"colvec_ktile"`` (per (row, k-tile)),
    ``"colvec_k64"`` / ``"colvec_k32"`` / ``"colvec_k16"`` (per (row, 64 /
    32 / 16 elements) — dense blockscaled-SF granularities), ``"kvec_m64"``
    (per (m64 row block, k-element) — the LCE dw strip; resolve
    coarser-than-64 M granularity host-side, e.g. a vocab-tile row per m64
    block). E.g. the linear-CE dx pow2 rescale::

        @a_transform(vec_size=8, args={"u": "colvec_ktile"})
        def dx_scale(x, u):
            return x * u

    The host passes A as the bundle
    ``host.transform_a_operand(mod, A, {"u": strip}, tile_m, tile_k)`` with
    ``strip`` a contiguous (ceil(K / tile_K) * g, M) tensor in A's dtype
    (one row per k-group; ragged K padded to whole k-tiles).

  There is deliberately NO k-invariant ``colvec`` kind: a per-row scale
  commutes through the GEMM — ``(u ⊙ A) @ B = u ⊙ (A @ B)`` — so it belongs
  in the EPILOGUE as an fp32 colvec (exact, measured free), and per-row
  additive terms commute via the rank-1 ``z · colsum(B)`` correction; only a
  fn NONLINEAR in x would need per-row mainloop values.

* DROPOUT (``dropout_a(p)``): not a fn — a dedicated mask-only transform
  (philox keep-mask ANDed onto the fragment at ~3 SASS per 2 elements; see
  :class:`~quack.operand_transform.transform.TransformADropout`). The
  (2,) int64 [seed, offset] tensor rides the same bundle
  (``host.transform_a_operand(dropout_a(p), A, {"seed": t}, tile_m)``, or
  ``TransformAOperand(A, t)`` at the direct-compile layer). The mask of
  (m, k) is a pure function of (m, k, seed, offset) — any kernel
  regenerates it — and is split-k invariant. Mask-only: fold 1/(1-p) into
  the epilogue.

* PACKED decodes (``packed=PackedInput(...)``): the fn IS the decode — the
  :meth:`~quack.operand_transform.formats.DecodeFormat.decode_k16` body,
  ``fn(xw, sfw, b, consts) -> 4 packed regs`` — and the ``PackedInput``
  carries the geometry (w8 / tile_k) and the host bundle (prepare /
  quantize_reference / dequant_reference) that must stay consistent with it.
  The mod mints a ``DecodeFormat``, so it slots into everything the
  class-based formats can (TransformAW4, the gemm_w4 wrapper, the roundtrip
  test fixture). ``sfw`` is always None for now (scale-factor strips need the
  aux A-side operand, not ported yet — sf_words must be 0).

A mod is a factory ``gemm -> TransformA`` — pass it straight to
``GemmSm90(transform_a=mod)``. ``__quack_semantic_key__`` fail-closed
fingerprints the fn (source + every capture, via the gemm_epilogue keyer), so
mods compose with the jit-cache machinery.

Torch surfaces: ``EpiMod.__call__``/``plan`` take ``transform_a=mod`` plus
``transform_operands={name: tensor}`` for runtime-operand mods (raw values —
the TransformAOperand bundle is built there from the resolved config; only
``mod.gemm`` takes pre-built bundles). Under torch.compile the same call
records the single ``quack::gemm_epi`` custom op: mods cross by
``semantic_digest`` (registered at construction; bind the mod to a module
global for graphs that must survive a fresh process) and operand tensors
ride the op's input list. See quack.gemm_runtime.torch_op.
"""

import hashlib
import inspect
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional


import cutlass.cute as cute

from torch._vendor.quack.operand_transform.formats import DecodeFormat, decode_format
from torch._vendor.quack.gemm_runtime.identity import (
    TORCH_OP_TRANSFORM_MODS,
    TransformARef,
    function_semantic_key as _function_semantic_key,
    module_locator,
    register_local_transform_mod,
    semantic_value_key as _semantic_value_key,
)
from torch._vendor.quack.operand_transform.kinds import ARG_KINDS
from torch._vendor.quack.operand_transform.transform import TransformAValue, TransformAW4

__all__ = ["a_transform", "ATransformMod", "PackedInput", "dropout_a", "w4_transform"]


@dataclass(frozen=True)
class PackedInput:
    """Storage geometry + host bundle for a packed-storage fn transform.
    Field meanings match :class:`~quack.operand_transform.formats.DecodeFormat`
    (w8: 32 B/thread raw words; tile_k: the k-tile the repack is built
    around). The host callables must stay consistent with the fn — the
    roundtrip test fixture is what pins that."""

    name: str
    w8: bool = False
    tile_k: int = 64
    make_consts: Optional[Callable] = None
    prepare: Optional[Callable] = None
    quantize_reference: Optional[Callable] = None
    dequant_reference: Optional[Callable] = None


class ResolvedOperands(NamedTuple):
    """EpiMod.gemm's slot resolution — ONE statement of the transform call
    contract: what crosses in the kernel's mA/mB slots at launch, the plain
    activations tensor for metadata/validation, the runtime-operand view
    (its metadata is a plan-key slot), the resolved tile_K, and the padded
    weight N for layout-owning transforms (None: derive N from B as
    usual)."""

    a_slot: object
    b_slot: object
    A: object
    arg_sf: object = None
    tile_k: object = None
    n_gemm: object = None


class TransformModBase:
    """Shared ``transform_a=`` handle protocol. The generic layers
    (gemm_runtime.host, EpiMod.gemm/__call__/plan, gemm_runtime.torch_op)
    never enumerate
    transform flavours: they normalize any handle to a mod
    (:func:`quack.operand_transform.host.as_transform_mod`) and call these
    methods — the layout-owning vs value branch below is its single
    statement."""

    args = ()
    packed = None
    consts = None
    regs = None

    @property
    def owned_fmt(self):
        """DecodeFormat when this transform owns A's layout (packed
        weights); None for value transforms (A stays a plain (M, K)
        operand)."""
        return None

    @property
    def needs_operands(self):
        """True when calls must carry runtime operand tensors (``args``)."""
        return bool(self.args)

    def plan_key(self):
        """Cheap plan-cache identity (the digest is precomputed)."""
        return ("mod", self.semantic_digest)

    def config_ok(self, cfg):
        """Cheap autotune prune: can this GemmConfig possibly run this
        transform? Geometry/kernel asserts still guard correctness (a
        mispruned config fails host validation and benches as inf) — this
        only avoids wasted compile attempts."""
        if cfg.swap_ab:
            return False  # transform_a + swap_ab is rejected at the call
        fmt = self.owned_fmt
        if fmt is None:
            return True
        return (
            not cfg.pingpong
            and cfg.cluster_m == 1
            and cfg.tile_m % 64 == 0
            and cfg.tile_k in (None, fmt.tile_k)
        )

    def compile_ref(self):
        """Picklable ref for the jit-cache / async-compile boundary
        (registers this mod for in-process resolution; workers get it as a
        cloudpickle payload)."""
        register_local_transform_mod(self.semantic_digest, self)
        return TransformARef("mod_local", semantic_digest=self.semantic_digest)

    def _module_locator(self):
        """Import anchor for cross-process torch.compile graphs (fn-authored
        mods override via their fn; format/dropout handles have no natural
        anchor)."""
        return None

    def default_config(self, A, B):
        """Transform-specific default GemmConfig, or None for the caller's
        per-arch default. Layout-owning formats get the measured W4
        coverage rule."""
        fmt = self.owned_fmt
        if fmt is None:
            return None
        from torch._vendor.quack.cute_dsl_utils import get_device_capacity
        from torch._vendor.quack.gemm_config import GemmConfig
        from torch._vendor.quack.operand_transform.host import pick_w4_cfg, pick_w4a8_cfg

        if fmt.promote:
            tm, tn, _sk = pick_w4a8_cfg(A.shape[-2], self.padded_n(B))
        else:
            # W4 runs atom_n == 1 on every arch (SM120's (4,1,1)/(8,1,1)
            # decode layouts included): tile_n floor is the 16-wide warp span
            tm, tn, _sk = pick_w4_cfg(
                A.shape[-2],
                self.padded_n(B),
                A.shape[-1] // fmt.tile_k,
                sm120=get_device_capacity(A.device)[0] == 12,
                device=A.device,
            )
        return GemmConfig(
            tile_m=tm,
            tile_n=tn,
            cluster_m=1,
            cluster_n=1,
            pingpong=False,
            is_dynamic_persistent=False,
        )

    def padded_n(self, B):
        """Padded weight N when B is this transform's repacked blob; None
        when B is a plain operand."""
        if self.owned_fmt is None:
            return None
        from torch._vendor.quack.operand_transform.host import w4_padded_n

        return w4_padded_n(B)

    def resolve_operands(self, A, B, transform_sf, tile_m, tile_k) -> ResolvedOperands:
        fmt = self.owned_fmt
        if fmt is None:
            if transform_sf is not None:
                raise ValueError("transform_sf without a layout-owning transform_a")
            if not self.args:
                return ResolvedOperands(A, B, A, tile_k=tile_k)
            from torch._vendor.quack.operand_transform.transform import TransformAOperand

            if not isinstance(A, TransformAOperand):
                raise ValueError(
                    "transform_a with runtime operands: pass A as "
                    "transform_a_operand(mod, A, values, tile_M, tile_K)"
                )
            # blob = plain (m, k) A for keys/validation, sf = the operand
            # view (must have been built with THIS call's tile_M/tile_K; a
            # mismatch fails at trace against the kind's fake).
            return ResolvedOperands(A, B, A.blob, arg_sf=A.sf, tile_k=tile_k)
        # Layout-owning: caller A = activations, caller B = the repacked blob.
        from torch._vendor.quack.operand_transform.host import w4_operand_views

        if A.ndim != 2 or A.stride(-1) != 1:
            raise ValueError("layout-owning transform_a: activations are (m, k), k-major")
        if tile_k is not None and tile_k != fmt.tile_k:
            raise ValueError(f"format {fmt.name!r} requires tile_K={fmt.tile_k}")
        if B.shape[1] * fmt.tile_k != A.shape[-1]:
            raise ValueError(
                f"K mismatch: activations K={A.shape[-1]}, blob K={B.shape[1] * fmt.tile_k}"
            )
        n_full = self.padded_n(B)
        if n_full % tile_m:
            raise ValueError(
                f"padded N ({n_full}) must be divisible by tile_M ({tile_m}): the blob's"
                " gmem view tiles kernel-M in whole CTA tiles"
            )
        bundle = w4_operand_views(fmt, B, transform_sf, tile_m)
        return ResolvedOperands(bundle, A, A, tile_k=fmt.tile_k, n_gemm=n_full)

    def bundle(self, A, operand_values, tile_m, tile_k):
        """TransformAOperand from RAW operand tensors (the __call__/plan
        convention; ``tile_k`` None resolves to the 16-bit kernel
        default)."""
        from torch._vendor.quack.operand_transform.host import transform_a_operand

        return transform_a_operand(
            self, A, operand_values, tile_m, 64 if tile_k is None else tile_k
        )

    def fake_bundle(self, mA_fake, a_dtype, tile_m, tile_k):
        """Trace-time twin of :meth:`bundle` (same tile_k fallback; the
        kind's smem box shape is derived from the real tile_K, so a
        mismatch fails at trace)."""
        from torch._vendor.quack.operand_transform.host import transform_a_fake_operand

        return transform_a_fake_operand(
            self, mA_fake, a_dtype, tile_m, 64 if tile_k is None else tile_k
        )

    def compile_dims(self, bundle):
        """(n_full, k) static problem geometry from a layout-owning bundle's
        blob view (256, wpt, tm64, Gt, Kt, 1); None for value transforms
        (their dims are symbolic)."""
        fmt = self.owned_fmt
        if fmt is None:
            return None
        blob = bundle.blob
        return (blob.shape[2] * blob.shape[3] * 64, blob.shape[4] * fmt.tile_k)

    def fake_operands(self, n_full, k, tile_m):
        """Trace-time fake bundle for a layout-owning transform."""
        from torch._vendor.quack.operand_transform.host import w4_fake_operands

        return w4_fake_operands(self.owned_fmt, n_full, k, tile_m)


class ATransformMod(TransformModBase):
    """A fn-authored A-operand transform; callable as the ``transform_a=``
    factory. See module docstring for the fn contracts."""

    def __init__(self, fn, vec_size, packed, consts=None, regs=None, args=None):
        self.fn = fn
        self.name = getattr(fn, "__name__", "a_transform")
        self.packed = packed
        if packed is not None:
            assert vec_size in (None, 8), "packed fn transforms decode whole k16 blocks"
            assert consts is None, "packed fns take consts via PackedInput.make_consts"
            assert not args, "runtime operands are value-fn only (packed decodes own A)"
            vec_size = 8
        else:
            vec_size = 2 if vec_size is None else vec_size
            assert vec_size in (2, 4, 8), "vec_size must be 2, 4 or 8 (one k16 block max)"
        self.vec_size = vec_size
        self.consts = consts
        self.regs = regs
        self.args = _normalize_args(fn, args)
        self._fmt = None
        self.semantic_digest = _digest(self.__quack_semantic_key__())
        TORCH_OP_TRANSFORM_MODS[self.semantic_digest] = self  # quack::gemm_epi resolution

    @property
    def owned_fmt(self):
        return self.as_decode_format() if self.packed is not None else None

    def __call__(self, gemm):
        if self.packed is not None:
            return TransformAW4(gemm, self.as_decode_format())
        return TransformAValue(gemm, self)

    def _module_locator(self):
        """(module, global_name) if this mod is reachable by import in a fresh
        process (the quack::gemm_epi custom op re-resolves it that way when a
        compiled graph crosses processes), else None — same contract as
        EpiMod._module_locator."""
        return module_locator(self, self.fn)

    def as_decode_format(self) -> DecodeFormat:
        """Mint the DecodeFormat backing a packed fn transform (cached)."""
        assert self.packed is not None, "value transforms do not define a decode format"
        if self._fmt is None:
            mod, spec = self, self.packed

            class _FnFormat(DecodeFormat):
                name = spec.name
                w8 = spec.w8
                tile_k = spec.tile_k

                def make_consts(self):
                    return spec.make_consts() if spec.make_consts is not None else None

                @cute.jit
                def decode_k16(self, xw, sfw, b, consts):
                    return mod.fn(xw, sfw, b, consts)

                def quantize_reference(self, w):
                    return spec.quantize_reference(w)

                def dequant_reference(self, q, sf):
                    return spec.dequant_reference(q, sf)

                def prepare(self, q, sf):
                    return spec.prepare(q, sf)

            _FnFormat.__qualname__ = f"FnFormat_{spec.name}"
            self._fmt = _FnFormat()
        return self._fmt

    def __quack_semantic_key__(self):
        packed_key = None
        if self.packed is not None:
            packed_key = (
                self.packed.name,
                self.packed.w8,
                self.packed.tile_k,
                # make_consts shapes device code; the other host callables
                # don't reach the kernel (repack consistency is pinned by the
                # roundtrip fixture, not the cache key).
                _function_semantic_key(self.packed.make_consts)
                if self.packed.make_consts is not None
                else None,
            )
        return (
            "a_transform",
            _semantic_value_key(self.fn, set()),
            self.vec_size,
            _function_semantic_key(self.consts) if self.consts is not None else None,
            self.regs,
            packed_key,
            self.args,
        )


def _normalize_args(fn, args) -> tuple:
    """Validate an ``args`` declaration ({fn param name: kind name}) against
    the fn signature and the kind registry; normalize to ((name, kind), ...)
    in fn-parameter order (the staging order in the kernel)."""
    if not args:
        return ()
    fn_kinds = {k for k, kind in ARG_KINDS.items() if kind.fn_facing}
    unknown = set(args.values()) - fn_kinds
    assert not unknown, f"unknown operand kind(s) {unknown}; have {fn_kinds}"
    params = list(inspect.signature(fn).parameters)
    missing = set(args) - set(params[1:])
    assert not missing, f"args {missing} are not parameters of {fn.__name__} (after x)"
    return tuple((name, args[name]) for name in params[1:] if name in args)


def _digest(key) -> str:
    return hashlib.sha256(repr(key).encode()).hexdigest()


class PackedFormatMod(TransformModBase):
    """``transform_a=`` handle for a DecodeFormat: runs a registered (or
    instance) packed dequant format, e.g.
    ``GemmSm90(transform_a=w4_transform("qtip2s"))``."""

    def __init__(self, fmt):
        self.fmt = decode_format(fmt)
        self.packed = self.fmt
        # registry names get a by-name compile ref (stable disk keys, no
        # payload shipping); instance formats resolve through the local
        # registry like fn mods
        self._registry_name = fmt if isinstance(fmt, str) else None
        self.name = f"w4_{self.fmt.name}"
        self.semantic_digest = _digest(
            (
                "w4_format",
                self.fmt.name,
                _function_semantic_key(type(self.fmt).decode_k16),
                _function_semantic_key(type(self.fmt).make_consts),
            )
        )
        TORCH_OP_TRANSFORM_MODS[self.semantic_digest] = self

    @property
    def owned_fmt(self):
        return self.fmt

    def compile_ref(self):
        if self._registry_name is not None:
            return TransformARef("w4_name", name=self._registry_name)
        return super().compile_ref()

    def __call__(self, gemm):
        return TransformAW4(gemm, self.fmt)

    def __quack_semantic_key__(self):
        return ("w4_format_mod", self.semantic_digest)


def w4_transform(fmt) -> PackedFormatMod:
    """A ``transform_a=`` handle for a packed dequant format (name from the
    W4_FORMATS registry, or a DecodeFormat instance)."""
    return PackedFormatMod(fmt)


class DropoutAMod(TransformModBase):
    """``transform_a=`` handle for dropout on A (see TransformADropout): the
    keep-mask of element (m, k) is a pure function of (m, k, seed, offset),
    reproducible by any kernel and invariant under split-k. MASK-ONLY — fold
    1/(1-p) into the epilogue. The (2,) int64 [seed, offset] CUDA tensor is
    the runtime operand, riding the TransformAOperand bundle's sf slot
    (``args`` declares it so the generic host plumbing — mod.gemm unpack,
    trace fakes, plan keys — treats it like any strip operand)."""

    args = (("seed", "seed_i64x2"),)

    def __init__(self, p: float, rounds: int = 7):
        assert 0.0 <= p < 1.0, "drop probability must be in [0, 1)"
        # keep iff byte >= threshold: P(drop) = threshold / 256, exactly
        self.p = p
        self.threshold = min(int(round(p * 256)), 255)
        self.rounds = rounds
        self.name = f"dropout_a_t{self.threshold}"
        # trailing int: scheme version
        self.semantic_digest = _digest(("dropout_a", self.threshold, rounds, 1))
        TORCH_OP_TRANSFORM_MODS[self.semantic_digest] = self

    def __call__(self, gemm):
        from torch._vendor.quack.operand_transform.transform import TransformADropout

        return TransformADropout(gemm, self)

    def __quack_semantic_key__(self):
        return ("dropout_a_mod", self.semantic_digest)


def dropout_a(p: float, rounds: int = 7) -> DropoutAMod:
    """A ``transform_a=`` handle for dropout on A with drop probability
    ``round(p * 256) / 256`` (mask-only; scale via the epilogue)."""
    return DropoutAMod(p, rounds)


def a_transform(
    vec_size: Optional[int] = None,
    packed=None,
    consts: Optional[Callable] = None,
    regs: Optional[tuple] = None,
    args: Optional[dict] = None,
):
    """Decorator: turn a plain fn into an A-operand transform mod. See the
    module docstring for the two fn contracts.

    ``consts=callable`` (value fns): called once per kernel (hoisted — LUTs,
    packed constants); its result is the fn's LAST parameter.
    ``args={param: kind}`` (value fns): runtime operands — each named fn
    parameter (between x and consts) receives its per-element values as a
    TensorSSA vector, staged by its kind (kinds.ARG_KINDS;
    the (M, K) mirror of EpiOps' operand kinds). A crosses as the
    ``host.transform_a_operand(mod, A, {param: tensor}, tile_m)`` bundle.
    ``regs=(load, mma)`` overrides the register budget split (multiples of 8,
    see setmaxnreg constraints in TransformAW4)."""

    def wrap(fn):
        return ATransformMod(
            fn, vec_size=vec_size, packed=packed, consts=consts, regs=regs, args=args
        )

    return wrap
