# Copyright (c) 2026, Tri Dao.
"""Runtime-operand KINDS for A transforms: a taxonomy over the transform's
(M, K) index space — the mainloop mirror of EpiOps' operand kinds over
(M, N) (scalar / colvec / rowvec / tile).

One kind = ONE object owning every side of the feature (the EpiOp rule):

* ``geometry`` — the (gran_m, g_m, gran_k, g_k, k_inner) resolution shared
  by the device staging and the host views (one statement, never mirrored);
* ``device_arg(gemm)`` — the kernel-side staging impl handed to
  :class:`~quack.operand_transform.transform.TransformAValue`;
* ``host_view(A, value, tile_m, tile_k)`` / ``host_fake(a_dtype, tile_m,
  tile_k)`` — the runtime torch view and its trace-time fake twin for the
  aux slot of the :class:`TransformAOperand` bundle;
* ``fn_facing`` — whether the kind may appear in ``@a_transform(args=...)``
  (``seed_i64x2`` is dropout-internal: it rides the bundle raw and is
  consumed by :class:`TransformADropout`, never by a value fn).

Shipped kinds: the strip family — one MMA-dtype value per (m-group of
``gran_m`` rows, k-group of ``gran_k`` elements), delivered via the aux
A-side TMA slot (per-stage smem under the AB mbarrier): ``colvec_ktile``
(per (row, k-tile)), ``colvec_k64/k32/k16`` (dense blockscaled-SF
granularities), ``kvec_m64`` (per (m64 row block, k-element) — the LCE dw
strip); plus ``seed_i64x2`` (dropout's raw (2,) int64 [seed, offset]).
There is deliberately NO k-invariant ``colvec`` kind: a per-row scale
commutes through the GEMM to the epilogue.
"""

import torch

import cutlass
import cutlass.cute as cute
from cutlass import const_expr

__all__ = ["ARG_KINDS", "SeedKind", "StripKind", "strip_geometry"]


def strip_geometry(gran_m, gran_k, tile_m, tile_k):
    """(gran_m, g_m, gran_k, g_k, k_inner) for a (gran_m, gran_k) strip at
    (tile_m, tile_k); ``None`` gran means the whole tile extent. THE single
    statement of the strip geometry — device staging, runtime views and
    trace fakes all call this. ``k_inner``: the box's stride-1 axis is the
    FINER one — TMA needs 16 B alignment on every non-inner stride, and only
    the finer axis's group count is large enough to guarantee it (e.g. the
    dw strip's per-m-tile stride is tile_M/64 elements = 4-16 B)."""
    gran_m = tile_m if gran_m is None else gran_m
    gran_k = tile_k if gran_k is None else gran_k
    assert tile_m % gran_m == 0, f"gran_m {gran_m} must divide tile_M ({tile_m})"
    assert tile_k % gran_k == 0, f"gran_k {gran_k} must divide tile_K ({tile_k})"
    return gran_m, tile_m // gran_m, gran_k, tile_k // gran_k, gran_k < gran_m


class _StripAux:
    """Dense per-(m-group, k-group) values riding the aux-operand slot: a
    (g_m x g_k)-element MMA-dtype box per k-tile TMA'd into per-stage smem
    under the AB mbarrier — the values arrive WITH A, no gather latency in
    the produce path (the AuxOperandA protocol, duck-typed by GemmSm90).
    Element-typed, so no byte/m64 factorization is needed (TMA box dims
    count ELEMENTS, <= 256 each: g_m <= tile_M <= 256 and g_k <= tile_K
    always hold). Box modes are (inner, outer) with the finer axis inner
    (see :func:`strip_geometry`); gmem view: (inner, outer, G_m, RestK, L)
    over a contiguous (outer-groups, inner-groups) tensor (see
    :meth:`StripKind.host_view`)."""

    multicast = False  # small boxes: dup loads beat mcast box-splitting

    def __init__(self, gemm, gran_m, gran_k):
        self.gemm = gemm
        self.gran_m, self.gran_k = gran_m, gran_k
        self.dtype = gemm.mma_a_dtype

    def _geometry(self):
        # resolved LAZILY — tile_K is 0 at transform-ctor time when the ctor
        # tile shape is (M, N) (resolved in _setup_tiled_mma); every consumer
        # runs after that.
        gemm = self.gemm
        return strip_geometry(
            self.gran_m, self.gran_k, gemm.cta_tile_shape_mnk[0], gemm.cta_tile_shape_mnk[2]
        )

    def _box(self):
        _, g_m, _, g_k, k_inner = self._geometry()
        return (g_k, g_m) if k_inner else (g_m, g_k)

    def bytes_per_stage(self):
        box = self._box()
        return box[0] * box[1] * self.dtype.width // 8

    def make_smem_layout_staged(self, ab_stage):
        return cute.make_ordered_layout((*self._box(), ab_stage), order=(0, 1, 2))

    def make_tma(self, mAux):
        gemm = self.gemm
        box = self._box()
        smem_layout = cute.make_ordered_layout(box, order=(0, 1))
        return gemm._make_tma_atoms_and_tensors(mAux, smem_layout, box, gemm.cluster_shape_mnk[1])

    def gmem_slice(self, mAux, tile_coord_mnkl, batch_idx):
        # (inner, outer, Gm, RestK, L) -> (inner, outer, RestK)
        return mAux[None, None, tile_coord_mnkl[0], None, batch_idx]


class _StripArg:
    """Kernel-side staging for the strip family, refreshed per k-tile.

    (the epi_ops VecLoad idiom, all partition algebra — no index math):
    broadcast the smem box to (tile_M, tile_K) with NESTED modes — m-mode
    (gran_m, g_m) and k-mode (gran_k, g_k), stride 0 on the inner
    (within-group) levels — partition it with the fragment's own tiled_mma,
    and cache a fragment-congruent rmem tensor whose zero-stride modes share
    registers, refreshed once per k-tile with one LDS per distinct value
    (filter_zeros). Per-element reads are identity indexing, so the staging
    is selects only and the fn math stays packed (HMUL2). Which fragment
    slots share a value falls out of the layout composition — any
    granularities dividing the tile work, including quad-varying ones."""

    def __init__(self, gemm, gran_m, gran_k):
        self.gemm = gemm
        self.gran_m, self.gran_k = gran_m, gran_k  # geometry resolves lazily
        self.aux = _StripAux(gemm, gran_m, gran_k)
        self._tCsS = None
        self._rvals = None

    @cute.jit
    def setup(self, tiled_mma, tidx, mma_m, sAux):
        """Once per kernel (inside make_copy_block)."""
        assert sAux is not None, "strip operands ride the aux slot (pass A as TransformAOperand)"
        gemm = self.gemm
        gran_m, g_m, gran_k, g_k, k_inner = self.aux._geometry()
        # Broadcast the (inner, outer, stage) box to (tile_M, tile_K, stage):
        # each axis is a nested (gran, g) mode — expand within a group
        # (stride 0), advance one box column across groups (the g factor is
        # the mode's second level, not a separate axis).
        sm, sk = (g_k, 1) if k_inner else (1, g_m)  # box strides of (m-group, k-group)
        sMK = cute.make_tensor(
            sAux.iterator,
            cute.make_layout(
                ((gran_m, g_m), (gran_k, g_k), gemm.ab_stage),
                stride=((0, sm), (0, sk), g_m * g_k),
            ),
        )
        # Partition with the fragment's own tiled_mma (epi_ops VecLoad
        # idiom): every value lands aligned with its fragment element — no
        # coordinate math. True lane: the fragment tensors are partitioned
        # from the per-warpgroup slice, but addressing needs the real thread.
        self._tCsS = tiled_mma.get_slice(tidx).partition_A(sMK)
        # fragment-congruent cache: make_rmem_tensor keeps the zero strides
        # (duplicates share a register) and compacts the rest, so the cache
        # holds exactly the per-lane distinct values
        self._rvals = cute.make_rmem_tensor(
            self._tCsS[None, None, None, 0].layout, gemm.mma_a_dtype
        )

    @cute.jit
    def on_block(self, stage_idx, b, mma_m):
        """Refresh the register cache — one LDS per DISTINCT value
        (filter_zeros pairs the deduped elements). k-coarse strips (a value
        spans >= one k16 block) load the whole tile's values once at b == 0;
        k-fine strips load block b's disjoint slice each block — the same
        produce rhythm as A itself, so the LDS spread across the WGMMA
        shadow and live ranges stay one block long (ptxas schedules within
        the unrolled body but won't restructure a whole-tile live range)."""
        gran_k = self.aux._geometry()[2]
        if const_expr(gran_k >= 16):
            if const_expr(b == 0):
                cute.autovec_copy(
                    cute.filter_zeros(self._tCsS[None, None, None, stage_idx]),
                    cute.filter_zeros(self._rvals),
                )
        else:
            cute.autovec_copy(
                cute.filter_zeros(self._tCsS[None, None, b, stage_idx]),
                cute.filter_zeros(self._rvals[None, None, b]),
            )

    def element(self, coord, m, b):
        """The operand value of fragment element (coord, m, b): the cache is
        fragment-congruent, so this is identity indexing — the zero-stride
        modes resolve duplicates to the same register."""
        return self._rvals[coord, m, b]


class StripKind:
    """One value per (m-group of ``gran_m`` A rows, k-group of ``gran_k``
    elements). Corners: (1, None=tile_K) = ``colvec_ktile`` (the LCE dx pow2
    rescale), (1, 16/32/64) = dense blockscaled-SF granularities, (64, 1) =
    ``kvec_m64`` (the LCE dw strip: per (vocab m64 block, token))."""

    fn_facing = True

    def __init__(self, gran_m, gran_k):
        self.gran_m, self.gran_k = gran_m, gran_k

    def device_arg(self, gemm):
        return _StripArg(gemm, self.gran_m, self.gran_k)

    def host_view(self, A, value, tile_m, tile_k):
        """The user tensor is (outer-groups, inner-groups) row-major with the
        FINER axis contiguous — (rk * g_k, M / gran_m) for m-fine strips
        (colvec family), (M / gran_m, rk * g_k) for k-fine strips (kvec
        family); rk = ceil(K / tile_k), so a ragged K tail is PADDED to whole
        k-tiles at group granularity. Returns the (inner, outer, G_m, rk, 1)
        element-typed VIEW for the aux TMA slot (one box per (m-tile,
        k-tile) — see :class:`_StripAux`)."""
        granm, g_m, grank, g_k, k_inner = strip_geometry(self.gran_m, self.gran_k, tile_m, tile_k)
        m, k = A.shape
        assert value.dtype == A.dtype and value.element_size() == 2
        assert value.is_contiguous(), "strip operands need a contiguous (outer, inner) tensor"
        assert m % tile_m == 0, f"M ({m}) must be divisible by tile_M ({tile_m})"
        rk = -(-k // tile_k)
        mg, kg = m // granm, rk * g_k
        shape = (mg, kg) if k_inner else (kg, mg)
        assert value.shape == torch.Size(shape), (
            f"strip shape {tuple(value.shape)} must be {shape} = "
            f"(M / {granm}, ceil(K / tile_K) * {g_k})"
            + ("" if k_inner else " transposed")
            + " (K padded to whole k-tiles)"
        )
        if k_inner:
            # (g_k, g_m, Gm, rk, 1); strides (1, KG, g_m*KG, g_k, 1)
            return value.view(m // tile_m, g_m, rk, g_k).permute(3, 1, 0, 2).unsqueeze(-1)
        # (g_m, g_k, Gm, rk, 1); strides (1, MG, g_m, g_k*MG, 1)
        return value.view(rk, g_k, m // tile_m, g_m).permute(3, 1, 2, 0).unsqueeze(-1)

    def host_fake(self, a_dtype, tile_m, tile_k):
        """Fake matching :meth:`host_view`'s strides exactly: contiguous
        inner run, one static stride (the box-outer axis's per-m-tile /
        per-k-tile step within the user tensor), symbolic G_m / rk extents
        and symbolic remaining strides (8-element-divisible: the 16 B TMA
        rule, guaranteed by M % tile_M == 0 and g_k >= 8 on the k-fine
        path)."""
        _granm, g_m, _grank, g_k, k_inner = strip_geometry(self.gran_m, self.gran_k, tile_m, tile_k)
        gm, rk = cute.sym_int(), cute.sym_int()
        sym8 = lambda: cute.sym_int64(divisibility=8)
        if k_inner:
            shape, stride = (g_k, g_m, gm, rk, 1), (1, sym8(), sym8(), g_k, 1)
        else:
            shape, stride = (g_m, g_k, gm, rk, 1), (1, sym8(), g_m, sym8(), 1)
        return cute.runtime.make_fake_tensor(a_dtype, shape, stride=stride, assumed_align=16)


class SeedKind:
    """Dropout's [seed, offset] operand: a (2,) int64 CUDA tensor crossing
    RAW in the bundle's sf slot (``aux_raw`` — no TMA box, no smem, consumed
    by :class:`TransformADropout`, never staged for a value fn)."""

    fn_facing = False

    def device_arg(self, gemm):
        raise NotImplementedError("seed_i64x2 is dropout-internal (aux_raw), not a value-fn kind")

    def host_view(self, A, value, tile_m, tile_k):
        assert value.dtype == torch.int64 and tuple(value.shape) == (2,), (
            "seed operand must be a (2,) int64 [seed, offset] tensor"
        )
        assert value.is_cuda and value.is_contiguous() and value.data_ptr() % 16 == 0
        return value

    def host_fake(self, a_dtype, tile_m, tile_k):
        return cute.runtime.make_fake_tensor(cutlass.Int64, (2,), stride=(1,), assumed_align=16)


ARG_KINDS = {
    "colvec_ktile": StripKind(gran_m=1, gran_k=None),
    "colvec_k64": StripKind(gran_m=1, gran_k=64),
    "colvec_k32": StripKind(gran_m=1, gran_k=32),
    "colvec_k16": StripKind(gran_m=1, gran_k=16),
    "kvec_m64": StripKind(gran_m=64, gran_k=1),
    "seed_i64x2": SeedKind(),
}
