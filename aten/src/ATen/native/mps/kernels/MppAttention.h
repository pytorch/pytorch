// MPP cooperative-tensor flash-attention prefill for MPS, on
// mpp::tensor_ops::matmul2d. Adapted from MLX. Emitted only with the
// Metal 4 SDK; runtime dispatch gates on macOS 26.2+.
#pragma once

#include <ATen/native/mps/kernels/PrefillAttention.h>

#if C10_METAL_HAS_MPP

#define MPP_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define MPP_CONST static constant constexpr const

// MSL's metal::integral_constant lacks arithmetic overloads, so add the
// minimum needed to multiply runtime indices by Int<1>{}
template <typename T, T Val, typename U>
METAL_FUNC constexpr auto operator*(U lhs, metal::integral_constant<T, Val>) {
  return lhs * Val;
}
template <typename T, T Val, typename U>
METAL_FUNC constexpr auto operator*(metal::integral_constant<T, Val>, U rhs) {
  return Val * rhs;
}

// 16x16 simdgroup tile, 8 cells/lane in 2 rows x 4 cols (rows fm, fm+8).
// Within-row col pattern is contig-quad {0,1,2,3} on Apple10 and split-pair
// {0,1,8,9} on Apple8, this matches what matmul2d::run() expects in each
// arch's ct_a/ct_b/ct_c slots so the mma below can register-copy directly.
template <bool IS_APPLE10>
struct BaseMPPFrag {
  MPP_CONST short kFragRows = 16;
  MPP_CONST short kFragCols = 16;
  MPP_CONST short kElemsPerFrag = (kFragRows * kFragCols) / 32; // 8
  MPP_CONST short kElemRows = 2;
  MPP_CONST short kElemCols = 4;
  MPP_CONST short kElemRowsJump = 8;

  static_assert(
      kElemRows * kElemCols == kElemsPerFrag,
      "MPPFrag shape inconsistent with size");

  template <typename U>
  using dtype_frag_t = metal::vec<U, kElemsPerFrag>;

  METAL_FUNC static short2 get_coord() {
    const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
    if (IS_APPLE10) {
      const short qid = lane >> 2;
      const short fm = (qid & 4) | ((lane >> 1) & 3);
      const short fn = ((qid & 2) | (lane & 1)) * 4;
      return short2{fn, fm};
    } else {
      const short fm = ((lane >> 4) & 1) * 4 + ((lane >> 1) & 3);
      const short fn = ((lane >> 3) & 1) * 4 + (lane & 1) * 2;
      return short2{fn, fm};
    }
  }

  // Slot j -> col offset: contig {0,1,2,3} on A10, split-pair {0,1,8,9} on A8.
  METAL_FUNC static constexpr short col_offset(short j) {
    if (IS_APPLE10) {
      return j;
    } else {
      return (j & 1) | ((j >> 1) << 3);
    }
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      if constexpr (metal::is_same_v<StrY, Int<1>>) {
        MPP_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[r * str_x + c + col_offset(j)]);
        }
      } else {
        MPP_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[r * str_x + (c + col_offset(j)) * str_y]);
        }
      }
    }
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_safe(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    auto ly = lim_y - sc.x;
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      MPP_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        const auto coff = col_offset(j);
        if (r < lx && (c + coff) < ly) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[r * str_x + (c + coff) * str_y]);
        } else {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      OffX off_x = {},
      OffY off_y = {}) {
    using U = pointer_element_t<DstPtrType>;
    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      if constexpr (metal::is_same_v<StrY, Int<1>>) {
        MPP_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[r * str_x + c + col_offset(j)] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      } else {
        MPP_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[r * str_x + (c + col_offset(j)) * str_y] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_safe(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = {},
      OffY off_y = {}) {
    using U = pointer_element_t<DstPtrType>;
    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    auto ly = lim_y - sc.x;
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      MPP_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        const auto coff = col_offset(j);
        if (r < lx && (c + coff) < ly) {
          dst[r * str_x + (c + coff) * str_y] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  // C[16,32] += A[16,16] * [Bn0|Bn1] (Bn0/Bn1 packed along N).
  // ct_b/ct_c slot order: A10 = (Bn0_0..7 | Bn1_0..7);
  // A8 = (Bn0[0..3] | Bn1[0..3] | Bn0[4..7] | Bn1[4..7]).
  // For Q@K^T (transpose_b=true), Bn0/Bn1 come from BaseMPPFragK so the
  // per-lane geometry already matches; slot copies are identical for both
  // transpose modes.
  template <
      typename CType,
      typename AType,
      typename BType,
      bool transpose_a = false,
      bool transpose_b = false>
  METAL_FUNC static void mma(
      thread dtype_frag_t<CType>& Cn0,
      thread dtype_frag_t<CType>& Cn1,
      const thread dtype_frag_t<AType>& A,
      metal::bool_constant<transpose_a>,
      const thread dtype_frag_t<BType>& Bn0,
      const thread dtype_frag_t<BType>& Bn1,
      metal::bool_constant<transpose_b>) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16,
        32,
        16,
        transpose_a,
        transpose_b,
        true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

    auto ct_a =
        gemm_op
            .template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto ct_b =
        gemm_op
            .template get_right_input_cooperative_tensor<AType, BType, CType>();
    auto ct_c = gemm_op.template get_destination_cooperative_tensor<
        decltype(ct_a),
        decltype(ct_b),
        CType>();

    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_a[i] = A[i];
    }

    if (IS_APPLE10) {
      MPP_PRAGMA_UNROLL
      for (short i = 0; i < kElemsPerFrag; i++) {
        ct_b[i] = Bn0[i];
        ct_b[kElemsPerFrag + i] = Bn1[i];
      }
      MPP_PRAGMA_UNROLL
      for (short i = 0; i < kElemsPerFrag; i++) {
        ct_c[i] = Cn0[i];
        ct_c[kElemsPerFrag + i] = Cn1[i];
      }

      gemm_op.run(ct_a, ct_b, ct_c);

      MPP_PRAGMA_UNROLL
      for (short i = 0; i < kElemsPerFrag; i++) {
        Cn0[i] = ct_c[i];
        Cn1[i] = ct_c[kElemsPerFrag + i];
      }
    } else {
      MPP_PRAGMA_UNROLL
      for (short i = 0; i < 4; i++) {
        ct_b[i] = Bn0[i];
        ct_b[4 + i] = Bn1[i];
        ct_b[8 + i] = Bn0[4 + i];
        ct_b[12 + i] = Bn1[4 + i];
      }
      MPP_PRAGMA_UNROLL
      for (short i = 0; i < 4; i++) {
        ct_c[i] = Cn0[i];
        ct_c[4 + i] = Cn1[i];
        ct_c[8 + i] = Cn0[4 + i];
        ct_c[12 + i] = Cn1[4 + i];
      }

      gemm_op.run(ct_a, ct_b, ct_c);

      MPP_PRAGMA_UNROLL
      for (short i = 0; i < 4; i++) {
        Cn0[i] = ct_c[i];
        Cn1[i] = ct_c[4 + i];
        Cn0[4 + i] = ct_c[8 + i];
        Cn1[4 + i] = ct_c[12 + i];
      }
    }
  }

  template <typename Op, typename T>
  METAL_FUNC static constexpr void row_reduce(
      thread const dtype_frag_t<T>& inp_vals,
      thread T* reduced_vals) {
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      T thr_reduce = Op::apply(
          Op::apply(inp_vals[i * kElemCols + 0], inp_vals[i * kElemCols + 1]),
          Op::apply(inp_vals[i * kElemCols + 2], inp_vals[i * kElemCols + 3]));
      T qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
      qgr_reduce = Op::apply(thr_reduce, qgr_reduce);
      T sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
      sgr_reduce = Op::apply(qgr_reduce, sgr_reduce);
      reduced_vals[i] = Op::apply(reduced_vals[i], sgr_reduce);
    }
  }

  template <typename Op, typename T>
  METAL_FUNC static constexpr void row_bin_op(
      thread dtype_frag_t<T>& inp_vals,
      thread T* row_vals) {
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      MPP_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        inp_vals[i * kElemCols + j] =
            Op::apply(inp_vals[i * kElemCols + j], row_vals[i]);
      }
    }
  }
};

// Second operand of Q@K^T (transpose_b=true) on Apple8: 4x2 split-pair per
// lane, rows {fm0, fm0+1, fm0+8, fm0+9}, cols {fn0, fn0+8}. K is read-only
// here, so only load / load_safe are implemented. On Apple10 the regular
// BaseMPPFrag layout already matchess
struct BaseMPPFragK_Apple8 {
  MPP_CONST short kFragRows = 16;
  MPP_CONST short kFragCols = 16;
  MPP_CONST short kElemsPerFrag = 8;
  // Rows aren't equally spaced ({fm0, fm0+1, fm0+8, fm0+9}); kElemRowsJump
  // is unused. kElemRows/kElemCols exist because MPPTile reads them.
  MPP_CONST short kElemRows = 4;
  MPP_CONST short kElemCols = 2;
  MPP_CONST short kElemRowsJump = 1;

  template <typename U>
  using dtype_frag_t = metal::vec<U, kElemsPerFrag>;

  METAL_FUNC static short2 get_coord() {
    const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
    const short fm0 = ((lane >> 3) & 1) * 4 + (lane & 1) * 2;
    const short fn0 = ((lane >> 4) & 1) * 4 + ((lane >> 1) & 3);
    return short2{fn0, fm0};
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    MPP_PRAGMA_UNROLL
    for (short s = 0; s < kElemsPerFrag; s++) {
      const short s_local = s & 3;
      const short r = off_x + (s_local & 1) + 8 * ((s_local >> 1) & 1);
      const short c = off_y + 8 * (s >> 2);
      if constexpr (metal::is_same_v<StrY, Int<1>>) {
        dst[s] = static_cast<T>(src[r * str_x + c]);
      } else {
        dst[s] = static_cast<T>(src[r * str_x + c * str_y]);
      }
    }
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_safe(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    auto ly = lim_y - sc.x;
    MPP_PRAGMA_UNROLL
    for (short s = 0; s < kElemsPerFrag; s++) {
      const short s_local = s & 3;
      const short r = off_x + (s_local & 1) + 8 * ((s_local >> 1) & 1);
      const short c = off_y + 8 * (s >> 2);
      if (r < lx && c < ly) {
        dst[s] = static_cast<T>(src[r * str_x + c * str_y]);
      } else {
        dst[s] = T(0);
      }
    }
  }
};

template <bool IS_APPLE10>
using BaseMPPFragK =
    metal::conditional_t<IS_APPLE10, BaseMPPFrag<true>, BaseMPPFragK_Apple8>;

template <typename T, short kTileRows_, short kTileCols_, class MPPFrag_>
struct MPPTile {
  using MPPFrag_t = MPPFrag_;
  using elem_type = T;

  MPP_CONST short kFragRows = MPPFrag_t::kFragRows;
  MPP_CONST short kFragCols = MPPFrag_t::kFragCols;
  MPP_CONST short kElemsPerFrag = MPPFrag_t::kElemsPerFrag;

  MPP_CONST short kTileRows = kTileRows_;
  MPP_CONST short kTileCols = kTileCols_;

  MPP_CONST short kRows = kTileRows * kFragRows;
  MPP_CONST short kCols = kTileCols * kFragCols;

  MPP_CONST short kNumFrags = kTileRows * kTileCols;
  MPP_CONST short kElemsPerTile = kNumFrags * kElemsPerFrag;

  MPP_CONST short kFragThrRows = MPPFrag_t::kElemRows;
  MPP_CONST short kFragThrCols = MPPFrag_t::kElemCols;
  MPP_CONST short kFragRowsJump = MPPFrag_t::kElemRowsJump;

  MPP_CONST short kRowsPerThread = kTileRows * MPPFrag_t::kElemRows;
  MPP_CONST short kColsPerThread = kTileCols * MPPFrag_t::kElemCols;

  using frag_type = typename MPPFrag_t::template dtype_frag_t<T>;

  frag_type val_frags[kNumFrags];

  METAL_FUNC MPPTile() thread {}

  METAL_FUNC constexpr void clear() {
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kNumFrags; ++i) {
      val_frags[i] = frag_type(0);
    }
  }

  METAL_FUNC constexpr thread frag_type& frag_at(const short i, const short j) {
    return val_frags[i * kTileCols + j];
  }
  METAL_FUNC constexpr const thread frag_type& frag_at(
      const short i,
      const short j) const {
    return val_frags[i * kTileCols + j];
  }

  METAL_FUNC thread elem_type* elems() {
    return reinterpret_cast<thread elem_type*>(val_frags);
  }
  METAL_FUNC const thread elem_type* elems() const {
    return reinterpret_cast<const thread elem_type*>(val_frags);
  }

  template <typename Op>
  METAL_FUNC void row_reduce(thread metal::vec<T, kRowsPerThread>& vals) const {
    auto vptr = (thread T*)(&vals);
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      MPP_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MPPFrag_t::template row_reduce<Op>(
            frag_at(i, j), &vptr[i * kFragThrRows]);
      }
    }
  }

  template <typename Op>
  METAL_FUNC void row_bin_op(thread metal::vec<T, kRowsPerThread>& vals) {
    auto vptr = (thread T*)(&vals);
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      MPP_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MPPFrag_t::template row_bin_op<Op>(
            frag_at(i, j), &vptr[i * kFragThrRows]);
      }
    }
  }

  template <typename U>
  METAL_FUNC void load(const device U* src, const int ld) {
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      MPP_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MPPFrag_t::load(
            frag_at(i, j), src, ld, Int<1>{}, i * kFragRows, j * kFragCols);
      }
    }
  }

  template <typename U>
  METAL_FUNC void load_safe(
      const device U* src,
      const int ld,
      const int n_rows,
      const int n_cols) {
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      MPP_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MPPFrag_t::load_safe(
            frag_at(i, j),
            src,
            ld,
            Int<1>{},
            n_rows,
            n_cols,
            i * kFragRows,
            j * kFragCols);
      }
    }
  }

  template <typename U>
  METAL_FUNC void store(device U* dst, const int ld) const {
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      MPP_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MPPFrag_t::store(
            frag_at(i, j), dst, ld, Int<1>{}, i * kFragRows, j * kFragCols);
      }
    }
  }

  template <typename U>
  METAL_FUNC void store_safe(
      device U* dst,
      const int ld,
      const int n_rows,
      const int n_cols) const {
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      MPP_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MPPFrag_t::store_safe(
            frag_at(i, j),
            dst,
            ld,
            Int<1>{},
            n_rows,
            n_cols,
            i * kFragRows,
            j * kFragCols);
      }
    }
  }
};

// Flash-attention prefill via matmul2d. Mirrors prefill_attention's contract
// (PrefillAttnParams + 4D mask, last-dim contiguous, [B, H, L, D] layout,
// upper-left causal). One threadgroup per (Q-block, head, batch, V/O slab).
template <
    typename T,
    int BD,
    bool HAS_MASK,
    bool DO_CAUSAL,
    bool IS_APPLE10,
    bool CHUNKED, // QK dim wider than BD, accumulated in BD-wide chunks
    typename MaskType>
[[kernel,
  max_total_threads_per_threadgroup(
      kPrefillSimdgroups* c10::metal::simdgroup_size)]] void
prefill_attention_mpp(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant PrefillAttnParams* params [[buffer(4)]],
    const constant PrefillAttnMaskParams* mask_params [[buffer(5)]],
    const device MaskType* mask [[buffer(6)]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  using AccumType = c10::metal::accum_t<T>;
  constexpr int BQ = kMppBQ;
  constexpr int BK = kMppBK;

  const int slab = tid.z / params->B;
  const int batch_idx = tid.z - slab * params->B;
  const int head_idx = tid.y;
  const int block_idx = tid.x;

  const int qL = params->qL;
  const int kL = params->kL;
  const int D = params->D;
  const int Dv = params->Dv;

  if (block_idx * BQ >= qL) {
    return;
  }

  const int NK_aligned = kL / BK;
  const int kL_rem = kL - NK_aligned * BK;
  const bool align_K = (kL_rem == 0);
  const int NK = align_K ? NK_aligned : (NK_aligned + 1);

  const int NQ_aligned = qL / BQ;
  const int qL_rem = qL - NQ_aligned * BQ;
  const bool align_Q = (qL_rem == 0);

  const int kv_head_idx = head_idx / params->gqa_factor;

  Q += batch_idx * params->Q_strides[0] + head_idx * params->Q_strides[1] +
      (block_idx * BQ) * params->Q_strides[2];
  K += batch_idx * params->K_strides[0] + kv_head_idx * params->K_strides[1];
  V += batch_idx * params->V_strides[0] + kv_head_idx * params->V_strides[1] +
      slab * Dv;
  O += batch_idx * params->O_strides[0] + head_idx * params->O_strides[1] +
      (block_idx * BQ) * params->O_strides[2] + slab * Dv;

  if constexpr (HAS_MASK) {
    mask += batch_idx * mask_params->M_strides[0] +
        head_idx * mask_params->M_strides[1];
  }

  // Narrowed back to 32-bit: these only ever multiply in-tile offsets, the
  // base pointers having already been advanced.
  const int q_seq_stride = int(params->Q_strides[2]);
  const int k_seq_stride = int(params->K_strides[2]);
  const int v_seq_stride = int(params->V_strides[2]);
  const int o_seq_stride = int(params->O_strides[2]);

  // Pre-multiply by 1/ln(2) so we can use exp2 in the inner softmax.
  const AccumType scale2 =
      static_cast<AccumType>(params->scale) * 1.44269504089f;

  constexpr short kU = 16;
  constexpr int kNWarps = kPrefillSimdgroups;
  static_assert(
      BQ >= (kNWarps * kU) && BQ % (kNWarps * kU) == 0,
      "Each simdgroup must host at least 1 MPP fragment along Q sequence.");

  constexpr int TQ = BQ / (kNWarps * kU);
  constexpr int TD = BD / kU;
  constexpr short TK = BK / kU;
  static_assert(TQ == 1, "Check TQ");
  static_assert(
      (TK & 1) == 0,
      "TK must be even: each Q@K mma writes 2 N-fragments at a time.");
  static_assert(
      (TD & 1) == 0,
      "TD must be even: each P@V mma writes 2 N-fragments at a time.");

  using QFrag_t = BaseMPPFrag<IS_APPLE10>;
  using KFrag_t = BaseMPPFragK<IS_APPLE10>;
  using otile_t = MPPTile<AccumType, TQ, TD, QFrag_t>;
  otile_t Otile;
  Otile.clear();

  const short tm = kU * TQ * simd_group_id;
  Q += tm * q_seq_stride;

  const short2 simd_coord = otile_t::MPPFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  constexpr short kRowsPT = otile_t::kRowsPerThread;
  metal::vec<AccumType, kRowsPT> max_score;
  metal::vec<AccumType, kRowsPT> sum_score{0};
  MPP_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = -INFINITY;
  }

  // PyTorch upper-left causal: query row r attends to key cols 0..r,
  // independent of qL vs kL.
  int kb_lim = NK;
  int kb_min_causal = NK;
  if constexpr (DO_CAUSAL) {
    int q_max = (block_idx + 1) * BQ;
    kb_lim = min(NK, (q_max + BK - 1) / BK);
    int q_min = block_idx * BQ;
    kb_min_causal = q_min / BK;
  }

  const bool is_last_q = (block_idx == NQ_aligned);
  const short q_rows = (!align_Q && is_last_q) ? short(qL_rem - tm) : BQ;

  for (int kb = 0; kb < kb_lim; kb++) {
    const bool is_last_k = (kb == NK_aligned);
    const short k_rows = (!align_K && is_last_k) ? short(kL_rem) : BK;

    using stile_t = MPPTile<AccumType, TQ, TK, QFrag_t>;
    stile_t Stile;
    Stile.clear();

    // S = Q @ K^T. K is loaded row-major and consumed transposed by mma.
    for (int d0 = 0; d0 < (CHUNKED ? D : BD); d0 += BD) {
      MPP_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        MPP_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik += 2) {
          MPP_PRAGMA_UNROLL
          for (short id = 0; id < TD; id++) {
            MPPTile<T, 1, 1, QFrag_t> Qtile;
            MPPTile<T, 2, 1, KFrag_t> Ktile;

            const int dcol = d0 + id * kU;
            const int Q_load_off = iq * kU * q_seq_stride + dcol;
            const int K_load_off = ik * kU * k_seq_stride + dcol;

            if (d0 + BD <= D && q_rows == BQ) {
              Qtile.load(Q + Q_load_off, q_seq_stride);
            } else {
              Qtile.load_safe(
                  Q + Q_load_off, q_seq_stride, q_rows - iq * kU, D - dcol);
            }

            if (d0 + BD <= D && k_rows == BK) {
              Ktile.load(K + K_load_off, k_seq_stride);
            } else {
              Ktile.load_safe(
                  K + K_load_off, k_seq_stride, k_rows - ik * kU, D - dcol);
            }

            QFrag_t::mma(
                Stile.frag_at(iq, ik),
                Stile.frag_at(iq, ik + 1),
                Qtile.frag_at(0, 0),
                metal::false_type{},
                Ktile.frag_at(0, 0),
                Ktile.frag_at(1, 0),
                metal::true_type{});
          }
        }
      }
    }

    // Scale (in log2 space).
    MPP_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale2;
    }

    // Mask out the partial K column when kL is not a multiple of 32 (BK).
    if (!align_K && is_last_k) {
      MPP_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        MPP_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const short col_pos = ik * kU + sn;
          thread auto& fg = Stile.frag_at(iq, ik);
          MPP_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            MPP_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto loc = ii * stile_t::kFragThrCols + jj;
              const short c_off = QFrag_t::col_offset(jj);
              if ((col_pos + c_off) >= kL_rem) {
                fg[loc] = -INFINITY;
              }
            }
          }
        }
      }
    }

    // Causal mask (upper-left).
    if constexpr (DO_CAUSAL) {
      if (kb >= kb_min_causal) {
        const int base_row = block_idx * BQ + tm;
        const int base_col = kb * BK;
        MPP_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          MPP_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            thread auto& fg = Stile.frag_at(iq, ik);
            MPP_PRAGMA_UNROLL
            for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
              MPP_PRAGMA_UNROLL
              for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
                const int r =
                    base_row + iq * kU + ii * stile_t::kFragRowsJump + sm;
                const int c = base_col + ik * kU + QFrag_t::col_offset(jj) + sn;
                const auto loc = ii * stile_t::kFragThrCols + jj;
                if (r < c) {
                  fg[loc] = -INFINITY;
                }
              }
            }
          }
        }
      }
    }

    // Optional additive / boolean attention mask.
    if constexpr (HAS_MASK) {
      constexpr bool is_bool = metal::is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, AccumType>;
      using mtile_t = MPPTile<melem_t, TQ, TK, QFrag_t>;
      using mfrag_t = typename mtile_t::frag_type;

      const int base_row = block_idx * BQ + tm;
      const int base_col = kb * BK;
      const bool inside = (base_row + BQ <= qL) && (base_col + BK <= kL);

      MPP_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        MPP_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const int row_pos = base_row + iq * kU;
          const int col_pos = base_col + ik * kU;
          mfrag_t mfrag;
          if (inside) {
            mtile_t::MPPFrag_t::load(
                mfrag,
                mask,
                mask_params->M_strides[2],
                Int<1>{},
                row_pos,
                col_pos);
          } else {
            mtile_t::MPPFrag_t::load_safe(
                mfrag,
                mask,
                mask_params->M_strides[2],
                Int<1>{},
                qL,
                kL,
                row_pos,
                col_pos);
          }
          thread auto& fg = Stile.frag_at(iq, ik);
          MPP_PRAGMA_UNROLL
          for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
            if constexpr (is_bool) {
              fg[jj] = mfrag[jj] ? fg[jj] : (AccumType)-INFINITY;
            } else {
              fg[jj] += mask_log2e(static_cast<AccumType>(mfrag[jj]));
            }
          }
        }
      }
    }

    // Online softmax update.
    metal::vec<AccumType, kRowsPT> new_max;
    metal::vec<AccumType, kRowsPT> factor;
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }
    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);

    // If new_max is -inf, the row is fully masked so far; keep state.
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = (new_max[i] == -INFINITY)
          ? (AccumType)1
          : fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }

    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i];
    }
    Stile.template row_reduce<SumOp>(sum_score);

    Otile.template row_bin_op<MulOp>(factor);

    simdgroup_barrier(mem_flags::mem_none);

    // O += P @ V.
    MPP_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      MPP_PRAGMA_UNROLL
      for (short id = 0; id < TD; id += 2) {
        if constexpr (BD == 128) {
          if (id == 4) {
            threadgroup_barrier(mem_flags::mem_none);
          }
        }
        MPP_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          MPPTile<T, 1, 2, QFrag_t> Vtile;
          const int V_load_off = ik * kU * v_seq_stride + id * kU;
          if (k_rows == BK && Dv == BD) {
            Vtile.load(V + V_load_off, v_seq_stride);
          } else {
            Vtile.load_safe(
                V + V_load_off, v_seq_stride, k_rows - ik * kU, Dv - id * kU);
          }
          QFrag_t::mma(
              Otile.frag_at(iq, id),
              Otile.frag_at(iq, id + 1),
              Stile.frag_at(iq, ik),
              metal::false_type{},
              Vtile.frag_at(0, 0),
              Vtile.frag_at(0, 1),
              metal::false_type{});
        }
      }
    }

    K += BK * k_seq_stride;
    V += BK * v_seq_stride;
  }

  // Drains the cooperative-tensor pipeline before the O registers are read
  // back below.
  threadgroup_barrier(mem_flags::mem_none);

  // Treat fully-masked rows as zero output.
  MPP_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    if (max_score[i] == -INFINITY) {
      sum_score[i] = (AccumType)1;
    }
  }
  metal::vec<AccumType, kRowsPT> rcp;
  MPP_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = 1.0f / sum_score[i];
  }
  Otile.template row_bin_op<MulOp>(rcp);

  O += tm * o_seq_stride;
  if (q_rows <= 0) {
    return;
  }
  if (q_rows >= otile_t::kRows && Dv >= otile_t::kCols) {
    Otile.store(O, o_seq_stride);
  } else {
    Otile.store_safe(O, o_seq_stride, q_rows, Dv);
  }
}

#define instantiate_mpp_attn_arch(arch, is_a10, dtype, bd, hm, dc, ck, mtype) \
  instantiate_kernel(                                                         \
      "prefill_attention_mpp_" arch "_" #dtype "_bd" #bd "_hm" #hm "_dc" #dc  \
      "_ck" #ck "_mask" #mtype,                                               \
      prefill_attention_mpp,                                                  \
      dtype,                                                                  \
      bd,                                                                     \
      hm,                                                                     \
      dc,                                                                     \
      is_a10,                                                                 \
      ck,                                                                     \
      mtype)

#define instantiate_mpp_attn(dtype, bd, hm, dc, ck, mtype)              \
  instantiate_mpp_attn_arch("a10", true, dtype, bd, hm, dc, ck, mtype); \
  instantiate_mpp_attn_arch("pre_a10", false, dtype, bd, hm, dc, ck, mtype)

// Keep the tiles in sync with kMppTiles / kMppWideTile in Attention.h: the 256
// tile only serves pre-Apple10, and only Apple10 chunks (on the 128 tile).
#define instantiate_mpp_attn_tiles(dtype, hm, dc, mtype)                     \
  instantiate_mpp_attn_arch("pre_a10", false, dtype, 256, hm, dc, 0, mtype); \
  instantiate_mpp_attn_arch("a10", true, dtype, 128, hm, dc, 1, mtype);      \
  instantiate_mpp_attn(dtype, 128, hm, dc, 0, mtype);                        \
  instantiate_mpp_attn(dtype, 96, hm, dc, 0, mtype);                         \
  instantiate_mpp_attn(dtype, 64, hm, dc, 0, mtype)

// Without a mask the mask name is the query dtype. A mask together with
// is_causal is rejected on the host, so there are no hm1_dc1 variants.
#define instantiate_mpp_attn_masks(dtype)         \
  instantiate_mpp_attn_tiles(dtype, 0, 0, dtype); \
  instantiate_mpp_attn_tiles(dtype, 0, 1, dtype); \
  instantiate_mpp_attn_tiles(dtype, 1, 0, dtype); \
  instantiate_mpp_attn_tiles(dtype, 1, 0, bool);  \
  instantiate_mpp_attn_tiles(dtype, 1, 0, float)

instantiate_mpp_attn_masks(half);
instantiate_mpp_attn_masks(bfloat);

#endif // C10_METAL_HAS_MPP
