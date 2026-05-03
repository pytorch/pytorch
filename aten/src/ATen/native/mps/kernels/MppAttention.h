// MPP (MetalPerformancePrimitives, cooperative-tensor) flash-attention
// prefill for MPS. Built on mpp::tensor_ops::matmul2d
//
// Adapted from MLX
//
// The kernel itself is only emitted when the Metal 4 SDK is available
// (matmul2d lives in <MetalPerformancePrimitives/...>). At runtime,
// dispatch additionally gates on macOS 26.0+ (see mpp_attention_available
// in Attention.mm).
#pragma once

#include <ATen/native/mps/kernels/PrefillAttention.h>

#if __METAL_VERSION__ >= 400

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#define MPP_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define MPP_CONST static constant constexpr const

// metal::integral_constant in MSL has no arithmetic operator overloads
// (unlike std::integral_constant or MLX's vendored version). Provide
// just enough so the load / store helpers below can multiply runtime
// indices by a compile-time stride like Int<1>{} without falling back
// to an explicit ::value access at every call site.
template <typename T, T Val, typename U>
METAL_FUNC constexpr auto operator*(U lhs, metal::integral_constant<T, Val>) {
  return lhs * Val;
}
template <typename T, T Val, typename U>
METAL_FUNC constexpr auto operator*(metal::integral_constant<T, Val>, U rhs) {
  return Val * rhs;
}

// 16x16 tile of a 32-lane simdgroup. Each thread owns 8 elements, laid out
// as 2 rows x 4 cols with row jump 8 (i.e. rows 0 and 8 of a 16x16 frag).
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
    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;
    return short2{fn, fm};
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
          dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + c + j]);
        }
      } else {
        MPP_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[r * str_x + (c + j) * str_y]);
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
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_rows(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      if (r < lx) {
        if constexpr (metal::is_same_v<StrY, Int<1>>) {
          MPP_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + c + j]);
          }
        } else {
          MPP_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[i * kElemCols + j] =
                static_cast<T>(src[r * str_x + (c + j) * str_y]);
          }
        }
      } else {
        MPP_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] = T(0);
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
        if (r < lx && (c + j) < ly) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[r * str_x + (c + j) * str_y]);
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
          dst[r * str_x + c + j] = static_cast<U>(src[i * kElemCols + j]);
        }
      } else {
        MPP_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[r * str_x + (c + j) * str_y] =
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
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_rows(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      OffX off_x = {},
      OffY off_y = {}) {
    using U = pointer_element_t<DstPtrType>;
    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      if (r < lx) {
        if constexpr (metal::is_same_v<StrY, Int<1>>) {
          MPP_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[r * str_x + c + j] = static_cast<U>(src[i * kElemCols + j]);
          }
        } else {
          MPP_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[r * str_x + (c + j) * str_y] =
                static_cast<U>(src[i * kElemCols + j]);
          }
        }
      }
    }
  }

  // C[16, 32] += A[16, 16] * B[16, 16]^T concatenated with B[16, 16]^T,
  // packed as two 16x16 destination fragments laid out along N.
  template <
      typename CType,
      typename AType,
      typename BType,
      bool transpose_a = false,
      bool transpose_b = false>
  METAL_FUNC static constexpr void mma(
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
  }

  // Same matmul shape, but two A fragments stacked along M (32x16) and a
  // single B fragment.
  template <
      typename CType,
      typename AType,
      typename BType,
      bool transpose_a = false,
      bool transpose_b = false>
  METAL_FUNC static constexpr void mma(
      thread dtype_frag_t<CType>& Cm0,
      thread dtype_frag_t<CType>& Cm1,
      const thread dtype_frag_t<AType>& Am0,
      const thread dtype_frag_t<AType>& Am1,
      metal::bool_constant<transpose_a>,
      const thread dtype_frag_t<BType>& B,
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
      ct_a[i] = Am0[i];
      ct_a[kElemsPerFrag + i] = Am1[i];
    }
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_b[i] = B[i];
    }
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_c[i] = Cm0[i];
      ct_c[kElemsPerFrag + i] = Cm1[i];
    }

    gemm_op.run(ct_a, ct_b, ct_c);

    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      Cm0[i] = ct_c[i];
      Cm1[i] = ct_c[kElemsPerFrag + i];
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

template <
    typename T,
    short kTileRows_,
    short kTileCols_,
    class MPPFrag_ = BaseMPPFrag>
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
  METAL_FUNC void load_rows(
      const device U* src,
      const int ld,
      const short n_rows) {
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      MPP_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MPPFrag_t::load_rows(
            frag_at(i, j),
            src,
            ld,
            Int<1>{},
            n_rows,
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
  METAL_FUNC void store_rows(device U* dst, const int ld, const short n_rows)
      const {
    MPP_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      MPP_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MPPFrag_t::store_rows(
            frag_at(i, j),
            dst,
            ld,
            Int<1>{},
            n_rows,
            i * kFragRows,
            j * kFragCols);
      }
    }
  }
};

// Flash-attention prefill kernel using cooperative-tensor matmul2d.
// Mirrors prefill_attention's contract (PrefillAttnParams + 4D
// PrefillAttnMaskParams, last-dim contiguous Q/K/V/mask, [B, H, L, D]
// layout) and PyTorch's upper-left causal alignment. Each threadgroup
// covers one (Q-block, head, batch) tile.
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    bool HAS_MASK,
    bool DO_CAUSAL,
    typename MaskType = float>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void
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

  const int batch_idx = tid.z;
  const int head_idx = tid.y;
  const int block_idx = tid.x;

  const int qL = params->qL;
  const int kL = params->kL;

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

  const ulong kv_head_idx = head_idx / params->gqa_factor;

  Q += batch_idx * params->Q_strides[0] + head_idx * params->Q_strides[1] +
      (block_idx * BQ) * params->Q_strides[2];
  K += batch_idx * params->K_strides[0] + kv_head_idx * params->K_strides[1];
  V += batch_idx * params->V_strides[0] + kv_head_idx * params->V_strides[1];
  O += batch_idx * params->O_strides[0] + head_idx * params->O_strides[1] +
      (block_idx * BQ) * params->O_strides[2];

  if constexpr (HAS_MASK) {
    mask += batch_idx * mask_params->M_strides[0] +
        head_idx * mask_params->M_strides[1];
  }

  // Pre-multiply by 1/ln(2) so we can use exp2 in the inner softmax.
  const AccumType scale2 =
      static_cast<AccumType>(params->scale) * 1.44269504089f;

  constexpr short kU = 16;
  constexpr int kNWarps = WM * WN;
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

  using otile_t = MPPTile<AccumType, TQ, TD>;
  otile_t Otile;
  Otile.clear();

  const short tm = kU * TQ * simd_group_id;
  Q += tm * params->Q_strides[2];

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
  const short lim_rows_q = qL_rem - tm;
  const short lim_rows_k = kL_rem;

  for (int kb = 0; kb < kb_lim; kb++) {
    const bool is_last_k = (kb == NK_aligned);

    using stile_t = MPPTile<AccumType, TQ, TK>;
    stile_t Stile;
    Stile.clear();

    // S = Q @ K^T. K is loaded row-major and consumed transposed by mma.
    MPP_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      MPP_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik += 2) {
        MPP_PRAGMA_UNROLL
        for (short id = 0; id < TD; id++) {
          MPPTile<T, 1, 1> Qtile;
          MPPTile<T, 2, 1> Ktile;

          const int Q_load_off = iq * kU * params->Q_strides[2] + id * kU;
          const int K_load_off = ik * kU * params->K_strides[2] + id * kU;

          if (!align_Q && is_last_q) {
            Qtile.load_rows(
                Q + Q_load_off, params->Q_strides[2], lim_rows_q - iq * kU);
          } else {
            Qtile.load(Q + Q_load_off, params->Q_strides[2]);
          }

          if (!align_K && is_last_k) {
            Ktile.load_rows(
                K + K_load_off, params->K_strides[2], lim_rows_k - ik * kU);
          } else {
            Ktile.load(K + K_load_off, params->K_strides[2]);
          }

          stile_t::MPPFrag_t::mma(
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

    // Scale (in log2 space).
    MPP_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale2;
    }

    // Mask out the partial K column when kL is not BK-aligned.
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
              if ((col_pos + jj) >= kL_rem) {
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
                const int c = base_col + ik * kU + jj + sn;
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
      using mtile_t = MPPTile<melem_t, TQ, TK>;
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
                int(mask_params->M_strides[2]),
                Int<1>{},
                row_pos,
                col_pos);
          } else {
            mtile_t::MPPFrag_t::load_safe(
                mfrag,
                mask,
                int(mask_params->M_strides[2]),
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
              fg[jj] += 1.44269504089f * static_cast<AccumType>(mfrag[jj]);
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
          MPPTile<T, 1, 2> Vtile;
          const int V_load_off = ik * kU * params->V_strides[2] + id * kU;
          if (!align_K && is_last_k) {
            Vtile.load_rows(
                V + V_load_off, params->V_strides[2], lim_rows_k - ik * kU);
          } else {
            Vtile.load(V + V_load_off, params->V_strides[2]);
          }
          otile_t::MPPFrag_t::mma(
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

    K += BK * params->K_strides[2];
    V += BK * params->V_strides[2];
  }

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

  O += tm * params->O_strides[2];
  if (!align_Q && is_last_q) {
    if (lim_rows_q <= 0) {
      return;
    }
    Otile.store_rows(O, params->O_strides[2], lim_rows_q);
  } else {
    Otile.store(O, params->O_strides[2]);
  }
}

#define instantiate_mpp_attn(                                                 \
    tname, dtype, bq, bk, bd, wm, wn, hm, dc, mname, mtype)                   \
  instantiate_kernel(                                                         \
      "prefill_attention_mpp_" #tname "_bq" #bq "_bk" #bk "_bd" #bd "_wm" #wm \
      "_wn" #wn "_hm" #hm "_dc" #dc "_mask" #mname,                           \
      prefill_attention_mpp,                                                  \
      dtype,                                                                  \
      bq,                                                                     \
      bk,                                                                     \
      bd,                                                                     \
      wm,                                                                     \
      wn,                                                                     \
      hm,                                                                     \
      dc,                                                                     \
      mtype)

#define instantiate_mpp_attn_shapes(iname, itype, hm, dc, mname, mtype)       \
  instantiate_mpp_attn(iname, itype, 64, 32, 256, 4, 1, hm, dc, mname, mtype) \
      instantiate_mpp_attn(                                                   \
          iname, itype, 64, 32, 128, 4, 1, hm, dc, mname, mtype)              \
          instantiate_mpp_attn(                                               \
              iname, itype, 64, 32, 96, 4, 1, hm, dc, mname, mtype)           \
              instantiate_mpp_attn(                                           \
                  iname, itype, 64, 32, 64, 4, 1, hm, dc, mname, mtype)       \
                  instantiate_mpp_attn(                                       \
                      iname, itype, 64, 64, 128, 4, 1, hm, dc, mname, mtype)  \
                      instantiate_mpp_attn(                                   \
                          iname,                                              \
                          itype,                                              \
                          64,                                                 \
                          64,                                                 \
                          64,                                                 \
                          4,                                                  \
                          1,                                                  \
                          hm,                                                 \
                          dc,                                                 \
                          mname,                                              \
                          mtype)

#define instantiate_mpp_attn_causal(iname, itype, mname, mtype)         \
  instantiate_mpp_attn_shapes(iname, itype, 0, 0, mname, mtype)         \
      instantiate_mpp_attn_shapes(iname, itype, 0, 1, mname, mtype)     \
          instantiate_mpp_attn_shapes(iname, itype, 1, 0, mname, mtype) \
              instantiate_mpp_attn_shapes(iname, itype, 1, 1, mname, mtype)

#define instantiate_mpp_attn_mask(iname, itype)           \
  instantiate_mpp_attn_causal(iname, itype, iname, itype) \
      instantiate_mpp_attn_causal(iname, itype, bool_, bool)

instantiate_mpp_attn_mask(float16, half)
    instantiate_mpp_attn_mask(bfloat16, bfloat)

#endif // __METAL_VERSION__ >= 400
