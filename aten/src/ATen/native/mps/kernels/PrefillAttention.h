// Flash-attention prefill kernel for MPS.
// Adapted from kernels-community/metal-flash-sdpa, which is itself
// adapted from MLX.
// Uses simdgroup-matrix MMA for high throughput; supports causal mask,
// optional additive/bool mask, GQA/MQA and any head dim: BD is the padded
// tile width, the QK dim is accumulated in BD-wide chunks and V/O carry a
// real width Dv <= BD.
// Operates on PyTorch's dense [B, H, L, D] layout via per-tensor strides.
//
// Included from Attention.metal, which supplies `using namespace metal`. Builds
// as part of Attention.metallib (one file, one bundled section), so
// Attention.mm only needs a single library handle.
#pragma once

#include <ATen/native/mps/kernels/Attention.h>

#define PREFILL_CONST static constant constexpr const
#define PREFILL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

template <typename T>
struct pointer_element {};
template <typename T>
struct pointer_element<thread T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<device T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<constant T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<threadgroup T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
using pointer_element_t = typename pointer_element<remove_cv_t<T>>::type;

template <int val>
using Int = integral_constant<int, val>;

template <
    typename T,
    short BROWS,
    short BCOLS,
    short kDstStrRow,
    short kDstStrCol,
    short tgp_size,
    short n_reads = (BCOLS * BROWS) / (tgp_size),
    short TCOLS = BCOLS / n_reads,
    short TROWS = tgp_size / TCOLS>
struct BlockLoaderT {
  PREFILL_CONST short n_rows = (BROWS + TROWS - 1) / TROWS;
  PREFILL_CONST short vec_size = n_reads;

  const int src_ld;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device T* src;

  METAL_FUNC BlockLoaderT(
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * kDstStrRow + bj * kDstStrCol),
        src(nullptr) {}

  METAL_FUNC void seek(const device T* src_) {
    src = src_ + bi * src_ld + bj;
  }

  // Loads the tile, zero-filling past src_tile_dim (cols, rows) when partial.
  METAL_FUNC void load(short2 src_tile_dim) const {
    if (src_tile_dim.x < BCOLS || src_tile_dim.y < BROWS) {
      load_safe(src_tile_dim);
    } else {
      load_unsafe();
    }
  }

  METAL_FUNC void load_unsafe() const {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * kDstStrRow + j * kDstStrCol] = src[i * src_ld + j];
      }
    }
  }

  METAL_FUNC void load_safe(short2 src_tile_dim) const {
    src_tile_dim = src_tile_dim - short2(bj, bi);

    if (src_tile_dim.x <= 0 || src_tile_dim.y <= 0) {
      PREFILL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        PREFILL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * kDstStrRow + j * kDstStrCol] = T(0);
        }
      }
      return;
    }

    bool tmp_idx[vec_size];
    T tmp_val[vec_size];

    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_idx[j] = (i < src_tile_dim.y) && (j < src_tile_dim.x);
      }
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = src[(tmp_idx[j] ? i * src_ld + j : 0)];
      }
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        tmp_val[j] = tmp_idx[j] ? tmp_val[j] : T(0);
      }
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        dst[i * kDstStrRow + j * kDstStrCol] = tmp_val[j];
      }
    }
  }
};

struct MaxOp {
  template <typename T>
  METAL_FUNC static T apply(T x, T y) {
    // c10::metal::max propagates NaN (plain max drops it -> row max -inf -> 0).
    return c10::metal::max(x, y);
  }
};

struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct ExpSubOp {
  // If y (the row max) is -inf, every score in this row was masked out
  // (e.g. an explicit additive mask of all -inf, or a fully causally-masked
  // row). exp(-inf - -inf) = exp(NaN) = NaN, which then poisons the running
  // sum and output. Returning 0 is the mathematically correct limit
  // (exp(-inf) = 0) and is what flash-attention implementations do.
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return (y == -INFINITY) ? T(0) : fast::exp2(x - y);
  }
};

struct DivOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x / y;
  }
};

template <typename T, int kFragRows_, int kFragCols_>
struct BaseMMAFrag {
  static_assert(kFragRows_ == 8, "Only 8x8 fragments are supported");
  static_assert(kFragCols_ == 8, "Only 8x8 fragments are supported");
};

template <typename T>
struct BaseMMAFrag<T, 8, 8> {
  PREFILL_CONST int kFragRows = 8;
  PREFILL_CONST int kFragCols = 8;

  PREFILL_CONST int kElemsPerFrag = (kFragRows * kFragCols) / 32;

  PREFILL_CONST int kElemRows = 1;
  PREFILL_CONST int kElemCols = 2;

  static_assert(
      kElemRows * kElemCols == kElemsPerFrag,
      "MMAFrag shape is not consistent with MMAFrag size");

  typedef metal::simdgroup_matrix<T, kFragRows, kFragCols> mat_type;
  typedef metal::vec<T, kElemsPerFrag> frag_type;
  typedef metal::vec<T, kElemRows> row_frag_type;
  typedef metal::vec<T, kElemCols> col_frag_type;

  template <typename U>
  using dtype_mat_t = typename metal::simdgroup_matrix<U, kFragRows, kFragCols>;

  template <typename U>
  using dtype_frag_t = typename metal::vec<U, kElemsPerFrag>;

  METAL_FUNC static constexpr short2 get_coord(ushort simd_lane_id
                                               [[thread_index_in_simdgroup]]) {
    const short qid = simd_lane_id / 4;
    const short fm = (qid & 4) + ((simd_lane_id / 2) % 4);
    const short fn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
    return short2{fn, fm};
  }

  template <typename SrcPtrType, typename StrX, typename StrY>
  METAL_FUNC static constexpr void load(
      thread frag_type& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y) {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        dst[i * kElemCols + j] =
            static_cast<T>(src[i * str_x.value + j * str_y.value]);
      }
    }
  }

  template <
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX,
      typename OffY>
  METAL_FUNC static constexpr void load_safe(
      thread frag_type& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = Int<0>{},
      OffY off_y = Int<0>{}) {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((off_x + i) < lim_x && (off_y + j) < lim_y) {
          dst[i * kElemCols + j] = static_cast<T>(
              src[(off_x + i) * str_x + (off_y + j) * str_y.value]);
        } else {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  template <typename DstPtrType, typename StrX, typename StrY>
  METAL_FUNC static constexpr void store(
      const thread frag_type& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y) {
    using U = pointer_element_t<DstPtrType>;
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        dst[i * str_x + j * str_y.value] =
            static_cast<U>(src[i * kElemCols + j]);
      }
    }
  }

  template <
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX,
      typename OffY>
  METAL_FUNC static constexpr void store_safe(
      const thread frag_type& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = Int<0>{},
      OffY off_y = Int<0>{}) {
    using U = pointer_element_t<DstPtrType>;
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((off_x + i) < lim_x && (off_y + j) < lim_y) {
          dst[(off_x + i) * str_x + (off_y + j) * str_y.value] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <typename Atype, typename Btype, typename Ctype>
  METAL_FUNC static constexpr void mma(
      thread frag_type& D,
      thread dtype_frag_t<Atype>& A,
      thread dtype_frag_t<Btype>& B,
      thread dtype_frag_t<Ctype>& C) {
    mat_type D_mat;
    dtype_mat_t<Atype> A_mat;
    dtype_mat_t<Btype> B_mat;
    dtype_mat_t<Ctype> C_mat;

    reinterpret_cast<thread dtype_frag_t<Atype>&>(A_mat.thread_elements()) = A;
    reinterpret_cast<thread dtype_frag_t<Btype>&>(B_mat.thread_elements()) = B;
    reinterpret_cast<thread dtype_frag_t<Ctype>&>(C_mat.thread_elements()) = C;

    simdgroup_multiply_accumulate(D_mat, A_mat, B_mat, C_mat);

    D = reinterpret_cast<thread frag_type&>(D_mat.thread_elements());
  }

  template <typename Op>
  METAL_FUNC static constexpr void row_reduce(
      thread const frag_type& inp_vals,
      thread T* reduced_vals) {
    T thr_reduce = Op::apply(inp_vals.x, inp_vals.y);

    T qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
    qgr_reduce = Op::apply(thr_reduce, qgr_reduce);

    T sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
    sgr_reduce = Op::apply(qgr_reduce, sgr_reduce);

    reduced_vals[0] = Op::apply(reduced_vals[0], sgr_reduce);
  }

  template <typename Op>
  METAL_FUNC static constexpr void row_bin_op(
      thread frag_type& inp_vals,
      thread T* row_vals) {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        inp_vals[i * kElemCols + j] =
            Op::apply(inp_vals[i * kElemCols + j], row_vals[i]);
      }
    }
  }
};

template <
    typename T,
    int kTileRows_,
    int kTileCols_,
    class MMAFrag_ = BaseMMAFrag<T, 8, 8>>
struct MMATile {
  using MMAFrag_t = MMAFrag_;
  using elem_type = T;
  PREFILL_CONST int kFragRows = MMAFrag_t::kFragRows;
  PREFILL_CONST int kFragCols = MMAFrag_t::kFragCols;
  PREFILL_CONST int kElemsPerFrag = MMAFrag_t::kElemsPerFrag;

  PREFILL_CONST int kTileRows = kTileRows_;
  PREFILL_CONST int kTileCols = kTileCols_;

  PREFILL_CONST int kRows = kTileRows * kFragRows;
  PREFILL_CONST int kCols = kTileCols * kFragCols;

  PREFILL_CONST int kNumFrags = kTileRows * kTileCols;
  PREFILL_CONST int kElemsPerTile = kNumFrags * kElemsPerFrag;

  PREFILL_CONST int kRowsPerThread = kTileRows * MMAFrag_t::kElemRows;
  PREFILL_CONST int kColsPerThread = kTileCols * MMAFrag_t::kElemCols;

  typedef typename MMAFrag_t::mat_type mat_type;
  typedef typename MMAFrag_t::frag_type frag_type;

  frag_type val_frags[kNumFrags];

  METAL_FUNC MMATile() thread {}

  METAL_FUNC constexpr void clear() {
    PREFILL_PRAGMA_UNROLL
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

  template <typename Op>
  METAL_FUNC void row_reduce(thread T vals[kRowsPerThread]) const {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::template row_reduce<Op>(
            frag_at(i, j), &vals[i * MMAFrag_t::kElemRows]);
      }
    }
  }

  template <typename Op>
  METAL_FUNC void row_bin_op(thread T vals[kRowsPerThread]) {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::template row_bin_op<Op>(
            frag_at(i, j), &vals[i * MMAFrag_t::kElemRows]);
      }
    }
  }

  template <typename U, int w_x, int w_y, int str_x, int str_y>
  METAL_FUNC void load(const threadgroup U* src) {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::load(
            frag_at(i, j),
            &(
                src[(i * kFragRows) * w_x * str_x +
                    (j * kFragCols) * w_y * str_y]),
            Int<str_x>{},
            Int<str_y>{});
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void store(device U* dst, const int ld) const {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store(
            frag_at(i, j),
            &(dst[(i * kFragRows) * w_x * ld + (j * kFragCols) * w_y]),
            ld,
            Int<1>{});
      }
    }
  }

  template <typename U, int w_x, int w_y>
  METAL_FUNC void store_safe(
      device U* dst,
      const int ld,
      const short2 dst_tile_dims) const {
    PREFILL_PRAGMA_UNROLL
    for (int i = 0; i < kTileRows; ++i) {
      PREFILL_PRAGMA_UNROLL
      for (int j = 0; j < kTileCols; ++j) {
        MMAFrag_t::store_safe(
            frag_at(i, j),
            dst,
            ld,
            Int<1>{},
            dst_tile_dims.y,
            dst_tile_dims.x,
            (i * kFragRows) * w_x,
            (j * kFragCols) * w_y);
      }
    }
  }
};

template <
    typename Dtype,
    typename Atype,
    typename Btype,
    typename Ctype,
    int M,
    int N,
    int K,
    class MMAFragD,
    class MMAFragA,
    class MMAFragB,
    class MMAFragC>
METAL_FUNC void tile_matmad(
    thread MMATile<Dtype, M, N, MMAFragD>& D,
    thread MMATile<Atype, M, K, MMAFragA>& A,
    thread MMATile<Btype, K, N, MMAFragB>& B,
    thread MMATile<Ctype, M, N, MMAFragC>& C) {
  PREFILL_PRAGMA_UNROLL
  for (short m = 0; m < M; ++m) {
    PREFILL_PRAGMA_UNROLL
    for (short n = 0; n < N; ++n) {
      short m_serp = m;
      short n_serp = (m % 2) ? (N - 1 - n) : n;

      PREFILL_PRAGMA_UNROLL
      for (short k = 0; k < K; ++k) {
        MMAFragD::mma(
            D.frag_at(m_serp, n_serp),
            A.frag_at(m_serp, k),
            B.frag_at(k, n_serp),
            C.frag_at(m_serp, n_serp));
      }
    }
  }
}

PREFILL_CONST int kThreadgroupMemoryBytes = 32768;

template <
    typename T,
    int BD,
    bool HAS_MASK,
    bool DO_CAUSAL,
    bool CHUNKED, // QK dim wider than BD, accumulated in BD-wide chunks
    bool PADDED, // head dims narrower than BD: bounds-checked columns
    typename MaskType>
[[kernel,
  max_total_threads_per_threadgroup(
      prefill_simdgroups(BD) * c10::metal::simdgroup_size)]] void
prefill_attention(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant PrefillAttnParams* params [[buffer(4)]],
    const constant PrefillAttnMaskParams* mask_params [[buffer(5)]],
    const device MaskType* mask [[buffer(6)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  using AccumType = c10::metal::accum_t<T>;
  constexpr bool has_mask = HAS_MASK;
  constexpr bool do_causal = DO_CAUSAL;
  constexpr int kThreads = prefill_simdgroups(BD) * c10::metal::simdgroup_size;
  constexpr int BQ = prefill_bq(BD);
  constexpr int BK = prefill_bk(BD);

  const int batch_idx = tid.z;
  const int head_idx = tid.y;
  const int block_idx = tid.x;

  const int q_seq_len = params->qL;
  const int k_seq_len = params->kL;
  const int D = params->D;
  const int Dv = params->Dv;

  if (block_idx * BQ >= q_seq_len) {
    return;
  }

  const ulong kv_head_idx = head_idx / params->gqa_factor;

  // Stride-based pointer offsets for [B, H, L, D] layout.
  Q += batch_idx * params->Q_strides[0] + head_idx * params->Q_strides[1] +
      (block_idx * BQ) * params->Q_strides[2];
  K += batch_idx * params->K_strides[0] + kv_head_idx * params->K_strides[1];
  V += batch_idx * params->V_strides[0] + kv_head_idx * params->V_strides[1];
  O += batch_idx * params->O_strides[0] + head_idx * params->O_strides[1] +
      (block_idx * BQ) * params->O_strides[2];

  if IF_CONSTEXPR (has_mask) {
    mask += batch_idx * mask_params->M_strides[0] +
        head_idx * mask_params->M_strides[1];
  }

  // Threadgroup memory, padded against bank conflicts when that still fits
  // the memory limit (it does not for the fp32 192 tile).
  constexpr short pad0 = 16 / sizeof(T);
  constexpr int kv_padded =
      (BK + pad0) * BD > (BD + pad0) * BK ? (BK + pad0) * BD : (BD + pad0) * BK;
  constexpr short pad =
      sizeof(T) * (BQ * (BD + pad0) + kv_padded) <= kThreadgroupMemoryBytes
      ? pad0
      : 0;

  constexpr short LDQ_tgp = BD + pad;
  constexpr short LDK_tgp = BK + pad;
  constexpr short LDV_tgp = BD + pad;

  constexpr short tgp_mem_0 = LDK_tgp * BD;
  constexpr short tgp_mem_1 = BK * LDV_tgp;
  constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;

  threadgroup T Q_smem[BQ * LDQ_tgp];
  threadgroup T KV_smem[tgp_mem_s];

  threadgroup T* Qs = Q_smem;
  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = KV_smem;

  // Q / K / V loaders. K is loaded transposed.
  using QBlockLoader = BlockLoaderT<T, BQ, BD, LDQ_tgp, 1, kThreads>;
  using KBlockLoader = BlockLoaderT<T, BK, BD, 1, LDK_tgp, kThreads>;
  using VBlockLoader = BlockLoaderT<T, BK, BD, LDV_tgp, 1, kThreads>;

  // Stride between consecutive sequence rows for Q/K/V/O.
  // Narrowed back to 32-bit: these only ever multiply in-tile offsets, the
  // base pointers having already been advanced.
  const int q_seq_stride = int(params->Q_strides[2]);
  const int k_seq_stride = int(params->K_strides[2]);
  const int v_seq_stride = int(params->V_strides[2]);
  const int o_seq_stride = int(params->O_strides[2]);

  QBlockLoader loader_q(q_seq_stride, Qs, simd_group_id, simd_lane_id);
  KBlockLoader loader_k(k_seq_stride, Ks, simd_group_id, simd_lane_id);
  VBlockLoader loader_v(v_seq_stride, Vs, simd_group_id, simd_lane_id);

  // 1.44269504089 = 1 / ln(2) so we can use exp2 instead of exp. Scale the
  // scores in AccumType, not Q in T: rounding the scale and the scaled Q to
  // the input dtype compounds into a visible error at large scales.
  const AccumType scale2 = params->scale * 1.44269504089f;

  constexpr short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  constexpr int kNWarps = prefill_simdgroups(BD);
  static_assert(
      BQ >= (kNWarps * kFragSize) && BQ % (kNWarps * kFragSize) == 0,
      "Each simdgroup must host at least 1 simdgroup matrix along Q sequence.");

  constexpr int TQ = BQ / (kNWarps * kFragSize);
  constexpr int TK = BK / kFragSize;
  constexpr int TD = BD / kFragSize;

  static_assert(TQ == 1, "Check TQ");

  MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
  MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile;
  MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Otile;

  Otile.clear();

  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * TQ * simd_group_id;

  const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
  const short Ks_offset = sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + sn;

  constexpr short Qs_tile_stride = kFragSize;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Load Q block (with safe load if last block is partial); a chunked pass
  // re-stages Q with every chunk.
  const int q_block_end = min(block_idx * BQ + BQ, q_seq_len);
  const int q_block_size = q_block_end - block_idx * BQ;

  if IF_CONSTEXPR (!CHUNKED) {
    loader_q.seek(Q);
    loader_q.load(short2(PADDED ? D : BD, q_block_size));
  }

  constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;

  AccumType max_score[kRowsPT];
  AccumType sum_score[kRowsPT] = {0};

  PREFILL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = -INFINITY;
  }

  // Causal masking can shorten the K range. PyTorch convention is upper-left
  // alignment: query at row r attends to keys at cols 0..r, regardless of qL
  // vs kL. So the last query in this block (row block_idx*BQ + q_block_size-1)
  // attends to cols 0..(block_idx*BQ + q_block_size - 1).
  int kb_lim = c10::metal::ceil_div(k_seq_len, BK);
  if IF_CONSTEXPR (do_causal) {
    int max_col = block_idx * BQ + q_block_size; // exclusive upper bound
    kb_lim = min(kb_lim, c10::metal::ceil_div(max_col, BK));
  }

  for (int kb = 0; kb < kb_lim; kb++) {
    const int k_block_end = min(kb * BK + BK, k_seq_len);
    const int k_block_size = k_block_end - kb * BK;

    Stile.clear();

    // S += Q K^T over the QK columns [d0, d0 + BD) of this K block.
    for (int d0 = 0; d0 < (CHUNKED ? D : BD); d0 += BD) {
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if IF_CONSTEXPR (CHUNKED) {
        loader_q.seek(Q + d0);
        loader_q.load(short2(min(D - d0, BD), q_block_size));
      }
      loader_k.seek(K + d0);
      loader_k.load(short2(PADDED ? min(D - d0, BD) : BD, k_block_size));

      threadgroup_barrier(mem_flags::mem_threadgroup);

      PREFILL_PRAGMA_UNROLL
      for (short dd = 0; dd < TD; dd++) {
        simdgroup_barrier(mem_flags::mem_none);

        Qtile.template load<T, 1, 1, LDQ_tgp, 1>(
            &Qs[Qs_offset + dd * Qs_tile_stride]);
        Ktile.template load<T, 1, 1, LDK_tgp, 1>(
            &Ks[Ks_offset + dd * Ks_tile_stride]);

        simdgroup_barrier(mem_flags::mem_none);

        tile_matmad(Stile, Qtile, Ktile, Stile);
      }
    }
    K += BK * k_seq_stride;

    {
      using stile_t = decltype(Stile);
      PREFILL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        PREFILL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          PREFILL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemsPerFrag; jj++) {
            Stile.frag_at(i, j)[jj] *= scale2;
          }
        }
      }
    }

    // Mask out partial K block.
    if (k_block_size < BK) {
      using stile_t = decltype(Stile);
      constexpr auto neg_inf = -INFINITY;

      PREFILL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        PREFILL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          short col_pos = sn + (j * stile_t::kFragCols);
          PREFILL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if ((col_pos + jj) >= k_block_size) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Causal mask (PyTorch upper-left convention: row r sees cols 0..r).
    if IF_CONSTEXPR (do_causal) {
      using stile_t = decltype(Stile);
      constexpr auto neg_inf = -INFINITY;

      PREFILL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        int row_pos = block_idx * BQ + tm + sm + (i * stile_t::kFragRows);
        PREFILL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);
          PREFILL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if (row_pos < (col_pos + jj)) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Optional additive / boolean mask.
    if IF_CONSTEXPR (has_mask) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = -INFINITY;

      constexpr bool is_bool = is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, selem_t>;

      using MMAFrag_mask_t = BaseMMAFrag<melem_t, kFragSize, kFragSize>;
      using frag_t = typename MMAFrag_mask_t::frag_type;

      PREFILL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos_in_seq =
            block_idx * BQ + tm + sm + (i * stile_t::kFragRows);
        PREFILL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos_in_seq = kb * BK + sn + (j * stile_t::kFragCols);

          frag_t mfrag;

          MMAFrag_mask_t::load_safe(
              mfrag,
              mask,
              mask_params->M_strides[2],
              Int<1>{},
              q_seq_len,
              k_seq_len,
              row_pos_in_seq,
              col_pos_in_seq);

          PREFILL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemsPerFrag; jj++) {
            if IF_CONSTEXPR (is_bool) {
              Stile.frag_at(i, j)[jj] =
                  mfrag[jj] ? Stile.frag_at(i, j)[jj] : neg_inf;
            } else {
              Stile.frag_at(i, j)[jj] += mask_log2e(mfrag[jj]);
            }
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    loader_v.seek(V);
    loader_v.load(short2(PADDED ? Dv : BD, k_block_size));
    V += BK * v_seq_stride;

    // Online softmax update.
    AccumType new_max[kRowsPT];
    AccumType factor[kRowsPT];
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);

    // If new_max is -inf, both old/new max are -inf (nothing valid yet).
    // Use factor=1 so we keep the running state unchanged for that row.
    // exp(-inf - -inf) would otherwise produce NaN.
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = (new_max[i] == -INFINITY)
          ? AccumType(1)
          : fast::exp2(max_score[i] - new_max[i]);
    }

    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = new_max[i];
    }

    AccumType sum_score_tmp[kRowsPT] = {0};
    Stile.template row_reduce<SumOp>(sum_score_tmp);

    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }

    Otile.template row_bin_op<MulOp>(factor);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    PREFILL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      PREFILL_PRAGMA_UNROLL
      for (short id = 0; id < TD; id++) {
        PREFILL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          if IF_CONSTEXPR (BD == 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          const short kk = ik * kFragSize;
          const short dd = id * kFragSize;

          Vtile.template load<T, 1, 1, LDV_tgp, 1>(
              &Vs[Vs_offset + kk * LDV_tgp + dd]);

          if IF_CONSTEXPR (BD == 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }

          MMAFrag_acc_t::mma(
              Otile.frag_at(iq, id),
              Stile.frag_at(iq, ik),
              Vtile.frag_at(0, 0),
              Otile.frag_at(iq, id));
        }
      }
    }
  }
  PREFILL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    if (max_score[i] == -INFINITY) {
      sum_score[i] = AccumType(1);
    }
  }

  // Normalize and store output.
  Otile.template row_bin_op<DivOp>(sum_score);
  threadgroup_barrier(mem_flags::mem_none);

  device T* O_tile = O + (tm + sm) * o_seq_stride + sn;

  const int o_cols = PADDED ? Dv : BD;
  if (q_block_size < BQ || o_cols < BD) {
    if ((tm + sm) < q_block_size && sn < o_cols) {
      auto dst_tile_dims = short2(o_cols - sn, q_block_size - (tm + sm));
      Otile.template store_safe<T, 1, 1>(O_tile, o_seq_stride, dst_tile_dims);
    }
  } else {
    Otile.template store<T, 1, 1>(O_tile, o_seq_stride);
  }
}

#define instantiate_attn_one(dtype, bd, hm, dc, ck, pd, mtype)            \
  instantiate_kernel(                                                     \
      "prefill_attention_" #dtype "_bd" #bd "_hm" #hm "_dc" #dc "_ck" #ck \
      "_pd" #pd "_mask" #mtype,                                           \
      prefill_attention,                                                  \
      dtype,                                                              \
      bd,                                                                 \
      hm,                                                                 \
      dc,                                                                 \
      ck,                                                                 \
      pd,                                                                 \
      mtype)

// Exact head dims (the tile width) and padded ones; chunked dims are padded.
#define instantiate_attn(dtype, bd, hm, dc, ck, mtype)   \
  instantiate_attn_one(dtype, bd, hm, dc, ck, 0, mtype); \
  instantiate_attn_one(dtype, bd, hm, dc, ck, 1, mtype)

// Keep the tiles in sync with kPrefillTiles in Attention.h. Head dims above
// 256 (the widest tile) run on 128 (kChunkTile), so only it gets the chunked
// variant.
#define instantiate_attn_tiles(dtype, hm, dc, mtype)     \
  instantiate_attn(dtype, 256, hm, dc, 0, mtype);        \
  instantiate_attn(dtype, 192, hm, dc, 0, mtype);        \
  instantiate_attn(dtype, 160, hm, dc, 0, mtype);        \
  instantiate_attn_one(dtype, 128, hm, dc, 1, 1, mtype); \
  instantiate_attn(dtype, 128, hm, dc, 0, mtype);        \
  instantiate_attn(dtype, 96, hm, dc, 0, mtype);         \
  instantiate_attn(dtype, 80, hm, dc, 0, mtype);         \
  instantiate_attn(dtype, 72, hm, dc, 0, mtype);         \
  instantiate_attn(dtype, 64, hm, dc, 0, mtype);         \
  instantiate_attn(dtype, 32, hm, dc, 0, mtype)

// Without a mask the mask name is the query dtype. A mask together with
// is_causal is rejected on the host, so there are no hm1_dc1 variants.
#define instantiate_attn_masks(dtype)         \
  instantiate_attn_tiles(dtype, 0, 0, dtype); \
  instantiate_attn_tiles(dtype, 0, 1, dtype); \
  instantiate_attn_tiles(dtype, 1, 0, dtype); \
  instantiate_attn_tiles(dtype, 1, 0, bool)

instantiate_attn_masks(half);
instantiate_attn_masks(bfloat);
instantiate_attn_masks(float);
// The public API allows an fp32 additive mask with any query dtype; half and
// bfloat take it as is and add it in fp32, like the CPU math path.
instantiate_attn_tiles(half, 1, 0, float);
instantiate_attn_tiles(bfloat, 1, 0, float);
