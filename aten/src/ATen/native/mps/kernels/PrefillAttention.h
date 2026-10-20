// Flash-attention prefill kernel for MPS.
// Adapted from kernels-community/metal-flash-sdpa, which is itself
// adapted from MLX.
// Uses simdgroup-matrix MMA for high throughput; supports causal mask,
// optional additive/bool mask, GQA/MQA and any head dim: BD is the padded
// tile width, the QK dim is accumulated in BD-wide chunks and V/O carry a
// real width Dv <= BD.
// Operates on PyTorch's dense [B, H, L, D] layout via per-tensor strides.
//
// Included from Attention.metal — no top-level includes / `using` here so the
// includer controls them. Builds as part of Attention.metallib (one file →
// one bundled section), so Attention.mm only needs a single library handle.
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
      OffY off_y = Int<0>{},
      T fill = T(0)) {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((off_x + i) < lim_x && (off_y + j) < lim_y) {
          dst[i * kElemCols + j] = static_cast<T>(
              src[(off_x + i) * str_x + (off_y + j) * str_y.value]);
        } else {
          dst[i * kElemCols + j] = fill;
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

PREFILL_CONST int kFragSize = 8;
PREFILL_CONST int kPrefillThreads = kPrefillSimdgroups * 32;
PREFILL_CONST int kThreadgroupMemoryBytes = 32768; // per threadgroup

// State of one prefill threadgroup: the Q block of a (batch, head) at `tid`,
// its share of the K blocks under split-K, and the running softmax of this
// simdgroup's kFragSize rows. prefill_attention below composes its passes
// from these steps.
template <
    typename T,
    int BD,
    bool HAS_MASK,
    bool DO_CAUSAL,
    bool CHUNKED, // QK dim wider than BD, accumulated in BD-wide chunks
    bool SPLIT, // split-K launch: O holds fp32 partial rows [row][split][BD]
    bool PV, // P @ V pass: no Q tile, V/O slabs along the grid's z dim
    typename MaskType>
struct PrefillBlock {
  using AccumType = c10::metal::accum_t<T>;
  using OutT = SplitOutT<T, SPLIT>;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;
  using MaskElem =
      metal::conditional_t<is_same_v<MaskType, bool>, bool, AccumType>;
  using MaskFrag = BaseMMAFrag<MaskElem, kFragSize, kFragSize>;
  PREFILL_CONST int BQ = kPrefillBQ;
  PREFILL_CONST int BK = prefill_bk(BD);
  PREFILL_CONST int TK = BK / kFragSize;
  PREFILL_CONST int TD = BD / kFragSize;
  using STile = MMATile<AccumType, 1, TK, MMAFrag_acc_t>;
  using OTile = MMATile<AccumType, 1, TD, MMAFrag_acc_t>;
  PREFILL_CONST short kRowsPT = STile::kRowsPerThread;

  // Threadgroup tiles, padded against bank conflicts when that still fits
  // the memory limit (it does not for the fp32 192 tile). K is staged
  // transposed; V reuses its tile, the two are never resident together.
  PREFILL_CONST short pad0 = 16 / sizeof(T);
  PREFILL_CONST int kv_padded = (BK + pad0) * BD > (BD + pad0) * BK
      ? (BK + pad0) * BD
      : (BD + pad0) * BK;
  PREFILL_CONST short pad =
      sizeof(T) * ((PV ? 0 : BQ * (BD + pad0)) + kv_padded) <=
          kThreadgroupMemoryBytes
      ? pad0
      : 0;
  PREFILL_CONST short LDQ = BD + pad;
  PREFILL_CONST short LDK = BK + pad;
  PREFILL_CONST short LDV = BD + pad;
  PREFILL_CONST int kQSmem = PV ? 1 : BQ * LDQ;
  PREFILL_CONST int kKVSmem = LDK * BD > LDV* BK ? LDK* BD : LDV* BK;
  using QLoader = BlockLoaderT<T, BQ, BD, LDQ, 1, kPrefillThreads>;
  using KLoader = BlockLoaderT<T, BK, BD, 1, LDK, kPrefillThreads>;
  using VLoader = BlockLoaderT<T, BK, BD, LDV, 1, kPrefillThreads>;

  const device T* Q;
  const device T* K;
  const device T* V;
  device OutT* O;
  const device MaskType* mask;
  const constant PrefillAttnMaskParams* mask_params;
  threadgroup T* Qs;
  threadgroup T* KVs;
  QLoader loader_q;
  KLoader loader_k;
  VLoader loader_v;
  STile Stile;
  OTile Otile;
  AccumType max_score[kRowsPT];
  AccumType sum_score[kRowsPT];
  AccumType scale2; // scale * log2(e): the softmax runs on exp2
  int block_idx, n_splits, q_seq_len, k_seq_len, D, Dv;
  int q_block_size; // valid rows of this Q block
  int kb0, kb1; // K blocks of this split
  int k_seq_stride, v_seq_stride, o_seq_stride, o_cols;
  short tm, sm, sn; // this simdgroup's first row; this thread's fragment coords
  short qs_off, ks_off, vs_off; // this thread's fragment origin in the tiles
  int rows_left; // valid rows from this thread's fragment row
  int row0; // this simdgroup's first row, flat over batch, head and query
  int part0; // (row0, split) in the split-K row stats

  METAL_FUNC PrefillBlock(
      const device T* Q_,
      const device T* K_,
      const device T* V_,
      device OutT* O_,
      const constant PrefillAttnParams* params,
      const constant PrefillAttnMaskParams* mask_params_,
      const device MaskType* mask_,
      threadgroup T* Qs_,
      threadgroup T* KVs_,
      uint simd_lane_id,
      uint simd_group_id,
      uint3 tid)
      : mask_params(mask_params_),
        Qs(Qs_),
        KVs(KVs_),
        loader_q(int(params->Q_strides[2]), Qs_, simd_group_id, simd_lane_id),
        loader_k(int(params->K_strides[2]), KVs_, simd_group_id, simd_lane_id),
        loader_v(int(params->V_strides[2]), KVs_, simd_group_id, simd_lane_id) {
    n_splits = params->n_splits;
    int slab = 0;
    int batch_split = tid.z;
    if IF_CONSTEXPR (PV) {
      const int batch_splits = params->B * n_splits;
      slab = tid.z / batch_splits;
      batch_split = tid.z - slab * batch_splits;
    }
    const int batch_idx = batch_split / n_splits;
    const int split = batch_split - batch_idx * n_splits;
    const int head_idx = tid.y;
    const int kv_head_idx = head_idx / params->gqa_factor;
    block_idx = tid.x;
    q_seq_len = params->qL;
    k_seq_len = params->kL;
    D = params->D;
    Dv = params->Dv;
    Q = Q_ + batch_idx * params->Q_strides[0] +
        head_idx * params->Q_strides[1] + block_idx * BQ * params->Q_strides[2];
    mask = mask_;
    if IF_CONSTEXPR (HAS_MASK) {
      mask += batch_idx * mask_params->M_strides[0] +
          head_idx * mask_params->M_strides[1];
    }
    // Narrowed to 32-bit: these only ever multiply in-tile offsets, the base
    // pointers having already been advanced.
    k_seq_stride = int(params->K_strides[2]);
    v_seq_stride = int(params->V_strides[2]);
    o_seq_stride = int(params->O_strides[2]);
    // Scale the scores in AccumType, not Q in T: rounding the scale and the
    // scaled Q to the input dtype compounds into a visible error at large
    // scales.
    scale2 = params->scale * M_LOG2E_F;
    const short2 coord = MMAFrag_acc_t::get_coord(simd_lane_id);
    sm = coord.y;
    sn = coord.x;
    tm = kFragSize * simd_group_id;
    qs_off = (tm + sm) * LDQ + sn;
    ks_off = sm * LDK + sn;
    vs_off = sm * LDV + sn;
    q_block_size = min(q_seq_len - block_idx * BQ, BQ);
    rows_left = q_block_size - (tm + sm);
    // Causal masking can shorten the K range. PyTorch convention is upper-left
    // alignment: query row r attends to keys 0..r regardless of qL vs kL, so
    // this block needs the keys below block_idx * BQ + q_block_size.
    int kb_lim = c10::metal::ceil_div(k_seq_len, BK);
    if IF_CONSTEXPR (DO_CAUSAL) {
      kb_lim =
          min(kb_lim, c10::metal::ceil_div(block_idx * BQ + q_block_size, BK));
    }
    kb0 = split * kb_lim / n_splits;
    kb1 = (split + 1) * kb_lim / n_splits;
    // K / V start at this split's first block and advance a block at a time.
    K = K_ + batch_idx * params->K_strides[0] +
        kv_head_idx * params->K_strides[1] + kb0 * BK * params->K_strides[2];
    V = V_ + batch_idx * params->V_strides[0] +
        kv_head_idx * params->V_strides[1] + kb0 * BK * params->V_strides[2] +
        slab * Dv;
    row0 = (batch_idx * params->H + head_idx) * q_seq_len + block_idx * BQ + tm;
    part0 = row0 * n_splits + split;
    o_cols = Dv;
    if IF_CONSTEXPR (PV) {
      if (n_splits > 1) {
        O = O_ + (long(part0 - tm * n_splits) * params->n_slabs + slab) * Dv;
        o_seq_stride = n_splits * params->n_slabs * Dv;
      } else {
        O = O_ + batch_idx * params->O_strides[0] +
            head_idx * params->O_strides[1] +
            block_idx * BQ * params->O_strides[2] + slab * Dv;
      }
    } else if (n_splits > 1) {
      O = O_ + long(part0 - tm * n_splits) * BD;
      o_seq_stride = n_splits * BD;
      o_cols = BD;
    } else {
      O = O_ + batch_idx * params->O_strides[0] +
          head_idx * params->O_strides[1] +
          block_idx * BQ * params->O_strides[2];
    }
    Otile.clear();
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = -INFINITY;
      sum_score[i] = 0;
    }
  }

  // Stages the Q block's columns [d0, d0 + BD).
  METAL_FUNC void stage_q(int d0) {
    loader_q.seek(Q + d0);
    loader_q.load(short2(min(D - d0, BD), q_block_size));
  }

  // S += Q K^T over the QK columns [d0, d0 + BD) of the current K block,
  // re-staging Q when CHUNKED.
  METAL_FUNC void qk_chunk(int d0, int k_rows) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if IF_CONSTEXPR (CHUNKED) {
      stage_q(d0);
    }
    loader_k.seek(K + d0);
    loader_k.load(short2(min(D - d0, BD), k_rows));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    MMATile<AccumType, 1, 1, MMAFrag_acc_t> Qtile;
    MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
    PREFILL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);
      Qtile.template load<T, 1, 1, LDQ, 1>(&Qs[qs_off + dd * kFragSize]);
      Ktile.template load<T, 1, 1, LDK, 1>(
          &KVs[ks_off + dd * (kFragSize * LDK)]);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(Stile, Qtile, Ktile, Stile);
    }
  }

  // Masks the columns of S from `bound` on.
  METAL_FUNC void mask_cols_from(int col0, int bound) {
    PREFILL_PRAGMA_UNROLL
    for (short j = 0; j < TK; j++) {
      PREFILL_PRAGMA_UNROLL
      for (short jj = 0; jj < MMAFrag_acc_t::kElemCols; jj++) {
        if (col0 + j * kFragSize + jj >= bound) {
          Stile.frag_at(0, j)[jj] = AccumType(-INFINITY);
        }
      }
    }
  }

  // S = scale * Q K^T for K block kb, masked: the columns past the K block,
  // above the causal diagonal (row r sees cols 0..r) and from the optional
  // additive / boolean mask.
  METAL_FUNC void scores(int kb) {
    const int k_end = min(kb * BK + BK, k_seq_len);
    Stile.clear();
    if IF_CONSTEXPR (CHUNKED) {
      for (int d0 = 0; d0 < D; d0 += BD) {
        qk_chunk(d0, k_end - kb * BK);
      }
    } else {
      qk_chunk(0, k_end - kb * BK);
    }
    K += BK * k_seq_stride;
    const int row = block_idx * BQ + tm + sm;
    const int col0 = kb * BK + sn;
    PREFILL_PRAGMA_UNROLL
    for (short j = 0; j < TK; j++) {
      PREFILL_PRAGMA_UNROLL
      for (short jj = 0; jj < MMAFrag_acc_t::kElemCols; jj++) {
        Stile.frag_at(0, j)[jj] *= scale2;
      }
    }
    if (k_end < kb * BK + BK) {
      mask_cols_from(col0, k_end);
    }
    if IF_CONSTEXPR (DO_CAUSAL) {
      mask_cols_from(col0, row + 1);
    }
    if IF_CONSTEXPR (HAS_MASK) {
      PREFILL_PRAGMA_UNROLL
      for (short j = 0; j < TK; j++) {
        typename MaskFrag::frag_type mfrag;
        MaskFrag::load_safe(
            mfrag,
            mask,
            mask_params->M_strides[2],
            Int<1>{},
            q_seq_len,
            k_seq_len,
            row,
            col0 + j * kFragSize);
        PREFILL_PRAGMA_UNROLL
        for (short jj = 0; jj < MMAFrag_acc_t::kElemCols; jj++) {
          if IF_CONSTEXPR (is_same_v<MaskType, bool>) {
            Stile.frag_at(0, j)[jj] =
                mfrag[jj] ? Stile.frag_at(0, j)[jj] : AccumType(-INFINITY);
          } else {
            Stile.frag_at(0, j)[jj] += AccumType(mfrag[jj]) * M_LOG2E_F;
          }
        }
      }
    }
  }

  // Stages V block kb over the K tile, once no simdgroup still reads it.
  METAL_FUNC void stage_v(int kb) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_v.seek(V);
    loader_v.load(short2(Dv, min(k_seq_len - kb * BK, BK)));
    V += BK * v_seq_stride;
  }

  // Online softmax step: folds the block's row max into the running max and
  // sum, rescales O and turns S into P = exp2(S - max).
  METAL_FUNC void softmax_update() {
    AccumType new_max[kRowsPT];
    AccumType factor[kRowsPT];
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }
    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);
    // A row with no valid key so far has both maxes -inf; factor 1 keeps its
    // state instead of exp2(-inf - -inf) = NaN.
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = new_max[i] == -INFINITY
          ? AccumType(1)
          : fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
      sum_score[i] *= factor[i];
    }
    Stile.template row_reduce<SumOp>(sum_score);
    Otile.template row_bin_op<MulOp>(factor);
  }

  // P = exp2(S - max) against the final row max (P @ V pass), summed into
  // the row sum.
  METAL_FUNC void exp_scores() {
    Stile.template row_bin_op<ExpSubOp>(max_score);
    Stile.template row_reduce<SumOp>(sum_score);
  }

  // O += P @ V.
  METAL_FUNC void pv() {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile;
    PREFILL_PRAGMA_UNROLL
    for (short id = 0; id < TD; id++) {
      PREFILL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        if IF_CONSTEXPR (BD == 128) {
          simdgroup_barrier(mem_flags::mem_none);
        }
        Vtile.template load<T, 1, 1, LDV, 1>(
            &KVs[vs_off + ik * (kFragSize * LDV) + id * kFragSize]);
        if IF_CONSTEXPR (BD == 128) {
          simdgroup_barrier(mem_flags::mem_none);
        }
        MMAFrag_acc_t::mma(
            Otile.frag_at(0, id),
            Stile.frag_at(0, ik),
            Vtile.frag_at(0, 0),
            Otile.frag_at(0, id));
      }
    }
  }

  // Row stats, laid out [row][split]: the sum and the max in log2 units.
  METAL_FUNC void store_stats(device float* sums, device float* maxs) const {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      const int row = sm + i * kFragSize;
      if (sn == 0 && tm + row < q_block_size) {
        sums[part0 + row * n_splits] = sum_score[i];
        maxs[part0 + row * n_splits] = max_score[i];
      }
    }
  }

  // The final row max over the scores pass's splits (P @ V pass); the row
  // sum is accumulated from the P this pass uses.
  METAL_FUNC void load_max(const device float* score_maxs, int n_score_splits) {
    PREFILL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      const int row = sm + i * kFragSize;
      if (tm + row < q_block_size) {
        const device float* m = score_maxs + long(row0 + row) * n_score_splits;
        for (int s = 0; s < n_score_splits; ++s) {
          max_score[i] = c10::metal::max(max_score[i], m[s]);
        }
      }
    }
  }

  // The scaled, masked scores S[row][kL] (rows flat over batch, head, query)
  // that prefill_attention_scores hands prefill_attention_pv, at this
  // thread's fragment origin in K block kb.
  METAL_FUNC device float* score_tile(device float* scores, int kb) const {
    return scores + long(row0 + sm) * k_seq_len + kb * BK + sn;
  }

  METAL_FUNC void store_scores(device float* scores, int kb) const {
    store_tile(
        Stile, score_tile(scores, kb), k_seq_len, min(k_seq_len - kb * BK, BK));
  }

  METAL_FUNC void load_scores(device float* scores, int kb) {
    const int cols_left = min(k_seq_len - kb * BK, BK) - sn;
    PREFILL_PRAGMA_UNROLL
    for (short j = 0; j < TK; j++) {
      MMAFrag_acc_t::load_safe(
          Stile.frag_at(0, j),
          score_tile(scores, kb),
          k_seq_len,
          Int<1>{},
          rows_left,
          cols_left,
          0,
          j * kFragSize,
          AccumType(-INFINITY));
    }
  }

  // Stores a tile at dst (this thread's fragment origin), clipped to the
  // block's valid rows and the first `cols` columns.
  template <typename U, typename Tile>
  METAL_FUNC void store_tile(
      const thread Tile& tile,
      device U* dst,
      int ld,
      int cols) const {
    if (q_block_size == BQ && cols == Tile::kCols) {
      tile.template store<U, 1, 1>(dst, ld);
    } else if (rows_left > 0 && sn < cols) {
      tile.template store_safe<U, 1, 1>(dst, ld, short2(cols - sn, rows_left));
    }
  }

  // Writes the normalized O block, or under split-K this split's unnormalized
  // partial rows plus the row stats for sdpa_merge_partials (every P @ V
  // slab computes the same stats and writes the same values).
  METAL_FUNC void store(device float* sums, device float* maxs) {
    if (n_splits > 1) {
      store_stats(sums, maxs);
    } else {
      // A fully masked row has sum 0 and all-zero O; keep it 0 instead of NaN.
      PREFILL_PRAGMA_UNROLL
      for (short i = 0; i < kRowsPT; ++i) {
        if (max_score[i] == -INFINITY) {
          sum_score[i] = AccumType(1);
        }
      }
      Otile.template row_bin_op<DivOp>(sum_score);
    }
    threadgroup_barrier(mem_flags::mem_none);
    store_tile(Otile, O + (tm + sm) * o_seq_stride + sn, o_seq_stride, o_cols);
  }
};

// Prefill passes: the fused kernel runs the online softmax and P @ V in one
// pass over the keys. For a V/O wider than the tile, the scores pass writes
// the scaled, masked scores and the row stats once (with no partials to merge
// it may split the keys further), so that each V/O slab only runs the P @ V
// pass instead of recomputing the scores.
enum PrefillStage { kFusedStage, kScoresStage, kPVStage };

// One threadgroup per (Q block, head, batch, split), and per V/O slab in the
// P @ V pass.
template <
    typename T,
    int BD,
    bool HAS_MASK,
    bool DO_CAUSAL,
    bool CHUNKED,
    bool SPLIT,
    PrefillStage STAGE,
    typename MaskType>
[[kernel, max_total_threads_per_threadgroup(kPrefillThreads)]] void
prefill_attention(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device SplitOutT<T, SPLIT>* O [[buffer(3)]],
    const constant PrefillAttnParams* params [[buffer(4)]],
    const constant PrefillAttnMaskParams* mask_params [[buffer(5)]],
    const device MaskType* mask [[buffer(6)]],
    device float* sums [[buffer(7)]],
    device float* maxs [[buffer(8)]],
    device float* scores [[buffer(9)]],
    const device float* score_maxs [[buffer(10)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  using Block = PrefillBlock<
      T,
      BD,
      HAS_MASK,
      DO_CAUSAL,
      CHUNKED,
      SPLIT,
      STAGE == kPVStage,
      MaskType>;
  threadgroup T Qs[Block::kQSmem];
  threadgroup T KVs[Block::kKVSmem];
  Block block(
      Q,
      K,
      V,
      O,
      params,
      mask_params,
      mask,
      Qs,
      KVs,
      simd_lane_id,
      simd_group_id,
      tid);
  if (block.q_block_size <= 0) {
    return;
  }
  if IF_CONSTEXPR (STAGE != kPVStage && !CHUNKED) {
    block.stage_q(0); // a chunked pass re-stages Q with every chunk
  }
  if IF_CONSTEXPR (STAGE == kFusedStage) {
    for (int kb = block.kb0; kb < block.kb1; kb++) {
      block.scores(kb);
      block.stage_v(kb);
      block.softmax_update();
      block.pv();
    }
    block.store(sums, maxs);
  } else if IF_CONSTEXPR (STAGE == kScoresStage) {
    for (int kb = block.kb0; kb < block.kb1; kb++) {
      block.scores(kb);
      block.store_scores(scores, kb);
      block.softmax_update();
    }
    block.store_stats(sums, maxs);
  } else {
    block.load_max(score_maxs, params->n_score_splits);
    for (int kb = block.kb0; kb < block.kb1; kb++) {
      block.load_scores(scores, kb);
      block.stage_v(kb);
      block.exp_scores();
      block.pv();
    }
    block.store(sums, maxs);
  }
}

#define instantiate_attn(suffix, stage, dtype, bd, hm, dc, ck, sp, mtype) \
  instantiate_kernel(                                                     \
      "prefill_attention" suffix "_" #dtype "_bd" #bd "_hm" #hm "_dc" #dc \
      "_ck" #ck "_split" #sp "_mask" #mtype,                              \
      prefill_attention,                                                  \
      dtype,                                                              \
      bd,                                                                 \
      hm,                                                                 \
      dc,                                                                 \
      ck,                                                                 \
      sp,                                                                 \
      stage,                                                              \
      mtype)

// The fused kernel also comes in a split-K variant writing fp32 partials.
#define instantiate_attn_fused(dtype, bd, hm, dc, ck, mtype)          \
  instantiate_attn("", kFusedStage, dtype, bd, hm, dc, ck, 0, mtype); \
  instantiate_attn("", kFusedStage, dtype, bd, hm, dc, ck, 1, mtype)

// Head dims above 192 are chunked / sliced on the 128 tile, so only it gets
// the chunked and the staged (wide V/O) variants. Keep the tiles in sync with
// kPrefillTiles in Attention.mm.
#define instantiate_attn_tiles(dtype, hm, dc, mtype)                          \
  instantiate_attn_fused(dtype, 192, hm, dc, 0, mtype);                       \
  instantiate_attn_fused(dtype, 160, hm, dc, 0, mtype);                       \
  instantiate_attn_fused(dtype, 128, hm, dc, 1, mtype);                       \
  instantiate_attn_fused(dtype, 128, hm, dc, 0, mtype);                       \
  instantiate_attn("_scores", kScoresStage, dtype, 128, hm, dc, 1, 0, mtype); \
  instantiate_attn("_scores", kScoresStage, dtype, 128, hm, dc, 0, 0, mtype); \
  instantiate_attn_fused(dtype, 96, hm, dc, 0, mtype);                        \
  instantiate_attn_fused(dtype, 80, hm, dc, 0, mtype);                        \
  instantiate_attn_fused(dtype, 72, hm, dc, 0, mtype);                        \
  instantiate_attn_fused(dtype, 64, hm, dc, 0, mtype);                        \
  instantiate_attn_fused(dtype, 32, hm, dc, 0, mtype)

// P @ V reads scores the scores pass already masked, so it has no mask
// variants; without a mask the mask name is the query dtype. Under split-K
// it writes fp32 partials like the fused kernel.
#define instantiate_attn_masks(dtype)                               \
  instantiate_attn_tiles(dtype, 0, 0, dtype);                       \
  instantiate_attn_tiles(dtype, 0, 1, dtype);                       \
  instantiate_attn_tiles(dtype, 1, 0, dtype);                       \
  instantiate_attn_tiles(dtype, 1, 1, dtype);                       \
  instantiate_attn_tiles(dtype, 1, 0, bool);                        \
  instantiate_attn_tiles(dtype, 1, 1, bool);                        \
  instantiate_attn("_pv", kPVStage, dtype, 128, 0, 0, 0, 0, dtype); \
  instantiate_attn("_pv", kPVStage, dtype, 128, 0, 1, 0, 0, dtype); \
  instantiate_attn("_pv", kPVStage, dtype, 128, 0, 0, 0, 1, dtype); \
  instantiate_attn("_pv", kPVStage, dtype, 128, 0, 1, 0, 1, dtype)

instantiate_attn_masks(half);
instantiate_attn_masks(bfloat);
instantiate_attn_masks(float);
