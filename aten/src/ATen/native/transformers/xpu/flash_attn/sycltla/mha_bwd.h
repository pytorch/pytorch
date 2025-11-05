#pragma once
#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <cutlass/numeric_conversion.h>
#include <sycl/sycl.hpp>

namespace cute {

template <
    class T_,
    int kHeadDim_,
    int kBlockM_,
    int kBlockN_,
    int kNSGs_,
    int AtomLayoutMSdP_ = 2,
    int AtomLayoutNdKV_ = 2,
    int AtomLayoutMdQ_ = 2,
    bool is_causal_ = false>
struct FAKernel {
  /*
    Q BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_QK
    K BATCH,NUM_HEAD_KV,SEQ_LEN_KV,HEAD_SIZE_QK
    V BATCH,NUM_HEAD_KV,SEQ_LEN_KV,HEAD_SIZE_VO
    P BATCH,NUM_HEAD_Q,SEQ_LEN_QO,SEQ_LEN_KV
    O BATCH,NUM_HEAD_Q,SEQ_LEN_QO,HEAD_SIZE_VO
  */
  using DType = T_;
  using VType = float; // accumulation
  using MMA_Atom_ARCH = std::conditional_t<
      std::is_same_v<DType, cutlass::half_t>,
      MMA_Atom<XE_8x16x16_F32F16F16F32_TT>,
      MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>>;
  static constexpr int kHeadDim = kHeadDim_;
  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kNSGs = kNSGs_;
  static constexpr int AtomLayoutMSdP = AtomLayoutMSdP_;
  static constexpr int AtomLayoutNdKV = AtomLayoutNdKV_;
  static constexpr int AtomLayoutMdQ = AtomLayoutMdQ_;
  static constexpr bool is_causal = is_causal_;
  using SubgroupLayoutSdP =
      Layout<Shape<Int<AtomLayoutMSdP>, Int<kNSGs / AtomLayoutMSdP>, _1>>;
  using SubgroupLayoutdKV =
      Layout<Shape<Int<AtomLayoutNdKV>, Int<kNSGs / AtomLayoutNdKV>, _1>>;
  using SubgroupLayoutdQ =
      Layout<Shape<Int<AtomLayoutMdQ>, Int<kNSGs / AtomLayoutMdQ>, _1>>;

  using TileShapeSdP =
      Tile<Int<16 * AtomLayoutMSdP>, Int<16 * kNSGs / AtomLayoutMSdP>, _16>;
  static_assert(
      size<0>(TileShapeSdP{}) <= kBlockM &&
      "tile size M must be smaller than or equal to kBlockM");
  static_assert(
      kBlockM % size<0>(TileShapeSdP{}) == 0 &&
      "kBlockM must be dividable by tile size M");
  static_assert(
      size<1>(TileShapeSdP{}) <= kBlockN &&
      "tile size N must be smaller than or equal to kBlockN");
  static_assert(
      kBlockN % size<1>(TileShapeSdP{}) == 0 &&
      "kBlockN must be dividable by tile size N ");

  using TileShapedKV =
      Tile<Int<16 * AtomLayoutNdKV>, Int<16 * kNSGs / AtomLayoutNdKV>, _16>;
  static_assert(
      size<0>(TileShapedKV{}) <= kBlockN &&
      "tile size M must be smaller than or equal to kBlockN");
  static_assert(
      kBlockN % size<0>(TileShapedKV{}) == 0 &&
      "kBlockN must be dividable by tile size M");
  static_assert(
      size<1>(TileShapedKV{}) <= kHeadDim &&
      "tile size N must be smaller than or equal to kHeadDim");
  static_assert(
      kHeadDim % size<1>(TileShapedKV{}) == 0 &&
      "kHeadDim must be dividable by tile size N");

  using TileShapedQ =
      Tile<Int<16 * AtomLayoutMdQ>, Int<16 * kNSGs / AtomLayoutMdQ>, _16>;
  static_assert(
      size<0>(TileShapedQ{}) <= kBlockM &&
      "tile size M must be smaller than or equal to kBlockM");
  static_assert(
      kBlockM % size<0>(TileShapedQ{}) == 0 &&
      "kBlockM must dividable by tile size M");
  static_assert(
      size<1>(TileShapedQ{}) <= kHeadDim &&
      "tile size N must be smaller than or equal to kHeadDim");
  static_assert(
      kHeadDim % size<1>(TileShapedQ{}) == 0 &&
      "kHeadDim must be dividable by tile size N");

  using TiledMmaSdP = typename TiledMMAHelper<
      MMA_Atom_ARCH,
      Layout<TileShapeSdP>,
      SubgroupLayoutSdP>::TiledMMA;

  using TiledMmadKV = typename TiledMMAHelper<
      MMA_Atom_ARCH,
      Layout<TileShapedKV>,
      SubgroupLayoutdKV>::TiledMMA;

  using TiledMmadQ = typename TiledMMAHelper<
      MMA_Atom_ARCH,
      Layout<TileShapedQ>,
      SubgroupLayoutdQ>::TiledMMA;
  static constexpr auto bP = Int<2>{}; // Pipeline

  using StrideR = cute::tuple<long, cute::C<1>>;
  using StrideC = cute::tuple<cute::C<1>, long>;

  // for load Q and Kt in S=QKt
  using TiledLoadQ = decltype(make_tiled_copy(
      Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_N, StrideR>, DType>{},
      Layout<Shape<_4, _16>>{}, // Thr layout 1x16 k-major
      Layout<Shape<_16, _4>>{})); // Val layout  16x1
  using TiledLoadKt = decltype(make_tiled_copy(
      Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_T, StrideR>, DType>{},
      Layout<Shape<_4, _16>>{}, // Thr layout 1x16 n-major
      Layout<Shape<_16, _4>>{})); // Val layout  16x1

  // for load dO and Vt in dP=dO*Vt
  using TiledLoaddO = decltype(make_tiled_copy(
      Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_N, StrideR>, DType>{},
      Layout<Shape<_4, _16>>{}, // Thr layout 1x16 k-major
      Layout<Shape<_16, _4>>{})); // Val layout  16x1

  using TiledLoadV = decltype(make_tiled_copy(
      Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_T, StrideR>, DType>{},
      Layout<Shape<_4, _16>>{}, // Thr layout 1x16 n-major
      Layout<Shape<_16, _4>>{})); // Val layout  16x1

  // for load Pt and dO in dV=Pt*dO
  using TiledLoadPt = decltype(make_tiled_copy(
      Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_T, StrideC>, DType>{},
      Layout<Shape<_4, _16>>{}, // Thr layout 1x16 m-major
      Layout<Shape<_16, _4>>{})); // // Val layout  8x1
  using TiledLoaddOt = decltype(make_tiled_copy(
      Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_V, StrideC>, DType>{}, // should
                                                                      // be V
                                                                      // here
      Layout<Shape<_4, _16>>{}, // Thr layout 1x16 n-major
      Layout<Shape<_16, _4>>{})); // val layout 16x1

  // for load dP, K and dQ in dQ=dP*K
  using TiledLoaddP = decltype(make_tiled_copy(
      Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_N, StrideR>, DType>{},
      Layout<Shape<_4, _16>>{}, // Thr layout 1x16 k-major
      Layout<Shape<_16, _4>>{})); // val layout 16x1
  using TiledLoadK = decltype(make_tiled_copy(
      Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_N, StrideC>, DType>{},
      Layout<Shape<_4, _16>>{}, // Thr layout 1x16 n-major
      Layout<Shape<_16, _4>>{})); // val layout 16x1

  using TiledLoaddQ = decltype(make_tiled_copy(
      Copy_Atom<Copy_Traits<XE_2D_U32x8x16_LD_N, StrideR>, VType>{},
      Layout<Shape<_4, _16>>{}, // Thr layout 1x16 n-major
      Layout<Shape<_8, _4>>{})); // val layout 8x1

  //  for load dPt, Q in dK=dPt*Q
  using TiledLoaddPt = decltype(make_tiled_copy(
      Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_T, StrideC>, DType>{},
      Layout<Shape<_4, _16>>{}, // Thr layout 1x16 k-major
      Layout<Shape<_16, _4>>{})); // Val layout  16x1
  using TiledLoadQt = decltype(make_tiled_copy(
      Copy_Atom<Copy_Traits<XE_2D_U16x16x16_LD_N, StrideC>, DType>{},
      Layout<Shape<_4, _16>>{}, // Thr layout 1x16 n-major
      Layout<Shape<_16, _4>>{})); // Val layout  16x1

  // for save S in S=QKt and P
  using TiledSaveS = decltype(make_tiled_copy(
      Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N, StrideR>, DType>{},
      Layout<Shape<_4, _16>>{}, // Thr layout 1x16 n-major
      Layout<Shape<_8, _4>>{})); // Val layout  8x1
  // for save dP in dP=dO*Vt
  using TiledSavedP = decltype(make_tiled_copy(
      Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N, StrideR>, DType>{},
      Layout<Shape<_4, _16>>{}, // Thr layout 1x16 n-major
      Layout<Shape<_8, _4>>{})); // Val layout  8x1
  // for save dV in dV=Pt*dO
  using TiledSavedV = decltype(make_tiled_copy(
      Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N, StrideR>, DType>{},
      Layout<Shape<_4, _16>>{}, // Thr layout 1x16 n-major
      Layout<Shape<_8, _4>>{})); // Val layout  8x1
  // for save dQ in dQ=dP*K
  using TiledSavedQ = decltype(make_tiled_copy(
      Copy_Atom<Copy_Traits<XE_2D_U32x8x16_ST_N, StrideR>, VType>{},
      Layout<Shape<_4, _16>>{}, // Thr layout 1x16 n-major
      Layout<Shape<_8, _4>>{})); // val layout 8x1
  // for save dK=dPt*Q
  using TiledSavedK = decltype(make_tiled_copy(
      Copy_Atom<Copy_Traits<XE_2D_U16x8x16_ST_N, StrideR>, DType>{},
      Layout<Shape<_4, _16>>{}, // Thr layout 1x16 n-major
      Layout<Shape<_8, _4>>{})); // Val layout  8x1

  static constexpr int SubgroupSize = 16;
  static constexpr int smem_size = 0;

  FAKernel() {}
};

using index_t = uint64_t;

template <typename T>
struct Param {
  Param(
      const T* dO,
      const T* o,
      const T* q,
      const T* k,
      const T* v,
      const float* lse,
      float* odo,
      float* dqaccum,
      T* dq,
      T* dk,
      T* dv,
      T* s,
      T* dp,
      T* pb,
      const float softmax_scale)
      : do_ptr(dO),
        o_ptr(o),
        q_ptr(q),
        k_ptr(k),
        v_ptr(v),
        lse_ptr(lse),
        odo_ptr(odo),
        dqaccum_ptr(dqaccum),
        dq_ptr(dq),
        dk_ptr(dk),
        dv_ptr(dv),
        s_ptr(s),
        dp_ptr(dp),
        pb_ptr(pb),
        scale_softmax(softmax_scale),
        scale_softmax_log2(softmax_scale * M_LOG2E),
        is_bhsd(true) {}
  // read only
  const T* do_ptr;
  const T* o_ptr;
  const T* q_ptr;
  const T* k_ptr;
  const T* v_ptr;
  const float* lse_ptr;
  const float scale_softmax;
  const float scale_softmax_log2;
  // write
  float* odo_ptr;
  float* dqaccum_ptr;
  T* dq_ptr;
  T* dk_ptr;
  T* dv_ptr;
  T* s_ptr;
  T* dp_ptr;
  T* pb_ptr;

  // const dimension
  int batch;
  int num_head_q;
  int num_head_kv;
  int seq_len_q;
  int seq_len_q_pad;
  int seq_len_kv;
  int seq_len_kv_pad;
  int head_dim;
  int n_block;
  int tail_n;
  int m_block;
  int tail_m;
  int num_qh_per_kvh;
  int q_r_stride;
  int q_h_stride;
  int q_b_stride;

  int k_r_stride;
  int k_h_stride;
  int k_b_stride;

  int dk_r_stride;
  int dk_h_stride;
  int dk_b_stride;

  int v_r_stride;
  int v_h_stride;
  int v_b_stride;

  int dv_r_stride;
  int dv_h_stride;
  int dv_b_stride;

  int o_r_stride;
  int o_h_stride;
  int o_b_stride;

  int s_r_stride;
  int s_s_stride;
  int s_b_stride;

  int dq_r_stride;
  int dq_h_stride;
  int dq_b_stride;
  /*
   * input output layout
   * true batch, numhead, seqlen, headsize
   * false batch, seqlen, numhead, headsize
   */
  bool is_bhsd;
};

template <typename T>
struct Boffset {
  Boffset(Param<T>& param_) : param(param_) {}
  index_t q_offset(const index_t b_id, const index_t h_id, const index_t s_id) {
    return b_id * param.q_b_stride + h_id * param.q_h_stride +
        s_id * param.q_r_stride;
  }
  index_t k_offset(const index_t b_id, const index_t h_id, const index_t s_id) {
    return b_id * param.k_b_stride + h_id * param.k_h_stride +
        s_id * param.k_r_stride;
  }
  index_t v_offset(const index_t b_id, const index_t h_id, const index_t s_id) {
    return b_id * param.v_b_stride + h_id * param.v_h_stride +
        s_id * param.v_r_stride;
  }
  index_t dk_offset(
      const index_t b_id,
      const index_t h_id,
      const index_t s_id) {
    return b_id * param.dk_b_stride + h_id * param.dk_h_stride +
        s_id * param.dk_r_stride;
  }
  index_t dv_offset(
      const index_t b_id,
      const index_t h_id,
      const index_t s_id) {
    return b_id * param.dv_b_stride + h_id * param.dv_h_stride +
        s_id * param.dv_r_stride;
  }
  index_t ps_offset(
      const index_t b_id,
      const index_t h_id,
      const index_t sq_id,
      const index_t sk_id) {
    return b_id * param.s_b_stride + h_id * param.s_s_stride +
        sq_id * param.s_r_stride + sk_id;
  }
  index_t lse_offset(
      const index_t b_id,
      const index_t h_id,
      const index_t s_id) {
    return b_id * param.seq_len_q * param.num_head_q + h_id * param.seq_len_q +
        s_id;
  }

  index_t o_offset(const index_t b_id, const index_t h_id, const index_t s_id) {
    return b_id * param.o_b_stride + h_id * param.o_h_stride +
        s_id * param.o_r_stride;
  }

  index_t dq_offset(
      const index_t b_id,
      const index_t h_id,
      const index_t s_id) {
    return b_id * param.dq_b_stride + h_id * param.dq_h_stride +
        s_id * param.dq_r_stride;
  }
  Param<T>& param;
};

template <typename T>
void setup_bhsd_stride(Param<T>& param) {
  param.q_r_stride = param.head_dim;
  param.q_h_stride = param.seq_len_q * param.head_dim;
  param.q_b_stride = param.num_head_q * param.seq_len_q * param.head_dim;

  // param.dq_r_stride = param.head_dim;
  // param.dq_h_stride = param.seq_len_q * param.head_dim;
  // param.dq_b_stride = param.num_head_q * param.seq_len_q * param.head_dim;

  param.k_r_stride = param.head_dim;
  param.k_h_stride = param.seq_len_kv * param.head_dim;
  param.k_b_stride = param.num_head_kv * param.seq_len_kv * param.head_dim;

  param.dk_r_stride = param.head_dim;
  param.dk_h_stride = param.seq_len_kv * param.head_dim;
  param.dk_b_stride = param.num_head_q * param.seq_len_kv * param.head_dim;

  param.v_r_stride = param.head_dim;
  param.v_h_stride = param.seq_len_kv * param.head_dim;
  param.v_b_stride = param.num_head_kv * param.seq_len_kv * param.head_dim;

  param.dv_r_stride = param.head_dim;
  param.dv_h_stride = param.seq_len_kv * param.head_dim;
  param.dv_b_stride = param.num_head_q * param.seq_len_kv * param.head_dim;

  param.o_r_stride = param.head_dim;
  param.o_h_stride = param.seq_len_q * param.head_dim;
  param.o_b_stride = param.num_head_q * param.seq_len_q * param.head_dim;

  // param.do_r_stride = param.head_dim;
  // param.do_h_stride = param.seq_len_q * param.head_dim;
  // param.do_b_stride = param.num_head_q * param.seq_len_q * param.head_dim;
  param.s_r_stride = param.seq_len_kv_pad;
  param.s_s_stride = param.seq_len_q_pad * param.seq_len_kv_pad;
  param.s_b_stride =
      param.num_head_q * param.seq_len_q_pad * param.seq_len_kv_pad;

  param.dq_r_stride = param.head_dim;
  param.dq_h_stride = param.seq_len_q_pad * param.head_dim;
  param.dq_b_stride = param.num_head_q * param.seq_len_q_pad * param.head_dim;
}

template <typename T>
void setup_bshd_stride(Param<T>& param) {
  param.q_r_stride = param.num_head_q * param.head_dim;
  param.q_h_stride = param.head_dim;
  param.q_b_stride = param.num_head_q * param.seq_len_q * param.head_dim;

  // param.dq_r_stride = param.head_dim;
  // param.dq_h_stride = param.seq_len_q * param.head_dim;
  // param.dq_b_stride = param.num_head_q * param.seq_len_q * param.head_dim;

  param.k_r_stride = param.num_head_kv * param.head_dim;
  param.k_h_stride = param.head_dim;
  param.k_b_stride = param.num_head_kv * param.seq_len_kv * param.head_dim;

  param.dk_r_stride = param.num_head_q * param.head_dim;
  param.dk_h_stride = param.head_dim;
  param.dk_b_stride = param.num_head_q * param.seq_len_kv * param.head_dim;

  param.v_r_stride = param.num_head_kv * param.head_dim;
  param.v_h_stride = param.head_dim;
  param.v_b_stride = param.num_head_kv * param.seq_len_kv * param.head_dim;

  param.dv_r_stride = param.num_head_q * param.head_dim;
  param.dv_h_stride = param.head_dim;
  param.dv_b_stride = param.num_head_q * param.seq_len_kv * param.head_dim;

  param.o_r_stride = param.num_head_q * param.head_dim;
  param.o_h_stride = param.head_dim;
  param.o_b_stride = param.num_head_q * param.seq_len_q * param.head_dim;

  // param.do_r_stride = param.head_dim;
  // param.do_h_stride = param.seq_len_q * param.head_dim;
  // param.do_b_stride = param.num_head_q * param.seq_len_q * param.head_dim;
  param.s_r_stride = param.seq_len_kv_pad;
  param.s_s_stride = param.seq_len_q_pad * param.seq_len_kv_pad;
  param.s_b_stride =
      param.num_head_q * param.seq_len_q_pad * param.seq_len_kv_pad;

  param.dq_r_stride = param.num_head_q * param.head_dim;
  param.dq_h_stride = param.head_dim;
  param.dq_b_stride = param.num_head_q * param.seq_len_q_pad * param.head_dim;
}

} // namespace cute
