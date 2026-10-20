// SDPA decode kernel, one query row per (batch, head). sdpa_vector<.., BLOCKS
// = 1> writes the normalized output in one pass; the kDecodeBlocks variant
// splits the keys across kDecodeBlocks threadgroups that write unnormalized
// fp32 partial rows for sdpa_merge_partials. Head dims are padded to BD: a
// lane's
// window of BD / 32 columns is clamped inside the real dims so its loads stay
// unconditional, and the columns a clamped window shares with the previous
// lane are zeroed in q and skipped at the store.
//
// Adapted from MLX:
//   https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/scaled_dot_product_attention.metal
#pragma once
#include <ATen/native/mps/kernels/Attention.h>

// Sums the simdgroups' partial rows into dst (scaled by inv_sum). Per column,
// the exchange through `outputs` puts simdgroup s's value of lane window w in
// slot [w][s]; thread t = (w, s) = (t / BN, t % BN) picks it up, the BN
// threads of a window reduce with xor shuffles and the s == 0 one stores.
template <int per_lane, int BN, typename U, typename DstT>
METAL_FUNC void decode_store(
    device DstT* dst,
    const thread U* o,
    threadgroup U* outputs,
    U lane_factor,
    U inv_sum,
    int Dv,
    uint simd_gid,
    uint simd_lid) {
  const int t = simd_gid * 32 + simd_lid;
  const int w = t / BN;
  const int s = t % BN;
  const int w_off = min(w * per_lane, Dv - per_lane);
  const U factor = simd_shuffle(lane_factor, ushort(s));
  for (int i = 0; i < per_lane; i++) {
    outputs[simd_lid * BN + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    U val = outputs[w * BN + s] * factor;
    for (int k = 1; k < BN; k <<= 1) {
      val += simd_shuffle_xor(val, ushort(k));
    }
    if (s == 0 && i >= w * per_lane - w_off) {
      dst[w_off + i] = static_cast<DstT>(val * inv_sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <
    typename T,
    int BD,
    bool is_causal,
    bool HAS_MASK,
    typename MaskType,
    int BLOCKS,
    int BN>
[[kernel, max_total_threads_per_threadgroup(BN * 32)]] void sdpa_vector(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant DecodeAttnParams& params [[buffer(4)]],
    const device MaskType* mask [[buffer(5)]],
    device float* partials [[buffer(6)]],
    device float* sums [[buffer(7)]],
    device float* maxs [[buffer(8)]],
    uint2 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int per_lane = BD / 32;
  constexpr bool is_bool_mask = HAS_MASK && is_same_v<MaskType, bool>;
  constexpr bool is_additive_mask = HAS_MASK && !is_bool_mask;
  typedef float U;

  const int N = params.N;
  const int Dv = params.Dv;
  const int qk_off = min(int(simd_lid) * per_lane, int(params.D) - per_lane);
  const int qk_skip = int(simd_lid) * per_lane - qk_off;
  const int v_off = min(int(simd_lid) * per_lane, Dv - per_lane);
  const int bh_idx = tid.x;
  const int batch_idx = bh_idx / int(params.num_heads);
  const int head_idx = bh_idx - batch_idx * int(params.num_heads);
  const int kv_head_idx = head_idx / int(params.gqa_factor);
  const int block_idx = tid.y;
  const int key0 = block_idx * BN + simd_gid;
  const long part = long(bh_idx) * BLOCKS + block_idx; // partial row
  const int k_seq_stride = params.K_strides[2];
  const int v_seq_stride = params.V_strides[2];
  const int mask_seq_stride = params.M_strides[2];

  queries += batch_idx * long(params.Q_strides[0]) +
      head_idx * long(params.Q_strides[1]) + qk_off;
  keys += batch_idx * long(params.K_strides[0]) +
      kv_head_idx * long(params.K_strides[1]) + key0 * k_seq_stride + qk_off;
  values += batch_idx * long(params.V_strides[0]) +
      kv_head_idx * long(params.V_strides[1]) + key0 * v_seq_stride + v_off;
  if IF_CONSTEXPR (HAS_MASK) {
    mask += batch_idx * long(params.M_strides[0]) +
        head_idx * long(params.M_strides[1]) + key0 * mask_seq_stride;
  }

  thread U q[per_lane];
  thread U k[per_lane];
  thread U o[per_lane];
  threadgroup U outputs[32 * BN];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Read the query and 0 the output accumulator
  for (int i = 0; i < per_lane; i++) {
    q[i] = i < qk_skip ? U(0) : params.scale * static_cast<U>(queries[i]);
  }
  for (int i = 0; i < per_lane; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;

  // For each key
  for (int i = key0; i < N; i += BLOCKS * BN) {
    bool use_key = true;
    if IF_CONSTEXPR (is_causal) {
      use_key = i == 0; // the query row sees key 0 (upper-left alignment)
    } else if IF_CONSTEXPR (is_bool_mask) {
      use_key = mask[0];
    }
    if (use_key) {
      // Read the key
      for (int j = 0; j < per_lane; j++) {
        k[j] = static_cast<U>(keys[j]);
      }

      // Compute the i-th score
      U score = 0;
      for (int j = 0; j < per_lane; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);

      if IF_CONSTEXPR (is_additive_mask) {
        score += static_cast<U>(mask[0]);
      }

      // c10::metal::max propagates NaN (plain max drops it).
      const U new_max = c10::metal::max(max_score, score);
      U factor = 1;
      U exp_score = 0;
      if (new_max != -INFINITY) {
        factor = fast::exp(max_score - new_max);
        exp_score = fast::exp(score - new_max);
      }

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int j = 0; j < per_lane; j++) {
        o[j] = o[j] * factor + exp_score * static_cast<U>(values[j]);
      }
    }

    // Move the pointers to the next kv
    keys += BLOCKS * BN * k_seq_stride;
    values += BLOCKS * BN * v_seq_stride;
    if IF_CONSTEXPR (HAS_MASK) {
      mask += BLOCKS * BN * mask_seq_stride;
    }
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Lane l takes simdgroup l's running max and sum (neutral past BN).
  const U m = simd_lid < BN ? max_scores[simd_lid] : U(-INFINITY);
  const U new_max = simd_max(m);
  const U lane_factor = new_max == -INFINITY ? U(1) : fast::exp(m - new_max);
  const U sum =
      simd_sum((simd_lid < BN ? sum_exp_scores[simd_lid] : U(0)) * lane_factor);

  // Two-pass blocks hand their row max (log2 units) and sum to the merge.
  if (BLOCKS > 1 && simd_gid == 0 && simd_lid == 0) {
    sums[part] = sum;
    maxs[part] = new_max * M_LOG2E_F;
  }
  // A fully masked row has sum == 0 and all-zero o; keep it 0 instead of NaN.
  if IF_CONSTEXPR (BLOCKS == 1) {
    decode_store<per_lane, BN>(
        out + bh_idx * long(Dv),
        o,
        outputs,
        lane_factor,
        1 / (sum == 0 ? U(1) : sum),
        Dv,
        simd_gid,
        simd_lid);
  } else {
    decode_store<per_lane, BN>(
        partials + part * BD,
        o,
        outputs,
        lane_factor,
        U(1),
        Dv,
        simd_gid,
        simd_lid);
  }
}

// Merges the unnormalized fp32 partial rows written by the two-pass decode
// and the split-K prefill kernels. kMergeLanes threads per output element
// each take every kMergeLanes-th block and reduce with xor shuffles:
//   out[row, c] = (sum over b of f_b P_b[row, c]) / (sum over b of f_b S_b),
//   f_b = 2^(max_b - M), M = max over b of max_b (maxs in log2 units).
// dims = (value head dim, blocks, out row stride, partial row width).
template <typename T>
[[kernel]] void sdpa_merge_partials(
    const device float* partials [[buffer(0)]],
    const device float* sums [[buffer(1)]],
    const device float* maxs [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant uint4& dims [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  const uint lane = gid.x;
  const uint col = gid.y;
  const uint row = gid.z;
  const uint blocks = dims.y;
  const uint width = dims.w;
  partials += long(row) * blocks * width + col;
  sums += long(row) * blocks;
  maxs += long(row) * blocks;
  float M = -INFINITY;
  for (uint b = 0; b < blocks; b++) {
    M = c10::metal::max(M, maxs[b]);
  }
  float num = 0;
  float den = 0;
  for (uint b = lane; b < blocks; b += kMergeLanes) {
    const float f = M == -INFINITY ? 1.0f : fast::exp2(maxs[b] - M);
    num += f * partials[b * width];
    den += f * sums[b];
  }
  for (ushort k = 1; k < kMergeLanes; k <<= 1) {
    num += simd_shuffle_xor(num, k);
    den += simd_shuffle_xor(den, k);
  }
  if (lane == 0) {
    out[row * long(dims.z) + col] =
        static_cast<T>(num / (den == 0 ? 1.0f : den));
  }
}

#define instantiate_decode(name, blocks, bn, dtype, bd) \
  instantiate_kernel(                                   \
      name "_" #dtype "_" #bd "_masknone",              \
      sdpa_vector,                                      \
      dtype,                                            \
      bd,                                               \
      false,                                            \
      false,                                            \
      float,                                            \
      blocks,                                           \
      bn);                                              \
  instantiate_kernel(                                   \
      name "_" #dtype "_" #bd "_masknone_causal",       \
      sdpa_vector,                                      \
      dtype,                                            \
      bd,                                               \
      true,                                             \
      false,                                            \
      float,                                            \
      blocks,                                           \
      bn);                                              \
  instantiate_kernel(                                   \
      name "_" #dtype "_" #bd "_maskbool",              \
      sdpa_vector,                                      \
      dtype,                                            \
      bd,                                               \
      false,                                            \
      true,                                             \
      bool,                                             \
      blocks,                                           \
      bn);                                              \
  instantiate_kernel(                                   \
      name "_" #dtype "_" #bd "_mask" #dtype,           \
      sdpa_vector,                                      \
      dtype,                                            \
      bd,                                               \
      false,                                            \
      true,                                             \
      dtype,                                            \
      blocks,                                           \
      bn)

#define instantiate_decode_tile(dtype, bd)                            \
  instantiate_decode("sdpa_vector", 1, kDecodeSimdgroups, dtype, bd); \
  instantiate_decode(                                                 \
      "sdpa_vector_2pass", kDecodeBlocks, kDecodeBlockSimdgroups, dtype, bd)

// Keep the tiles in sync with decode_bucket in Attention.mm.
#define instantiate_decode_tiles(dtype) \
  instantiate_decode_tile(dtype, 32);   \
  instantiate_decode_tile(dtype, 64);   \
  instantiate_decode_tile(dtype, 96);   \
  instantiate_decode_tile(dtype, 128);  \
  instantiate_decode_tile(dtype, 256);  \
  instantiate_decode_tile(dtype, 512);  \
  instantiate_kernel("sdpa_merge_partials_" #dtype, sdpa_merge_partials, dtype)

instantiate_decode_tiles(float);
instantiate_decode_tiles(half);
instantiate_decode_tiles(bfloat);
