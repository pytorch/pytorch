// SDPA decode kernels (short Q sequence). Three variants:
//   sdpa_vector            - one-pass, for short qL with moderate kL
//   sdpa_vector_2pass_1    - two-pass pass 1, splits the K loop across blocks
//   sdpa_vector_2pass_2    - two-pass pass 2, aggregates per-block partials
//
// Adapted from MLX:
//   https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/scaled_dot_product_attention.metal
#pragma once

template <typename T, int D, int V = D, bool is_causal = false>
[[kernel]] void sdpa_vector(const device T* queries [[buffer(0)]],
                            const device T* keys [[buffer(1)]],
                            const device T* values [[buffer(2)]],
                            device T* out [[buffer(3)]],
                            const constant uint& gqa_factor [[buffer(4)]],
                            const constant uint& N [[buffer(5)]],
                            const constant uint3& qkv_head_strides [[buffer(6)]],
                            const constant uint3& qkv_seq_strides [[buffer(7)]],
                            const constant float& scale [[buffer(8)]],
                            const device bool* mask [[buffer(9)]],
                            const constant uint3& mask_strides [[buffer(10)]],
                            const constant bool& has_mask [[buffer(11)]],
                            const constant uint4& qkv_batch_strides_heads [[buffer(12)]],
                            uint3 tid [[threadgroup_position_in_grid]],
                            uint3 tpg [[threadgroups_per_grid]],
                            uint simd_gid [[simdgroup_index_in_threadgroup]],
                            uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr uint BN = 32;
  constexpr uint BD = 32;
  constexpr uint qk_per_thread = D / BD;
  constexpr uint v_per_thread = V / BD;
  const uint q_head_stride = qkv_head_strides.x;
  const uint q_seq_stride = qkv_seq_strides.x;
  const uint q_batch_stride = qkv_batch_strides_heads.x;
  const uint k_head_stride = qkv_head_strides.y;
  const uint k_seq_stride = qkv_seq_strides.y;
  const uint k_batch_stride = qkv_batch_strides_heads.y;
  const uint v_head_stride = qkv_head_strides.z;
  const uint v_seq_stride = qkv_seq_strides.z;
  const uint v_batch_stride = qkv_batch_strides_heads.z;
  const uint num_heads = qkv_batch_strides_heads.w;
  const uint mask_kv_seq_stride = mask_strides.x;
  const uint mask_q_seq_stride = mask_strides.y;
  const uint mask_head_stride = mask_strides.z;
  uint inner_k_stride = BN * int(k_seq_stride);
  uint inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  const int bh_idx = tid.x;
  const int batch_idx = bh_idx / int(num_heads);
  const int head_idx = bh_idx - batch_idx * int(num_heads);
  const int q_seq_idx = tid.y;
  const int kv_head_idx = head_idx / gqa_factor;
  const int Q = tpg.y;
  const int group_offset = bh_idx * Q + q_seq_idx;
  const int o_offset = group_offset;
  queries +=
      batch_idx * q_batch_stride + head_idx * q_head_stride + q_seq_idx * q_seq_stride + simd_lid * qk_per_thread;
  keys += batch_idx * k_batch_stride + kv_head_idx * k_head_stride + simd_gid * k_seq_stride + simd_lid * qk_per_thread;
  values +=
      batch_idx * v_batch_stride + kv_head_idx * v_head_stride + simd_gid * v_seq_stride + simd_lid * v_per_thread;
  if (has_mask) {
    mask += bh_idx * mask_head_stride + simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }

  out += o_offset * V + simd_gid * v_per_thread;

  // Read the query and 0 the output accumulator
  for (uint i = 0; i < qk_per_thread; i++) {
    q[i] = scale * static_cast<U>(queries[i]);
  }
  for (uint i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;

  // For each key
  for (uint i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (is_causal) {
      use_key = int(i) <= q_seq_idx;
    } else if (has_mask) {
      use_key = mask[0];
    }
    if (use_key) {
      // Read the key
      for (uint j = 0; j < qk_per_thread; j++) {
        k[j] = static_cast<U>(keys[j]);
      }

      // Compute the i-th score
      U score = 0;
      for (uint j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = metal::fast::exp(max_score - new_max);
      U exp_score = metal::fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (uint j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * static_cast<U>(values[j]);
      }
    }

    // Move the pointers to the next kv
    keys += inner_k_stride;
    values += inner_v_stride;
    if (has_mask) {
      mask += BN * mask_kv_seq_stride;
    }
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = metal::fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // Now we need to aggregate all the outputs
  for (uint i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const U safe_sum = (sum_exp_score == 0 ? 1e-6f : sum_exp_score);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor) / safe_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (uint i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

template <typename T, int D, int V = D, bool is_causal = false>
[[kernel]] void sdpa_vector_2pass_1(const device T* queries [[buffer(0)]],
                                    const device T* keys [[buffer(1)]],
                                    const device T* values [[buffer(2)]],
                                    device T* out [[buffer(3)]],
                                    device float* sums [[buffer(4)]],
                                    device float* maxs [[buffer(5)]],
                                    const constant uint& gqa_factor [[buffer(6)]],
                                    const constant uint& N [[buffer(7)]],
                                    const constant uint3& qkv_head_strides [[buffer(8)]],
                                    const constant uint3& qkv_seq_strides [[buffer(9)]],
                                    const constant float& scale [[buffer(10)]],
                                    const device bool* mask [[buffer(11)]],
                                    const constant uint3& mask_strides [[buffer(12)]],
                                    const constant bool& has_mask [[buffer(13)]],
                                    const constant uint4& qkv_batch_strides_heads [[buffer(14)]],
                                    uint3 tid [[threadgroup_position_in_grid]],
                                    uint3 tpg [[threadgroups_per_grid]],
                                    uint simd_gid [[simdgroup_index_in_threadgroup]],
                                    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 8;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  const int q_head_stride = qkv_head_strides.x;
  const int q_seq_stride = qkv_seq_strides.x;
  const int q_batch_stride = qkv_batch_strides_heads.x;
  const int k_head_stride = qkv_head_strides.y;
  const int k_seq_stride = qkv_seq_strides.y;
  const int k_batch_stride = qkv_batch_strides_heads.y;
  const int v_head_stride = qkv_head_strides.z;
  const int v_seq_stride = qkv_seq_strides.z;
  const int v_batch_stride = qkv_batch_strides_heads.z;
  const int num_heads = qkv_batch_strides_heads.w;
  const int mask_kv_seq_stride = mask_strides.x;
  const int mask_q_seq_stride = mask_strides.y;
  const int mask_head_stride = mask_strides.z;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);
  constexpr int blocks = 32;

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  const int block_idx = tid.z;
  const int bh_idx = tid.x;
  const int batch_idx = bh_idx / int(num_heads);
  const int head_idx = bh_idx - batch_idx * int(num_heads);
  const int q_seq_idx = tid.y;
  const int o_offset = bh_idx * tpg.y + q_seq_idx;
  const int kv_head_idx = head_idx / gqa_factor;

  queries +=
      batch_idx * q_batch_stride + head_idx * q_head_stride + q_seq_idx * q_seq_stride + simd_lid * qk_per_thread;
  keys += batch_idx * k_batch_stride + kv_head_idx * k_head_stride + (block_idx * BN + simd_gid) * k_seq_stride +
      simd_lid * qk_per_thread;
  values += batch_idx * v_batch_stride + kv_head_idx * v_head_stride + (block_idx * BN + simd_gid) * v_seq_stride +
      simd_lid * v_per_thread;
  out += o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
  if (has_mask) {
    mask +=
        bh_idx * mask_head_stride + (block_idx * BN + simd_gid) * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  sums += o_offset * blocks + block_idx;
  maxs += o_offset * blocks + block_idx;

  // Read the query and 0 the output accumulator
  for (uint i = 0; i < qk_per_thread; i++) {
    q[i] = scale * static_cast<U>(queries[i]);
  }
  for (uint i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;

  // For each key
  for (uint i = block_idx * BN + simd_gid; i < N; i += blocks * BN) {
    bool use_key = true;
    if (is_causal) {
      use_key = int(i) <= q_seq_idx;
    } else if (has_mask) {
      use_key = mask[0];
    }
    if (use_key) {
      // Read the key
      for (uint j = 0; j < qk_per_thread; j++) {
        k[j] = static_cast<U>(keys[j]);
      }

      // Compute the i-th score
      U score = 0;
      for (uint j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (uint j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * static_cast<U>(values[j]);
      }
    }

    // Move the pointers to the next kv
    keys += blocks * inner_k_stride;
    values += blocks * inner_v_stride;
    if (has_mask) {
      mask += BN * blocks * mask_kv_seq_stride;
    }
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = (simd_lid < BN) ? max_scores[simd_lid] : -1e9;
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = (simd_lid < BN) ? sum_exp_scores[simd_lid] : 0;
  sum_exp_score = simd_sum(sum_exp_score * factor);

  // Write the sum and new max
  if (simd_gid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = new_max;
  }

  // Now we need to aggregate all the outputs
  for (uint i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BN + simd_gid] = o[i] * fast::exp(max_scores[simd_gid] - new_max);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // And write the output
    if (simd_gid == 0) {
      U output = outputs[simd_lid * BN];
      for (uint j = 1; j < BN; j++) {
        output += outputs[simd_lid * BN + j];
      }
      out[i] = static_cast<T>(output);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <typename T, int D>
[[kernel]] void sdpa_vector_2pass_2(const device T* partials [[buffer(0)]],
                                    const device float* sums [[buffer(1)]],
                                    const device float* maxs [[buffer(2)]],
                                    device T* out [[buffer(3)]],
                                    uint3 tid [[threadgroup_position_in_grid]],
                                    uint3 tpg [[threadgroups_per_grid]],
                                    uint simd_gid [[simdgroup_index_in_threadgroup]],
                                    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;
  constexpr int blocks = 32;

  typedef float U;

  thread U o[elem_per_thread];
  threadgroup U outputs[BN * BD];

  // Adjust positions
  const int head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int hq_offset = head_idx * tpg.y + q_seq_idx;
  partials += hq_offset * blocks * D + simd_gid * D + simd_lid * elem_per_thread;
  sums += hq_offset * blocks;
  maxs += hq_offset * blocks;
  out += hq_offset * D + simd_gid * elem_per_thread;

  // First every thread reads the max and sum_exp
  U max_score = maxs[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  U sum_exp_score = simd_sum(sums[simd_lid] * factor);

  // Now read the block into registers and then use shared memory to transpose
  // it
  for (uint i = 0; i < elem_per_thread; i++) {
    o[i] = partials[i];
  }
  for (uint i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const U safe_sum = (sum_exp_score == 0 ? 1e-6f : sum_exp_score);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor) / safe_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (uint i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

#define INSTANTIATE_SDPA_VECTOR_ONE(DTYPE, QK_DIM, VALUE_DIM, CAUSAL, NAME_SUFFIX)                            \
  template[[host_name("sdpa_vector_" #DTYPE "_" #QK_DIM "_" #VALUE_DIM NAME_SUFFIX)]] kernel void             \
  sdpa_vector<DTYPE, QK_DIM, VALUE_DIM, CAUSAL>(const device DTYPE* queries [[buffer(0)]],                    \
                                                const device DTYPE* keys [[buffer(1)]],                       \
                                                const device DTYPE* values [[buffer(2)]],                     \
                                                device DTYPE* out [[buffer(3)]],                              \
                                                const constant uint& gqa_factor [[buffer(4)]],                \
                                                const constant uint& N [[buffer(5)]],                         \
                                                const constant uint3& qkv_head_strides [[buffer(6)]],         \
                                                const constant uint3& qkv_seq_strides [[buffer(7)]],          \
                                                const constant float& scale [[buffer(8)]],                    \
                                                const device bool* mask [[buffer(9)]],                        \
                                                const constant uint3& mask_strides [[buffer(10)]],            \
                                                const constant bool& has_mask [[buffer(11)]],                 \
                                                const constant uint4& qkv_batch_strides_heads [[buffer(12)]], \
                                                uint3 tid [[threadgroup_position_in_grid]],                   \
                                                uint3 tpg [[threadgroups_per_grid]],                          \
                                                uint simd_gid [[simdgroup_index_in_threadgroup]],             \
                                                uint simd_lid [[thread_index_in_simdgroup]]);

#define INSTANTIATE_SDPA_VECTOR(DTYPE, QK_DIM, VALUE_DIM)           \
  INSTANTIATE_SDPA_VECTOR_ONE(DTYPE, QK_DIM, VALUE_DIM, false, ""); \
  INSTANTIATE_SDPA_VECTOR_ONE(DTYPE, QK_DIM, VALUE_DIM, true, "_causal");

#define INSTANTIATE_SDPA_VECTOR_2PASS_1_ONE(DTYPE, QK_DIM, VALUE_DIM, CAUSAL, NAME_SUFFIX)                            \
  template[[host_name("sdpa_vector_2pass_1_" #DTYPE "_" #QK_DIM "_" #VALUE_DIM NAME_SUFFIX)]] kernel void             \
  sdpa_vector_2pass_1<DTYPE, QK_DIM, VALUE_DIM, CAUSAL>(const device DTYPE* queries [[buffer(0)]],                    \
                                                        const device DTYPE* keys [[buffer(1)]],                       \
                                                        const device DTYPE* values [[buffer(2)]],                     \
                                                        device DTYPE* out [[buffer(3)]],                              \
                                                        device float* sums [[buffer(4)]],                             \
                                                        device float* maxs [[buffer(5)]],                             \
                                                        const constant uint& gqa_factor [[buffer(6)]],                \
                                                        const constant uint& N [[buffer(7)]],                         \
                                                        const constant uint3& qkv_head_strides [[buffer(8)]],         \
                                                        const constant uint3& qkv_seq_strides [[buffer(9)]],          \
                                                        const constant float& scale [[buffer(10)]],                   \
                                                        const device bool* mask [[buffer(11)]],                       \
                                                        const constant uint3& mask_strides [[buffer(12)]],            \
                                                        const constant bool& has_mask [[buffer(13)]],                 \
                                                        const constant uint4& qkv_batch_strides_heads [[buffer(14)]], \
                                                        uint3 tid [[threadgroup_position_in_grid]],                   \
                                                        uint3 tpg [[threadgroups_per_grid]],                          \
                                                        uint simd_gid [[simdgroup_index_in_threadgroup]],             \
                                                        uint simd_lid [[thread_index_in_simdgroup]]);

#define INSTANTIATE_SDPA_VECTOR_2PASS_1(DTYPE, QK_DIM, VALUE_DIM)           \
  INSTANTIATE_SDPA_VECTOR_2PASS_1_ONE(DTYPE, QK_DIM, VALUE_DIM, false, ""); \
  INSTANTIATE_SDPA_VECTOR_2PASS_1_ONE(DTYPE, QK_DIM, VALUE_DIM, true, "_causal");

#define INSTANTIATE_SDPA_VECTOR_AGGREGATION(DTYPE, VALUE_DIM)                             \
  template [[host_name("sdpa_vector_2pass_2_" #DTYPE "_" #VALUE_DIM)]] kernel void        \
  sdpa_vector_2pass_2<DTYPE, VALUE_DIM>(const device DTYPE* partials [[buffer(0)]],       \
                                        const device float* sums [[buffer(1)]],           \
                                        const device float* maxs [[buffer(2)]],           \
                                        device DTYPE* out [[buffer(3)]],                  \
                                        uint3 tid [[threadgroup_position_in_grid]],       \
                                        uint3 tpg [[threadgroups_per_grid]],              \
                                        uint simd_gid [[simdgroup_index_in_threadgroup]], \
                                        uint simd_lid [[thread_index_in_simdgroup]]);

#define INSTANTIATE_SDPA_VECTOR_HEADS(DTYPE)        \
  INSTANTIATE_SDPA_VECTOR(DTYPE, 64, 64);           \
  INSTANTIATE_SDPA_VECTOR(DTYPE, 96, 96);           \
  INSTANTIATE_SDPA_VECTOR(DTYPE, 128, 128);         \
  INSTANTIATE_SDPA_VECTOR(DTYPE, 256, 256);         \
  INSTANTIATE_SDPA_VECTOR_2PASS_1(DTYPE, 64, 64);   \
  INSTANTIATE_SDPA_VECTOR_2PASS_1(DTYPE, 96, 96);   \
  INSTANTIATE_SDPA_VECTOR_2PASS_1(DTYPE, 128, 128); \
  INSTANTIATE_SDPA_VECTOR_2PASS_1(DTYPE, 256, 256); \
  INSTANTIATE_SDPA_VECTOR_AGGREGATION(DTYPE, 64);   \
  INSTANTIATE_SDPA_VECTOR_AGGREGATION(DTYPE, 96);   \
  INSTANTIATE_SDPA_VECTOR_AGGREGATION(DTYPE, 128);  \
  INSTANTIATE_SDPA_VECTOR_AGGREGATION(DTYPE, 256);

INSTANTIATE_SDPA_VECTOR_HEADS(float);
INSTANTIATE_SDPA_VECTOR_HEADS(half);
INSTANTIATE_SDPA_VECTOR_HEADS(bfloat);
