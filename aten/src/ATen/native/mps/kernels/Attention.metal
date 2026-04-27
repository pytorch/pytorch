// Largely influeneced by
// https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/scaled_dot_product_attention.metal
#include <c10/metal/utils.h>
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector(
    const device T* queries [[buffer(0)]],
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
  const uint mask_head_stride = mask_strides.x;
  const uint mask_kv_seq_stride = mask_strides.y;
  const uint mask_q_seq_stride = mask_strides.z;
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
  queries += batch_idx * q_batch_stride + head_idx * q_head_stride +
      q_seq_idx * q_seq_stride + simd_lid * qk_per_thread;
  keys += batch_idx * k_batch_stride + kv_head_idx * k_head_stride +
      simd_gid * k_seq_stride + simd_lid * qk_per_thread;
  values += batch_idx * v_batch_stride + kv_head_idx * v_head_stride +
      simd_gid * v_seq_stride + simd_lid * v_per_thread;
  if (has_mask) {
    mask += bh_idx * mask_head_stride + simd_gid * mask_kv_seq_stride +
        q_seq_idx * mask_q_seq_stride;
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
    if (!has_mask || mask[0]) {
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
      U factor = metal::precise::exp(max_score - new_max);
      U exp_score = metal::precise::exp(score - new_max);

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
  U factor = metal::precise::exp(max_score - new_max);
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

template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector_2pass_1(
    const device T* queries [[buffer(0)]],
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

  queries += batch_idx * q_batch_stride + head_idx * q_head_stride +
      q_seq_idx * q_seq_stride + simd_lid * qk_per_thread;
  keys += batch_idx * k_batch_stride + kv_head_idx * k_head_stride +
      (block_idx * BN + simd_gid) * k_seq_stride + simd_lid * qk_per_thread;
  values += batch_idx * v_batch_stride + kv_head_idx * v_head_stride +
      (block_idx * BN + simd_gid) * v_seq_stride + simd_lid * v_per_thread;
  out += o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
  if (has_mask) {
    mask += bh_idx * mask_head_stride +
        (block_idx * BN + simd_gid) * mask_kv_seq_stride +
        q_seq_idx * mask_q_seq_stride;
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
    if (!has_mask || mask[0]) {
      // Read the key
      for (uint i = 0; i < qk_per_thread; i++) {
        k[i] = static_cast<U>(keys[i]);
      }

      // Compute the i-th score
      U score = 0;
      for (uint i = 0; i < qk_per_thread; i++) {
        score += q[i] * k[i];
      }
      score = simd_sum(score);

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = metal::precise::exp(max_score - new_max);
      U exp_score = metal::precise::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (uint i = 0; i < v_per_thread; i++) {
        o[i] = o[i] * factor + exp_score * static_cast<U>(values[i]);
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
  U factor = metal::precise::exp(max_score - new_max);
  sum_exp_score = (simd_lid < BN) ? sum_exp_scores[simd_lid] : 0;
  sum_exp_score = simd_sum(sum_exp_score * factor);

  // Write the sum and new max
  if (simd_gid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = new_max;
  }

  // Now we need to aggregate all the outputs
  for (uint i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BN + simd_gid] =
        o[i] * metal::precise::exp(max_scores[simd_gid] - new_max);
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
[[kernel]] void sdpa_vector_2pass_2(
    const device T* partials [[buffer(0)]],
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
  partials +=
      hq_offset * blocks * D + simd_gid * D + simd_lid * elem_per_thread;
  sums += hq_offset * blocks;
  maxs += hq_offset * blocks;
  out += hq_offset * D + simd_gid * elem_per_thread;

  // First every thread reads the max and sum_exp
  U max_score = maxs[simd_lid];
  U new_max = simd_max(max_score);
  U factor = metal::precise::exp(max_score - new_max);
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

template <typename T, int BQ, int BK, int BD, int WM, int WN>
kernel void attention(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant uint& qL [[buffer(4)]],
    const constant uint& kL [[buffer(5)]],
    const constant uint& gqa_factor [[buffer(6)]],
    const constant float& scale [[buffer(7)]],
    const constant uint& NK [[buffer(8)]],
    const constant uint3& Q_strides [[buffer(9)]],
    const constant uint3& K_strides [[buffer(10)]],
    const constant uint3& V_strides [[buffer(11)]],
    const constant uint3& O_strides [[buffer(12)]],
    uint3 group_pos [[threadgroup_position_in_grid]],
    uint3 local_pos [[thread_position_in_threadgroup]]) {
  // 1. Compute a full linear thread id from the 3D local id.
  constexpr int THREADGROUP_DIM_X = 32;
  constexpr int THREADGROUP_DIM_Y = WM;
  constexpr int THREADGROUP_DIM_Z = WN;
  const int threads_in_group =
      THREADGROUP_DIM_X * THREADGROUP_DIM_Y * THREADGROUP_DIM_Z;
  int tid = local_pos.x + local_pos.y * THREADGROUP_DIM_X +
      local_pos.z * (THREADGROUP_DIM_X * THREADGROUP_DIM_Y);

  // 2. Compute the effective number of Q (query) rows for this tile.
  const int query_seq_length = qL;
  int start_q = group_pos.x * BQ;
  uint tile_rows =
      (start_q + BQ <= query_seq_length) ? BQ : (query_seq_length - start_q);

  // 3. Compute Global Pointers Offsets for Q and O.
  uint batch = group_pos.z;
  uint head = group_pos.y;
  uint seq_tile = group_pos.x;

  const device T* Q_tile_ptr = Q + batch * Q_strides.x + head * Q_strides.y +
      seq_tile * BQ * Q_strides.z;
  device T* O_tile_ptr = O + batch * O_strides.x + head * O_strides.y +
      seq_tile * BQ * O_strides.z;

  // Adjust head index for K and V using gqa_factor.
  uint kv_head = head / gqa_factor;
  const device T* K_ptr = K + batch * K_strides.x + kv_head * K_strides.y;
  const device T* V_ptr = V + batch * V_strides.x + kv_head * V_strides.y;

  // 4. Declare Threadgroup (Shared) Memory for tiles.
  // qTile covers BQ rows (each of length BD), kTile and vTile cover BK rows.
  threadgroup T qTile[BQ * BD];
  threadgroup T kTile[BK * BD];
  threadgroup T vTile[BK * BD];

  // 5. Load Q from global memory into threadgroup memory & apply scaling.
  uint tile_q_elements = tile_rows * BD;
  for (uint i = tid; i < tile_q_elements; i += threads_in_group) {
    int row = i / BD;
    int col = i % BD;
    qTile[i] = Q_tile_ptr[row * Q_strides.z + col] * (T)scale;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 6. Initialize accumulation buffers for output and softmax reduction.
  float oAcc[BQ * BD]; // Only first tile_q_elements are used
  float row_max[BQ]; // For each valid query row
  float row_sum[BQ]; // For each valid query row
  for (uint i = 0; i < tile_rows; i++) {
    row_max[i] = -FLT_MAX;
    row_sum[i] = 0.0f;
  }
  for (uint i = 0; i < tile_q_elements; i++) {
    oAcc[i] = 0.0f;
  }

  // 7. Loop over the Key/Value (KV) sequence tiles.
  for (uint kb_tile = 0; kb_tile < NK; ++kb_tile) {
    uint kv_base = kb_tile * BK; // first KV row in this tile
    uint total_kv_elements = BK * BD;

    // --- Load K and V tiles into threadgroup memory.
    // For positions that are out-of-bound (padded) set K to -INFINITY.
    for (uint i = tid; i < total_kv_elements; i += threads_in_group) {
      int row = i / BD;
      int col = i % BD;
      if ((kv_base + row) < kL) {
        kTile[i] = K_ptr[(kv_base + row) * K_strides.z + col];
        vTile[i] = V_ptr[(kv_base + row) * V_strides.z + col];
      } else {
        // For invalid keys, assign a very negative value so that exp(-inf)=0
        kTile[i] = static_cast<T>(-INFINITY);
        vTile[i] = 0;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 8. Compute the score matrix S = Q x (K)^T for this KV tile.
    float S[BQ * BK];
    for (uint i = 0; i < tile_rows; i++) {
      for (int j = 0; j < BK; j++) {
        float dot = 0.0f;
        // Only compute dot product if this tile row corresponds to a valid key.
        if ((kv_base + j) < kL) {
          for (int d = 0; d < BD; d++) {
            dot += qTile[i * BD + d] * kTile[j * BD + d];
          }
        } else {
          dot = -INFINITY;
        }
        S[i * BK + j] = dot;
      }
    }

    // 9. Update softmax statistics (row-wise) using an online reduction.
    for (uint i = 0; i < tile_rows; i++) {
      float old_max = row_max[i];
      float new_max = old_max;
      for (int j = 0; j < BK; j++) {
        float val = S[i * BK + j];
        if (val > new_max) {
          new_max = val;
        }
      }
      float factor = exp(old_max - new_max);
      row_max[i] = new_max;
      // Scale the accumulated numerator for this row.
      for (int d = 0; d < BD; d++) {
        oAcc[i * BD + d] *= factor;
      }
      // Exponentiate the scores and accumulate the sums.
      float exp_sum = 0.0f;
      for (int j = 0; j < BK; j++) {
        float s_val = exp(S[i * BK + j] - new_max);
        S[i * BK + j] = s_val;
        exp_sum += s_val;
      }
      row_sum[i] = row_sum[i] * factor + exp_sum;
    }

    // 10. Use the softmax weights to compute the weighted sum of V.
    for (uint i = 0; i < tile_rows; i++) {
      for (int d = 0; d < BD; d++) {
        float acc = 0.0f;
        for (int j = 0; j < BK; j++) {
          acc += S[i * BK + j] * vTile[j * BD + d];
        }
        oAcc[i * BD + d] += acc;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  } // End of KV tile loop

  // 11. Normalize the accumulated output and store the results to global
  // memory.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (local_pos.x == 0 && local_pos.y == 0 && local_pos.z == 0) {
    for (uint i = 0; i < tile_rows; i++) {
      for (int d = 0; d < BD; d++) {
        O_tile_ptr[i * O_strides.z + d] =
            static_cast<T>(oAcc[i * BD + d] / row_sum[i]);
      }
    }
  }
}

#define INSTANTIATE_SDPA_VECTOR(DTYPE, QK_DIM, VALUE_DIM)           \
  template [[host_name("sdpa_vector_" #DTYPE "_" #QK_DIM            \
                       "_" #VALUE_DIM)]] kernel void                \
  sdpa_vector<DTYPE, QK_DIM, VALUE_DIM>(                            \
      const device DTYPE* queries [[buffer(0)]],                    \
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

#define INSTANTIATE_SDPA_VECTOR_2PASS_1(DTYPE, QK_DIM, VALUE_DIM)   \
  template [[host_name("sdpa_vector_2pass_1_" #DTYPE "_" #QK_DIM    \
                       "_" #VALUE_DIM)]] kernel void                \
  sdpa_vector_2pass_1<DTYPE, QK_DIM, VALUE_DIM>(                    \
      const device DTYPE* queries [[buffer(0)]],                    \
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

#define INSTANTIATE_SDPA_VECTOR_AGGREGATION(DTYPE, VALUE_DIM)                 \
  template                                                                    \
      [[host_name("sdpa_vector_2pass_2_" #DTYPE "_" #VALUE_DIM)]] kernel void \
      sdpa_vector_2pass_2<DTYPE, VALUE_DIM>(                                  \
          const device DTYPE* partials [[buffer(0)]],                         \
          const device float* sums [[buffer(1)]],                             \
          const device float* maxs [[buffer(2)]],                             \
          device DTYPE* out [[buffer(3)]],                                    \
          uint3 tid [[threadgroup_position_in_grid]],                         \
          uint3 tpg [[threadgroups_per_grid]],                                \
          uint simd_gid [[simdgroup_index_in_threadgroup]],                   \
          uint simd_lid [[thread_index_in_simdgroup]]);

#define INSTANTIATE_SDPA_VECTOR_HEADS(DTYPE)        \
  INSTANTIATE_SDPA_VECTOR(DTYPE, 64, 64);           \
  INSTANTIATE_SDPA_VECTOR(DTYPE, 96, 96);           \
  INSTANTIATE_SDPA_VECTOR(DTYPE, 128, 128);         \
  INSTANTIATE_SDPA_VECTOR_2PASS_1(DTYPE, 64, 64);   \
  INSTANTIATE_SDPA_VECTOR_2PASS_1(DTYPE, 96, 96);   \
  INSTANTIATE_SDPA_VECTOR_2PASS_1(DTYPE, 128, 128); \
  INSTANTIATE_SDPA_VECTOR_AGGREGATION(DTYPE, 64);   \
  INSTANTIATE_SDPA_VECTOR_AGGREGATION(DTYPE, 96);   \
  INSTANTIATE_SDPA_VECTOR_AGGREGATION(DTYPE, 128);

INSTANTIATE_SDPA_VECTOR_HEADS(float);
INSTANTIATE_SDPA_VECTOR_HEADS(half);
INSTANTIATE_SDPA_VECTOR_HEADS(bfloat);

#define INSTANTIATE_ATTN(DTYPE, bq, bk, bd, wm, wn)                      \
  template [[host_name("attention_" #DTYPE "_bq" #bq "_bk" #bk "_bd" #bd \
                       "_wm" #wm "_wn" #wn)]] [[kernel]] void            \
  attention<DTYPE, bq, bk, bd, wm, wn>(                                  \
      const device DTYPE* Q [[buffer(0)]],                               \
      const device DTYPE* K [[buffer(1)]],                               \
      const device DTYPE* V [[buffer(2)]],                               \
      device DTYPE* O [[buffer(3)]],                                     \
      const constant uint& qL [[buffer(4)]],                             \
      const constant uint& kL [[buffer(5)]],                             \
      const constant uint& gqa_factor [[buffer(6)]],                     \
      const constant float& scale [[buffer(7)]],                         \
      const constant uint& NK [[buffer(8)]],                             \
      const constant uint3& Q_strides [[buffer(9)]],                     \
      const constant uint3& K_strides [[buffer(10)]],                    \
      const constant uint3& V_strides [[buffer(11)]],                    \
      const constant uint3& O_strides [[buffer(12)]],                    \
      uint3 group_pos [[threadgroup_position_in_grid]],                  \
      uint3 local_pos [[thread_position_in_threadgroup]]);

#define INSTANTIATE_ATTN_SHAPES_HELPER(dtype) \
  INSTANTIATE_ATTN(dtype, 32, 16, 128, 4, 1)  \
  INSTANTIATE_ATTN(dtype, 32, 32, 80, 4, 1)   \
  INSTANTIATE_ATTN(dtype, 32, 32, 64, 4, 1)

INSTANTIATE_ATTN_SHAPES_HELPER(float);
INSTANTIATE_ATTN_SHAPES_HELPER(half);
INSTANTIATE_ATTN_SHAPES_HELPER(bfloat);
// FlashAttention-2 Metal kernels for MPS backend
// Forward + Backward (dQ kernel, dKdV kernel)
// Reference: Dao et al. 2022 (https://arxiv.org/abs/2205.14135)
//
// Thread organization:
//   Forward / dQ:  one thread per query row,  grid=(B*H, ceil(qL/BQ), 1)
//   dKdV:          one thread per key row,     grid=(B*H, ceil(kL/BQ), 1)
//
// D is a compile-time template parameter (64, 128).
// BK = key-tile rows loaded into threadgroup memory per inner-loop iteration.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

template <typename T, int D, int BQ = 64, int BK = 32>
[[kernel]] void flash_attn_fwd(
    const device T*      Q          [[buffer(0)]],
    const device T*      K          [[buffer(1)]],
    const device T*      V          [[buffer(2)]],
    device       T*      O          [[buffer(3)]],
    device       float*  LSE        [[buffer(4)]],   // (B*H, qL)
    const constant uint& qL         [[buffer(5)]],
    const constant uint& kL         [[buffer(6)]],
    const constant uint& gqa_factor [[buffer(7)]],
    const constant uint& num_heads  [[buffer(8)]],
    const constant float& scale     [[buffer(9)]],
    const constant bool&  is_causal [[buffer(10)]],
    const constant uint4& Q_str     [[buffer(11)]],  // (batch, head, seq, 1)
    const constant uint4& K_str     [[buffer(12)]],
    const constant uint4& V_str     [[buffer(13)]],
    const constant uint4& O_str     [[buffer(14)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint bh  = tgid.x;
    const uint q_tile = tgid.y;
    const uint q_row  = q_tile * BQ + tid;
    if (q_row >= qL) return;

    const uint batch    = bh / num_heads;
    const uint head     = bh % num_heads;
    const uint kv_head  = head / gqa_factor;

    const device T* Q_ptr = Q + batch * Q_str[0] + head    * Q_str[1];
    const device T* K_ptr = K + batch * K_str[0] + kv_head * K_str[1];
    const device T* V_ptr = V + batch * V_str[0] + kv_head * V_str[1];
    device       T* O_ptr = O + batch * O_str[0] + head    * O_str[1];

    // Load query row into registers
    float q[D];
    for (uint d = 0; d < D; d++)
        q[d] = float(Q_ptr[q_row * Q_str[2] + d]);

    // Threadgroup memory for K/V tiles
    threadgroup float K_smem[BK * D];
    threadgroup float V_smem[BK * D];

    // Per-row online-softmax state
    float m   = -INFINITY;
    float l   = 0.0f;
    float acc[D];
    for (uint d = 0; d < D; d++) acc[d] = 0.0f;

    for (uint kb = 0; kb < kL; kb += BK) {
        // Cooperative load of K and V tiles
        for (uint i = tid; i < BK * D; i += BQ) {
            const uint r = kb + i / D;
            const uint d = i % D;
            K_smem[i] = (r < kL) ? float(K_ptr[r * K_str[2] + d]) : 0.0f;
            V_smem[i] = (r < kL) ? float(V_ptr[r * V_str[2] + d]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute scores and apply causal mask
        float scores[BK];
        for (uint j = 0; j < BK; j++) {
            const uint k_row = kb + j;
            if (k_row >= kL || (is_causal && k_row > q_row)) {
                scores[j] = -INFINITY;
                continue;
            }
            float dot = 0.0f;
            for (uint d = 0; d < D; d++)
                dot += q[d] * K_smem[j * D + d];
            scores[j] = dot * scale;
        }

        // Online softmax update
        float m_new = m;
        for (uint j = 0; j < BK; j++) m_new = max(m_new, scores[j]);

        const float alpha = metal::precise::exp(m - m_new);
        float l_new = l * alpha;

        float p[BK];
        for (uint j = 0; j < BK; j++) {
            p[j] = (scores[j] == -INFINITY) ? 0.0f
                                             : metal::precise::exp(scores[j] - m_new);
            l_new += p[j];
        }

        // Rescale and accumulate
        for (uint d = 0; d < D; d++) acc[d] *= alpha;
        for (uint j = 0; j < BK; j++)
            for (uint d = 0; d < D; d++)
                acc[d] += p[j] * V_smem[j * D + d];

        m = m_new;
        l = l_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    const float l_safe = (l == 0.0f) ? 1.0f : l;
    for (uint d = 0; d < D; d++)
        O_ptr[q_row * O_str[2] + d] = T(acc[d] / l_safe);

    // Write log-sum-exp for backward
    LSE[bh * qL + q_row] = m + metal::precise::log(l_safe);
}

// ---------------------------------------------------------------------------
// Backward preprocess: D_vec[i] = rowsum(dO[i] * O[i])
// One thread per query row.
// ---------------------------------------------------------------------------

template <typename T, int D, int BQ = 64>
[[kernel]] void flash_attn_bwd_preprocess(
    const device T*      dO         [[buffer(0)]],
    const device T*      O          [[buffer(1)]],
    device       float*  D_vec      [[buffer(2)]],   // (B*H, qL)
    const constant uint& qL         [[buffer(3)]],
    const constant uint& num_heads  [[buffer(4)]],
    const constant uint4& dO_str    [[buffer(5)]],
    const constant uint4& O_str     [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint bh    = tgid.x;
    const uint q_row = tgid.y * BQ + tid;
    if (q_row >= qL) return;

    const uint batch = bh / num_heads;
    const uint head  = bh % num_heads;

    const device T* dO_ptr = dO + batch * dO_str[0] + head * dO_str[1];
    const device T* O_ptr  = O  + batch * O_str[0]  + head * O_str[1];

    float d = 0.0f;
    for (uint k = 0; k < D; k++)
        d += float(dO_ptr[q_row * dO_str[2] + k])
           * float(O_ptr [q_row * O_str[2]  + k]);
    D_vec[bh * qL + q_row] = d;
}

// ---------------------------------------------------------------------------
// Backward dQ kernel
// Outer loop over Q-tiles; inner loop over K-tiles.
// No atomics needed — each thread owns its own dQ row.
// ---------------------------------------------------------------------------

template <typename T, int D, int BQ = 64, int BK = 32>
[[kernel]] void flash_attn_bwd_dq(
    const device T*      Q          [[buffer(0)]],
    const device T*      K          [[buffer(1)]],
    const device T*      V          [[buffer(2)]],
    const device T*      O          [[buffer(3)]],
    const device T*      dO         [[buffer(4)]],
    const device float*  LSE        [[buffer(5)]],
    const device float*  D_vec      [[buffer(6)]],
    device       T*      dQ         [[buffer(7)]],
    const constant uint& qL         [[buffer(8)]],
    const constant uint& kL         [[buffer(9)]],
    const constant uint& gqa_factor [[buffer(10)]],
    const constant uint& num_heads  [[buffer(11)]],
    const constant float& scale     [[buffer(12)]],
    const constant bool&  is_causal [[buffer(13)]],
    const constant uint4& Q_str     [[buffer(14)]],
    const constant uint4& K_str     [[buffer(15)]],
    const constant uint4& V_str     [[buffer(16)]],
    const constant uint4& O_str     [[buffer(17)]],
    const constant uint4& dO_str    [[buffer(18)]],
    const constant uint4& dQ_str    [[buffer(19)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint bh    = tgid.x;
    const uint q_row = tgid.y * BQ + tid;
    if (q_row >= qL) return;

    const uint batch   = bh / num_heads;
    const uint head    = bh % num_heads;
    const uint kv_head = head / gqa_factor;

    const device T* Q_ptr  = Q  + batch * Q_str[0]  + head    * Q_str[1];
    const device T* K_ptr  = K  + batch * K_str[0]  + kv_head * K_str[1];
    const device T* V_ptr  = V  + batch * V_str[0]  + kv_head * V_str[1];
    const device T* dO_ptr = dO + batch * dO_str[0] + head    * dO_str[1];
    device       T* dQ_ptr = dQ + batch * dQ_str[0] + head    * dQ_str[1];

    // Load my query row and dO row
    float q[D], do_row[D];
    for (uint d = 0; d < D; d++) {
        q[d]      = float(Q_ptr [q_row * Q_str[2]  + d]);
        do_row[d] = float(dO_ptr[q_row * dO_str[2] + d]);
    }

    const float lse_i = LSE[bh * qL + q_row];
    const float di    = D_vec[bh * qL + q_row];

    threadgroup float K_smem[BK * D];
    threadgroup float V_smem[BK * D];

    float dq[D];
    for (uint d = 0; d < D; d++) dq[d] = 0.0f;

    for (uint kb = 0; kb < kL; kb += BK) {
        for (uint i = tid; i < BK * D; i += BQ) {
            const uint r = kb + i / D;
            const uint d = i % D;
            K_smem[i] = (r < kL) ? float(K_ptr[r * K_str[2] + d]) : 0.0f;
            V_smem[i] = (r < kL) ? float(V_ptr[r * V_str[2] + d]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = 0; j < BK; j++) {
            const uint k_row = kb + j;
            if (k_row >= kL) break;
            if (is_causal && k_row > q_row) break;

            // Recompute S_ij = dot(q, K[j]) * scale
            float S = 0.0f;
            for (uint d = 0; d < D; d++) S += q[d] * K_smem[j * D + d];
            S *= scale;

            // P_ij = exp(S_ij - LSE_i)
            const float P = metal::precise::exp(S - lse_i);

            // dP_ij = dot(dO_i, V[j])
            float dP = 0.0f;
            for (uint d = 0; d < D; d++) dP += do_row[d] * V_smem[j * D + d];

            // dS_ij = P_ij * (dP_ij - D_i)
            const float dS = P * (dP - di) * scale;

            // dQ_i += dS_ij * K[j]
            for (uint d = 0; d < D; d++) dq[d] += dS * K_smem[j * D + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint d = 0; d < D; d++)
        dQ_ptr[q_row * dQ_str[2] + d] = T(dq[d]);
}

// ---------------------------------------------------------------------------
// Backward dK + dV kernel
// Outer loop over K-tiles; inner loop over Q-tiles.
// No atomics needed — each thread owns its own dK / dV row.
// ---------------------------------------------------------------------------

template <typename T, int D, int BQ = 32, int BK = 64>
[[kernel]] void flash_attn_bwd_dkdv(
    const device T*      Q          [[buffer(0)]],
    const device T*      K          [[buffer(1)]],
    const device T*      V          [[buffer(2)]],
    const device T*      O          [[buffer(3)]],
    const device T*      dO         [[buffer(4)]],
    const device float*  LSE        [[buffer(5)]],
    const device float*  D_vec      [[buffer(6)]],
    device       T*      dK         [[buffer(7)]],
    device       T*      dV         [[buffer(8)]],
    const constant uint& qL         [[buffer(9)]],
    const constant uint& kL         [[buffer(10)]],
    const constant uint& gqa_factor [[buffer(11)]],
    const constant uint& num_heads  [[buffer(12)]],
    const constant float& scale     [[buffer(13)]],
    const constant bool&  is_causal [[buffer(14)]],
    const constant uint4& Q_str     [[buffer(15)]],
    const constant uint4& K_str     [[buffer(16)]],
    const constant uint4& V_str     [[buffer(17)]],
    const constant uint4& O_str     [[buffer(18)]],
    const constant uint4& dO_str    [[buffer(19)]],
    const constant uint4& dK_str    [[buffer(20)]],
    const constant uint4& dV_str    [[buffer(21)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint bh    = tgid.x;
    const uint k_row = tgid.y * BK + tid;
    if (k_row >= kL) return;

    const uint batch   = bh / num_heads;
    const uint head    = bh % num_heads;
    const uint kv_head = head / gqa_factor;

    const device T* Q_ptr  = Q  + batch * Q_str[0]  + head    * Q_str[1];
    const device T* K_ptr  = K  + batch * K_str[0]  + kv_head * K_str[1];
    const device T* V_ptr  = V  + batch * V_str[0]  + kv_head * V_str[1];
    const device T* dO_ptr = dO + batch * dO_str[0] + head    * dO_str[1];
    device       T* dK_ptr = dK + batch * dK_str[0] + kv_head * dK_str[1];
    device       T* dV_ptr = dV + batch * dV_str[0] + kv_head * dV_str[1];

    // Load my key and value rows
    float k[D], v[D];
    for (uint d = 0; d < D; d++) {
        k[d] = float(K_ptr[k_row * K_str[2] + d]);
        v[d] = float(V_ptr[k_row * V_str[2] + d]);
    }

    threadgroup float Q_smem [BQ * D];
    threadgroup float dO_smem[BQ * D];

    float dk[D], dv[D];
    for (uint d = 0; d < D; d++) { dk[d] = 0.0f; dv[d] = 0.0f; }

    for (uint qb = 0; qb < qL; qb += BQ) {
        for (uint i = tid; i < BQ * D; i += BK) {
            const uint r = qb + i / D;
            const uint d = i % D;
            Q_smem [i] = (r < qL) ? float(Q_ptr [r * Q_str[2]  + d]) : 0.0f;
            dO_smem[i] = (r < qL) ? float(dO_ptr[r * dO_str[2] + d]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < BQ; i++) {
            const uint q_row = qb + i;
            if (q_row >= qL) break;
            if (is_causal && q_row < k_row) {
                // causal: query can only attend to earlier/same keys
                continue;
            }

            // Recompute S_ij = dot(Q[q_row], k) * scale
            float S = 0.0f;
            for (uint d = 0; d < D; d++) S += Q_smem[i * D + d] * k[d];
            S *= scale;

            const float lse_i = LSE[bh * qL + q_row];
            const float P     = metal::precise::exp(S - lse_i);
            const float di    = D_vec[bh * qL + q_row];

            // dV_j += P_ij * dO_i
            for (uint d = 0; d < D; d++) dv[d] += P * dO_smem[i * D + d];

            // dP_ij = dot(dO_i, v_j)
            float dP = 0.0f;
            for (uint d = 0; d < D; d++) dP += dO_smem[i * D + d] * v[d];

            // dS_ij = P_ij * (dP_ij - D_i) * scale
            const float dS = P * (dP - di) * scale;

            // dK_j += dS_ij * Q_i
            for (uint d = 0; d < D; d++) dk[d] += dS * Q_smem[i * D + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint d = 0; d < D; d++) {
        dK_ptr[k_row * dK_str[2] + d] = T(dk[d]);
        dV_ptr[k_row * dV_str[2] + d] = T(dv[d]);
    }
}

// ---------------------------------------------------------------------------
// Instantiations
// ---------------------------------------------------------------------------

#define INST_FLASH_FWD(T, D) \
  template [[host_name("flash_attn_fwd_" #T "_" #D)]] [[kernel]] \
  void flash_attn_fwd<T, D>(                                       \
      const device T*       Q   [[buffer(0)]],                     \
      const device T*       K   [[buffer(1)]],                     \
      const device T*       V   [[buffer(2)]],                     \
      device       T*       O   [[buffer(3)]],                     \
      device       float*   LSE [[buffer(4)]],                     \
      const constant uint&  qL  [[buffer(5)]],                     \
      const constant uint&  kL  [[buffer(6)]],                     \
      const constant uint&  gqa [[buffer(7)]],                     \
      const constant uint&  nh  [[buffer(8)]],                     \
      const constant float& sc  [[buffer(9)]],                     \
      const constant bool&  ic  [[buffer(10)]],                    \
      const constant uint4& qs  [[buffer(11)]],                    \
      const constant uint4& ks  [[buffer(12)]],                    \
      const constant uint4& vs  [[buffer(13)]],                    \
      const constant uint4& os  [[buffer(14)]],                    \
      uint3 tgid [[threadgroup_position_in_grid]],                 \
      uint  tid  [[thread_index_in_threadgroup]]);

#define INST_FLASH_BWD_PRE(T, D) \
  template [[host_name("flash_attn_bwd_pre_" #T "_" #D)]] [[kernel]] \
  void flash_attn_bwd_preprocess<T, D>(                            \
      const device T*       dO   [[buffer(0)]],                    \
      const device T*       O    [[buffer(1)]],                    \
      device       float*   Dv   [[buffer(2)]],                    \
      const constant uint&  qL   [[buffer(3)]],                    \
      const constant uint&  nh   [[buffer(4)]],                    \
      const constant uint4& dos  [[buffer(5)]],                    \
      const constant uint4& os   [[buffer(6)]],                    \
      uint3 tgid [[threadgroup_position_in_grid]],                 \
      uint  tid  [[thread_index_in_threadgroup]]);

#define INST_FLASH_BWD_DQ(T, D) \
  template [[host_name("flash_attn_bwd_dq_" #T "_" #D)]] [[kernel]] \
  void flash_attn_bwd_dq<T, D>(                                    \
      const device T*       Q    [[buffer(0)]],                    \
      const device T*       K    [[buffer(1)]],                    \
      const device T*       V    [[buffer(2)]],                    \
      const device T*       O    [[buffer(3)]],                    \
      const device T*       dO   [[buffer(4)]],                    \
      const device float*   LSE  [[buffer(5)]],                    \
      const device float*   Dv   [[buffer(6)]],                    \
      device       T*       dQ   [[buffer(7)]],                    \
      const constant uint&  qL   [[buffer(8)]],                    \
      const constant uint&  kL   [[buffer(9)]],                    \
      const constant uint&  gqa  [[buffer(10)]],                   \
      const constant uint&  nh   [[buffer(11)]],                   \
      const constant float& sc   [[buffer(12)]],                   \
      const constant bool&  ic   [[buffer(13)]],                   \
      const constant uint4& qs   [[buffer(14)]],                   \
      const constant uint4& ks   [[buffer(15)]],                   \
      const constant uint4& vs   [[buffer(16)]],                   \
      const constant uint4& os   [[buffer(17)]],                   \
      const constant uint4& dos  [[buffer(18)]],                   \
      const constant uint4& dqs  [[buffer(19)]],                   \
      uint3 tgid [[threadgroup_position_in_grid]],                 \
      uint  tid  [[thread_index_in_threadgroup]]);

#define INST_FLASH_BWD_DKDV(T, D) \
  template [[host_name("flash_attn_bwd_dkdv_" #T "_" #D)]] [[kernel]] \
  void flash_attn_bwd_dkdv<T, D>(                                  \
      const device T*       Q    [[buffer(0)]],                    \
      const device T*       K    [[buffer(1)]],                    \
      const device T*       V    [[buffer(2)]],                    \
      const device T*       O    [[buffer(3)]],                    \
      const device T*       dO   [[buffer(4)]],                    \
      const device float*   LSE  [[buffer(5)]],                    \
      const device float*   Dv   [[buffer(6)]],                    \
      device       T*       dK   [[buffer(7)]],                    \
      device       T*       dV   [[buffer(8)]],                    \
      const constant uint&  qL   [[buffer(9)]],                    \
      const constant uint&  kL   [[buffer(10)]],                   \
      const constant uint&  gqa  [[buffer(11)]],                   \
      const constant uint&  nh   [[buffer(12)]],                   \
      const constant float& sc   [[buffer(13)]],                   \
      const constant bool&  ic   [[buffer(14)]],                   \
      const constant uint4& qs   [[buffer(15)]],                   \
      const constant uint4& ks   [[buffer(16)]],                   \
      const constant uint4& vs   [[buffer(17)]],                   \
      const constant uint4& os   [[buffer(18)]],                   \
      const constant uint4& dos  [[buffer(19)]],                   \
      const constant uint4& dks  [[buffer(20)]],                   \
      const constant uint4& dvs  [[buffer(21)]],                   \
      uint3 tgid [[threadgroup_position_in_grid]],                 \
      uint  tid  [[thread_index_in_threadgroup]]);

#define INST_FLASH_ALL(T) \
  INST_FLASH_FWD(T, 64)     \
  INST_FLASH_FWD(T, 128)    \
  INST_FLASH_BWD_PRE(T, 64) \
  INST_FLASH_BWD_PRE(T, 128)\
  INST_FLASH_BWD_DQ(T, 64)  \
  INST_FLASH_BWD_DQ(T, 128) \
  INST_FLASH_BWD_DKDV(T, 64)  \
  INST_FLASH_BWD_DKDV(T, 128)

INST_FLASH_ALL(float)
INST_FLASH_ALL(half)
INST_FLASH_ALL(bfloat)
