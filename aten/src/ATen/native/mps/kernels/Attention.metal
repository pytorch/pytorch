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
// Forward + Backward (preprocess, dQ, dK+dV)
// Reference: Dao et al. 2022 (https://arxiv.org/abs/2205.14135)
//
// Thread organization (all kernels):
//   TG   = (32, BQ, 1)  —  32 SIMD lanes × BQ Q-rows (or K-rows) per TG
//   Grid = (B*H, ceil(qL/BQ), 1)
//
// Each SIMD group (row 0..BQ-1, 32 lanes) owns one Q-row.
// 32 threads collaborate via simd_sum() on every dot-product.
// Each thread holds EPL = D/32 elements of its row in registers (float).
//
// Threadgroup memory budget (32 KB guaranteed on all Apple Silicon):
//   K_smem + V_smem  =  BKV * D * sizeof(float) * 2  ≤  32 KB
//   → BKV = (D == 64) ? 64 : 32
//
//   D=64  : BKV=64  → ≤8 outer loop iterations for S=512  (was 32)
//   D=128 : BKV=32  → ≤16 outer loop iterations for S=512 (was 32)
//
// Note: BKV is sized for float32 (worst case); half/bfloat get the same
// BKV but only use half the smem budget — safe on all Apple GPUs.
//
// Stage-parameter convention: Metal rejects mixing uint3 + uint2 in
// explicit [[host_name]] instantiations, so all kernels use a single
// flat uint [[thread_index_in_threadgroup]]; lane and row are derived
// as (tid % 32) and (tid / 32) respectively.

// ── forward ──────────────────────────────────────────────────────────────────

template<typename T, int D>
[[kernel]] void flash_attn_fwd(
    const device T*       Q   [[buffer(0)]],
    const device T*       K   [[buffer(1)]],
    const device T*       V   [[buffer(2)]],
    device       T*       O   [[buffer(3)]],
    device       float*   LSE [[buffer(4)]],
    const constant uint&  qL  [[buffer(5)]],
    const constant uint&  kL  [[buffer(6)]],
    const constant uint&  gqa [[buffer(7)]],
    const constant uint&  nh  [[buffer(8)]],
    const constant float& sc  [[buffer(9)]],
    const constant bool&  ic  [[buffer(10)]],
    const constant uint4& qs  [[buffer(11)]],
    const constant uint4& ks  [[buffer(12)]],
    const constant uint4& vs  [[buffer(13)]],
    const constant uint4& os  [[buffer(14)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    constexpr int EPL = D / 32;
    constexpr int BQ  = 32;
    constexpr int BKV = (D == 64) ? 64 : 32;

    threadgroup float K_smem[BKV * D];
    threadgroup float V_smem[BKV * D];

    const uint lane    = tid % 32;
    const uint q_local = tid / 32;
    const uint bh      = tgid.x;
    const uint q_row   = tgid.y * BQ + q_local;
    const uint q_max   = tgid.y * BQ + BQ - 1;

    const uint b    = bh / nh;
    const uint h    = bh % nh;
    const uint kv_h = h / gqa;

    const device T* Q_ptr = Q + b * qs[0] + h    * qs[1];
    const device T* K_ptr = K + b * ks[0] + kv_h * ks[1];
    const device T* V_ptr = V + b * vs[0] + kv_h * vs[1];
    device       T* O_ptr = O + b * os[0] + h    * os[1];

    const bool valid_q = (q_row < qL);

    float q_reg[EPL];
    for (int e = 0; e < EPL; e++)
        q_reg[e] = valid_q ? float(Q_ptr[q_row * qs[2] + lane * EPL + e]) : 0.0f;

    float acc[EPL] = {};
    float m = -INFINITY, l = 0.0f;

    const uint tg_size = 32 * BQ;  // 1024

    for (uint kb = 0; kb < kL; kb += BKV) {
        if (ic && kb > q_max) break;

        for (uint i = tid; i < (uint)(BKV * D); i += tg_size) {
            uint r = kb + i / D;
            uint d = i % D;
            bool in = (r < kL);
            K_smem[i] = in ? float(K_ptr[r * ks[2] + d]) : 0.0f;
            V_smem[i] = in ? float(V_ptr[r * vs[2] + d]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint tile_end = min(kb + (uint)BKV, kL);

        for (uint k_row = kb; k_row < tile_end; ++k_row) {
            const bool cv = !ic || (k_row <= q_row);
            int j = (int)(k_row - kb);

            float partial = 0.0f;
            for (int e = 0; e < EPL; e++)
                partial += q_reg[e] * K_smem[j * D + lane * EPL + e];
            float score = cv ? (simd_sum(partial) * sc) : -INFINITY;

            float m_new = max(m, score);
            float alpha = metal::precise::exp(m - m_new);
            float p_j   = metal::precise::exp(score - m_new);
            m = m_new;
            l = l * alpha + p_j;

            for (int e = 0; e < EPL; e++)
                acc[e] = acc[e] * alpha + p_j * V_smem[j * D + lane * EPL + e];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!valid_q) return;

    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int e = 0; e < EPL; e++)
        O_ptr[q_row * os[2] + lane * EPL + e] = T(acc[e] * inv_l);
    if (lane == 0)
        LSE[bh * qL + q_row] = m + log(l);
}

// ── backward preprocess ───────────────────────────────────────────────────────
// D_vec[i] = rowsum(dO_i * O_i).  Pure register reduction via simd_sum().
// Grid  : (B*H, ceil(qL/BQ), 1),  TG : (32, BQ, 1)

template<typename T, int D>
[[kernel]] void flash_attn_bwd_preprocess(
    const device T*       dO  [[buffer(0)]],
    const device T*       O   [[buffer(1)]],
    device       float*   Dv  [[buffer(2)]],
    const constant uint&  qL  [[buffer(3)]],
    const constant uint&  nh  [[buffer(4)]],
    const constant uint4& dos [[buffer(5)]],
    const constant uint4& os  [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    constexpr int EPL = D / 32;
    constexpr int BQ  = 32;

    const uint lane    = tid % 32;
    const uint q_local = tid / 32;
    const uint bh      = tgid.x;
    const uint q_row   = tgid.y * BQ + q_local;

    if (q_row >= qL) return;

    const uint b = bh / nh;
    const uint h = bh % nh;

    const device T* dO_ptr = dO + b * dos[0] + h * dos[1];
    const device T* O_ptr  = O  + b * os[0]  + h * os[1];

    float partial = 0.0f;
    for (int e = 0; e < EPL; e++)
        partial += float(dO_ptr[q_row * dos[2] + lane * EPL + e])
                 * float(O_ptr[q_row *  os[2]  + lane * EPL + e]);

    if (lane == 0)
        Dv[bh * qL + q_row] = simd_sum(partial);
}

// ── backward dQ ──────────────────────────────────────────────────────────────
// Recomputes attention weights from saved LSE; accumulates dQ.
// Same smem strategy as forward: K+V both in threadgroup memory.
// Grid  : (B*H, ceil(qL/BQ), 1),  TG : (32, BQ, 1)

template<typename T, int D>
[[kernel]] void flash_attn_bwd_dq(
    const device T*       Q   [[buffer(0)]],
    const device T*       K   [[buffer(1)]],
    const device T*       V   [[buffer(2)]],
    const device T*       O   [[buffer(3)]],
    const device T*       dO  [[buffer(4)]],
    const device float*   LSE [[buffer(5)]],
    const device float*   Dv  [[buffer(6)]],
    device       T*       dQ  [[buffer(7)]],
    const constant uint&  qL  [[buffer(8)]],
    const constant uint&  kL  [[buffer(9)]],
    const constant uint&  gqa [[buffer(10)]],
    const constant uint&  nh  [[buffer(11)]],
    const constant float& sc  [[buffer(12)]],
    const constant bool&  ic  [[buffer(13)]],
    const constant uint4& qs  [[buffer(14)]],
    const constant uint4& ks  [[buffer(15)]],
    const constant uint4& vs  [[buffer(16)]],
    const constant uint4& os  [[buffer(17)]],
    const constant uint4& dos [[buffer(18)]],
    const constant uint4& dqs [[buffer(19)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    constexpr int EPL = D / 32;
    constexpr int BQ  = 32;
    constexpr int BKV = (D == 64) ? 64 : 32;

    threadgroup float K_smem[BKV * D];
    threadgroup float V_smem[BKV * D];

    const uint lane    = tid % 32;
    const uint q_local = tid / 32;
    const uint bh      = tgid.x;
    const uint q_row   = tgid.y * BQ + q_local;
    const uint q_max   = tgid.y * BQ + BQ - 1;

    const uint b    = bh / nh;
    const uint h    = bh % nh;
    const uint kv_h = h / gqa;

    const device T*  Q_ptr  = Q  + b * qs[0]  + h    * qs[1];
    const device T*  K_ptr  = K  + b * ks[0]  + kv_h * ks[1];
    const device T*  V_ptr  = V  + b * vs[0]  + kv_h * vs[1];
    const device T*  dO_ptr = dO + b * dos[0] + h    * dos[1];
    device       T*  dQ_ptr = dQ + b * dqs[0] + h    * dqs[1];

    const bool valid_q = (q_row < qL);

    float q_reg[EPL]  = {};
    float do_reg[EPL] = {};
    float dq_acc[EPL] = {};
    float lse_val = 0.0f, d_vec = 0.0f;

    if (valid_q) {
        for (int e = 0; e < EPL; e++) {
            q_reg[e]  = float(Q_ptr[q_row  * qs[2]  + lane * EPL + e]);
            do_reg[e] = float(dO_ptr[q_row * dos[2] + lane * EPL + e]);
        }
        lse_val = LSE[bh * qL + q_row];
        d_vec   = Dv[bh * qL + q_row];
    }

    const uint tg_size = 32 * BQ;

    for (uint kb = 0; kb < kL; kb += BKV) {
        if (ic && kb > q_max) break;

        for (uint i = tid; i < (uint)(BKV * D); i += tg_size) {
            uint r = kb + i / D;
            uint d = i % D;
            bool in = (r < kL);
            K_smem[i] = in ? float(K_ptr[r * ks[2] + d]) : 0.0f;
            V_smem[i] = in ? float(V_ptr[r * vs[2] + d]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint tile_end = min(kb + (uint)BKV, kL);

        for (uint k_row = kb; k_row < tile_end; ++k_row) {
            const bool cv = !ic || (k_row <= q_row);
            int j = (int)(k_row - kb);

            float partial = 0.0f;
            for (int e = 0; e < EPL; e++)
                partial += q_reg[e] * K_smem[j * D + lane * EPL + e];
            float score = cv ? (simd_sum(partial) * sc) : -INFINITY;
            float p_ij  = metal::precise::exp(score - lse_val);
            if (!valid_q) p_ij = 0.0f;

            float dov = 0.0f;
            for (int e = 0; e < EPL; e++)
                dov += do_reg[e] * V_smem[j * D + lane * EPL + e];
            float ds_ij = p_ij * (simd_sum(dov) - d_vec);

            for (int e = 0; e < EPL; e++)
                dq_acc[e] += ds_ij * K_smem[j * D + lane * EPL + e];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!valid_q) return;

    for (int e = 0; e < EPL; e++)
        dQ_ptr[q_row * dqs[2] + lane * EPL + e] = T(dq_acc[e] * sc);
}

// ── backward dK + dV ─────────────────────────────────────────────────────────
// K and V live in per-simdgroup registers.  Q and dO are tiled through smem.
// Grid  : (B*H, ceil(kL/BK), 1),  TG : (32, BK, 1)
//
// Smem budget: Q_smem + dO_smem = BQS * D * sizeof(float) * 2 ≤ 32 KB
// → BQS = (D == 64) ? 64 : 32

template<typename T, int D>
[[kernel]] void flash_attn_bwd_dkdv(
    const device T*       Q   [[buffer(0)]],
    const device T*       K   [[buffer(1)]],
    const device T*       V   [[buffer(2)]],
    const device T*       O   [[buffer(3)]],   // unused, kept for dispatch compat
    const device T*       dO  [[buffer(4)]],
    const device float*   LSE [[buffer(5)]],
    const device float*   Dv  [[buffer(6)]],
    device       T*       dK  [[buffer(7)]],
    device       T*       dV  [[buffer(8)]],
    const constant uint&  qL  [[buffer(9)]],
    const constant uint&  kL  [[buffer(10)]],
    const constant uint&  gqa [[buffer(11)]],
    const constant uint&  nh  [[buffer(12)]],
    const constant float& sc  [[buffer(13)]],
    const constant bool&  ic  [[buffer(14)]],
    const constant uint4& qs  [[buffer(15)]],
    const constant uint4& ks  [[buffer(16)]],
    const constant uint4& vs  [[buffer(17)]],
    const constant uint4& os  [[buffer(18)]],
    const constant uint4& dos [[buffer(19)]],
    const constant uint4& dks [[buffer(20)]],
    const constant uint4& dvs [[buffer(21)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    constexpr int EPL  = D / 32;
    constexpr int BK   = 32;
    constexpr int BQS  = (D == 64) ? 64 : 32;

    threadgroup float  Q_smem[BQS * D];
    threadgroup float dO_smem[BQS * D];

    const uint lane    = tid % 32;
    const uint k_local = tid / 32;
    const uint bh      = tgid.x;
    const uint k_row   = tgid.y * BK + k_local;
    const uint k_min   = tgid.y * BK;

    // bh encodes (batch, kv_head): bh = b * nh + kv_h  (nh == kvH here)
    const uint kv_h = bh % nh;
    const uint b    = bh / nh;

    const device T*  K_ptr  = K  + b * ks[0]  + kv_h * ks[1];
    const device T*  V_ptr  = V  + b * vs[0]  + kv_h * vs[1];
    device       T*  dK_ptr = dK + b * dks[0] + kv_h * dks[1];
    device       T*  dV_ptr = dV + b * dvs[0] + kv_h * dvs[1];

    const bool valid_k = (k_row < kL);

    float k_reg[EPL] = {};
    float v_reg[EPL] = {};
    if (valid_k) {
        for (int e = 0; e < EPL; e++) {
            k_reg[e] = float(K_ptr[k_row * ks[2] + lane * EPL + e]);
            v_reg[e] = float(V_ptr[k_row * vs[2] + lane * EPL + e]);
        }
    }

    float dk_acc[EPL] = {};
    float dv_acc[EPL] = {};

    const uint tg_size = 32 * BK;

    for (uint g = 0; g < gqa; g++) {
        const uint q_head = kv_h * gqa + g;
        const device T* Q_ptr  = Q  + b * qs[0]  + q_head * qs[1];
        const device T* dO_ptr = dO + b * dos[0] + q_head * dos[1];
        const uint bh_lse = b * (nh * gqa) + q_head;

    for (uint qb = 0; qb < qL; qb += BQS) {
        if (ic && qb + (uint)BQS - 1 < k_min) continue;

        for (uint i = tid; i < (uint)(BQS * D); i += tg_size) {
            uint r = qb + i / D;
            uint d = i % D;
            bool in = (r < qL);
            Q_smem[i]  = in ? float( Q_ptr[r * qs[2]  + d]) : 0.0f;
            dO_smem[i] = in ? float(dO_ptr[r * dos[2] + d]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint tile_end = min(qb + (uint)BQS, qL);

        for (uint q_row = qb; q_row < tile_end; ++q_row) {
            if (ic && k_row > q_row) continue;

            float lse_i   = LSE[bh_lse * qL + q_row];
            float d_vec_i = Dv[bh_lse * qL + q_row];
            int i = (int)(q_row - qb);

            float qk = 0.0f;
            for (int e = 0; e < EPL; e++)
                qk += Q_smem[i * D + lane * EPL + e] * k_reg[e];
            float p_ij = metal::precise::exp(simd_sum(qk) * sc - lse_i);
            if (!valid_k) p_ij = 0.0f;

            float dov = 0.0f;
            for (int e = 0; e < EPL; e++)
                dov += dO_smem[i * D + lane * EPL + e] * v_reg[e];
            float ds_ij = p_ij * (simd_sum(dov) - d_vec_i);

            for (int e = 0; e < EPL; e++)
                dv_acc[e] += p_ij * dO_smem[i * D + lane * EPL + e];

            for (int e = 0; e < EPL; e++)
                dk_acc[e] += ds_ij * Q_smem[i * D + lane * EPL + e];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    } // end gqa g-loop

    if (!valid_k) return;

    for (int e = 0; e < EPL; e++) {
        dK_ptr[k_row * dks[2] + lane * EPL + e] = T(dk_acc[e] * sc);
        dV_ptr[k_row * dvs[2] + lane * EPL + e] = T(dv_acc[e]);
    }
}

// ── explicit instantiation macros ────────────────────────────────────────────

#define INST_FLASH_FWD(T, D) \
  template [[host_name("flash_attn_fwd_" #T "_" #D)]] [[kernel]] \
  void flash_attn_fwd<T, D>( \
      const device T*        Q   [[buffer(0)]],  \
      const device T*        K   [[buffer(1)]],  \
      const device T*        V   [[buffer(2)]],  \
      device       T*        O   [[buffer(3)]],  \
      device       float*    LSE [[buffer(4)]],  \
      const constant uint&   qL  [[buffer(5)]],  \
      const constant uint&   kL  [[buffer(6)]],  \
      const constant uint&   gqa [[buffer(7)]],  \
      const constant uint&   nh  [[buffer(8)]],  \
      const constant float&  sc  [[buffer(9)]],  \
      const constant bool&   ic  [[buffer(10)]], \
      const constant uint4&  qs  [[buffer(11)]], \
      const constant uint4&  ks  [[buffer(12)]], \
      const constant uint4&  vs  [[buffer(13)]], \
      const constant uint4&  os  [[buffer(14)]], \
      uint3 tgid [[threadgroup_position_in_grid]], \
      uint  tid  [[thread_index_in_threadgroup]]);

#define INST_FLASH_BWD_PRE(T, D) \
  template [[host_name("flash_attn_bwd_pre_" #T "_" #D)]] [[kernel]] \
  void flash_attn_bwd_preprocess<T, D>( \
      const device T*        dO  [[buffer(0)]], \
      const device T*        O   [[buffer(1)]], \
      device       float*    Dv  [[buffer(2)]], \
      const constant uint&   qL  [[buffer(3)]], \
      const constant uint&   nh  [[buffer(4)]], \
      const constant uint4&  dos [[buffer(5)]], \
      const constant uint4&  os  [[buffer(6)]], \
      uint3 tgid [[threadgroup_position_in_grid]], \
      uint  tid  [[thread_index_in_threadgroup]]);

#define INST_FLASH_BWD_DQ(T, D) \
  template [[host_name("flash_attn_bwd_dq_" #T "_" #D)]] [[kernel]] \
  void flash_attn_bwd_dq<T, D>( \
      const device T*        Q   [[buffer(0)]],  \
      const device T*        K   [[buffer(1)]],  \
      const device T*        V   [[buffer(2)]],  \
      const device T*        O   [[buffer(3)]],  \
      const device T*        dO  [[buffer(4)]],  \
      const device float*    LSE [[buffer(5)]],  \
      const device float*    Dv  [[buffer(6)]],  \
      device       T*        dQ  [[buffer(7)]],  \
      const constant uint&   qL  [[buffer(8)]],  \
      const constant uint&   kL  [[buffer(9)]],  \
      const constant uint&   gqa [[buffer(10)]], \
      const constant uint&   nh  [[buffer(11)]], \
      const constant float&  sc  [[buffer(12)]], \
      const constant bool&   ic  [[buffer(13)]], \
      const constant uint4&  qs  [[buffer(14)]], \
      const constant uint4&  ks  [[buffer(15)]], \
      const constant uint4&  vs  [[buffer(16)]], \
      const constant uint4&  os  [[buffer(17)]], \
      const constant uint4&  dos [[buffer(18)]], \
      const constant uint4&  dqs [[buffer(19)]], \
      uint3 tgid [[threadgroup_position_in_grid]], \
      uint  tid  [[thread_index_in_threadgroup]]);

#define INST_FLASH_BWD_DKDV(T, D) \
  template [[host_name("flash_attn_bwd_dkdv_" #T "_" #D)]] [[kernel]] \
  void flash_attn_bwd_dkdv<T, D>( \
      const device T*        Q   [[buffer(0)]],  \
      const device T*        K   [[buffer(1)]],  \
      const device T*        V   [[buffer(2)]],  \
      const device T*        O   [[buffer(3)]],  \
      const device T*        dO  [[buffer(4)]],  \
      const device float*    LSE [[buffer(5)]],  \
      const device float*    Dv  [[buffer(6)]],  \
      device       T*        dK  [[buffer(7)]],  \
      device       T*        dV  [[buffer(8)]],  \
      const constant uint&   qL  [[buffer(9)]],  \
      const constant uint&   kL  [[buffer(10)]], \
      const constant uint&   gqa [[buffer(11)]], \
      const constant uint&   nh  [[buffer(12)]], \
      const constant float&  sc  [[buffer(13)]], \
      const constant bool&   ic  [[buffer(14)]], \
      const constant uint4&  qs  [[buffer(15)]], \
      const constant uint4&  ks  [[buffer(16)]], \
      const constant uint4&  vs  [[buffer(17)]], \
      const constant uint4&  os  [[buffer(18)]], \
      const constant uint4&  dos [[buffer(19)]], \
      const constant uint4&  dks [[buffer(20)]], \
      const constant uint4&  dvs [[buffer(21)]], \
      uint3 tgid [[threadgroup_position_in_grid]], \
      uint  tid  [[thread_index_in_threadgroup]]);

#define INST_FLASH_ALL(T) \
  INST_FLASH_FWD(T, 64)      \
  INST_FLASH_FWD(T, 128)     \
  INST_FLASH_BWD_PRE(T, 64)  \
  INST_FLASH_BWD_PRE(T, 128) \
  INST_FLASH_BWD_DQ(T, 64)   \
  INST_FLASH_BWD_DQ(T, 128)  \
  INST_FLASH_BWD_DKDV(T, 64) \
  INST_FLASH_BWD_DKDV(T, 128)

INST_FLASH_ALL(float)
INST_FLASH_ALL(half)
INST_FLASH_ALL(bfloat)
