// Largely influeneced by
// https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/scaled_dot_product_attention.metal
#include <ATen/native/mps/kernels/Attention.h>
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
  const uint k_head_stride = qkv_head_strides.y;
  const uint k_seq_stride = qkv_seq_strides.y;
  const uint v_head_stride = qkv_head_strides.z;
  const uint v_seq_stride = qkv_seq_strides.z;
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

  // Adjust positions
  const int head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = head_idx / gqa_factor;
  const int Q = tpg.y;
  const int group_offset = head_idx * Q + q_seq_idx;
  const int o_offset = group_offset;
  queries += head_idx * q_head_stride + q_seq_idx * q_seq_stride +
      simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  if (has_mask) {
    mask += head_idx * mask_head_stride + simd_gid * mask_kv_seq_stride +
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
  const int k_head_stride = qkv_head_strides.y;
  const int k_seq_stride = qkv_seq_strides.y;
  const int v_head_stride = qkv_head_strides.z;
  const int v_seq_stride = qkv_seq_strides.z;
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

  // Adjust positions
  const int block_idx = tid.z;
  const int head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int o_offset = head_idx * tpg.y + q_seq_idx;
  const int kv_head_idx = head_idx / gqa_factor;

  queries += head_idx * q_head_stride + q_seq_idx * q_seq_stride +
      simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride +
      (block_idx * BN + simd_gid) * k_seq_stride + simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride +
      (block_idx * BN + simd_gid) * v_seq_stride + simd_lid * v_per_thread;
  out += o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
  if (has_mask) {
    mask += head_idx * mask_head_stride +
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
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

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
    outputs[simd_lid * BN + simd_gid] =
        o[i] * fast::exp(max_scores[simd_gid] - new_max);
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

// simdgroup_multiply_accumulate operates on 8x8 sub-tiles
#define SUBTILE_SIZE 8
// number of 8-wide sub-tiles across one tile dimension
#define SUBTILE_GRID_SIZE (TILE_SIZE / SUBTILE_SIZE)
// Threadgroup memory needed: two TILE_SIZExTILE_SIZE buffers. For 32x32 tiles,
// this is 8 KB
#define SMEM_FLOATS (2 * TILE_SIZE * TILE_SIZE)

// Scaled matrix multiplication `r = scale * a @ b`.
// `a` is size (M, N)
// `b` is size (N, K)
// `r` is size (M, K)
//
// For performance, this function uses a tiled matmul algorithm where each
// threadgroup operates on one pair of TILE_SIZExTILE_SIZE tiles of the inputs
// at a time, so that work can be done in `threadgroup` memory, which is faster
// than `device` or `constant` memory.
//
// Each pair of threadgroup tiles is further broken up into 8x8 sub-tiles, whose
// partial results are calculated with `simdgroup_multiply_accumulate`, which is
// a performant way to calculate a matmul between two 8x8 matrices and
// accumulate with previous results.
//
// Note: the pointer type for `a` is templated because both `constant` and
// `device` pointer types need to be supported for this argument.
template <typename T, typename A_ptr>
static void mm_simdgroup(
    device T* r,
    uint32_t r_stride0,
    uint32_t r_stride1,
    A_ptr a,
    uint32_t a_stride0,
    uint32_t a_stride1,
    constant T* b,
    uint32_t b_stride0,
    uint32_t b_stride1,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    float scale,
    uint32_t threadgroup_row,
    uint32_t threadgroup_col,
    uint simdgroup_idx,
    threadgroup float* smem) {
  threadgroup float* tile_a = smem;
  threadgroup float* tile_b = smem + TILE_SIZE * TILE_SIZE;

  const uint32_t num_M = (M + TILE_SIZE - 1) / TILE_SIZE;
  const uint32_t num_K = (K + TILE_SIZE - 1) / TILE_SIZE;
  const uint32_t num_N = (N + TILE_SIZE - 1) / TILE_SIZE;

  // simdgroup_idx 0..15 are active, one for each 4x4 grid position of 8x8
  // output sub-tiles.
  // simdgroup_idx 16..31 are inactive of the simd multiply-accumulate, but they
  // still help load tile data.
  const bool simd_active =
      (simdgroup_idx < (uint32_t)(SUBTILE_GRID_SIZE * SUBTILE_GRID_SIZE));
  const uint32_t subtile_m =
      simdgroup_idx / SUBTILE_GRID_SIZE; // 0..3 are active
  const uint32_t subtile_k =
      simdgroup_idx % SUBTILE_GRID_SIZE; // 0..3 are active

  for (uint32_t tile_m = 0; tile_m < num_M; tile_m++) {
    for (uint32_t tile_k = 0; tile_k < num_K; tile_k++) {
      simdgroup_float8x8 subtile_r =
          make_filled_simdgroup_matrix<float, 8, 8>(0.f);

      for (uint32_t tile_n = 0; tile_n < num_N; tile_n++) {
        // All 1024 threads cooperatively fill tile_a and tile_b
        uint32_t a_row = tile_m * TILE_SIZE + threadgroup_row,
                 a_col = tile_n * TILE_SIZE + threadgroup_col;
        tile_a[threadgroup_row * TILE_SIZE + threadgroup_col] =
            (a_row < M && a_col < N)
            ? float(a[a_row * a_stride0 + a_col * a_stride1])
            : 0.f;
        uint32_t b_row = tile_n * TILE_SIZE + threadgroup_row,
                 b_col = tile_k * TILE_SIZE + threadgroup_col;
        tile_b[threadgroup_row * TILE_SIZE + threadgroup_col] =
            (b_row < N && b_col < K)
            ? float(b[b_row * b_stride0 + b_col * b_stride1])
            : 0.f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Active simdgroups compute their 8x8 sub-tile and accumulate into
        // subtile_r.
        if (simd_active) {
          for (uint32_t subtile_n = 0; subtile_n < (uint32_t)SUBTILE_GRID_SIZE;
               subtile_n++) {
            simdgroup_float8x8 subtile_a, subtile_b;
            // subtile_a <-- tile_a[subtile_m*8 .. subtile_m*8+7][subtile_n*8
            // .. subtile_n*8+7]
            simdgroup_load(
                subtile_a,
                tile_a,
                TILE_SIZE,
                ulong2(subtile_n * SUBTILE_SIZE, subtile_m * SUBTILE_SIZE));
            // subtile_b <-- tile_b[subtile_n*8 .. subtile_n*8+7][subtile_k*8 ..
            // subtile_k*8+7]
            simdgroup_load(
                subtile_b,
                tile_b,
                TILE_SIZE,
                ulong2(subtile_k * SUBTILE_SIZE, subtile_n * SUBTILE_SIZE));
            simdgroup_multiply_accumulate(
                subtile_r, subtile_a, subtile_b, subtile_r);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }

      // Scatter result back through smem, then write to r. The active
      // simdgroups cover all sub-tiles of the output tile, so every position in
      // tile_a is written.
      if (simd_active) {
        simdgroup_store(
            subtile_r,
            tile_a,
            TILE_SIZE,
            ulong2(subtile_k * SUBTILE_SIZE, subtile_m * SUBTILE_SIZE));
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      uint32_t r_row = tile_m * TILE_SIZE + threadgroup_row;
      uint32_t r_col = tile_k * TILE_SIZE + threadgroup_col;
      if (r_row < M && r_col < K) {
        r[r_row * r_stride0 + r_col * r_stride1] =
            T(tile_a[threadgroup_row * TILE_SIZE + threadgroup_col] * scale);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}

// Find the batch and head offset of `mask` only if the mask is enabled for the
// kernel.
struct MaskBatchOffset {
  template <typename T_MASK>
  inline static constant T_MASK* apply(
      constant T_MASK* mask,
      uint32_t stride0,
      uint32_t stride1,
      uint32_t batch_idx,
      uint32_t head_idx) {
    return mask + stride0 * batch_idx + stride1 * head_idx;
  }

  template <>
  inline constant void* apply(
      constant void* mask,
      uint32_t stride0,
      uint32_t stride1,
      uint32_t batch_idx,
      uint32_t head_idx) {
    return mask;
  }
};

// Apply the attention mask if enabled for the kernel. The mask can either be a
// float or a bool. void indicates that the mask is disabled.
struct AttnMask {
  template <typename T_MASK>
  inline static float apply(float value, constant T_MASK* mask, uint32_t idx) {
    auto masked_value = value + mask[idx];
    return ::metal::isnan(masked_value) ? -INFINITY : masked_value;
  }

  template <>
  inline float apply(float value, constant bool* mask, uint32_t idx) {
    return mask[idx] ? value : -INFINITY;
  }

  template <>
  inline float apply(float value, constant void* mask, uint32_t idx) {
    return value;
  }
};

// Apply the causal mask if enabled for the kernel. If enabled, the upper right
// elements are masked out.
struct CausalMask {
  template <bool is_causal, enable_if_t<is_causal, bool> = true>
  inline static float apply(float value, uint32_t row, uint32_t col) {
    return (col <= row) ? value : -INFINITY;
  }

  template <bool is_causal, enable_if_t<!is_causal, bool> = true>
  inline static float apply(float value, uint32_t row, uint32_t col) {
    return value;
  }
};

// Load a value from `attn` and apply masks
template <typename T, typename T_MASK, bool is_causal>
inline float load_attn_value(
    device T* attn,
    uint32_t attn_stride0,
    uint32_t attn_stride1,
    constant T_MASK* mask,
    uint32_t mask_stride0,
    uint32_t mask_stride1,
    uint32_t row,
    uint32_t col) {
  auto attn_idx = row * attn_stride0 + col * attn_stride1;
  auto mask_idx = row * mask_stride0 + col * mask_stride1;
  return CausalMask::apply<is_causal>(
      AttnMask::apply(static_cast<float>(attn[attn_idx]), mask, mask_idx),
      row,
      col);
}

// In-place softmax `attn = softmax(attn, dim=-1)`.
// `attn` is size (L, S)
//
// Within each row of the input, the following steps are performed:
//  1) Find `row_max`, the maximum value in the row
//  2) Find `row_sum`, sum of `exp(value - row_max)` for each value.
//  3) Write normalized `exp(value - row_max) / row_sum` to each value in place.
//
// The `mask` is applied when values are read from `attn`.
//
// For performance, the input is broken up into TILE_SIZE x TILE_SIZE tiles.
// During step 1 and 2, each threadgroup operates on one tile of the input at a
// time, so that the reduction work can be performed in threadgroup memory.
// First, each thread in the threadgroup accumulates the max/sum of the values
// in its assigned position in the tile and writes it into its spot in
// threadgroup memory. Then, a binary reduction is performed on the rows of the
// tile.
template <typename T, typename T_MASK, bool is_causal>
static void softmax_rows(
    device T* attn,
    uint32_t attn_stride0,
    uint32_t attn_stride1,
    constant T_MASK* mask,
    uint32_t mask_stride0,
    uint32_t mask_stride1,
    uint32_t L,
    uint32_t S,
    uint32_t threadgroup_row,
    uint32_t threadgroup_col,
    threadgroup float* smem) {
  const uint32_t num_tile_rows = (L + TILE_SIZE - 1) / TILE_SIZE;

  // Iterate over each row of tiles.
  for (uint32_t tile_row_idx = 0; tile_row_idx < num_tile_rows;
       tile_row_idx++) {
    const uint32_t row = tile_row_idx * TILE_SIZE + threadgroup_row;
    const bool valid = row < L;

    // Step 1- Find the max value in each row
    float local_max = -INFINITY;
    if (valid) {
      for (uint32_t col = threadgroup_col; col < S; col += TILE_SIZE) {
        float value = load_attn_value<T, T_MASK, is_causal>(
            attn,
            attn_stride0,
            attn_stride1,
            mask,
            mask_stride0,
            mask_stride1,
            row,
            col);
        local_max = max(local_max, value);
      }
    }
    smem[threadgroup_row * TILE_SIZE + threadgroup_col] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Reduce the partial max values in threadgroup memory
    for (uint32_t stride = TILE_SIZE / 2; stride > 0; stride >>= 1) {
      if (threadgroup_col < stride)
        smem[threadgroup_row * TILE_SIZE + threadgroup_col] =
            max(smem[threadgroup_row * TILE_SIZE + threadgroup_col],
                smem[threadgroup_row * TILE_SIZE + threadgroup_col + stride]);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float row_max = smem[threadgroup_row * TILE_SIZE];

    // Step 2 - Find the exp sum in each row
    float local_sum = 0.f;
    if (valid) {
      for (uint32_t col = threadgroup_col; col < S; col += TILE_SIZE) {
        float value = load_attn_value<T, T_MASK, is_causal>(
            attn,
            attn_stride0,
            attn_stride1,
            mask,
            mask_stride0,
            mask_stride1,
            row,
            col);
        float e = precise::exp(value - row_max);
        local_sum += ::metal::isnan(e) ? 0 : e;
      }
    }
    smem[threadgroup_row * TILE_SIZE + threadgroup_col] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Reduce the partial sum values in threadgroup memory
    for (uint32_t stride = TILE_SIZE / 2; stride > 0; stride >>= 1) {
      if (threadgroup_col < stride)
        smem[threadgroup_row * TILE_SIZE + threadgroup_col] +=
            smem[threadgroup_row * TILE_SIZE + threadgroup_col + stride];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float row_sum = smem[threadgroup_row * TILE_SIZE];

    // Step 3 - Normalize in place
    if (valid) {
      for (uint32_t col = threadgroup_col; col < S; col += TILE_SIZE) {
        float value = load_attn_value<T, T_MASK, is_causal>(
            attn,
            attn_stride0,
            attn_stride1,
            mask,
            mask_stride0,
            mask_stride1,
            row,
            col);
        float e = precise::exp(value - row_max);
        auto attn_idx = row * attn_stride0 + col * attn_stride1;
        attn[attn_idx] = row_sum == 0 ? 0 : T(e / row_sum);
      }
    }
  }
}

template <typename T, typename T_MASK, bool is_causal>
kernel void sdpa(
    device T* out [[buffer(0)]],
    device T* attn [[buffer(1)]],
    constant T* q [[buffer(2)]],
    constant T* k [[buffer(3)]],
    constant T* v [[buffer(4)]],
    constant T_MASK* mask [[buffer(5)]],
    constant SDPAParams<>& params [[buffer(6)]],
    uint2 lid [[thread_position_in_threadgroup]], // (x=col, y=row)
    uint2 tgid [[threadgroup_position_in_grid]], // one group per (batch, head)
    uint simdgroup_idx [[simdgroup_index_in_threadgroup]] // 0..31
) {
  threadgroup float smem[SMEM_FLOATS];

  // Find the batch and head offsets of each of the inputs and outputs
  const uint32_t head_idx = tgid.x % params.num_heads;
  const uint32_t batch_idx = tgid.x / params.num_heads;
  out += params.out_strides[0] * batch_idx + params.out_strides[1] * head_idx;
  attn +=
      params.attn_strides[0] * batch_idx + params.attn_strides[1] * head_idx;
  q += params.q_strides[0] * batch_idx + params.q_strides[1] * head_idx;
  k += params.k_strides[0] * batch_idx + params.k_strides[1] * head_idx;
  v += params.v_strides[0] * batch_idx + params.v_strides[1] * head_idx;
  mask = MaskBatchOffset::apply(
      mask,
      params.mask_strides[0],
      params.mask_strides[1],
      batch_idx,
      head_idx);

  const uint32_t threadgroup_row = lid.y, threadgroup_col = lid.x;

  // Matmul `q`, size (L, E), and `k^T`, size (E, S), then multiply by `scale`,
  // and write output to `attn`.
  mm_simdgroup<T, constant T*>(
      attn,
      params.attn_strides[2],
      params.attn_strides[3],
      q,
      params.q_strides[2],
      params.q_strides[3],
      k,
      // k needs to be transposed, which is accomplished by swapping the strides
      params.k_strides[3],
      params.k_strides[2],
      params.L,
      params.E,
      params.S,
      params.scale,
      threadgroup_row,
      threadgroup_col,
      simdgroup_idx,
      smem);

  threadgroup_barrier(mem_flags::mem_device);

  // Perform softmax to `attn` in-place.
  softmax_rows<T, T_MASK, is_causal>(
      attn,
      params.attn_strides[2],
      params.attn_strides[3],
      mask,
      params.mask_strides[2],
      params.mask_strides[3],
      params.L,
      params.S,
      threadgroup_row,
      threadgroup_col,
      smem);

  threadgroup_barrier(mem_flags::mem_device);

  // Matmul `attn`, size (L, S), and `v`, size (S, Ev), and write output to
  // `out`.
  mm_simdgroup<T, device T*>(
      out,
      params.out_strides[2],
      params.out_strides[3],
      attn,
      params.attn_strides[2],
      params.attn_strides[3],
      v,
      params.v_strides[2],
      params.v_strides[3],
      params.L,
      params.S,
      params.Ev,
      /*scale=*/1.f,
      threadgroup_row,
      threadgroup_col,
      simdgroup_idx,
      smem);
}

#define INSTANTIATE_SDPA_VECTOR(DTYPE, QK_DIM, VALUE_DIM)   \
  template [[host_name("sdpa_vector_" #DTYPE "_" #QK_DIM    \
                       "_" #VALUE_DIM)]] kernel void        \
  sdpa_vector<DTYPE, QK_DIM, VALUE_DIM>(                    \
      const device DTYPE* queries [[buffer(0)]],            \
      const device DTYPE* keys [[buffer(1)]],               \
      const device DTYPE* values [[buffer(2)]],             \
      device DTYPE* out [[buffer(3)]],                      \
      const constant uint& gqa_factor [[buffer(4)]],        \
      const constant uint& N [[buffer(5)]],                 \
      const constant uint3& qkv_head_strides [[buffer(6)]], \
      const constant uint3& qkv_seq_strides [[buffer(7)]],  \
      const constant float& scale [[buffer(8)]],            \
      const device bool* mask [[buffer(9)]],                \
      const constant uint3& mask_strides [[buffer(10)]],    \
      const constant bool& has_mask [[buffer(11)]],         \
      uint3 tid [[threadgroup_position_in_grid]],           \
      uint3 tpg [[threadgroups_per_grid]],                  \
      uint simd_gid [[simdgroup_index_in_threadgroup]],     \
      uint simd_lid [[thread_index_in_simdgroup]]);

#define INSTANTIATE_SDPA_VECTOR_2PASS_1(DTYPE, QK_DIM, VALUE_DIM) \
  template [[host_name("sdpa_vector_2pass_1_" #DTYPE "_" #QK_DIM  \
                       "_" #VALUE_DIM)]] kernel void              \
  sdpa_vector_2pass_1<DTYPE, QK_DIM, VALUE_DIM>(                  \
      const device DTYPE* queries [[buffer(0)]],                  \
      const device DTYPE* keys [[buffer(1)]],                     \
      const device DTYPE* values [[buffer(2)]],                   \
      device DTYPE* out [[buffer(3)]],                            \
      device float* sums [[buffer(4)]],                           \
      device float* maxs [[buffer(5)]],                           \
      const constant uint& gqa_factor [[buffer(6)]],              \
      const constant uint& N [[buffer(7)]],                       \
      const constant uint3& qkv_head_strides [[buffer(8)]],       \
      const constant uint3& qkv_seq_strides [[buffer(9)]],        \
      const constant float& scale [[buffer(10)]],                 \
      const device bool* mask [[buffer(11)]],                     \
      const constant uint3& mask_strides [[buffer(12)]],          \
      const constant bool& has_mask [[buffer(13)]],               \
      uint3 tid [[threadgroup_position_in_grid]],                 \
      uint3 tpg [[threadgroups_per_grid]],                        \
      uint simd_gid [[simdgroup_index_in_threadgroup]],           \
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

#define CAUSAL_SUFFIX_true "_causal"
#define CAUSAL_SUFFIX_false ""

#define REGISTER_SDPA(T, T_MASK, IS_CAUSAL)                                \
  template[[host_name("sdpa_" #T                                           \
                      "_" #T_MASK CAUSAL_SUFFIX_##IS_CAUSAL)]] kernel void \
  sdpa<T, T_MASK, IS_CAUSAL>(                                              \
      device T * out [[buffer(0)]],                                        \
      device T * attn [[buffer(1)]],                                       \
      constant T * q [[buffer(2)]],                                        \
      constant T * k [[buffer(3)]],                                        \
      constant T * v [[buffer(4)]],                                        \
      constant T_MASK * mask [[buffer(5)]],                                \
      constant SDPAParams<> & params [[buffer(6)]],                        \
      uint2 lid [[thread_position_in_threadgroup]],                        \
      uint2 tgid [[threadgroup_position_in_grid]],                         \
      uint simdgroup_idx [[simdgroup_index_in_threadgroup]]);

#define REGISTER_SDPA_MASK_TYPES(T) \
  REGISTER_SDPA(T, void, false);    \
  REGISTER_SDPA(T, void, true);     \
  REGISTER_SDPA(T, bool, false);    \
  REGISTER_SDPA(T, T, false);

REGISTER_SDPA_MASK_TYPES(float);
REGISTER_SDPA_MASK_TYPES(half);
REGISTER_SDPA_MASK_TYPES(bfloat);
