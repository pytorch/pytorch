#include <metal_atomic>
#include <metal_stdlib>
using namespace metal;

kernel void flatten_indices_kernel(
    device const int64_t* indices [[buffer(0)]],
    device const int64_t* strides [[buffer(1)]],
    device int64_t* flat_indices [[buffer(2)]],
    constant uint& sparse_dim [[buffer(3)]],
    constant uint& nnz [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  int64_t flat_idx = 0;
  for (uint d = 0; d < sparse_dim; d++) {
    flat_idx += indices[d * nnz + gid] * strides[d];
  }
  flat_indices[gid] = flat_idx;
}

kernel void compute_output_positions_kernel(
    device const bool* is_unique [[buffer(0)]],
    device int* positions [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  int pos = 0;
  for (uint i = 0; i < gid; i++) {
    if (is_unique[i])
      pos++;
  }
  positions[gid] = pos;
}

kernel void mark_unique_positions_and_count_kernel(
    device const int64_t* flat_indices [[buffer(0)]],
    device bool* is_unique [[buffer(1)]],
    device atomic_int* count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  bool unique = (tid == 0) || (flat_indices[tid] != flat_indices[tid - 1]);
  is_unique[tid] = unique;

  if (unique) {
    atomic_fetch_add_explicit(count, 1, memory_order_relaxed);
  }
}

// Kogge-Stone parallel prefix sum step
kernel void kogge_stone_step(
    device const int* input [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant uint& stride [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  int val = input[gid];
  if (gid >= stride) {
    val += input[gid - stride];
  }
  output[gid] = val;
}

// Shift right for exclusive scan
kernel void shift_right_kernel(
    device const int* input [[buffer(0)]],
    device int* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  output[gid] = (gid == 0) ? 0 : input[gid - 1];
}

template <typename T>
kernel void coalesce_with_positions_kernel(
    device const int64_t* flat_indices [[buffer(0)]],
    device const int64_t* indices [[buffer(1)]],
    device const T* in_values [[buffer(2)]],
    device const bool* is_unique [[buffer(3)]],
    device const int* output_positions [[buffer(4)]],
    device int64_t* out_indices [[buffer(5)]],
    device T* out_values [[buffer(6)]],
    constant uint& nnz [[buffer(7)]],
    constant uint& value_size [[buffer(8)]],
    constant uint& sparse_dim [[buffer(9)]],
    constant uint& total_unique [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
  if (!is_unique[gid])
    return;

  int out_pos = output_positions[gid];

  for (uint d = 0; d < sparse_dim; d++) {
    out_indices[d * total_unique + out_pos] = indices[d * nnz + gid];
  }

  int64_t current_index = flat_indices[gid];
  uint end = gid + 1;
  while (end < nnz && flat_indices[end] == current_index) {
    end++;
  }

  for (uint elem = 0; elem < value_size; elem++) {
    T sum = 0;
    for (uint j = gid; j < end; j++) {
      sum += in_values[j * value_size + elem];
    }
    out_values[out_pos * value_size + elem] = sum;
  }
}

#define INSTANTIATE_COALESCE_WITH_POSITIONS(DTYPE)                            \
  template                                                                    \
      [[host_name("coalesce_with_positions_kernel_" #DTYPE)]] [[kernel]] void \
      coalesce_with_positions_kernel<DTYPE>(                                  \
          device const int64_t* flat_indices [[buffer(0)]],                   \
          device const int64_t* indices [[buffer(1)]],                        \
          device const DTYPE* in_values [[buffer(2)]],                        \
          device const bool* is_unique [[buffer(3)]],                         \
          device const int* output_positions [[buffer(4)]],                   \
          device int64_t* out_indices [[buffer(5)]],                          \
          device DTYPE* out_values [[buffer(6)]],                             \
          constant uint& nnz [[buffer(7)]],                                   \
          constant uint& value_size [[buffer(8)]],                            \
          constant uint& sparse_dim [[buffer(9)]],                            \
          constant uint& total_unique [[buffer(10)]],                         \
          uint gid [[thread_position_in_grid]]);

INSTANTIATE_COALESCE_WITH_POSITIONS(float);
INSTANTIATE_COALESCE_WITH_POSITIONS(half);
INSTANTIATE_COALESCE_WITH_POSITIONS(bfloat);
INSTANTIATE_COALESCE_WITH_POSITIONS(bool);
INSTANTIATE_COALESCE_WITH_POSITIONS(long);
INSTANTIATE_COALESCE_WITH_POSITIONS(char);
INSTANTIATE_COALESCE_WITH_POSITIONS(uchar);
INSTANTIATE_COALESCE_WITH_POSITIONS(short);
INSTANTIATE_COALESCE_WITH_POSITIONS(int);