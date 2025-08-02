#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

kernel void flatten_indices_kernel(
    device const int64_t* indices    [[buffer(0)]],  // shape: (sparse_dim, nnz)
    device const int64_t* strides    [[buffer(1)]],  // shape: (sparse_dim,)
    device int64_t* flat_indices     [[buffer(2)]],  // shape: (nnz,)
    constant uint& sparse_dim        [[buffer(3)]],
    constant uint& nnz               [[buffer(4)]],
    uint gid                         [[thread_position_in_grid]]
) { 
    flat_indices[gid] = (indices[gid] * strides[0] + indices[nnz + gid] * strides[1]);
}

// Kernel to mark unique positions (without counting)
kernel void mark_unique_positions_kernel(
    device const int64_t* indices    [[buffer(0)]],  // sorted flat indices
    device bool* is_unique          [[buffer(1)]],  // output: true if start of unique group
    constant uint& nnz              [[buffer(2)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= nnz) return;
    
    // First element is always unique, others are unique if different from previous
    bool unique = (gid == 0) || (indices[gid] != indices[gid - 1]);
    is_unique[gid] = unique;
}

// Kernel to compute output positions via prefix sum
kernel void compute_output_positions_kernel(
    device const bool* is_unique     [[buffer(0)]],  // input: marks unique positions
    device int* positions           [[buffer(1)]],  // output: position in output array
    constant uint& nnz              [[buffer(2)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= nnz) return;
    
    // Simple serial prefix sum - could be optimized with parallel scan
    int pos = 0;
    for (uint i = 0; i < gid; i++) {
        if (is_unique[i]) pos++;
    }
    positions[gid] = pos;
}

// Coalesce kernel using precomputed positions
template <typename T>
kernel void coalesce_with_positions_kernel(
    device const int64_t* flat_indices   [[buffer(0)]],  // sorted flat indices
    device const int64_t* indices        [[buffer(1)]],  // original multi-dim indices (sparse_dim x nnz)
    device const T* in_values            [[buffer(2)]],  // input values
    device const bool* is_unique         [[buffer(3)]],  // marks start of unique groups
    device const int* output_positions   [[buffer(4)]],  // precomputed output positions
    device int64_t* out_indices          [[buffer(5)]],  // output indices (sparse_dim x newNnz)
    device T* out_values                 [[buffer(6)]],  // output values
    constant uint& nnz                   [[buffer(7)]],
    constant uint& value_size            [[buffer(8)]],
    constant uint& sparse_dim            [[buffer(9)]],
    constant uint& total_unique          [[buffer(10)]], // total number of unique elements
    uint gid                            [[thread_position_in_grid]]
) {
    if (gid >= nnz) return;
    
    // Only process if this is the start of a unique group
    if (!is_unique[gid]) return;
    
    // Get output position from precomputed array
    int out_pos = output_positions[gid];
    
    // Copy multi-dimensional indices
    for (uint d = 0; d < sparse_dim; d++) {
        out_indices[d * total_unique + out_pos] = indices[d * nnz + gid];
    }
    
    // Accumulate values for this unique index
    int64_t current_index = flat_indices[gid];
    for (uint elem = 0; elem < value_size; elem++) {
        T accumulator = 0;
        for (uint j = gid; j < nnz && flat_indices[j] == current_index; j++) {
            accumulator += in_values[j * value_size + elem];
        }
        out_values[out_pos * value_size + elem] = accumulator;
    }
}

// Instantiate the kernels
#define INSTANTIATE_COALESCE_WITH_POSITIONS(DTYPE) \
  template [[host_name("coalesce_with_positions_kernel_" #DTYPE)]] [[kernel]] void coalesce_with_positions_kernel<DTYPE>( \
      device const int64_t* flat_indices [[buffer(0)]], \
      device const int64_t* indices     [[buffer(1)]], \
      device const DTYPE* in_values     [[buffer(2)]], \
      device const bool* is_unique      [[buffer(3)]], \
      device const int* output_positions [[buffer(4)]], \
      device int64_t* out_indices       [[buffer(5)]], \
      device DTYPE* out_values          [[buffer(6)]], \
      constant uint& nnz                [[buffer(7)]], \
      constant uint& value_size         [[buffer(8)]], \
      constant uint& sparse_dim         [[buffer(9)]], \
      constant uint& total_unique       [[buffer(10)]], \
      uint gid                         [[thread_position_in_grid]]);

INSTANTIATE_COALESCE_WITH_POSITIONS(float);
INSTANTIATE_COALESCE_WITH_POSITIONS(half);
#if __METAL_VERSION__ >= 310
INSTANTIATE_COALESCE_WITH_POSITIONS(bfloat);
#endif
INSTANTIATE_COALESCE_WITH_POSITIONS(bool);
