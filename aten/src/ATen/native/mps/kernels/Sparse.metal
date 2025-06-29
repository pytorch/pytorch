#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// Template kernel to coalesce sorted indices and sum duplicate values
template <typename T>
kernel void coalesce_kernel(
    device const ulong* in_indices       [[buffer(0)]],  // flat indices array (length = nnz)
    device const T*    in_values         [[buffer(1)]],  // values array (length = nnz)
    device ulong*      out_indices       [[buffer(2)]],  // output unique indices (length <= nnz)
    device T*          out_values        [[buffer(3)]],  // output summed values (length <= nnz)
    constant uint&     nnz               [[buffer(4)]],  // number of input non-zero entries
    device atomic_uint* unique_count     [[buffer(5)]],  // atomic counter for output length
    uint gid                            [[thread_position_in_grid]]
) {
  uint i = gid;
  if (i >= nnz) return;
  
  // Determine if this thread is at the start of a group of identical indices
  bool is_group_start = (i == 0) || (in_indices[i] != in_indices[i - 1]);
  if (!is_group_start) {
    return;
  }
  
  // This thread will accumulate all values for the index `current_index`
  ulong current_index = in_indices[i];
  float accumulator = 0.0f;  // accumulate in higher precision (float)
  for (uint j = i; j < nnz && in_indices[j] == current_index; ++j) {
    accumulator += (float)in_values[j];
  }
  
  // Atomically reserve an output slot and write the result
  uint out_pos = atomic_fetch_add_explicit(unique_count, 1, memory_order_relaxed);
  out_indices[out_pos] = current_index;
  out_values[out_pos]   = (T)accumulator;
}


// Instantiate the kernel for the types we need (float, half, bool, bfloat16, etc.)
#define INSTANTIATE_COALESCE(DTYPE) \
  template [[host_name("coalesce_kernel_" #DTYPE)]] [[kernel]] void coalesce_kernel<DTYPE>( \
      device const ulong* in_indices [[buffer(0)]], \
      device const DTYPE*  in_values  [[buffer(1)]], \
      device ulong*       out_indices[[buffer(2)]], \
      device DTYPE*        out_values [[buffer(3)]], \
      constant uint&      nnz        [[buffer(4)]], \
      device atomic_uint* unique_count[[buffer(5)]], \
      uint gid                      [[thread_position_in_grid]]);

INSTANTIATE_COALESCE(float);
INSTANTIATE_COALESCE(half);
#if __METAL_VERSION__ >= 310
INSTANTIATE_COALESCE(bfloat);  // bfloat16 supported on Metal 3.1+
#endif
INSTANTIATE_COALESCE(bool);