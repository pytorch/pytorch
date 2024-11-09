#include <metal_stdlib>

ulong offset_from_coord(thread ulong* idx, constant ulong* strides, uint ndim) {
   ulong rc = 0;
   for(uint i = 0; i < ndim; ++i) {
     rc += idx[i] * strides[i];
   }
   return rc;
}

void pos_from_index(ulong idx, thread ulong* pos, constant ulong* sizes, uint ndim) {
  for(uint i = 0; i < ndim; ++i) {
    pos[i] = idx % sizes[i];
    idx /= sizes[i];
  }
}


// Consider out = in.unfold(dim, size, step), then
// out.shape[dim] == (in.shape[dim] - size) / step + 1,
// out.shape[-1] == size.
// out.ndim) == in.ndim) + 1
//
// unfold_backward receives grad_in and returns grad_out such that
// grad_in.shape == out.shape,
// grad_out.shape == in.shape.

kernel void unfold_backward_float(
    constant float *grad_in,
    device float* grad_out,
    constant ulong* input_strides,
    constant ulong* output_sizes,
    constant ulong* output_strides,
    constant uint4& dim_size_step_ndim,
    uint thread_index [[thread_position_in_grid]]) {
    auto dim_idx = dim_size_step_ndim.x;
    auto size = dim_size_step_ndim.y;
    auto step = dim_size_step_ndim.z;
    auto ndim = dim_size_step_ndim.w;
    auto dim_idx_size = (output_sizes[dim_idx] - size) / step + 1;
    ulong pos[16];
    pos_from_index(thread_index, pos, output_sizes, ndim);
    auto output_offs = offset_from_coord(pos, output_strides, ndim);
    pos[ndim] = size;
    grad_out[output_offs] = 1.0 * thread_index;
}
