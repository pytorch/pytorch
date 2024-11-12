#include <metal_stdlib>
using namespace metal;

ulong offset_from_coord(thread ulong* idx, constant ulong* strides, uint ndim) {
   ulong rc = 0;
   for(uint i = 0; i < ndim; ++i) {
     rc += idx[i] * strides[i];
   }
   return rc;
}

ulong2 divmod(long x, ulong y) {
  return ulong2(x/y, x%y);
}

void pos_from_index(ulong idx, thread ulong* pos, constant ulong* sizes, uint ndim) {
  for(uint i = 0; i < ndim; ++i) {
    auto rc = divmod(idx, sizes[i]);
    pos[i] = rc.y;
    idx = rc.x;
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

template<typename T>
kernel void unfold_backward(
    constant T *grad_in,
    device T* grad_out,
    constant ulong* input_strides,
    constant ulong* output_sizes,
    constant ulong* output_strides,
    constant uint4& dim_size_step_ndim,
    uint thread_index [[thread_position_in_grid]]) {
    auto dim = dim_size_step_ndim.x;
    auto size = dim_size_step_ndim.y;
    auto step = dim_size_step_ndim.z;
    auto ndim = dim_size_step_ndim.w;
    ulong pos[16];
    pos_from_index(thread_index, pos, output_sizes, ndim);
    const auto output_offs = offset_from_coord(pos, output_strides, ndim);
    auto grad_in_dim_size = (output_sizes[dim] - size) / step + 1;
    float rc = 0;
    const auto grad_out_idx = pos[dim];
    auto left_fold_idx = grad_out_idx > size ? (grad_out_idx - size) / step : 0UL;
    auto right_fold_idx = min(grad_in_dim_size - 1, grad_out_idx / step);
    for(auto idx = left_fold_idx; idx <= right_fold_idx; ++idx) {
        pos[dim] = idx;
        pos[ndim] = grad_out_idx - idx * step;
        if (pos[ndim] >= size) continue;
         auto input_offset = offset_from_coord(pos, input_strides, ndim + 1);
         rc += grad_in[input_offset];
    }
    grad_out[output_offs] = static_cast<T>(rc);
}

#define INSTANTIATE_UNFOLD_BACKWARD(DTYPE)                        \
  template [[host_name("unfold_backward_" #DTYPE)]] kernel void \
  unfold_backward<DTYPE>(                                       \
      constant DTYPE *,                    \
      device DTYPE *,                     \
      constant ulong*, \
      constant ulong*, \
      constant ulong*, \
      constant uint4&, \
      uint thread_index [[thread_position_in_grid]])

INSTANTIATE_UNFOLD_BACKWARD(float);
INSTANTIATE_UNFOLD_BACKWARD(half);
#if __METAL_VERSION__ >= 310
INSTANTIATE_UNFOLD_BACKWARD(bfloat);
#endif
