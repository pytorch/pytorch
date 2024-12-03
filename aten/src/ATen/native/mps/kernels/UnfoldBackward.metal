#include <metal_stdlib>
using namespace metal;

// Given coordinates and strides, calculates offset from the start of the
// tensors
long offset_from_coord(thread long* idx, constant long* strides, uint ndim) {
  long rc = 0;
  for (uint i = 0; i < ndim; ++i) {
    rc += idx[i] * strides[i];
  }
  return rc;
}

// Given thread index calculates position in the ndim tensor
void pos_from_thread_index(
    long idx,
    thread long* pos,
    constant long* sizes,
    uint ndim) {
  for (uint i = 0; i < ndim; ++i) {
    pos[i] = idx % sizes[i];
    idx /= sizes[i];
  }
}

// Consider out = in.unfold(dim, size, step), then
// out.shape[dim] == (in.shape[dim] - size) / step + 1,
// out.shape[-1] == size.
// out.ndim == in.ndim + 1
//
// unfold_backward receives grad_in and returns grad_out such that
// grad_in.shape == out.shape,
// grad_out.shape == in.shape.

// For each index in grad_out find the elements contributing to it and sum them
// up. Such algorithm requires no synchronization between threads. I.e.
// grad_out[...,out_dim_idx,...] accumulates all values
// grad_in[...,in_dim_idx,...,in_last_idx], where in_dim_idx is range
// [(out_dim_idx - size) / step, out_dim_idx / step] clamped to (0, in_dim_size)
// and in_last_idx is out_dim_idx - in_dim_idx * step.
// Accumulation step is skipped if in_last_idx is outside of [0, size] range
template <typename T>
kernel void unfold_backward(
    constant T* grad_in,
    device T* grad_out,
    constant long* input_strides,
    constant long* output_sizes,
    constant long* output_strides,
    constant uint4& dim_size_step_ndim,
    uint thread_index [[thread_position_in_grid]]) {
  auto dim = dim_size_step_ndim.x;
  auto size = dim_size_step_ndim.y;
  auto step = dim_size_step_ndim.z;
  auto ndim = dim_size_step_ndim.w;
  long pos[16];
  pos_from_thread_index(thread_index, pos, output_sizes, ndim);
  const auto output_offs = offset_from_coord(pos, output_strides, ndim);
  const auto in_dim_size = max(1L, (output_sizes[dim] - size) / step + 1);
  const auto out_dim_idx = pos[dim];
  const auto left_fold_idx = max(0L, (out_dim_idx - size) / step);
  const auto right_fold_idx = min(in_dim_size - 1, out_dim_idx / step);
  // Shift grad_in to start of unfold windows
  pos[dim] = 0;
  grad_in += offset_from_coord(pos, input_strides, ndim);
  float rc = 0;
  const auto in_dim_stride = input_strides[dim];
  const auto in_last_dim_stride = input_strides[ndim];
  for (auto in_dim_idx = left_fold_idx; in_dim_idx <= right_fold_idx;
       ++in_dim_idx) {
    const auto in_last_idx = out_dim_idx - in_dim_idx * step;
    if (in_last_idx < 0 || in_last_idx >= size) {
      continue;
    }
    rc +=
        grad_in[in_dim_idx * in_dim_stride + in_last_idx * in_last_dim_stride];
  }
  grad_out[output_offs] = static_cast<T>(rc);
}

#define INSTANTIATE_UNFOLD_BACKWARD(DTYPE)                      \
  template [[host_name("unfold_backward_" #DTYPE)]] kernel void \
  unfold_backward<DTYPE>(                                       \
      constant DTYPE*,                                          \
      device DTYPE*,                                            \
      constant long*,                                           \
      constant long*,                                           \
      constant long*,                                           \
      constant uint4&,                                          \
      uint thread_index [[thread_position_in_grid]])

INSTANTIATE_UNFOLD_BACKWARD(float);
INSTANTIATE_UNFOLD_BACKWARD(half);
#if __METAL_VERSION__ >= 310
INSTANTIATE_UNFOLD_BACKWARD(bfloat);
#endif
