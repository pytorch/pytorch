#include <ATen/native/cuda/SegmentReduceKernels.h>

namespace at::native {

// Declaration of main kernel function
Tensor _segment_reduce_lengths_offsets_cuda_kernel(
  ReductionType reduction,
  const Tensor& data,
  const Tensor& lengths_or_offsets,
  int64_t axis,
  const std::optional<Scalar>& initial,
  bool is_offsets_like);

Tensor _segment_reduce_offsets_cuda_kernel(
  ReductionType reduction,
  const Tensor& data,
  const Tensor& offsets,
  int64_t axis,
  const std::optional<Scalar>& initial) {
  return _segment_reduce_lengths_offsets_cuda_kernel(
    reduction, data, offsets, axis, initial, /*is_offsets_like=*/true);
}

} // namespace at::native 