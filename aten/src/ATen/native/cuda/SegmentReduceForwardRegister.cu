#include <ATen/native/cuda/SegmentReduceKernels.h>

namespace at::native {

// Declarations
Tensor _segment_reduce_lengths_cuda_kernel(
  ReductionType reduction,
  const Tensor& data,
  const Tensor& lengths,
  int64_t axis,
  const std::optional<Scalar>& initial);

Tensor _segment_reduce_offsets_cuda_kernel(
  ReductionType reduction,
  const Tensor& data,
  const Tensor& offsets,
  int64_t axis,
  const std::optional<Scalar>& initial);

REGISTER_DISPATCH(_segment_reduce_lengths_stub, &_segment_reduce_lengths_cuda_kernel)
REGISTER_DISPATCH(_segment_reduce_offsets_stub, &_segment_reduce_offsets_cuda_kernel)

} // namespace at::native 