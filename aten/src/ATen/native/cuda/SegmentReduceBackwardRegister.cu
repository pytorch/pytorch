#include <ATen/native/cuda/SegmentReduceKernels.h>

namespace at::native {

// Declarations
Tensor _segment_reduce_lengths_backward_cuda_kernel(
  const Tensor& grad_contig,
  const Tensor& output_contig,
  const Tensor& data_contig,
  ReductionType reduction,
  const Tensor& lengths_contig,
  int64_t axis,
  const std::optional<Scalar>& initial);

Tensor _segment_reduce_offsets_backward_cuda_kernel(
  const Tensor& grad_contig,
  const Tensor& output_contig,
  const Tensor& data_contig,
  ReductionType reduction,
  const Tensor& offsets_contig,
  int64_t axis,
  const std::optional<Scalar>& initial);

REGISTER_DISPATCH(
    _segment_reduce_lengths_backward_stub,
    &_segment_reduce_lengths_backward_cuda_kernel);
REGISTER_DISPATCH(
  _segment_reduce_offsets_backward_stub,
  &_segment_reduce_offsets_backward_cuda_kernel);

} // namespace at::native 