#include <ATen/native/cuda/SegmentReduceKernels.h>

namespace at::native {

// Declaration of main backward kernel function
Tensor _segment_reduce_lengths_offsets_backward_cuda_kernel(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    ReductionType reduction,
    const Tensor& lengths_or_offsets_contig,
    int64_t axis,
    const std::optional<Scalar>& initial,
    bool is_offsets_like);

Tensor _segment_reduce_lengths_backward_cuda_kernel(
  const Tensor& grad_contig,
  const Tensor& output_contig,
  const Tensor& data_contig,
  ReductionType reduction,
  const Tensor& lengths_contig,
  int64_t axis,
  const std::optional<Scalar>& initial) {
  return _segment_reduce_lengths_offsets_backward_cuda_kernel(
    grad_contig, output_contig, data_contig, reduction, lengths_contig, axis, initial, /*is_offsets_like=*/false);
}

} // namespace at::native 