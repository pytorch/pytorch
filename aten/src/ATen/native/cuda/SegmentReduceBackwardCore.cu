#include <ATen/native/cuda/SegmentReduceKernels.h>

namespace at::native {

template <typename scalar_t, typename index_t>
void segment_reduce_backward_impl(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    ReductionType reduction,
    const Tensor& lengths_or_offsets_contig,
    int64_t axis,
    const std::optional<Scalar>& initial,
    bool is_offsets_like,
    Tensor& grad_input,
    const index_t* offsets_data,
    const index_t* lengths_data) {
  
  int64_t segment_count = is_offsets_like ?
                          lengths_or_offsets_contig.size(axis) - 1 :
                          lengths_or_offsets_contig.size(axis);
  int64_t lengths_stride_axis = lengths_or_offsets_contig.stride(axis);

  // outer_offset is the size of the outer dimensions of output (before axis)
  // inner_offset is the size of the inner dimensions of output (after axis)
  int64_t outer_offset = 1, inner_offset = 1;
  for (int64_t d = 0; d < axis; d++) {
    outer_offset *= output_contig.size(d);
  }
  for (int64_t d = axis + 1; d < output_contig.dim(); d++) {
    inner_offset *= output_contig.size(d);
  }

  constexpr int threads_per_block = 256;
  int64_t num_blocks = (outer_offset * inner_offset * segment_count + threads_per_block - 1) / threads_per_block;
  num_blocks = std::max(num_blocks, (int64_t)1);

  auto data_stride_axis = data_contig.stride(axis);
  auto data_size_axis = data_contig.size(axis);
  auto output_stride_axis = output_contig.stride(axis);
  auto output_size_axis = output_contig.size(axis);
  auto offsets_stride_axis = lengths_or_offsets_contig.stride(axis);

  auto* output_data = output_contig.const_data_ptr<scalar_t>();
  auto* grad_data = grad_contig.const_data_ptr<scalar_t>();
  auto* grad_input_data = grad_input.mutable_data_ptr<scalar_t>();
  const auto* values_data = data_contig.const_data_ptr<scalar_t>();

  scalar_t initial_prod_value;
  if (initial.has_value()) {
    initial_prod_value = initial.value().to<scalar_t>();
  } else {
    initial_prod_value = 1;
  }

  segment_reduce_backward_kernel<scalar_t>
      <<<num_blocks,
         threads_per_block,
         0,
         at::cuda::getCurrentCUDAStream()>>>(
          reduction,
          grad_input_data,
          grad_data,
          output_data,
          values_data,
          lengths_data,
          offsets_data,
          segment_count,
          lengths_stride_axis,
          initial_prod_value,
          outer_offset,
          inner_offset,
          data_stride_axis,
          data_size_axis,
          output_stride_axis,
          output_size_axis,
          offsets_stride_axis
        );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Explicit instantiations
template void segment_reduce_backward_impl<float, int32_t>(const Tensor&, const Tensor&, const Tensor&, ReductionType, const Tensor&, int64_t, const std::optional<Scalar>&, bool, Tensor&, const int32_t*, const int32_t*);
template void segment_reduce_backward_impl<float, int64_t>(const Tensor&, const Tensor&, const Tensor&, ReductionType, const Tensor&, int64_t, const std::optional<Scalar>&, bool, Tensor&, const int64_t*, const int64_t*);
template void segment_reduce_backward_impl<double, int32_t>(const Tensor&, const Tensor&, const Tensor&, ReductionType, const Tensor&, int64_t, const std::optional<Scalar>&, bool, Tensor&, const int32_t*, const int32_t*);
template void segment_reduce_backward_impl<double, int64_t>(const Tensor&, const Tensor&, const Tensor&, ReductionType, const Tensor&, int64_t, const std::optional<Scalar>&, bool, Tensor&, const int64_t*, const int64_t*);

} // namespace at::native 