#include <ATen/native/cuda/SegmentReduceKernels.h>

namespace at::native {

// Declaration of external CUB function
template <typename scalar_t, typename index_t>
void segment_reduce_cub_calls(
    ReductionType reduction,
    const scalar_t* data_data_ptr,
    scalar_t* output_data_ptr,
    int64_t segment_count,
    const index_t* offsets_data_ptr,
    scalar_t initial_value);

template <typename scalar_t, typename index_t>
void segment_reduce_forward_impl(
    ReductionType reduction,
    const Tensor& data,
    const Tensor& lengths_or_offsets,
    int64_t axis,
    const std::optional<Scalar>& initial,
    bool is_offsets_like,
    Tensor& output,
    const index_t* offsets_data_ptr,
    const index_t* lengths_data_ptr) {
  
  int64_t segment_count = is_offsets_like ? lengths_or_offsets.size(axis) - 1 : lengths_or_offsets.size(axis);
  int64_t lengths_stride_axis = lengths_or_offsets.stride(axis);
  auto output_shape = data.sizes().vec();
  
  constexpr int threads_per_block = 256;
  int64_t num_blocks = (output.numel() + threads_per_block - 1) / threads_per_block;
  num_blocks = std::max(num_blocks, (int64_t)1);

  auto data_stride_axis = data.stride(axis);
  auto data_size_axis = data.size(axis);
  auto output_stride_axis = output.stride(axis);
  auto output_size_axis = output.size(axis);
  auto offsets_stride_axis = lengths_or_offsets.stride(axis);

  auto* data_data_ptr = data.const_data_ptr<scalar_t>();
  auto* output_data_ptr = output.mutable_data_ptr<scalar_t>();

  // initialize starting value
  scalar_t initial_value = 0;
  if (initial.has_value()) {
    initial_value = initial.value().to<scalar_t>();
  } else if (reduction == ReductionType::MAX) {
    initial_value = -std::numeric_limits<scalar_t>::infinity();
  } else if (
      reduction == ReductionType::MEAN ||
      reduction == ReductionType::SUM) {
    initial_value = 0;
  } else if (reduction == ReductionType::MIN) {
    initial_value = std::numeric_limits<scalar_t>::infinity();
  } else if (reduction == ReductionType::PROD) {
    initial_value = 1;
  }

  if (output_shape.size() > 1) {
    // outer_offset is the size of the outer dimensions of output (before axis)
    // inner_offset is the size of the inner dimensions of output (after axis)
    int64_t outer_offset = 1, inner_offset = 1;
    for (int64_t d = 0; d < axis; d++) {
      outer_offset *= output.size(d);
    }
    for (int64_t d = axis + 1; d < output.dim(); d++) {
      inner_offset *= output.size(d);
    }

    segment_reduce_forward_kernel<scalar_t>
        <<<num_blocks,
           threads_per_block,
           0,
           at::cuda::getCurrentCUDAStream()>>>(
            reduction,
            output_data_ptr,
            data_data_ptr,
            lengths_data_ptr,
            offsets_data_ptr,
            segment_count,
            lengths_stride_axis,
            initial.has_value(),
            initial_value,
            outer_offset,
            inner_offset,
            data_stride_axis,
            data_size_axis,
            output_stride_axis,
            output_size_axis,
            offsets_stride_axis
          );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    segment_reduce_cub_calls<scalar_t, index_t>(
        reduction,
        data_data_ptr,
        output_data_ptr,
        segment_count,
        offsets_data_ptr,
        initial_value);

    if (reduction == ReductionType::MEAN) {
      post_sum_div_kernel<scalar_t>
          <<<num_blocks,
             threads_per_block,
             0,
             at::cuda::getCurrentCUDAStream()>>>(
              output_data_ptr,
              lengths_data_ptr,
              segment_count,
              initial.has_value(),
              initial_value);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
}

// Explicit instantiations
template void segment_reduce_forward_impl<float, int32_t>(ReductionType, const Tensor&, const Tensor&, int64_t, const std::optional<Scalar>&, bool, Tensor&, const int32_t*, const int32_t*);
template void segment_reduce_forward_impl<float, int64_t>(ReductionType, const Tensor&, const Tensor&, int64_t, const std::optional<Scalar>&, bool, Tensor&, const int64_t*, const int64_t*);
template void segment_reduce_forward_impl<double, int32_t>(ReductionType, const Tensor&, const Tensor&, int64_t, const std::optional<Scalar>&, bool, Tensor&, const int32_t*, const int32_t*);
template void segment_reduce_forward_impl<double, int64_t>(ReductionType, const Tensor&, const Tensor&, int64_t, const std::optional<Scalar>&, bool, Tensor&, const int64_t*, const int64_t*);

} // namespace at::native 