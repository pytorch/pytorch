#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/SegmentReduceKernels.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/zeros.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/cumsum.h>
#endif

namespace at::native {

// Declaration of implementation function
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
    const index_t* lengths_data);

Tensor _segment_reduce_lengths_offsets_backward_cuda_kernel(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    ReductionType reduction,
    const Tensor& lengths_or_offsets_contig,
    int64_t axis,
    const std::optional<Scalar>& initial,
    bool is_offsets_like) {
  
  axis = lengths_or_offsets_contig.dim() - 1;
  auto grad_input = at::zeros({data_contig.sizes()}, grad_contig.options());

  auto offsets = lengths_or_offsets_contig;
  auto lengths = lengths_or_offsets_contig;
  if (is_offsets_like) {
    lengths = lengths.diff();
  } else {
    auto zeros_shape = offsets.sizes().vec();
    zeros_shape[axis] = 1;
    offsets = at::cat({at::zeros(zeros_shape, offsets.options()), offsets}, axis);
    offsets.cumsum_(axis);
  }

  AT_DISPATCH_INDEX_TYPES(
      lengths_or_offsets_contig.scalar_type(), "_segment_reduce_cuda_lengths_offsets_backward_kernel1", ([&] {
        const auto* lengths_data = lengths.const_data_ptr<index_t>();
        auto* offsets_data = offsets.const_data_ptr<index_t>();

        AT_DISPATCH_FLOATING_TYPES_AND2(
            kBFloat16,
            kHalf,
            data_contig.scalar_type(),
            "_segment_reduce_cpu",
            ([&]() {
              segment_reduce_backward_impl<scalar_t, index_t>(
                  grad_contig, output_contig, data_contig, reduction,
                  lengths_or_offsets_contig, axis, initial, is_offsets_like,
                  grad_input, offsets_data, lengths_data);
            }));
      }));
  return grad_input;
}

} // namespace at::native 