#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/SegmentReduceKernels.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/cumsum.h>
#endif

namespace at::native {

// Declaration of implementation function
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
    const index_t* lengths_data_ptr);

Tensor _segment_reduce_lengths_offsets_cuda_kernel(
  ReductionType reduction,
  const Tensor& data,
  const Tensor& lengths_or_offsets,
  int64_t axis,
  const std::optional<Scalar>& initial,
  bool is_offsets_like) {
  
  TORCH_CHECK(data.is_contiguous());
  TORCH_CHECK(lengths_or_offsets.is_contiguous());
  axis = lengths_or_offsets.dim() - 1;
  int64_t segment_count = is_offsets_like ? lengths_or_offsets.size(axis) - 1 : lengths_or_offsets.size(axis);
  auto output_shape = data.sizes().vec();
  output_shape[axis] = segment_count;
  auto output = at::empty(output_shape, data.options());

  auto offsets = lengths_or_offsets;
  auto lengths = lengths_or_offsets;
  if (is_offsets_like) {
    lengths = lengths.diff();
  } else {
    auto zeros_shape = offsets.sizes().vec();
    zeros_shape[axis] = 1;
    offsets = at::cat({at::zeros(zeros_shape, offsets.options()), offsets}, axis);
    offsets.cumsum_(axis);
  }

  AT_DISPATCH_INDEX_TYPES(
      lengths_or_offsets.scalar_type(), "_segment_reduce_cuda_kernel1", ([&] {
        auto* offsets_data_ptr = offsets.const_data_ptr<index_t>();
        auto* lengths_data_ptr = lengths.const_data_ptr<index_t>();
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            data.scalar_type(),
            "segment_reduce_cuda",
            [&]() {
              segment_reduce_forward_impl<scalar_t, index_t>(
                  reduction, data, lengths_or_offsets, axis, initial, is_offsets_like,
                  output, offsets_data_ptr, lengths_data_ptr);
            });
      }));

  return output;
}

} // namespace at::native 