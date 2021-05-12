
#include <ATen/native/SegmentReduce.h>

#include <ATen/ATen.h>
#include <ATen/NumericUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/cub.cuh>

namespace at {
namespace native {

struct CustomMax {
  template <typename OutputT>
  __host__ __device__ __forceinline__ OutputT
  operator()(const OutputT& a, const OutputT& b) const {
    if (at::_isnan(a)) {
      return a;
    } else if (at::_isnan(b)) {
      return b;
    }
    return std::max<OutputT>(a, b);
  }
};

Tensor _get_complete_sum(const Tensor& lengths) {
  int64_t segment_count = lengths.numel();
  TORCH_CHECK(segment_count < INT_MAX);
  auto offsets = at::empty({segment_count + 1}, lengths.options());
  offsets[0].zero_();
  auto* lengths_data_ptr = lengths.data_ptr<int64_t>();
  auto* offsets_data_ptr = offsets.data_ptr<int64_t>();

  CUB_WRAPPER(
      cub::DeviceScan::InclusiveSum,
      lengths_data_ptr,
      offsets_data_ptr + 1,
      segment_count,
      at::cuda::getCurrentCUDAStream());

  return offsets;
}

Tensor _segment_reduce_cuda_kernel(
    const Tensor& data,
    const Tensor& lengths,
    int64_t axis,
    const c10::optional<Scalar>& initial) {
  int64_t segment_count = lengths.numel();
  auto output = at::empty({segment_count}, data.options());

  auto offsets = _get_complete_sum(lengths);
  auto* offsets_data_ptr = offsets.data_ptr<int64_t>();

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      data.scalar_type(),
      "segment_reduce_cuda",
      [&]() {
        auto* data_data_ptr = data.data_ptr<scalar_t>();
        auto* output_data_ptr = output.data_ptr<scalar_t>();

        CustomMax max_op{};
        scalar_t initial_value = initial.has_value()
            ? initial.value().to<scalar_t>()
            : std::numeric_limits<scalar_t>::lowest();
        CUB_WRAPPER(
            cub::DeviceSegmentedReduce::Reduce,
            data_data_ptr,
            output_data_ptr,
            segment_count,
            offsets_data_ptr,
            offsets_data_ptr + 1,
            max_op,
            initial_value,
            at::cuda::getCurrentCUDAStream());
      });

  return output;
}

REGISTER_DISPATCH(_segment_reduce_stub, &_segment_reduce_cuda_kernel);

} // namespace native
} // namespace at
