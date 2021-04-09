
#include <ATen/native/SegmentReduce.h>

#include <ATen/ATen.h>
#include <ATen/NumericUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CubUtils.cuh>
#include <iostream>

namespace at {
namespace native {

struct CustomMax {
  template <typename OutputT>
  __host__ __device__ __forceinline__ OutputT
  operator()(const OutputT& a, const OutputT& b) {
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
  auto offsets = at::empty({segment_count + 1}, lengths.options());
  offsets[0].zero_();
  auto* lengths_data_ptr = lengths.data_ptr<int64_t>();
  auto* offsets_data_ptr = offsets.data_ptr<int64_t>();
  size_t temp_storage_bytes = 0;
  AT_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
                    nullptr,
                    temp_storage_bytes,
                    lengths_data_ptr,
                    offsets_data_ptr + 1,
                    segment_count,
                    at::cuda::getCurrentCUDAStream()););

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(temp_storage_bytes);

  AT_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
                    dataPtr.get(),
                    temp_storage_bytes,
                    lengths_data_ptr,
                    offsets_data_ptr + 1,
                    segment_count,
                    at::cuda::getCurrentCUDAStream()););

  return offsets;
}

Tensor _segment_reduce_cuda_kernel(
    const Tensor& data,
    const Tensor& lengths,
    int64_t axis,
    bool unsafe) {
  if (!unsafe) {
    TORCH_CHECK(
        (lengths.min().item<int64_t>() > 0),
        "lengths contains non positive value!");
    TORCH_CHECK(lengths.sum().item<int64_t>() == data.numel());
  }

  int64_t segment_count = lengths.numel();
  const auto data_contig = data.contiguous();
  auto output = at::empty({segment_count}, data.options());

  const auto lengths_contig = lengths.contiguous();
  auto offsets = _get_complete_sum(lengths_contig);
  auto* offsets_data_ptr = offsets.data_ptr<int64_t>();

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      data.scalar_type(),
      "segment_reduce_cuda",
      [&]() {
        auto* data_contig_data_ptr = data_contig.data_ptr<scalar_t>();
        auto* output_data_ptr = output.data_ptr<scalar_t>();

        CustomMax max_op{};
        size_t temp_storage_bytes = 0;
        scalar_t initial_value = std::numeric_limits<scalar_t>::lowest();
        AT_CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(
                          nullptr,
                          temp_storage_bytes,
                          data_contig_data_ptr,
                          output_data_ptr,
                          segment_count,
                          offsets_data_ptr,
                          offsets_data_ptr + 1,
                          max_op,
                          initial_value,
                          at::cuda::getCurrentCUDAStream()););

        auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
        auto dataPtr = allocator.allocate(temp_storage_bytes);

        AT_CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(
                          dataPtr.get(),
                          temp_storage_bytes,
                          data_contig_data_ptr,
                          output_data_ptr,
                          segment_count,
                          offsets_data_ptr,
                          offsets_data_ptr + 1,
                          max_op,
                          initial_value,
                          at::cuda::getCurrentCUDAStream()););
      });
  return output;
}

REGISTER_DISPATCH(_segment_reduce_stub, &_segment_reduce_cuda_kernel);

} // namespace native
} // namespace at
