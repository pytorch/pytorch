
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

struct CustomSum {
  template <typename OutputT>
  __host__ __device__ __forceinline__ OutputT
  operator()(const OutputT& a, const OutputT& b) const {
    return a + b;
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

template <typename scalar_t>
__global__ static void post_sum_div_kernel(
    scalar_t* output_data,
    const int64_t* lengths_data,
    const int64_t segment_count,
    bool is_initial_set,
    scalar_t initial) {
  CUDA_KERNEL_LOOP(index, segment_count) {
    CUDA_KERNEL_ASSERT(lengths_data[index] >= 0);
    if (lengths_data[index] == 0) {
      if (is_initial_set) {
        output_data[index] = initial;
      } else {
        output_data[index] = NAN;
      }
    } else if (!at::_isnan(output_data[index])) {
      output_data[index] = output_data[index] / lengths_data[index];
    }
  }
}

Tensor _segment_reduce_cuda_kernel(
    SegmentReductionType reduction,
    const Tensor& data,
    const Tensor& lengths,
    int64_t axis,
    const c10::optional<Scalar>& initial) {
  int64_t segment_count = lengths.numel();
  auto output = at::empty({segment_count}, data.options());

  auto offsets = _get_complete_sum(lengths);
  auto* offsets_data_ptr = offsets.data_ptr<int64_t>();

  constexpr int threads_per_block = 256;
  int64_t num_blocks =
      (segment_count + threads_per_block - 1) / threads_per_block;
  num_blocks = std::max(num_blocks, (int64_t)1);
  auto* lengths_data_ptr = lengths.data_ptr<int64_t>();

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      data.scalar_type(),
      "segment_reduce_cuda",
      [&]() {
        auto* data_data_ptr = data.data_ptr<scalar_t>();
        auto* output_data_ptr = output.data_ptr<scalar_t>();

        if (reduction == SegmentReductionType::MAX) {
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
        } else if (reduction == SegmentReductionType::MEAN) {
          CustomSum sum_op{};
          scalar_t initial_value = initial.has_value()
              ? initial.value().to<scalar_t>()
              : (scalar_t)0;
          CUB_WRAPPER(
              cub::DeviceSegmentedReduce::Reduce,
              data_data_ptr,
              output_data_ptr,
              segment_count,
              offsets_data_ptr,
              offsets_data_ptr + 1,
              sum_op,
              initial_value,
              at::cuda::getCurrentCUDAStream());

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
      });

  return output;
}

REGISTER_DISPATCH(_segment_reduce_stub, &_segment_reduce_cuda_kernel);

} // namespace native
} // namespace at
