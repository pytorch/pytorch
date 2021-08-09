
#include <ATen/native/SegmentReduce.h>

#include <ATen/ATen.h>
#include <ATen/NumericUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/cub.cuh>

namespace at {
namespace native {

namespace {
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

struct CustomMin {
  template <typename OutputT>
  __host__ __device__ __forceinline__ OutputT
  operator()(const OutputT& a, const OutputT& b) const {
    if (at::_isnan(a)) {
      return a;
    } else if (at::_isnan(b)) {
      return b;
    }
    return std::min<OutputT>(a, b);
  }
};

Tensor _get_complete_sum(const Tensor& lengths) {
  int64_t segment_count = lengths.numel();
  TORCH_CHECK(segment_count < INT_MAX);
  auto offsets = at::empty({segment_count + 1}, lengths.options());
  offsets[0].zero_();

  AT_DISPATCH_INDEX_TYPES(
      lengths.type(), "_segment_reduce_cuda_backward_kernel1", ([&] {
        auto* lengths_data_ptr = lengths.data_ptr<index_t>();
        auto* offsets_data_ptr = offsets.data_ptr<index_t>();
        CUB_WRAPPER(
            cub::DeviceScan::InclusiveSum,
            lengths_data_ptr,
            offsets_data_ptr + 1,
            segment_count,
            at::cuda::getCurrentCUDAStream());
      }));
  return offsets;
}

template <typename scalar_t, typename index_t>
__global__ static void post_sum_div_kernel(
    scalar_t* output_data,
    const index_t* lengths_data,
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

template <typename scalar_t, typename index_t>
__global__ void segment_reduce_forward_kernel(
    SegmentReductionType reduction,
    scalar_t* output_data,
    scalar_t* values_data,
    const index_t* lengths_data,
    const index_t* lengths_cumsum_data,
    const int64_t segment_count,
    const int64_t stride_count,
    bool is_initial_set,
    scalar_t initial_value) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t row_id = idx / stride_count;
  int64_t lane_id = idx % stride_count;
  if (idx >= (segment_count * stride_count)) {
    return;
  }
  int64_t offset_start = lengths_cumsum_data[row_id];
  int64_t offset_end = lengths_cumsum_data[row_id + 1];

  // ===== step2: apply reduction
  for (int64_t j = offset_start; j < offset_end; ++j) {
    int64_t starting_index = (j * stride_count) + lane_id;
    const auto data = values_data[starting_index];
    // TODO: There is no need to branch with every element
    if (reduction == SegmentReductionType::MAX) {
      initial_value =
          at::_isnan(data) ? data : std::max<scalar_t>(initial_value, data);
    } else if (
        reduction == SegmentReductionType::MEAN ||
        reduction == SegmentReductionType::SUM) {
      initial_value = initial_value + data;
    } else if (reduction == SegmentReductionType::MIN) {
      initial_value =
          at::_isnan(data) ? data : std::min<scalar_t>(initial_value, data);
    }
  }

  // ===== step3: finalize reduction
  CUDA_KERNEL_ASSERT(lengths_data[row_id] >= 0);
  if (lengths_data[row_id] == 0 && !is_initial_set &&
      reduction == SegmentReductionType::MEAN) {
    initial_value = static_cast<scalar_t>(NAN);
  } else if (
      reduction == SegmentReductionType::MEAN && lengths_data[row_id] > 0 &&
      !at::_isnan(initial_value)) {
    initial_value = initial_value / lengths_data[row_id];
  }
  int64_t output_index = (row_id * stride_count) + lane_id;
  output_data[output_index] = initial_value;
}

template <typename scalar_t, typename index_t>
__global__ void segment_reduce_backward_kernel(
    SegmentReductionType reduction,
    scalar_t* grad_input_data,
    scalar_t* grad_data,
    scalar_t* output_data,
    const scalar_t* values_data,
    const index_t* lengths_data,
    const index_t* lengths_cumsum_data,
    const int64_t segment_count,
    const int64_t stride_count) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t row_id = idx / stride_count;
  int64_t lane_id = idx % stride_count;

  if (idx >= (segment_count * stride_count)) {
    return;
  }
  if (lengths_data[row_id] == 0) {
    return;
  }

  int64_t offset_start = lengths_cumsum_data[row_id];
  int64_t offset_end = lengths_cumsum_data[row_id + 1];

  int64_t output_index = (row_id * stride_count) + lane_id;

  if (reduction == SegmentReductionType::MAX ||
      reduction == SegmentReductionType::MIN) {
    int64_t counter = 0;
    for (int64_t j = offset_start; j < offset_end; ++j) {
      int64_t starting_index = (j * stride_count) + lane_id;
      if (at::_isnan(values_data[starting_index]) ||
          values_data[starting_index] == output_data[output_index]) {
        grad_input_data[starting_index] = grad_data[output_index];
        counter++;
      }
    }
    // Average gradient based on number of maximum elements in the
    // segment
    if (counter < 2) {
      return;
    }
    for (int64_t j = offset_start; j < offset_end; ++j) {
      int64_t starting_index = (j * stride_count) + lane_id;
      if (grad_input_data[starting_index] > 0) {
        grad_input_data[starting_index] =
            grad_input_data[starting_index] / counter;
      }
    }
  } else if (reduction == SegmentReductionType::MEAN) {
    auto grad_val = grad_data[output_index] / lengths_data[row_id];
    for (int64_t j = offset_start; j < offset_end; ++j) {
      int64_t starting_index = (j * stride_count) + lane_id;
      grad_input_data[starting_index] = grad_val;
    }
  } else if (reduction == SegmentReductionType::SUM) {
    const auto& grad_val = grad_data[output_index];
    for (int64_t j = offset_start; j < offset_end; ++j) {
      int64_t starting_index = (j * stride_count) + lane_id;
      grad_input_data[starting_index] = grad_val;
    }
  }
}

} // namespace

Tensor _segment_reduce_cuda_backward_kernel(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    SegmentReductionType reduction,
    const Tensor& lengths_contig,
    int64_t axis) {
  int64_t segment_count = lengths_contig.numel();
  auto output_shape = data_contig.sizes().vec();
  output_shape[axis] = segment_count;
  auto grad_input = at::zeros({data_contig.sizes()}, grad_contig.options());

  int64_t stride_count = data_contig.numel() / data_contig.size(axis);

  auto offsets = _get_complete_sum(lengths_contig);

  constexpr int threads_per_block = 256;
  int64_t num_blocks =
      ((segment_count * stride_count) + threads_per_block - 1) /
      threads_per_block;

  num_blocks = std::max(num_blocks, (int64_t)1);

  AT_DISPATCH_INDEX_TYPES(
      lengths_contig.type(), "_segment_reduce_cuda_backward_kernel1", ([&] {
        const auto* lengths_data = lengths_contig.data_ptr<index_t>();
        auto* offsets_data = offsets.data_ptr<index_t>();

        // TODO: Swtich to TensorIterator for better maintainablility and
        // readability
        AT_DISPATCH_FLOATING_TYPES_AND2(
            kBFloat16,
            kHalf,
            data_contig.scalar_type(),
            "_segment_reduce_cpu",
            ([&]() {
              auto* output_data = output_contig.data_ptr<scalar_t>();
              auto* grad_data = grad_contig.data_ptr<scalar_t>();
              auto* grad_input_data = grad_input.data_ptr<scalar_t>();
              const auto* values_data = data_contig.data_ptr<scalar_t>();

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
                      stride_count);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            }));
      }));
  return grad_input;
}

Tensor _segment_reduce_cuda_kernel(
    SegmentReductionType reduction,
    const Tensor& data,
    const Tensor& lengths,
    int64_t axis,
    const c10::optional<Scalar>& initial) {
  int64_t segment_count = lengths.numel();
  auto output_shape = data.sizes().vec();
  output_shape[axis] = segment_count;
  auto output = at::empty(output_shape, data.options());

  int64_t stride_count = data.numel() / data.size(axis);

  auto offsets = _get_complete_sum(lengths);

  constexpr int threads_per_block = 256;
  int64_t num_blocks =
      ((segment_count * stride_count) + threads_per_block - 1) /
      threads_per_block;

  num_blocks = std::max(num_blocks, (int64_t)1);

  AT_DISPATCH_INDEX_TYPES(
      lengths.type(), "_segment_reduce_cuda_kernel1", ([&] {
        auto* offsets_data_ptr = offsets.data_ptr<index_t>();
        auto* lengths_data_ptr = lengths.data_ptr<index_t>();
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            data.scalar_type(),
            "segment_reduce_cuda",
            [&]() {
              auto* data_data_ptr = data.data_ptr<scalar_t>();
              auto* output_data_ptr = output.data_ptr<scalar_t>();

              // initialize starting value
              scalar_t initial_value;
              if (initial.has_value()) {
                initial_value = initial.value().to<scalar_t>();
              } else if (reduction == SegmentReductionType::MAX) {
                initial_value = -std::numeric_limits<scalar_t>::infinity();
              } else if (
                  reduction == SegmentReductionType::MEAN ||
                  reduction == SegmentReductionType::SUM) {
                initial_value = 0;
              } else if (reduction == SegmentReductionType::MIN) {
                initial_value = std::numeric_limits<scalar_t>::infinity();
              }

              if (output_shape.size() > 1) {
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
                        stride_count,
                        initial.has_value(),
                        initial_value);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              } else {
                if (reduction == SegmentReductionType::MAX) {
                  CustomMax max_op{};
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
                } else if (reduction == SegmentReductionType::MIN) {
                  CustomMin min_op{};
                  CUB_WRAPPER(
                      cub::DeviceSegmentedReduce::Reduce,
                      data_data_ptr,
                      output_data_ptr,
                      segment_count,
                      offsets_data_ptr,
                      offsets_data_ptr + 1,
                      min_op,
                      initial_value,
                      at::cuda::getCurrentCUDAStream());
                } else if (reduction == SegmentReductionType::SUM) {
                  CustomSum sum_op{};
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
                }
              }
            });
      }));

  return output;
}

REGISTER_DISPATCH(_segment_reduce_stub, &_segment_reduce_cuda_kernel);
REGISTER_DISPATCH(
    _segment_reduce_backward_stub,
    &_segment_reduce_cuda_backward_kernel);

} // namespace native
} // namespace at
