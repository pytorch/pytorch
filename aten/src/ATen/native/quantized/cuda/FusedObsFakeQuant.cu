#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/ceil_div.h>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/cuda/CUDAGuard.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/aminmax.h>
#include <ATen/ops/_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.h>
#include <ATen/ops/fake_quantize_per_channel_affine.h>
#include <ATen/ops/fake_quantize_per_channel_affine_cachemask.h>
#include <ATen/ops/fake_quantize_per_tensor_affine.h>
#include <ATen/ops/fused_moving_avg_obs_fake_quant_native.h>
#include <ATen/ops/ones_like.h>
#endif

#include <cmath>

namespace at {
namespace native {

namespace {
__global__ void ChooseQuantizationParamsKernelImpl(
    const int64_t* fake_quant_on,
    const float* x_min,
    const float* x_max,
    int32_t qmin,
    int32_t qmax,
    int size,
    bool preserve_sparsity,
    float* scale,
    int32_t* zero_point) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size && *fake_quant_on == 1) {
    float min_val = x_min[i];
    float max_val = x_max[i];

    if (min_val < 0 && max_val > 0 && preserve_sparsity) {
      int symmetric_qmin = -((qmax - qmin) / 2 + 1);
      int symmetric_qmax = (qmax - qmin) / 2;
      double max_scale = std::max(
          fabs(min_val / symmetric_qmin), fabs(max_val / symmetric_qmax));
      min_val = max_scale * symmetric_qmin;
      max_val = max_scale * symmetric_qmax;
    }

    // We extend the [min, max] interval to ensure that it contains 0.
    // Otherwise, we would not meet the requirement that 0 be an exactly
    // representable value.
    min_val = std::min(min_val, 0.f);
    max_val = std::max(max_val, 0.f);
    scale[i] = (static_cast<double>(max_val) - min_val) / (qmax - qmin);

    // Moving this check outside this function would result in extra Device to
    // Host copy of the min and max val which would result in a perf hit.
    if (scale[i] == 0.0f || ::isinf(1.0f / scale[i])) {
      scale[i] = 0.1;
    }

    double zero_point_from_min = qmin - min_val / static_cast<double>(scale[i]);
    double zero_point_from_max = qmax - max_val / static_cast<double>(scale[i]);
    double zero_point_from_min_error =
        std::abs(qmin) + std::abs(min_val / static_cast<double>(scale[i]));
    double zero_point_from_max_error =
        std::abs(qmax) + std::abs(max_val / static_cast<double>(scale[i]));
    double initial_zero_point =
        zero_point_from_min_error < zero_point_from_max_error
        ? zero_point_from_min
        : zero_point_from_max;

    // Note: preserve_sparsity here means symmetric quantization.
    // for symmetric quantization, we force zero_point
    // to be a middle value between qmin and qmax.
    // If either min or max is 0, then we just use 0 as zero_point.
    if (min_val < 0 && max_val > 0 && preserve_sparsity) {
      initial_zero_point = static_cast<double>(qmin + qmax) / 2;
    }
    // Now we need to nudge the zero point to be an integer
    // (our zero points are integer, and this is motivated by the
    // requirement to be able to represent the real value "0" exactly as a
    // quantized value, which is required in multiple places, for example in
    // Im2col with zero padding).
    int32_t nudged_zero_point = 0;
    if (initial_zero_point < qmin) {
      nudged_zero_point = qmin;
    } else if (initial_zero_point > qmax) {
      nudged_zero_point = qmax;
    } else {
      nudged_zero_point = nearbyint(initial_zero_point);
    }
    zero_point[i] = nudged_zero_point;
  }
}

// CUDA kernel to compute Moving Average Min/Max of the tensor.
// It uses the running_min and running_max along with averaging const, c.
// The formula used to compute the new min/max is as follows
//
// running_min = (1 - c) * running_min + c * x_min, if running_min != inf
// running_min = x_min, if running_min == inf
__global__ void MovingAverageMinMax(
    const int64_t* observer_on,
    const float* x_min,
    const float* x_max,
    float* running_min,
    float* running_max,
    const float averaging_const,
    const int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (*observer_on == 1) {
    if (i < size) {
      float curr_min = x_min[i];
      float curr_max = x_max[i];

      float adjusted_min = ::isinf(running_min[i])
          ? curr_min
          : (running_min[i]) + averaging_const * (curr_min - (running_min[i]));

      float adjusted_max = ::isinf(running_max[i])
          ? curr_max
          : (running_max[i]) + averaging_const * (curr_max - (running_max[i]));

      running_min[i] = adjusted_min;
      running_max[i] = adjusted_max;
    }
  }
}

void _calculate_moving_average(
    const at::Tensor& x,
    const at::Tensor& observer_on,
    at::Tensor& running_min,
    at::Tensor& running_max,
    const float averaging_const,
    const int64_t size,
    bool per_row_fq) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(x.get_device());

  at::Tensor x_min, x_max;

  int64_t* observer_on_data = observer_on.data_ptr<int64_t>();
  float* running_min_data = running_min.data_ptr<float>();
  float* running_max_data = running_max.data_ptr<float>();
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

  if (per_row_fq) {
    std::tie(x_min, x_max) = at::aminmax(x, 1);
    float* x_min_data = x_min.data_ptr<float>();
    float* x_max_data = x_max.data_ptr<float>();
    int num_threads = std::min(size, (int64_t)512);
    const uint64_t num_blocks = ceil_div<uint64_t>(size, num_threads);

    // Moving Average Min/Max observer for activations
    MovingAverageMinMax<<<num_blocks, num_threads, 0, cuda_stream>>>(
        observer_on_data,
        x_min_data,
        x_max_data,
        running_min_data,
        running_max_data,
        averaging_const,
        size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    std::tie(x_min, x_max) = at::aminmax(x);
    float* x_min_data = x_min.data_ptr<float>();
    float* x_max_data = x_max.data_ptr<float>();
    // Moving Average Min/Max observer for activations
    MovingAverageMinMax<<<1, 1, 0, cuda_stream>>>(
        observer_on_data,
        x_min_data,
        x_max_data,
        running_min_data,
        running_max_data,
        averaging_const,
        1 /*size*/);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void _calc_moving_avg_qparams_helper(
    const at::Tensor& x,
    const at::Tensor fake_quant_on,
    at::Tensor& running_min,
    at::Tensor& running_max,
    float* scale_ptr,
    int32_t* zp_ptr,
    int32_t qmin,
    int32_t qmax,
    bool symmetric_quant,
    const int64_t size,
    bool per_row_fq = false) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(x.get_device());

  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  int64_t* fake_quant_on_data = fake_quant_on.data_ptr<int64_t>();
  if (per_row_fq) {
    float* running_min_data = running_min.data_ptr<float>();
    float* running_max_data = running_max.data_ptr<float>();
    int num_threads = std::min(size, (int64_t)512);
    const uint64_t num_blocks = ceil_div<uint64_t>(size, num_threads);
    ChooseQuantizationParamsKernelImpl<<<num_blocks, num_threads, 0, cuda_stream>>>(
        fake_quant_on_data,
        running_min_data,
        running_max_data,
        qmin,
        qmax,
        size,
        symmetric_quant,
        scale_ptr,
        zp_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    float* running_min_data = running_min.data_ptr<float>();
    float* running_max_data = running_max.data_ptr<float>();
    ChooseQuantizationParamsKernelImpl<<<1, 1, 0, cuda_stream>>>(
        fake_quant_on_data,
        running_min_data,
        running_max_data,
        qmin,
        qmax,
        1, // size
        symmetric_quant, // preserve_sparsity
        scale_ptr,
        zp_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

} // namespace

std::tuple<at::Tensor, at::Tensor> fused_moving_avg_obs_fake_quant_cuda(
    const at::Tensor& x,
    const at::Tensor& observer_on,
    const at::Tensor& fake_quant_on,
    at::Tensor& running_min,
    at::Tensor& running_max,
    at::Tensor& scale,
    at::Tensor& zero_point,
    const double averaging_const,
    const int64_t qmin,
    const int64_t qmax,
    const int64_t ch_axis,
    bool per_row_fq,
    bool symmetric_quant) {
  TORCH_CHECK(ch_axis < x.dim(), "Error in fused_moving_avg_obs_fake_quant_cpu: ch_axis must be < self.dim()");
  const auto x_contig = x.contiguous();
  // Calculate the size of the dimension we need to quantize over,
  // For per-channel quant we default to axis 0, since it is only for
  // weight quantization currently.
  int64_t size = 1;
  if (per_row_fq) {
    at::Tensor y = x;
    if (x.dim() != 2) {
      auto res = DimVector(x.sizes());
      std::iota(res.begin(), res.end(), 0);
      res[ch_axis] = 0;
      res[0] = ch_axis;

      y = x.permute(res);
      y = y.flatten(1);
    }
    size = x.size(ch_axis);
    if (running_min.numel() == 0) {
      float inf = std::numeric_limits<float>::infinity();
      running_min.resize_(size).fill_(inf);
      running_max.resize_(size).fill_(-inf);
      scale.resize_(size);
      zero_point.resize_(size);
    }
    _calculate_moving_average(
        y,
        observer_on,
        running_min,
        running_max,
        averaging_const,
        size,
        per_row_fq);
  } else {
    _calculate_moving_average(
        x_contig,
        observer_on,
        running_min,
        running_max,
        averaging_const,
        size,
        per_row_fq);
  }

  float* scale_ptr = scale.data_ptr<float>();
  int32_t* zp_ptr = zero_point.data_ptr<int32_t>();

  _calc_moving_avg_qparams_helper(
      x_contig,
      fake_quant_on,
      running_min,
      running_max,
      scale_ptr,
      zp_ptr,
      qmin,
      qmax,
      symmetric_quant,
      size,
      per_row_fq);

  if (per_row_fq) {
    if (fake_quant_on.item().toInt()) {
      return at::fake_quantize_per_channel_affine_cachemask(
          x, scale, zero_point, 0, qmin, qmax);
    } else {
      auto mask = at::ones_like(x, at::kBool, MemoryFormat::Preserve);
      return std::make_tuple(x.clone(), mask);
    }
  } else {
    return at::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        x, scale, zero_point, fake_quant_on, qmin, qmax);
  }
}
} // namespace native
} // namespace at
