#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <cmath>

namespace at {
namespace native {

namespace {
__global__ void ChooseQuantizationParamsKernelImpl(
    const float* x_min,
    const float* x_max,
    int32_t qmin,
    int32_t qmax,
    int size,
    bool preserve_sparsity,
    float* scale,
    int32_t* zero_point) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
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
    if (scale[i] == 0.0f || std::isinf(1.0f / scale[i])) {
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
    const float* averaging_const,
    const bool validate) {
  if (validate && *observer_on == 0) {
    if (std::isinf(*running_min) || std::isinf(*running_max)) {
      CUDA_KERNEL_ASSERT(
          false &&
          "Expected running_min/max values to not be inf when FakeQuant is ON and Observer is OFF");
    }
  }

  if (*observer_on == 1) {
    float curr_min = *x_min;
    float curr_max = *x_max;

    float adjusted_min = std::isinf(*running_min)
        ? curr_min
        : (*running_min) + *averaging_const * (curr_min - (*running_min));

    float adjusted_max = std::isinf(*running_max)
        ? curr_max
        : (*running_max) + *averaging_const * (curr_max - (*running_max));

    *running_min = adjusted_min;
    *running_max = adjusted_max;
  }
}

void _calculate_moving_average(
    const at::Tensor& x,
    const at::Tensor& observer_on,
    const at::Tensor& averaging_constant,
    at::Tensor& running_min,
    at::Tensor& running_max,
    const bool& validate = false) {
  at::Tensor x_min, x_max, curr_min, curr_max;
  std::tie(x_min, x_max) = at::_aminmax(x);
  float* x_min_data = x_min.data_ptr<float>();
  float* x_max_data = x_max.data_ptr<float>();
  int64_t* observer_on_data = observer_on.data_ptr<int64_t>();
  float* running_min_data = running_min.data_ptr<float>();
  float* running_max_data = running_max.data_ptr<float>();
  float* averaging_const = averaging_constant.data_ptr<float>();
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
  // Moving Average Min/Max observer for activations
  MovingAverageMinMax<<<1, 1, 0, cuda_stream>>>(
      observer_on_data,
      x_min_data,
      x_max_data,
      running_min_data,
      running_max_data,
      averaging_const,
      validate);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void _calc_moving_avg_qparams_helper(
    const at::Tensor& x,
    at::Tensor& running_min,
    at::Tensor& running_max,
    float* scale_ptr,
    int32_t* zp_ptr,
    int32_t qmin,
    int32_t qmax,
    bool symmetric_quant,
    bool per_row_fq = false) {
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

  if (per_row_fq) {
    at::Tensor w_min, w_max;
    std::tie(w_min, w_max) = at::_aminmax(x, 1);
    float* w_min_data = w_min.data_ptr<float>();
    float* w_max_data = w_max.data_ptr<float>();
    int64_t size = x.size(0);

    int num_threads = std::min(size, (int64_t)512);
    const uint64_t num_blocks = cuda::ATenCeilDiv<uint64_t>(size, num_threads);
    ChooseQuantizationParamsKernelImpl<<<num_blocks, num_threads, 0, cuda_stream>>>(
        w_min_data,
        w_max_data,
        qmin,
        qmax,
        size,
        symmetric_quant /*preserve_sparsity*/,
        scale_ptr,
        zp_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    float* running_min_data = running_min.data_ptr<float>();
    float* running_max_data = running_max.data_ptr<float>();
    ChooseQuantizationParamsKernelImpl<<<1, 1, 0, cuda_stream>>>(
        running_min_data,
        running_max_data,
        qmin,
        qmax,
        /*size=*/1,
        symmetric_quant, /*preserve_sparsity=*/
        scale_ptr,
        zp_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

at::Tensor _observe_and_fake_quant_gpu(
    const at::Tensor& x,
    const at::Tensor& observer_on,
    const at::Tensor& fake_quant_on,
    const at::Tensor& averaging_const,
    at::Tensor& running_min,
    at::Tensor& running_max,
    at::Tensor& scale,
    at::Tensor& zero_point,
    const int64_t qmin,
    const int64_t qmax,
    const int64_t ch_axis,
    bool per_row_fq,
    bool symmetric_quant) {
  const auto x_contig = x.contiguous();
  _calculate_moving_average(
      x_contig, observer_on, averaging_const, running_min, running_max);

  auto fake_quant = fake_quant_on.item().toInt();

  if (fake_quant == 1) {
    int64_t size = per_row_fq ? x.size(0) : 1;
    float* scale_ptr = scale.data_ptr<float>();
    int32_t* zp_ptr = zero_point.data_ptr<int32_t>();

    _calc_moving_avg_qparams_helper(
        x_contig,
        running_min,
        running_max,
        scale_ptr,
        zp_ptr,
        qmin,
        qmax,
        symmetric_quant, /* preserve_sparsity */
        per_row_fq);

    at::Tensor output = at::empty_like(x, x.options(), MemoryFormat::Preserve);
    if (per_row_fq) {
      return at::fake_quantize_per_channel_affine(
          x, scale, zero_point, 0, qmin, qmax);
    } else {
      return at::fake_quantize_per_tensor_affine(
          x, scale, zero_point, qmin, qmax);
    }
  }
  return x;
}
} // namespace

Tensor fused_moving_avg_obs_fake_quant_cuda(
    const at::Tensor& x,
    const at::Tensor& observer_on,
    const at::Tensor& fake_quant_on,
    const at::Tensor& averaging_const,
    at::Tensor& running_min,
    at::Tensor& running_max,
    at::Tensor& scale,
    at::Tensor& zero_point,
    const int64_t qmin,
    const int64_t qmax,
    const int64_t ch_axis,
    bool per_row_fq,
    bool symmetric_quant) {
  const auto x_contig = x.contiguous();
  _calculate_moving_average(
      x_contig, observer_on, averaging_const, running_min, running_max);

  auto fake_quant = fake_quant_on.item().toInt();

  if (fake_quant == 1) {
    int64_t size = per_row_fq ? x.size(0) : 1;
    float* scale_ptr = scale.data_ptr<float>();
    int32_t* zp_ptr = zero_point.data_ptr<int32_t>();

    _calc_moving_avg_qparams_helper(
        x_contig,
        running_min,
        running_max,
        scale_ptr,
        zp_ptr,
        qmin,
        qmax,
        symmetric_quant, /* preserve_sparsity */
        per_row_fq);

    at::Tensor output = at::empty_like(x, x.options(), MemoryFormat::Preserve);
    if (per_row_fq) {
      return at::fake_quantize_per_channel_affine(
          x, scale, zero_point, 0, qmin, qmax);
    } else {
      return at::fake_quantize_per_tensor_affine(
          x, scale, zero_point, qmin, qmax);
    }
  }
  return x;
}
} // namespace native
} // namespace at
