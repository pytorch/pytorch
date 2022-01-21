#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <c10/util/irange.h>
#include <cmath>
#include <tuple>
#include <vector>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.h>
#include <ATen/ops/_fused_moving_avg_obs_fq_helper.h>
#include <ATen/ops/_fused_moving_avg_obs_fq_helper_native.h>
#include <ATen/ops/aminmax.h>
#include <ATen/ops/fake_quantize_per_channel_affine_cachemask.h>
#include <ATen/ops/fused_moving_avg_obs_fake_quant_native.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/ones_like.h>
#endif

#ifdef USE_FBGEMM
#include <fbgemm/QuantUtils.h>
#endif
#include <ATen/native/quantized/cpu/quant_utils.h>

namespace {
void calculate_moving_average(
    const at::Tensor& x,
    at::Tensor& running_min,
    at::Tensor& running_max,
    float averaging_const,
    bool per_row_fake_quant,
    int ch_axis) {
  at::Tensor x_min, x_max;
  if (per_row_fake_quant) {
    TORCH_CHECK(
        ch_axis == 0,
        "Per-channel FakeQuant in fused_moving_avg_obs_fake_quant is only supported on axis == 0");
    std::tie(x_min, x_max) = at::aminmax(x, 1);
  } else {
    std::tie(x_min, x_max) = at::aminmax(x);
  }
  const float* min_curr_val = x_min.data_ptr<float>();
  const float* max_curr_val = x_max.data_ptr<float>();
  // Moving Average Min/Max observer for input tensor
  float* running_min_val = running_min.data_ptr<float>();
  float* running_max_val = running_max.data_ptr<float>();
  for (const auto i : c10::irange(x_min.numel())) {
    running_min_val[i] = std::isinf(running_min_val[i]) ? min_curr_val[i]
                                                        : running_min_val[i] +
            averaging_const * (min_curr_val[i] - running_min_val[i]);
    running_max_val[i] = std::isinf(running_max_val[i]) ? max_curr_val[i]
                                                        : running_max_val[i] +
            averaging_const * (max_curr_val[i] - running_max_val[i]);
  }

  return;
}

std::tuple<at::Tensor, at::Tensor> choose_qparams_fake_quant(
    const at::Tensor& x,
    const at::Tensor& inp_running_min,
    const at::Tensor& inp_running_max,
    at::Tensor& scale,
    at::Tensor& zero_point,
    bool per_row_fake_quant,
    bool symmetric_quant,
    int qmin,
    int qmax,
    int ch_axis) {
  std::tuple<at::Tensor, at::Tensor> fake_quant_out;
  at::Tensor x_min, x_max;
  if (per_row_fake_quant) {
    float* x_min_data = inp_running_min.data_ptr<float>();
    float* x_max_data = inp_running_max.data_ptr<float>();
    for (const auto i : c10::irange(inp_running_min.numel())) {
#ifdef USE_FBGEMM
      fbgemm::TensorQuantizationParams x_qparams{};
      x_qparams = fbgemm::ChooseQuantizationParams(
          x_min_data[i],
          x_max_data[i],
          qmin,
          qmax,
          symmetric_quant, // preserve sparsity
          false // force power of two
      );
      scale[i] = x_qparams.scale;
      zero_point[i] = x_qparams.zero_point;
#else
      quant_utils::TensorQuantizationParams x_qparams{};
      x_qparams = quant_utils::ChooseQuantizationParams(
          x_min_data[i],
          x_max_data[i],
          qmin,
          qmax,
          symmetric_quant, // preserve sparsity
          false // force power of two
      );
      scale[i] = x_qparams.scale;
      zero_point[i] = x_qparams.zero_point;
#endif
    }
    fake_quant_out = at::fake_quantize_per_channel_affine_cachemask(
        x, scale, zero_point, ch_axis, qmin, qmax);
  } else {
#ifdef USE_FBGEMM
    fbgemm::TensorQuantizationParams x_qparams{};
    // compute quantization parameters using min-max values
    x_qparams = fbgemm::ChooseQuantizationParams(
        inp_running_min.item().toFloat(),
        inp_running_max.item().toFloat(),
        qmin,
        qmax,
        symmetric_quant, // bool preserve_sparsity
        false // force power of two
    );

    scale[0] = x_qparams.scale;
    zero_point[0] = x_qparams.zero_point;
#else
    quant_utils::TensorQuantizationParams x_qparams{};
    // compute quantization parameters using min-max values
    x_qparams = quant_utils::ChooseQuantizationParams(
        inp_running_min.item().toFloat(),
        inp_running_max.item().toFloat(),
        qmin,
        qmax,
        symmetric_quant, // bool preserve_sparsity
        false // force power of two
    );
    scale[0] = x_qparams.scale;
    zero_point[0] = x_qparams.zero_point;
#endif
    auto fake_quant_enabled = at::ones(1, x.options().dtype(at::kLong));
    fake_quant_out =
        at::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
            x, scale, zero_point, fake_quant_enabled, qmin, qmax);
  }
  return fake_quant_out;
}
} // namespace

namespace at {
namespace native {

std::tuple<at::Tensor, at::Tensor> fused_moving_avg_obs_fake_quant_cpu(
    const at::Tensor& self,
    const at::Tensor& observer_on,
    const at::Tensor& fake_quant_on,
    at::Tensor& running_min,
    at::Tensor& running_max,
    at::Tensor& scale,
    at::Tensor& zero_point,
    const double averaging_const,
    const int64_t quant_min,
    const int64_t quant_max,
    const int64_t ch_axis,
    bool per_row_fake_quant,
    bool symmetric_quant) {
  // Calculate min/max
  auto observe = observer_on.item().toInt();
  // Calculate the size of the dimension we need to quantize over,
  // For per-channel quant we default to axis 0, since it is only for
  // weight quantization currently.
  if (per_row_fake_quant) {
    at::Tensor y = self;
    if (self.dim() != 2) {
      auto res = DimVector(self.sizes());
      std::iota(res.begin(), res.end(), 0);
      res[ch_axis] = 0;
      res[0] = ch_axis;

      y = self.permute(res);
      y = y.flatten(1);
    }
    int64_t size = self.size(ch_axis);
    if (running_min.numel() == 0) {
      float inf = std::numeric_limits<float>::infinity();
      running_min.resize_(size).fill_(inf);
      running_max.resize_(size).fill_(-inf);
      scale.resize_(size);
      zero_point.resize_(size);
    }
    if (observe) {
      calculate_moving_average(
          y,
          running_min,
          running_max,
          averaging_const,
          per_row_fake_quant,
          ch_axis);
    }
  } else {
    if (observe) {
      calculate_moving_average(
          self,
          running_min,
          running_max,
          averaging_const,
          per_row_fake_quant,
          ch_axis);
    }
  }
  // Calculate qparams and fake_quantize
  auto fake_quant = fake_quant_on.item().toInt();
  if (fake_quant) {
    return choose_qparams_fake_quant(
        self,
        running_min,
        running_max,
        scale,
        zero_point,
        per_row_fake_quant,
        symmetric_quant,
        quant_min,
        quant_max,
        ch_axis);
  }
  auto mask = at::ones_like(self, at::kBool, MemoryFormat::Preserve);
  return std::make_tuple(self.clone(), mask);
}

at::Tensor fused_moving_avg_obs_fake_quant(
    const at::Tensor& self,
    const at::Tensor& observer_on,
    const at::Tensor& fake_quant_on,
    at::Tensor& running_min,
    at::Tensor& running_max,
    at::Tensor& scale,
    at::Tensor& zero_point,
    const double averaging_const,
    const int64_t quant_min,
    const int64_t quant_max,
    const int64_t ch_axis,
    bool per_row_fake_quant,
    bool symmetric_quant) {
  if (self.numel() == 0) {
    return self.clone();
  }
  const auto res = at::_fused_moving_avg_obs_fq_helper(
      self,
      observer_on,
      fake_quant_on,
      running_min,
      running_max,
      scale,
      zero_point,
      averaging_const,
      quant_min,
      quant_max,
      ch_axis,
      per_row_fake_quant,
      symmetric_quant);
  return std::get<0>(res);
}
} // namespace native
} // namespace at
