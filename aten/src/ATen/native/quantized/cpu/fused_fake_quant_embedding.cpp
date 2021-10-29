#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/quantized/cpu/quant_utils.h>
#include <torch/library.h>
#include <c10/util/irange.h>

#include <cstring>
#include <memory>
#include <sstream>
#include <vector>

namespace at {
namespace native {

namespace {

struct TensorQuantizationFloatParams {
  double scale;
  float zero_point;
  int precision;
};

TensorQuantizationFloatParams calc_per_channel_affine_float_qparams(
    float min_val,
    float max_val,
    int32_t qmin,
    int32_t qmax) {

  TORCH_CHECK(
      min_val <= max_val,
      "In ChooseQuantizationParams, min should be less than or equal to max");

  float min_val_neg = std::fmin(min_val, 0.0);
  float min_val_pos = std::fmax(max_val, 0.0);

  float scale = (max_val - min_val) / float(qmax - qmin);
  if(scale <= std::numeric_limits<float>().epsilon()){
    scale = 1.0;
  }
  float zero_point = -1.0 * min_val / scale;

  TensorQuantizationFloatParams result;
  result.scale = scale;
  result.zero_point = zero_point;
  return result;
}

Tensor fused_fake_quant_embedding(const Tensor & weight, const Tensor & indices,
                                  int64_t quant_min, int64_t quant_max,
                                  int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  TORCH_CHECK(weight.dim() == 2,  "'weight' must be 2-D");
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarTypes("embedding", indices_arg, {kLong, kInt});

  // TODO: use tensor.index() after improving perf
  if (indices.dim() == 1) {
    return weight.index_select(0, indices);
  }

  auto size = indices.sizes().vec();
  for (auto d : weight.sizes().slice(1)) {
    size.push_back(d);
  }

  auto weights = weight.index_select(0, indices.reshape(-1));

  at::Tensor fake_quant_w;
  at::Tensor w_min, w_max;
  TensorQuantizationFloatParams w_qparams{};

  // Always per_column
  std::tie(w_min, w_max) = at::_aminmax(weights, 1);
  float* w_min_data = w_min.data_ptr<float>();
  float* w_max_data = w_max.data_ptr<float>();

  at::Tensor scales = at::empty_like(w_min);
  // TODO: Adjust numerics to match per_channel_affine_float_qparams case in torch.ao.quantization.observer.
  at::Tensor zero_points =
      at::empty_like(w_min, w_min.options().dtype(at::kFloat));
  for (const auto i : c10::irange(w_min.numel())) {
    w_qparams = calc_per_channel_affine_float_qparams(
        w_min_data[i],
        w_max_data[i],
        quant_min,
        quant_max);
    scales[i] = w_qparams.scale;
    zero_points[i] = w_qparams.zero_point;
  }

  fake_quant_w = at::fake_quantize_per_channel_affine(weights, scales, zero_points, 0, quant_min, quant_max);
  return fake_quant_w.view(size);
}

TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::fused_fake_quant_embedding"),
      TORCH_FN(fused_fake_quant_embedding));
}

} // namespace
} // namespace native
} // namespace at
