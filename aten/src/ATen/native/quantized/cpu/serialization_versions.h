#pragma once

#include <ATen/ATen.h>
#include <ATen/core/List.h>

#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>

#include <tuple>

using SerializationType = std::tuple<
  // version
  std::string,
  // non-optional tensors
  std::vector<at::Tensor>,
  // optional tensors
  std::vector<c10::optional<at::Tensor>>>;

/* Convolution prepacked parameters serialization.
 *
 * Files that need to be updated if version changes:
 * - ATen/native/quantized/cpu/fbgemm_utils.cpp
 * - jit/passes/onnx/unpack_quantized_weights.cpp
 *
 * Version 1
 * - Fields:
 *  1. weight
 *  2. bias
 *  3. stride x kSpatialDim
 *  4. padding x kSpatialDim
 *  5. dilation x kSpatialDim
 *  6. groups
 *
 * Version 2
 * - Fields:
 *  0. version (string)
 *  1. list of non-optional tensors (int64_t)
 *    0: packed parameters
 *      - kSpatialDim
 *      - stride x kSpatialDim
 *      - padding x kSpatialDim
 *      - dilation x kSpatialDim
 *      - groups
 *    1: weight
 *  2. list of optional tensors
 *    0: bias
 */
const std::string kConvName = "conv";
const std::vector<int64_t> kConvPackedParamsSerializationVersion = {2};
template <uint32_t kSpatialDim>
SerializationType serialize_conv(
    const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& params) {

  std::string version = "2";
  std::vector<at::Tensor> non_optional;
  std::vector<c10::optional<at::Tensor>> optional;

  // create a packed int64_t tensor for conv params
  std::vector<int64_t> params_vec;
  params_vec.push_back(kSpatialDim);
  auto stride = params->stride().vec();
  params_vec.insert(params_vec.end(), stride.begin(), stride.end());
  auto padding = params->padding().vec();
  params_vec.insert(params_vec.end(), padding.begin(), padding.end());
  auto dilation = params->dilation().vec();
  params_vec.insert(params_vec.end(), dilation.begin(), dilation.end());
  params_vec.push_back(params->groups());
  int64_t vec_size = params_vec.size();
  at::Tensor params_tensor = at::from_blob(
      params_vec.data(), {vec_size},
      at::TensorOptions().dtype(at::kLong));

  at::Tensor weight;
  c10::optional<at::Tensor> bias;
  std::tie(weight, bias) = params->unpack();

  // create the non_optional tensor
  // TODO(before land): fix clone
  non_optional.emplace_back(params_tensor.clone());
  non_optional.emplace_back(weight.clone());
  optional.emplace_back(std::move(bias));

  return std::tie(version, non_optional, optional);
}

template <uint32_t kSpatialDim>
c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> deserialize_conv(
    SerializationType state) {

  std::string version;
  std::vector<at::Tensor> non_optional;
  std::vector<c10::optional<at::Tensor>> optional;

  std::tie(version, non_optional, optional) = state;

  // c10::optional<at::Tensor> bias;
  torch::List<int64_t> stride, padding, dilation;
  int64_t groups;

  at::Tensor conv_params_packed = non_optional[0];
  at::Tensor weight = non_optional[1];
  c10::optional<at::Tensor> bias = optional[0];

  // skip kSpatialDim
  int idx = 1;
  for (; idx < kSpatialDim + 1; ++idx) {
    stride.emplace_back(conv_params_packed[idx].item<int64_t>());
  }
  for (; idx < 2 * kSpatialDim + 1; ++idx) {
    padding.emplace_back(conv_params_packed[idx].item<int64_t>());
  }
  for (; idx < 3 * kSpatialDim + 1; ++idx) {
    dilation.emplace_back(conv_params_packed[idx].item<int64_t>());
  }
  groups = conv_params_packed[idx].item<int64_t>();

  auto& ctx = at::globalContext();

#ifdef USE_FBGEMM
  if (ctx.qEngine() == at::QEngine::FBGEMM) {
    return PackedConvWeight<kSpatialDim>::prepack(
      weight,
      bias,
      stride,
      padding,
      dilation,
      groups
    );
  }
#endif // USE_FBGEMM
#ifdef USE_PYTORCH_QNNPACK
  if (ctx.qEngine() == at::QEngine::QNNPACK) {
    TORCH_CHECK(
        kSpatialDim == 2,
        "prepack/__setstate__: QNNPACK only supports Conv2d "
        "now.");
    return PackedConvWeightsQnnp<kSpatialDim>::prepack(
      weight,
      bias,
      stride,
      padding,
      dilation,
      groups
    );
  }
#endif // USE_PYTORCH_QNNPACK
TORCH_CHECK(
  false,
  "Didn't find engine for when deserializing ConvPackedParams: ",
  toString(ctx.qEngine()));
}
