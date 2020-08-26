#pragma once

#include <ATen/ATen.h>
#include <ATen/core/List.h>

#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>

#include <tuple>

/* Convolution prepacked parameters serialization.
 *
 * Version 1
 *
 * - Fields:
 *  1. weight
 *  2. bias
 *  3. stride x kSpatialDim
 *  4. padding x kSpatialDim
 *  5. dilation x kSpatialDim
 *  6. groups
 *
 * Version 2
 *
 * - Fields:
 *  0. version (string)
 *  1. list of non-optional tensors
 *    0: packed parameters (int8_t)
 *      - kSpatialDim
 *      - stride x kSpatialDim
 *      - padding x kSpatialDim
 *      - dilation x kSpatialDim
 *      - groups
 *    1: weight
 *  2. list of optional tensors
 *    0: bias
 *
 *  Note: version is a string and conv params are packed into a Tensor
 *    to make ONNX happy (ints and containers of ints are not supported).
 */
using ConvParamsSerializationType = std::tuple<
  // version
  std::string,
  // non-optional tensors
  std::vector<at::Tensor>,
  // optional tensors
  std::vector<c10::optional<at::Tensor>>>;

// Parses any historical conv packed params format into
// the current format.
template <uint32_t kSpatialDim>
ConvParamsSerializationType parse_conv_serialized_state(c10::IValue v) {

  // determine the version based on IValue contents
  int version = -1;
  if (v.isTuple()) {
    auto elements = v.toTuple()->elements();
    if (elements.size() > 0) {
      auto firstElement = elements[0];
      if (firstElement.isTensor()) {
        version = 1;
      } else if (firstElement.isString()) {
        std::string version_str = firstElement.toStringRef();
        // note: not parsing the string to automatically handle bad
        // inputs
        if (version_str == "2") {
          version = 2;
        }
      }
    }
  }
  TORCH_INTERNAL_ASSERT(version != -1, "Unable to parse serialization version");

  ConvParamsSerializationType state;

  if (version == 1) {
    // version 1 - convert to version 2 manually

    auto elements = v.toTuple()->elements();

    at::Tensor weight = elements[0].toTensor();
    c10::optional<at::Tensor> bias = elements[1].toOptional<at::Tensor>();
    torch::List<at::Tensor> stride_x_kSpatialDim = elements[2].toTensorList();
    torch::List<at::Tensor> padding_x_kSpatialDim = elements[3].toTensorList();
    torch::List<at::Tensor> dilation_x_kSpatialDim = elements[4].toTensorList();
    at::Tensor groups = elements[5].toTensor();

    std::string version = "2";
    std::vector<at::Tensor> non_optional;
    std::vector<c10::optional<at::Tensor>> optional;

    std::vector<int8_t> params_vec;
    params_vec.push_back(kSpatialDim);
    for (int i = 0; i < stride_x_kSpatialDim.size(); i++) {
      auto stride = stride_x_kSpatialDim.get(i);
      params_vec.push_back(stride[0].item<int8_t>());
    }
    for (int i = 0; i < padding_x_kSpatialDim.size(); i++) {
      auto padding = padding_x_kSpatialDim.get(i);
      params_vec.push_back(padding[0].item<int8_t>());
    }
    for (int i = 0; i < dilation_x_kSpatialDim.size(); i++) {
      auto dilation = dilation_x_kSpatialDim.get(i);
      params_vec.push_back(dilation[0].item<int8_t>());
    }
    params_vec.push_back(groups[0].item<int8_t>());
    int64_t vec_size = params_vec.size();
    at::Tensor params_tensor = at::from_blob(params_vec.data(),
        {vec_size}, at::TensorOptions().dtype(at::kChar))
      // clone to retain ownership of the data
      .clone();

    non_optional.emplace_back(std::move(params_tensor));
    non_optional.emplace_back(std::move(weight));
    optional.emplace_back(std::move(bias));

    state = std::tie(version, non_optional, optional);

  } else {
    // version 2
    state = v.to<ConvParamsSerializationType>();
  }

  return state;
}

template <uint32_t kSpatialDim>
ConvParamsSerializationType serialize_conv(
    const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& params) {

  std::string version = "2";
  std::vector<at::Tensor> non_optional;
  std::vector<c10::optional<at::Tensor>> optional;

  // create a packed int8_t tensor for conv params
  std::vector<int8_t> params_vec;
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
      at::TensorOptions().dtype(at::kChar))
    // clone to retain ownership of the data
    .clone();

  at::Tensor weight;
  c10::optional<at::Tensor> bias;
  std::tie(weight, bias) = params->unpack();

  non_optional.emplace_back(std::move(params_tensor));
  non_optional.emplace_back(std::move(weight));
  optional.emplace_back(std::move(bias));

  return std::tie(version, non_optional, optional);
}

template <uint32_t kSpatialDim>
c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> deserialize_conv(
    ConvParamsSerializationType state) {

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
