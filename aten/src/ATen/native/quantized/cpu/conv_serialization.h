#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/OnednnUtils.h>
#include <c10/util/irange.h>
#if !defined(__s390x__) && !defined(__powerpc__)
#include <cpuinfo.h>
#endif

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/from_blob.h>
#endif


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
 *    0: packed parameters (int16_t)
 *      - kSpatialDim
 *      - stride x kSpatialDim
 *      - padding x kSpatialDim
 *      - dilation x kSpatialDim
 *      - output_padding x kSpatialDim
 *      - groups
 *      - transpose (0 or 1)
 *    1: weight
 *  2. list of optional tensors
 *    0: bias
 *
 * Version 3
 *
 * - Fields:
 *  0. version (int64_t)
 *  1. list of int64_t configuration values
 *    - kSpatialDim
 *    - stride x kSpatialDim
 *    - padding x kSpatialDim
 *    - dilation x kSpatialDim
 *    - output_padding x kSpatialDim
 *    - groups
 *    - flags (bitmask)
 *      - (1 << 0) transpose (1 = yes)
 *  2. list of optional tensors
 *    0: None (helps with type inference)
 *    1: weight (this must be present)
 *    2: bias
 */

using ConvParamsSerializationTypeV2 = std::tuple<
  // version, for versions 2 and up
  std::string,
  // non-optional tensors
  std::vector<at::Tensor>,
  // optional tensors
  std::vector<c10::optional<at::Tensor>>>;

using ConvParamsSerializationTypeV3 = std::tuple<
  // version, int for versions 3 and up
  int64_t,
  // configuration values
  std::vector<int64_t>,
  // optional tensors
  std::vector<c10::optional<at::Tensor>>>;

// Parses any historical conv packed params format into
// the current format.
template <uint32_t kSpatialDim>
ConvParamsSerializationTypeV3 parse_conv_serialized_state(c10::IValue v) {

  // determine the version based on IValue contents
  int version = -1;
  if (v.isTuple()) {
    const auto& elements = v.toTupleRef().elements();
    if (!elements.empty()) {
      auto firstElement = elements[0];
      if (firstElement.isTensor()) {
        version = 1;
      } else if (firstElement.isString()) {
        const std::string& version_str = firstElement.toStringRef();
        // note: not parsing the string to automatically handle bad
        // inputs
        if (version_str == "2") {
          version = 2;
        }
      } else if (firstElement.isInt()) {
        auto raw_version = firstElement.toInt();
        if (raw_version == 3) {
          version = 3;
        }
      }
    }
  }
  TORCH_INTERNAL_ASSERT(version != -1, "Unable to parse serialization version");

  if (version == 1) {
    // version 1 - convert to version 3 manually

    const auto& elements = v.toTupleRef().elements();

    at::Tensor weight = elements[0].toTensor();
    c10::optional<at::Tensor> bias = elements[1].toOptional<at::Tensor>();
    torch::List<at::Tensor> stride_x_kSpatialDim = elements[2].toTensorList();
    torch::List<at::Tensor> padding_x_kSpatialDim = elements[3].toTensorList();
    torch::List<at::Tensor> dilation_x_kSpatialDim = elements[4].toTensorList();
    at::Tensor groups = elements[5].toTensor();

    std::vector<int64_t> config_vals;
    config_vals.reserve(
        stride_x_kSpatialDim.size() + padding_x_kSpatialDim.size() +
        dilation_x_kSpatialDim.size() + kSpatialDim + 3);
    config_vals.push_back(kSpatialDim);
    for (const auto i : c10::irange(stride_x_kSpatialDim.size())) {
      auto stride = stride_x_kSpatialDim.get(i);
      config_vals.push_back(stride[0].item<int16_t>());
    }
    for (const auto i : c10::irange(padding_x_kSpatialDim.size())) {
      auto padding = padding_x_kSpatialDim.get(i);
      config_vals.push_back(padding[0].item<int16_t>());
    }
    for (const auto i : c10::irange(dilation_x_kSpatialDim.size())) {
      auto dilation = dilation_x_kSpatialDim.get(i);
      config_vals.push_back(dilation[0].item<int16_t>());
    }
    // output_padding does not exist in v1, so we fill in a default value
    for (C10_UNUSED const auto i : c10::irange(kSpatialDim)) {
      config_vals.push_back(0);
    }
    config_vals.push_back(groups[0].item<int16_t>());
    // transpose does not exist in v1, so we fill in a default value
    config_vals.push_back(0);

    std::vector<c10::optional<at::Tensor>> tensors;
    tensors.emplace_back();
    tensors.emplace_back(weight);
    tensors.emplace_back(bias);

    int64_t version = 3;
    return std::tie(version, config_vals, tensors);
  } else if (version == 2) {
    // version 2
    const auto& elements = v.toTupleRef().elements();
    std::vector<at::Tensor> non_optional = elements[1].toTensorList().vec();
    std::vector<c10::optional<at::Tensor>> optional;

    if (elements[2].isTensorList()) {
      for (const auto& elem : elements[2].toTensorList()) {
        optional.emplace_back(static_cast<at::Tensor>(elem));
      }
    } else {
      for (const auto& elem : elements[2].toList()) {
        optional.emplace_back(static_cast<c10::IValue>(elem).toOptional<at::Tensor>());
      }
    }
    // create default optional value for bias
    if (optional.empty()) {
      optional.emplace_back();
    }

    auto config_a = non_optional[0].accessor<int16_t, 1>();
    std::vector<int64_t> config_vals;
    config_vals.reserve(config_a.size(0));
    for (const auto i : c10::irange(config_a.size(0))) {
      config_vals.emplace_back(config_a[i]);
    }

    auto weight = non_optional[1];
    auto bias = optional[0];

    std::vector<c10::optional<at::Tensor>> tensors;
    tensors.emplace_back();
    tensors.emplace_back(weight);
    tensors.emplace_back(bias);

    int64_t version = 3;
    return std::tie(version, config_vals, tensors);
  } else if (version == 3) {
    return v.to<ConvParamsSerializationTypeV3>();
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unexpected serialized qconv version: ",
        version);
  }
}

#define QCONV_SERIALIZATION_VERSION 2

#if QCONV_SERIALIZATION_VERSION == 2
using ConvParamsSerializationType = ConvParamsSerializationTypeV2;

template <uint32_t kSpatialDim>
ConvParamsSerializationTypeV2 serialize_conv(
    const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& params) {

  std::string version = "2";
  std::vector<at::Tensor> non_optional;
  std::vector<c10::optional<at::Tensor>> optional;

  // create a packed int8_t tensor for conv params
  std::vector<int16_t> params_vec;
  params_vec.push_back(kSpatialDim);
  auto stride = params->stride().vec();
  params_vec.insert(params_vec.end(), stride.begin(), stride.end());
  auto padding = params->padding().vec();
  params_vec.insert(params_vec.end(), padding.begin(), padding.end());
  auto dilation = params->dilation().vec();
  params_vec.insert(params_vec.end(), dilation.begin(), dilation.end());
  auto output_padding = params->output_padding().vec();
  params_vec.insert(params_vec.end(), output_padding.begin(),
                    output_padding.end());
  params_vec.push_back(params->groups());
  params_vec.push_back(params->transpose());
  int64_t vec_size = params_vec.size();
  at::Tensor params_tensor = at::from_blob(
      params_vec.data(), {vec_size},
      at::TensorOptions().dtype(at::kShort))
    // clone to retain ownership of the data
    .clone();

  auto [weight, bias] = params->unpack();

  non_optional.emplace_back(std::move(params_tensor));
  non_optional.emplace_back(std::move(weight));
  optional.emplace_back(std::move(bias));

  return std::tie(version, non_optional, optional);
}

#elif QCONV_SERIALIZATION_VERSION == 3
using ConvParamsSerializationType = ConvParamsSerializationTypeV3;

template <uint32_t kSpatialDim>
ConvParamsSerializationTypeV3 serialize_conv(
    const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& params) {
  std::vector<int64_t> config_vals;
  config_vals.push_back(kSpatialDim);
  auto stride = params->stride().vec();
  config_vals.insert(config_vals.end(), stride.begin(), stride.end());
  auto padding = params->padding().vec();
  config_vals.insert(config_vals.end(), padding.begin(), padding.end());
  auto dilation = params->dilation().vec();
  config_vals.insert(config_vals.end(), dilation.begin(), dilation.end());
  auto output_padding = params->output_padding().vec();
  config_vals.insert(config_vals.end(), output_padding.begin(),
                    output_padding.end());
  config_vals.push_back(params->groups());
  config_vals.push_back(params->transpose());

  auto [weight, bias] = params->unpack();

  std::vector<c10::optional<at::Tensor>> tensors;
  tensors.emplace_back();
  tensors.emplace_back(weight);
  tensors.emplace_back(bias);

  int64_t version = 3;
  return std::tie(version, config_vals, tensors);
}

#else
#error "Invalid qconv serialization version."
#endif

template <uint32_t kSpatialDim>
c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> deserialize_conv(
    ConvParamsSerializationTypeV3 state) {
  auto [version, config_vals, tensors] = state;
  TORCH_INTERNAL_ASSERT(version == 3, "Unexpected serialized qconv version: ", version);

  TORCH_CHECK(tensors.size() == 3, "Wrong number of tensors", tensors.size());
  c10::optional<at::Tensor> weight = tensors[1];
  c10::optional<at::Tensor> bias = tensors[2];
  TORCH_INTERNAL_ASSERT(weight, "Weight should always be present in serialized qconv.");

  torch::List<int64_t> stride, padding, output_padding, dilation;
  // skip kSpatialDim
  int idx = 1;
  for (C10_UNUSED const auto i : c10::irange(kSpatialDim)) {
    stride.emplace_back(config_vals.at(idx));
    idx++;
  }
  for (C10_UNUSED const auto i : c10::irange(kSpatialDim)) {
    padding.emplace_back(config_vals.at(idx));
    idx++;
  }
  for (C10_UNUSED const auto i : c10::irange(kSpatialDim)) {
    dilation.emplace_back(config_vals.at(idx));
    idx++;
  }
  for (C10_UNUSED const auto i : c10::irange(kSpatialDim)) {
    TORCH_INTERNAL_ASSERT(idx < static_cast<int64_t>(config_vals.size()),
        "Unexpected index = ", idx, " for config_vals of size ",
        config_vals.size());
    output_padding.emplace_back(config_vals.at(idx));
    idx++;
  }
  int64_t groups = config_vals.at(idx);
  idx++;
  int64_t flags = config_vals.at(idx);
  idx++;
  TORCH_INTERNAL_ASSERT(idx == static_cast<int64_t>(config_vals.size()),
      "Unexpected length of config_vals, expected ",
      idx,
      " got ",
      config_vals.size());

  bool transpose = flags & (1 << 0);

  int64_t other_flags = flags & ~(1 << 0);
  TORCH_INTERNAL_ASSERT(other_flags == 0, "Unexpected flags set in ", flags, ".");

  auto& ctx = at::globalContext();

#ifdef USE_FBGEMM
  if (ctx.qEngine() == at::QEngine::X86) {
#if AT_MKLDNN_ENABLED()
    bool use_onednn = onednn_utils::should_use_onednn_quant(
        weight.value(), transpose, groups, output_padding);
    if (use_onednn) {
      return PackedConvWeightsOnednn<kSpatialDim>::prepack(
        weight.value(),
        bias,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        transpose
      );
    }
#endif
    return PackedConvWeight<kSpatialDim>::prepack(
      weight.value(),
      bias,
      stride,
      padding,
      output_padding,
      dilation,
      groups,
      transpose
    );
  } // x86
#endif

#ifdef USE_FBGEMM
  if (ctx.qEngine() == at::QEngine::FBGEMM) {
    return PackedConvWeight<kSpatialDim>::prepack(
      weight.value(),
      bias,
      stride,
      padding,
      output_padding,
      dilation,
      groups,
      transpose
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
      weight.value(),
      bias,
      stride,
      padding,
      output_padding,
      dilation,
      groups,
      transpose
    );
  }
#endif // USE_PYTORCH_QNNPACK
#if AT_MKLDNN_ENABLED()
  if (ctx.qEngine() == at::QEngine::ONEDNN) {
    return PackedConvWeightsOnednn<kSpatialDim>::prepack(
      weight.value(),
      bias,
      stride,
      padding,
      output_padding,
      dilation,
      groups,
      transpose
    );
  }
#endif // AT_MKLDNN_ENABLED()
TORCH_CHECK(
  false,
  "Didn't find engine for when deserializing ConvPackedParams: ",
  toString(ctx.qEngine()));
}
