#pragma once

#include <ATen/ATen.h>
#include <ATen/core/List.h>

#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>

#include <tuple>

using SerializationType = std::tuple<
  std::string /*name of the op that is serialized*/,
  int64_t /*version*/,
  std::vector<at::Tensor> /*list of all non-optional tensors*/,
  std::vector<c10::optional<at::Tensor>> /*list of all optional tensors*/,
  std::vector<double> /*list of all doubles*/,
  std::vector<int64_t> /*list of all integers*/
>;

/* Legacy serialization types */


/* Convolution prepacked parameters serialization.
 *
 * Files that need to be updated if version changes:
 * - ATen/native/quantized/cpu/fbgemm_utils.cpp
 * - jit/passes/onnx/unpack_quantized_weights.cpp
 *
 * Version 1 (Legacy)
 * - Fields:
 *  1. weight
 *  2. bias
 *  3. stride x kSpatialDim
 *  4. padding x kSpatialDim
 *  5. dilation x kSpatialDim
 *  6. groups
 *
 * Version 2+
 * - Fields:
 *  1. name of the op ("conv")
 *  2. version number (int64_t)
 *  3. list of all non-optional tensors (vector<Tensor>)
 *    1. unpacked weight
 *  4. list of all optional tensors (vector<optional<Tensor>>)
 *    1. bias
 *  5. list of all doubles
 *    EMPTY
 *  6. list of all integers (vector<int64_t> of size (1 + 4 * kSpatialDim + 1))
 *    1. kSpatialDim
 *    2. stride x kSpatialDim
 *    3. padding x kSpatialDim
 *    4. dilation x kSpatialDim
 *    5. groups
 */
const std::string kConvName = "conv";
constexpr int64_t kConvPackedParamsSerializationVersion = 2;
template <uint32_t kSpatialDim>
SerializationType serialize_conv(
    const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& params) {
  std::vector<at::Tensor> non_optional;
  std::vector<c10::optional<at::Tensor>> optional;
  std::vector<double> doubles;
  std::vector<int64_t> longs;
  constexpr int64_t kConvLongsSize = 1 + 4 * kSpatialDim + 1;
  longs.reserve(kConvLongsSize);

  at::Tensor weight;
  c10::optional<at::Tensor> bias;
  std::tie(weight, bias) = params->unpack();

  non_optional.emplace_back(std::move(weight));
  optional.emplace_back(std::move(bias));

  longs.push_back(kSpatialDim);

  auto stride = params->stride().vec();
  longs.insert(longs.end(), stride.begin(), stride.end());

  auto padding = params->padding().vec();
  longs.insert(longs.end(), padding.begin(), padding.end());

  auto dilation = params->dilation().vec();
  longs.insert(longs.end(), dilation.begin(), dilation.end());

  longs.push_back(params->groups());


  return std::make_tuple(
    kConvName,
    kConvPackedParamsSerializationVersion,
    non_optional,
    optional,
    doubles,
    longs
  );
}

template <uint32_t kSpatialDim>
c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> deserialize_conv(
    SerializationType state) {
  std::string name;
  int64_t version;
  std::vector<at::Tensor> non_optional;
  std::vector<c10::optional<at::Tensor>> optional;
  std::vector<double> doubles;
  std::vector<int64_t> longs;

  std::tie(name, version, non_optional, optional, doubles, longs) = state;

  at::Tensor weight;
  c10::optional<at::Tensor> bias;
  torch::List<int64_t> stride, padding, dilation;
  int64_t groups;

  weight = non_optional[0];
  bias = optional[0];

  int idx = 0;
  for (; idx < kSpatialDim; ++idx) {
    stride.emplace_back(longs[idx]);
  }
  for (; idx < 2 * kSpatialDim; ++idx) {
    padding.emplace_back(longs[idx]);
  }
  for (; idx < 3 * kSpatialDim; ++idx) {
    dilation.emplace_back(longs[idx]);
  }
  groups = longs[idx];

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
