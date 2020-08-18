#pragma once

#include <ATen/ATen.h>
#include <ATen/core/List.h>

#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>

#include <torch/torch.h>

#include <tuple>

namespace at {
namespace native {
namespace serialization {

constexpr int64_t kConvPackedParamsCurrentVersion = 2;

namespace {
template <uint32_t kSpatialDim>
using ConvPackedParamsBasePtr =
    c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>;

template <uint32_t kSpatialDim>
ConvPackedParamsBasePtr<kSpatialDim> call_prepack(
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const torch::List<int64_t>& stride,
    const torch::List<int64_t>& padding,
    const torch::List<int64_t>& output_padding,
    const torch::List<int64_t>& dilation,
    int64_t groups,
    bool transposed) {
  auto& ctx = at::globalContext();
#ifdef USE_FBGEMM
  if (ctx.qEngine() == at::QEngine::FBGEMM) {
    return PackedConvWeight<kSpatialDim>::prepack(
        weight,
        bias,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        transposed);
  }
#endif // USE_FBGEMM
#ifdef USE_PYTORCH_QNNPACK
  if (ctx.qEngine() == at::QEngine::QNNPACK) {
    TORCH_CHECK(
        kSpatialDim == 2,
        "prepack/__setstate__: QNNPACK only supports Conv2d now.");
    return PackedConvWeightsQnnp<kSpatialDim>::prepack(
        weight,
        bias,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        transposed);
  }
#endif // USE_PYTORCH_QNNPACK
  TORCH_CHECK(
      false,
      "Didn't find engine for when deserializing ConvPackedParams: ",
      toString(ctx.qEngine()));
}

}  // namespace

/* Convolution packed params version history
For all versions >1 reuse the lists of tensors and adding the version as a first
element to the last tensor in the serialization tuple.
Because of the https://github.com/pytorch/pytorch/issues/43168 we cannot just
replace the serialization type with a generic one -- it will break the BC.

NOTE: Changing the serialization also requires to change:
- torch/csrc/jit/passes/onnx/unpack_quantized_weights.cpp
TODO: Can we factor out the code from jit/passes/onnx to this file?

Version 1 ======================================================================
- weight (at::Tensor)
- bias (c10::optional<at::Tensor>)
- strides (torch::List<at::Tensor>)
- padding (torch::List<at::Tensor>)
- dilation (torch::List<at::Tensor>)
- groups (at::Tensor)

Notes:
- Version 1 is detected by checking if the last tensor's size == 1
- The strides/padding/dilation is stored as a separate tensor for each element.
  That is there will be kSpatialDims single-element tensors in each of them.
  For example, for 2d convolution, the "strides" list will be 2 1-size tensors.

Version 2 ======================================================================
- weight (at::Tensor)
- bias (c10::optional<at::Tensor>)
- strides, padding, output_padding, dilation (torch::List<at::Tensor>)
- unused (torch::List<at::Tensor>)
- unused (torch::List<at::Tensor>)
- version, groups, transposed (at::Tensor)

Notes:
- Version 2+ is detected by checking if the last tensors's size > 1
- The strides/padding/output_padding/dilation is stored as separate tensors for
  each parameter. The size of each tensor will be `kSpatialDim`.
- The list of tensors at locations 4, 5 in the tuple are unused and are kept for
  the BC and FC purposes.
*/
using ConvPackedParamsSerializationType = std::tuple<
  at::Tensor,
  c10::optional<at::Tensor>,
  // these are meant to be torch::List<int64_t> but
  // it's not supported by onnx, so we'll use Tensor as
  // a workaround
  torch::List<at::Tensor>,
  torch::List<at::Tensor>,
  torch::List<at::Tensor>,
  at::Tensor
>;

// __setstate__
template <uint32_t kSpatialDim>
ConvPackedParamsSerializationType conv_packed_params_v1(
    const ConvPackedParamsBasePtr<kSpatialDim>& params) {
  at::Tensor weight;
  c10::optional<at::Tensor> bias;
  std::tie(weight, bias) = params->unpack();
  torch::List<at::Tensor> stride;
  torch::List<at::Tensor> padding;
  torch::List<at::Tensor> dilation;
  at::Tensor groups;
  for (int64_t s : params->stride()) {
    stride.emplace_back(at::tensor(s));
  }
  for (int64_t p : params->padding()) {
    padding.emplace_back(at::tensor(p));
  }
  for (int64_t d : params->dilation()) {
    dilation.emplace_back(at::tensor(d));
  }
  groups = at::tensor(params->groups());
  return std::make_tuple(
      std::move(weight),
      std::move(bias),
      stride,
      padding,
      dilation,
      groups);
}

// __getstate__
template <uint32_t kSpatialDim>
ConvPackedParamsBasePtr<kSpatialDim> conv_packed_params_v1(
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const torch::List<at::Tensor>& stride_tensor,
    const torch::List<at::Tensor>& padding_tensor,
    const torch::List<at::Tensor>& dilation_tensor,
    const at::Tensor& groups_tensor) {
  torch::List<int64_t> stride, padding, output_padding, dilation;
  for (at::Tensor s : stride_tensor) {
    stride.emplace_back(s[0].item<int64_t>());
  }
  for (at::Tensor p : padding_tensor) {
    padding.emplace_back(p[0].item<int64_t>());
  }
  for (at::Tensor d : dilation_tensor) {
    dilation.emplace_back(d[0].item<int64_t>());
  }
  int64_t groups = groups_tensor[0].item<int64_t>();

  return call_prepack<kSpatialDim>(weight, bias, stride, padding,
                                   output_padding, dilation, groups,
                                   /*transposed=*/false);
}

// __setstate__
template <uint32_t kSpatialDim>
ConvPackedParamsSerializationType conv_packed_params_v2(
    const ConvPackedParamsBasePtr<kSpatialDim>& params) {
  at::Tensor weight;
  c10::optional<at::Tensor> bias;
  std::tie(weight, bias) = params->unpack();
  torch::List<at::Tensor> unused_list;

  // version, groups, transposed
  at::Tensor params_tensor = torch::tensor({kConvPackedParamsCurrentVersion,
                                            params->groups(),
                                            int64_t(params->transpose())});
  auto strides = at::empty({kSpatialDim},
    at::TensorOptions(weight.device()).dtype(kLong).requires_grad(false));
  auto padding = at::empty_like(strides);
  auto output_padding = at::empty_like(strides);
  auto dilation = at::empty_like(strides);

  for (int idx = 0; idx < kSpatialDim; ++idx) {
    strides[idx] = params->stride().get(idx);
    padding[idx] = params->padding().get(idx);
    output_padding[idx] = params->output_padding().get(idx);
    dilation[idx] = params->dilation().get(idx);
  }
  torch::List<at::Tensor> params_list {strides, padding, output_padding,
                                       dilation};
  return std::make_tuple(
      std::move(weight),
      std::move(bias),
      params_list,
      unused_list,
      unused_list,
      params_tensor);
}

// __getstate__
template <uint32_t kSpatialDim>
ConvPackedParamsBasePtr<kSpatialDim> conv_packed_params_v2(
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const torch::List<at::Tensor>& params_list,
    const at::Tensor& params_tensor) {
  auto version = params_tensor[0].item<int64_t>();
  TORCH_CHECK(version == 2, "Expecting version 2, found ", version);
  torch::List<int64_t> stride, padding, dilation, output_padding;

  auto stride_tensor = params_list.get(0);
  for (int idx = 0; idx < kSpatialDim; ++idx) {
    stride.emplace_back(stride_tensor[idx].item<int64_t>());
  }
  auto padding_tensor = params_list.get(1);
  for (int idx = 0; idx < kSpatialDim; ++idx) {
    padding.emplace_back(padding_tensor[idx].item<int64_t>());
  }
  auto output_padding_tensor = params_list.get(2);
  for (int idx = 0; idx < kSpatialDim; ++idx) {
    output_padding.emplace_back(output_padding_tensor[idx].item<int64_t>());
  }
  auto dilation_tensor = params_list.get(3);
  for (int idx = 0; idx < kSpatialDim; ++idx) {
    dilation.emplace_back(dilation_tensor[idx].item<int64_t>());
  }
  int64_t groups = params_tensor[1].item<int64_t>();
  bool transposed = params_tensor[2].item<bool>();

  return call_prepack<kSpatialDim>(weight, bias, stride, padding,
                                   output_padding, dilation, groups,
                                   transposed);
}


// __getstate__
template <uint32_t kSpatialDim>
ConvPackedParamsSerializationType conv_packed_params(
    const ConvPackedParamsBasePtr<kSpatialDim>& params) {
  // Compiler should optimize out everything but the current version.
  // This is kept here for debugging, as we only want to serialize into the
  // latest version.
  switch (kConvPackedParamsCurrentVersion) {
    case 1: return conv_packed_params_v1<kSpatialDim>(params);
    case 2: return conv_packed_params_v2<kSpatialDim>(params);
    default: TORCH_CHECK(false, "Unknown serialization version ",
                         kConvPackedParamsCurrentVersion);
  }
}

// __setstate__
template <uint32_t kSpatialDim>
ConvPackedParamsBasePtr<kSpatialDim> conv_packed_params(
    const ConvPackedParamsSerializationType& state) {
  at::Tensor field_1;
  c10::optional<at::Tensor> field_2;
  torch::List<at::Tensor> field_3, field_4, field_5;
  at::Tensor field_6;
  std::tie(field_1, field_2, field_3, field_4, field_5, field_6) = state;

  int64_t version = 1;
  if (field_6.size(0) > 1) {
    version = field_6[0].item<int64_t>();
    TORCH_CHECK(version > 1, "The explicit version should be > 1");
  }

  switch (version) {
    case 1: return conv_packed_params_v1<kSpatialDim>(
      field_1, field_2, field_3, field_4, field_5, field_6);
    case 2: return conv_packed_params_v2<kSpatialDim>(
      field_1, field_2, field_3, field_6); // Fields 4 and 5 are unused
    default: TORCH_CHECK(false, "Unknown deserialization version ", version);
  }
}

}}} // at::native::serialization
