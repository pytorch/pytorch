#pragma once

#include <ATen/Tensor.h>
#include <ATen/native/mkldnn/Common.h>
#include <ATen/native/mkldnn/OpContext.h>
#include <c10/util/string_view.h>

namespace at {
namespace native {
namespace mkldnn {
namespace internal {
namespace convolution2d {

// These constants control the fusion behavior of convolution.
enum AttrType {
  None, // No fusion
  ReLU, // ReLU fusion
  END
};

static AttrType get_attrtype_enum(const c10::string_view attr) {
  if (attr == "none") {
    return AttrType::None;
  } else if (attr == "relu") {
    return AttrType::ReLU;
  } else {
    TORCH_CHECK(false, "unknown attr argument: ", attr);
  }
}

c10::intrusive_ptr<mkldnn::Conv2dOpContext> createConv2dPrePackOpContext(
    Tensor weight,
    c10::optional<Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups,
    std::vector<int64_t> input_size,
    c10::string_view attr);

Tensor conv2d_run(
    const Tensor& input,
    const c10::intrusive_ptr<mkldnn::Conv2dOpContext>& op_context);

ContextConv2D create(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const int64_t groups,
    const IntArrayRef input_size,
    const ideep::attr_t& attr);

Tensor run(ContextConv2D& context, const Tensor& input);

} // namespace convolution2d
} // namespace internal
} // namespace mkldnn
} // namespace native
} // namespace at
