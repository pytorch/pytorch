#include <ATen/Tensor.h>
#include <ATen/core/Reduction.h>
#include <ATen/native/ConvUtils.h>
#include <torch/csrc/jit/frontend/sugared_value.h>
#include <torch/csrc/lazy/core/permutation_util.h>
#include <torch/csrc/lazy/core/shape.h>
#include <torch/jit.h>

#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/ops/as_strided.h"
#include "lazy_tensor_core/csrc/ops/as_strided_view_update.h"
#include "lazy_tensor_core/csrc/ops/cast.h"
#include "lazy_tensor_core/csrc/ops/constant.h"
#include "lazy_tensor_core/csrc/ops/constant_pad_nd.h"
#include "lazy_tensor_core/csrc/ops/convolution_backward_overrideable.h"
#include "lazy_tensor_core/csrc/ops/convolution_overrideable.h"
#include "lazy_tensor_core/csrc/ops/device_data.h"
#include "lazy_tensor_core/csrc/ops/expand.h"
#include "lazy_tensor_core/csrc/ops/generic_slice.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensor_core/csrc/ops/permute.h"
#include "lazy_tensor_core/csrc/ops/repeat.h"
#include "lazy_tensor_core/csrc/ops/scalar.h"
#include "lazy_tensor_core/csrc/ops/select.h"
#include "lazy_tensor_core/csrc/ops/squeeze.h"
#include "lazy_tensor_core/csrc/ops/stack.h"
#include "lazy_tensor_core/csrc/ops/ts_native_batch_norm_backward.h"
#include "lazy_tensor_core/csrc/ops/ts_native_batch_norm_forward.h"
#include "lazy_tensor_core/csrc/ops/unselect.h"
#include "lazy_tensor_core/csrc/ops/unsqueeze.h"
#include "lazy_tensor_core/csrc/ops/update_slice.h"
#include "lazy_tensor_core/csrc/ops/view.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_lowering_context.h"

namespace torch_lazy_tensors {
namespace compiler {

torch::lazy::Shape InferComparison(const torch::lazy::Node* node) {
  const torch::lazy::Output& lhs = node->operand(0);
  const torch::lazy::Output& rhs = node->operand(1);
  return torch::lazy::Shape(
      c10::ScalarType::Bool,
      Helpers::GetPromotedShape(ir::GetShapeFromTsOutput(lhs).sizes(),
                                ir::GetShapeFromTsOutput(rhs).sizes()));
}

torch::lazy::Shape InferConvolutionOverrideable(
    const ir::ops::ConvolutionOverrideable* conv) {
  const auto& operands = conv->operands();
  CHECK(!operands.empty());

  // TODO: Shape::sizes() returns a Span and converting it to
  // a vector of int is awkard. Clean up this after we switch to a
  // PyTorch shape.
  const auto input_shape = ir::GetShapeFromTsOutput(operands[0]);
  const auto& input_size = std::vector<int64_t>(
      input_shape.sizes().begin(), input_shape.sizes().end());
  const auto weight_shape = ir::GetShapeFromTsOutput(operands[1]);
  const auto& weight_size = std::vector<int64_t>(
      weight_shape.sizes().begin(), weight_shape.sizes().end());
  const auto& dilation = conv->dilation();
  const auto& padding = conv->padding();
  const auto& stride = conv->stride();
  const auto& output_padding = conv->output_padding();
  const auto& groups = conv->groups();

  if (!conv->transposed()) {
    return torch::lazy::Shape(
        input_shape.scalar_type(),
        at::native::conv_output_size(input_size, weight_size, padding, stride,
                                     dilation));
  } else {
    return torch::lazy::Shape(
        input_shape.scalar_type(),
        at::native::conv_input_size(input_size, weight_size, padding,
                                    output_padding, stride, dilation, groups));
  }
}

torch::lazy::Shape InferRepeat(const ir::ops::Repeat* repeat) {
  const torch::lazy::Output& input = repeat->operand(0);
  const torch::lazy::Shape& input_shape = ir::GetShapeFromTsOutput(input);
  const auto& repeats = repeat->repeats();
  CHECK_GE(repeats.size(), input_shape.dim());

  int64_t num_new_dimensions = repeats.size() - input_shape.dim();
  std::vector<int64_t> padded_size(num_new_dimensions, 1);
  padded_size.insert(padded_size.end(), input_shape.sizes().begin(),
                     input_shape.sizes().end());
  std::vector<int64_t> target_size(repeats.size());
  for (const auto idx : c10::irange(repeats.size())) {
    target_size[idx] = padded_size[idx] * repeats[idx];
  }
  return torch::lazy::Shape(input_shape.scalar_type(), target_size);
}

torch::lazy::Shape InferStack(const ir::ops::Stack* stack) {
  const auto& inputs = stack->operands();
  CHECK(!inputs.empty());
  const torch::lazy::Shape& input_shape = ir::GetShapeFromTsOutput(inputs[0]);
  for (const torch::lazy::Output& input : inputs) {
    CHECK_EQ(ir::GetShapeFromTsOutput(input), input_shape);
  }
  const auto input_dimensions = input_shape.sizes();
  std::vector<int64_t> output_dimensions(input_dimensions.begin(),
                                         input_dimensions.end());
  CHECK_GE(stack->dim(), 0);
  CHECK_LE(stack->dim(), output_dimensions.size());
  output_dimensions.insert(output_dimensions.begin() + stack->dim(),
                           inputs.size());
  return torch::lazy::Shape(input_shape.scalar_type(), output_dimensions);
}
torch::lazy::Shape InferShape(const torch::lazy::Node* node) {
  if (node->op() == *ir::ops::ltc_generic_slice) {
    auto generic_slice = torch::lazy::NodeCast<ir::ops::GenericSlice>(
        node, *ir::ops::ltc_generic_slice);
    const torch::lazy::Output& argument = node->operand(0);
    return torch::lazy::Shape(
        ir::GetShapeFromTsOutput(argument).scalar_type(),
        generic_slice->sizes());
  }
  if (node->op() == *ir::ops::ltc_update_slice) {
    const torch::lazy::Output& argument = node->operand(0);
    return ir::GetShapeFromTsOutput(argument);
  }
  switch (node->op().op) {
    case at::aten::expand: {
      auto expand = torch::lazy::NodeCast<ir::ops::Expand>(
          node, ir::OpKind(at::aten::expand));
      const torch::lazy::Output& argument = node->operand(0);
      return torch::lazy::Shape(
          ir::GetShapeFromTsOutput(argument).scalar_type(), expand->size());
    }
    case at::aten::convolution_overrideable: {
      return InferConvolutionOverrideable(
          torch::lazy::NodeCast<ir::ops::ConvolutionOverrideable>(
              node, ir::OpKind(at::aten::convolution_overrideable)));
    }
    case at::aten::permute: {
      auto permute = torch::lazy::NodeCast<ir::ops::Permute>(
          node, ir::OpKind(at::aten::permute));
      const torch::lazy::Output& argument = node->operand(0);
      return ir::ops::Permute::MakePermuteShape(
          ir::GetShapeFromTsOutput(argument), permute->dims());
    }
    // activation and unary op do not change shape
    case at::aten::pow: {
      const torch::lazy::Output& argument = node->operand(0);
      return ir::GetShapeFromTsOutput(argument);
    }
    case at::aten::repeat: {
      return InferRepeat(torch::lazy::NodeCast<ir::ops::Repeat>(
          node, ir::OpKind(at::aten::repeat)));
    }
    case at::aten::stack: {
      return InferStack(torch::lazy::NodeCast<ir::ops::Stack>(
          node, ir::OpKind(at::aten::stack)));
    }
    case at::aten::constant_pad_nd: {
      auto constant_pad_nd = torch::lazy::NodeCast<ir::ops::ConstantPadNd>(
          node, ir::OpKind(at::aten::constant_pad_nd));
      const torch::lazy::Output& argument = node->operand(0);
      const torch::lazy::Shape& argument_shape =
          ir::GetShapeFromTsOutput(argument);
      const auto argument_dimensions = argument_shape.sizes();
      const auto& pad = constant_pad_nd->pad();
      CHECK_EQ(argument_dimensions.size() * 2, pad.size());
      std::vector<int64_t> padded_dimensions(argument_dimensions.begin(),
                                             argument_dimensions.end());
      size_t i = 0;
      for (auto rit = pad.rbegin(); rit != pad.rend(); rit += 2, ++i) {
        padded_dimensions[i] += (*rit + *(rit + 1));
      }
      return torch::lazy::Shape(argument_shape.scalar_type(),
                                 padded_dimensions);
    }
    case at::aten::eq:
    case at::aten::ge:
    case at::aten::gt:
    case at::aten::le:
    case at::aten::lt:
    case at::aten::ne: {
      return InferComparison(node);
    }
    default:
      LOG(FATAL) << *node << "Not implemented yet.";
  }
}
}  // namespace compiler
}  // namespace torch_lazy_tensors
