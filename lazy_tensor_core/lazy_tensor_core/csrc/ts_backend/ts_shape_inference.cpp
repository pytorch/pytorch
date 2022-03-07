#include <ATen/Tensor.h>
#include <ATen/core/Reduction.h>
#include <ATen/native/ConvUtils.h>
#include <torch/csrc/jit/frontend/sugared_value.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/ts_backend/ops/cast.h>
#include <torch/csrc/lazy/ts_backend/ops/device_data.h>
#include <torch/csrc/lazy/ts_backend/ops/expand.h>
#include <torch/csrc/lazy/ts_backend/ops/scalar.h>
#include <torch/csrc/lazy/core/permutation_util.h>
#include <torch/csrc/lazy/core/shape.h>
#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>
#include <torch/jit.h>

#include "lazy_tensor_core/csrc/ops/squeeze.h"
#include "lazy_tensor_core/csrc/ops/stack.h"
#include "lazy_tensor_core/csrc/ops/ts_native_batch_norm_backward.h"
#include "lazy_tensor_core/csrc/ops/ts_native_batch_norm_forward.h"
#include "lazy_tensor_core/csrc/ops/unsqueeze.h"

namespace torch_lazy_tensors {
namespace compiler {



torch::lazy::Shape InferStack(const ir::ops::Stack* stack) {
  const auto& inputs = stack->operands();
  CHECK_EQ(inputs.size(), 1);
  auto* tensorlist = torch::lazy::NodeCast<torch::lazy::TensorList>(
          inputs[0].node, torch::lazy::tensor_list_opkind);
  auto operands = tensorlist->operands();
  const torch::lazy::Shape& input_shape =
      torch::lazy::GetShapeFromTsOutput(operands[0]);
  for (const torch::lazy::Output& operand : operands) {
    CHECK_EQ(torch::lazy::GetShapeFromTsOutput(operand), input_shape);
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
  switch (node->op().op) {
    // activation and unary op do not change shape
    case at::aten::stack: {
      return InferStack(torch::lazy::NodeCast<ir::ops::Stack>(
          node, torch::lazy::OpKind(at::aten::stack)));
    }
    default:
      LOG(FATAL) << *node << "Not implemented yet.";
  }
}
}  // namespace compiler
}  // namespace torch_lazy_tensors
