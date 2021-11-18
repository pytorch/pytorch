#include "lazy_tensor_core/csrc/ops/ops.h"

#include <c10/util/Half.h>

#include <cmath>

#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/ops/arithmetic_ir_ops.h"
#include "lazy_tensor_core/csrc/ops/expand.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/ts_backend/LazyLazyIr.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"
#include "lazy_tensor_core/csrc/view_ops/permute.h"
#include "lazy_tensors/computation_client/util.h"
#include "torch/csrc/lazy/core/ir_metadata.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

#define PTLTC_BINARY_OP(name, sym)                                          \
  torch::lazy::NodePtr name(const torch::lazy::Value& input0,               \
                            const torch::lazy::Value& input1) {             \
    torch::lazy::NodePtr node =                                             \
        GenericOp(torch::lazy::OpKind(sym), {input0, input1});              \
    std::dynamic_pointer_cast<torch::lazy::TsNode>(node)->SetShapeDeferred( \
        [&]() { return compiler::InferShape(node.get()); });                \
    return node;                                                            \
  }

PTLTC_BINARY_OP(Pow, at::aten::pow);

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
