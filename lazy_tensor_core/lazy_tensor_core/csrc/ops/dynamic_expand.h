#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_node_lowering.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class DynamicExpand2 : public TsNode {
 public:
  DynamicExpand2(torch::lazy::Value lhs, torch::lazy::Value sz);

  TSOpVector Lower(std::shared_ptr<torch::jit::GraphFunction> function,
                   ts_backend::TSLoweringContext* loctx) const override {

    CHECK(operands().size() == 2);
    auto graph = function->graph();
    auto sz_val = loctx->GetOutputOp(operand(1));
    auto expand = graph->insert(at::aten::expand, {loctx->GetOutputOp(operands().at(0)), sz_val});
    return {expand};
  }

};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors



