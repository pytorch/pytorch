#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_node_lowering.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class DynamicSize2 : public TsNode {
 public:
  DynamicSize2(torch::lazy::Value lhs);

  TSOpVector Lower(std::shared_ptr<torch::jit::GraphFunction> function,
                   ts_backend::TSLoweringContext* loctx) const override {

    CHECK(operands().size() == 1);
    auto graph = function->graph();

    auto size_val = graph->insert(at::aten::size, {loctx->GetOutputOp(operands().at(0))});
    return {size_val};

  }

};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
