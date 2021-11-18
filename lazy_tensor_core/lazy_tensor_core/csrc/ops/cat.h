#pragma once

#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Cat : public torch::lazy::TsNode {
 public:
  Cat(std::vector<torch::lazy::Value> values, int64_t dim,
      std::vector<torch::lazy::Shape>&& shapes);

  std::string ToString() const override;

  int64_t dim() const { return dim_; };

  torch::lazy::TSOpVector Lower(
      std::shared_ptr<torch::jit::GraphFunction> function,
      torch::lazy::TSLoweringContext* loctx) const override {
    std::vector<torch::jit::NamedValue> arguments;
    std::vector<torch::jit::NamedValue> kwarguments;
    arguments.reserve(2);
    kwarguments.reserve(0);

    std::vector<torch::jit::Value*> tensor_list;
    CHECK(!operands().empty());
    for (const torch::lazy::Output& operand : operands()) {
      tensor_list.emplace_back(loctx->GetOutputOp(operand));
    }
    auto graph = function->graph();
    arguments.emplace_back(
        graph
            ->insertNode(graph->createList(tensor_list[0]->type(), tensor_list))
            ->output());
    arguments.emplace_back(dim());
    torch::lazy::TSOpVector cat_out =
        torch::lazy::LowerTSBuiltin(function, op().op, arguments, kwarguments);
    CHECK_EQ(cat_out.size(), 1);

    return cat_out;
  }

 private:
  int64_t dim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
