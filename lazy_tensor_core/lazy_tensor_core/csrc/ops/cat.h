#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_node_lowering.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Cat : public TsNode {
 public:
  Cat(std::vector<torch::lazy::Value> values, int64_t dim, const std::vector<at::ScalarType>& out_dtypes,
      const std::vector<std::vector<int64_t>>& out_shapes);

  std::string ToString() const override;

  int64_t dim() const { return dim_; };

  TSOpVector Lower(std::shared_ptr<torch::jit::GraphFunction> function,
                   ts_backend::TSLoweringContext* loctx) const override {
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
    TSOpVector cat_out = compiler::LowerTSBuiltin(function, op().op, arguments, kwarguments);
    CHECK_EQ(cat_out.size(), 1);

    return cat_out;

  }


 private:
  int64_t dim_;
  std::vector<at::ScalarType> at_dtypes_;
  std::vector<std::vector<int64_t>> at_shapes_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
