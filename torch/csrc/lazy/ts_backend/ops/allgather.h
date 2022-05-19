#pragma once

#include <c10d/ProcessGroup.hpp>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

struct Allgather : public TsNode {
  Allgather(const Value& output_tensors, const Value& input_tensor, const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
      int64_t timeout, std::vector<Shape>&& shapes, int64_t num_outputs)
    : TsNode(OpKind(c10::Symbol::fromQualString("c10d::allgather_")),
          {output_tensors, input_tensor}, std::move(shapes), num_outputs,
          MHash(timeout)),
      process_group(process_group),
      timeout(timeout) {}

  std::string ToString() const override {
    std::stringstream ss;
    ss << TsNode::ToString();
    ss << ", process_group=" << process_group->getBackendName()
        << ", timeout=" << timeout;
    return ss.str();
  }

  TSOpVector Lower(std::shared_ptr<torch::jit::GraphFunction> function,
      TSLoweringContext* loctx) const override {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.reserve(4);
    arguments.emplace_back(loctx->GetOutputOp(operand(0)));
    arguments.emplace_back(loctx->GetOutputOp(operand(1)));
    arguments.emplace_back("process_group", process_group);
    arguments.emplace_back("timeout", timeout);

    auto out = torch::lazy::LowerTSBuiltin(function, op().op, arguments);
    TORCH_INTERNAL_ASSERT(out.size(), 1);

    // We turn the TensorList output into a list of tensors to be compatible with Lazy IR.
    auto graph = function->graph();
    auto listnode = graph->insertNode(graph->createListUnpack(out[0], num_outputs()));
    return listnode->outputs().vec();
  }

  c10::intrusive_ptr<c10d::ProcessGroup> process_group;
  int64_t timeout;
};

}  // namespace lazy
}  // namespace torch
