#pragma once

#include <torch/csrc/distributed/c10d/ProcessGroup.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

struct Broadcast : public TsNode {
  Broadcast(c10::intrusive_pointer<c10d::ProcessGroup> process_group, const Value& input, int64_t root_rank,
      int64_t root_tensor, int64_t timeout, std::vector<Shape>&& shapes)
    : TsNode(OpKind(c10::Symbol::fromQualString("c10d::broadcast")),
          {input}, std::move(shapes), /* num_outputs */ 1,
          MHash(root_rank, root_tensor, timeout)),
      root_rank(root_rank),
      root_tensor(root_tensor),
      timeout(timeout) {}

  std::string ToString() const override {
    std::stringstream ss;
    ss << TsNode::ToString();
    ss << ", process_group=" << process_group->getBackendName() ", root_rank=" << root_rank << ", root_tensor="
        << root_tensor << ", timeout=" << timeout;
    return ss.str();
  }

  TSOpVector Lower(std::shared_ptr<torch::jit::GraphFunction> function,
      TSLoweringContext* loctx) const override {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.reserve(5);
    arguments.emplace_back("process_group", process_group);
    arguments.emplace_back(loctx->GetOutputOp(operand(0)));
    arguments.emplace_back("root_rank", root_rank);
    arguments.emplace_back("root_tensor", root_tensor);
    arguments.emplace_back("timeout", timeout);

    auto out = torch::lazy::LowerTSBuiltin(function, op().op, arguments);
    TORCH_INTERNAL_ASSERT(out.size(), 1);

    return out;
  }

  c10::intrusive_pointer<c10d::ProcessGroup> process_group;
  int64_t root_rank;
  int64_t root_tensor;
  int64_t timeout;
};

}  // namespace lazy
}  // namespace torch
