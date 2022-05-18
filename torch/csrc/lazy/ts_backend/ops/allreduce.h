#pragma once

#include <c10d/ProcessGroup.hpp>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

struct Allreduce : public TsNode {
  Allreduce(const Value& input, const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
      int64_t reduce_op, int64_t timeout, std::vector<Shape>&& shapes)
    : TsNode(OpKind(c10::Symbol::fromQualString("c10d::allreduce_")),
          {input}, std::move(shapes), /* num_outputs */ 1,
          MHash(reduce_op, timeout)),
      process_group(process_group),
      reduce_op(reduce_op),
      timeout(timeout) {}

  std::string ToString() const override {
    std::stringstream ss;
    ss << TsNode::ToString();
    ss << ", process_group=" << process_group->getBackendName() << ", reduce_op=" << reduce_op
        << ", timeout=" << timeout;
    return ss.str();
  }

  TSOpVector Lower(std::shared_ptr<torch::jit::GraphFunction> function,
      TSLoweringContext* loctx) const override {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.reserve(4);
    arguments.emplace_back(loctx->GetOutputOp(operand(0)));
    arguments.emplace_back("process_group", process_group);
    arguments.emplace_back("reduce_op", reduce_op);
    arguments.emplace_back("timeout", timeout);

    auto out = torch::lazy::LowerTSBuiltin(function, op().op, arguments);
    TORCH_INTERNAL_ASSERT(out.size(), 1);

    return out;
  }

  c10::intrusive_ptr<c10d::ProcessGroup> process_group;
  int64_t reduce_op;
  int64_t timeout;
};

}  // namespace lazy
}  // namespace torch
