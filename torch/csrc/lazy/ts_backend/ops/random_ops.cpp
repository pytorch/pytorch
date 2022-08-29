#include <torch/csrc/lazy/core/util.h>
#include <torch/csrc/lazy/ts_backend/ops/random_ops.h>

namespace torch {
namespace lazy {

Normal::Normal(
    const torch::lazy::Value& self,
    const double& mean,
    const double& std,
    std::vector<torch::lazy::Shape>&& shapes)
    : torch::lazy::TsNode(
          ClassOpKind(),
          {self},
          std::move(shapes),
          /* num_outputs */ 1,
          torch::lazy::MHash(mean, std)),
      mean_(mean),
      std_(std) {}

std::string Normal::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString();
  ss << ", mean=" << mean_;
  ss << ", std=" << std_;
  return ss.str();
}

torch::lazy::TSOpVector Normal::Lower(
    std::shared_ptr<torch::jit::GraphFunction> function,
    torch::lazy::TSLoweringContext* loctx) const {
  std::vector<torch::jit::NamedValue> arguments;
  std::vector<torch::jit::NamedValue> kwarguments;
  arguments.reserve(3);
  size_t i = 0;
  arguments.emplace_back(loctx->GetOutputOp(operand(i++)));
  arguments.emplace_back("mean", mean_);
  arguments.emplace_back("std", std_);
  torch::lazy::TSOpVector normal__out =
      torch::lazy::LowerTSBuiltin(function, op().op, arguments, kwarguments);
  TORCH_CHECK_EQ(normal__out.size(), 1);

  return normal__out;
}

} // namespace lazy
} // namespace torch
