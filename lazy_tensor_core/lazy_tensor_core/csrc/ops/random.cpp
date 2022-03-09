#include "lazy_tensor_core/csrc/ops/random.h"

#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>

#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"
#include <torch/csrc/lazy/core/util.h>

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

// aten::random builtin symbol cannot be recognized as a builtin function
// since random only has in-place versions. Therefore we force the symbol to
// be "aten::random_" here.
Random::Random(const torch::lazy::Value& input,
               const c10::optional<int64_t>& from,
               const c10::optional<int64_t>& to)
    : torch::lazy::TsNode(
          torch::lazy::OpKind(c10::Symbol::fromQualString("aten::random_")),
          {input}, {input.shape()}),
      from(from),
      to(to) {}

std::string Random::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString();
  if (from) {
    ss << ", from=" << *from;
  }
  if (to) {
    ss << ", to=" << *to;
  }
  return ss.str();
}

torch::lazy::TSOpVector Random::Lower(
    std::shared_ptr<torch::jit::GraphFunction> function,
    torch::lazy::TSLoweringContext* loctx) const {
  std::vector<torch::jit::NamedValue> arguments;
  arguments.emplace_back(loctx->GetOutputOp(operand(0)));
  if (from) {
    arguments.push_back(*from);
  }
  if (to) {
    arguments.push_back(*to);
  }

  return torch::lazy::LowerTSBuiltin(function, op().op, arguments);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
