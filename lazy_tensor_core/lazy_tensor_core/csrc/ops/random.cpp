#include "lazy_tensor_core/csrc/ops/random.h"

#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/str_join.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

// aten::random builtin symbol cannot be recognized as a builtin function
// since random only has in-place versions. Therefore we force the symbol to
// be "aten::random_" here.
Random::Random(const torch::lazy::Value& input, const c10::optional<int64_t>& from, const c10::optional<int64_t>& to)
    : TsNode(torch::lazy::OpKind(c10::Symbol::fromQualString("aten::random_")),
        {input}, ir::GetShapeFromTsValue(input))
    , from(from)
    , to(to) {}

std::string Random::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString();
  if (from) {
    lazy_tensors::ToString("from", *from, ss);
  }
  if (to) {
    lazy_tensors::ToString("to", *to, ss);
  }
  return ss.str();
}

TSOpVector Random::Lower(TSNodeLoweringInterface& tsLoweringInterface,
    std::shared_ptr<torch::jit::GraphFunction> function, ts_backend::TSLoweringContext* loctx) const {
  std::vector<torch::jit::NamedValue> arguments;
  arguments.emplace_back(loctx->GetOutputOp(operand(0)));
  if (from) {
    arguments.push_back(*from);
  }
  if (to) {
    arguments.push_back(*to);
  }

  return tsLoweringInterface.LowerBuiltin(op().op, arguments);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
