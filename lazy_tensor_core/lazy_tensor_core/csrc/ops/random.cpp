#include "lazy_tensor_core/csrc/ops/random.h"

#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

// aten::random builtin symbol cannot be recognized as a builtin function
// since random only has in-place versions. Therefore we force the symbol to
// be "aten::random_" here.
Random::Random(const torch::lazy::Value& input)
    : TsNode(torch::lazy::OpKind(c10::Symbol::fromQualString("aten::random_")),
        {input}, ir::GetShapeFromTsValue(input)) {}

NodePtr Random::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Random>(operands.at(0));
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
