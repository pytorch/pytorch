#include "lazy_tensor_core/csrc/ops/random.h"

#include "lazy_tensors/computation_client/ltc_util.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

// aten::random builtin symbol cannot be recognized as a builtin function
// since random only has in-place versions. Therefore we force the symbol to
// be "aten::random_" here.
Random::Random(const Value& input)
    : Node(ir::OpKind(c10::Symbol::fromQualString("aten::random_")),
        {input}, input.shape()) {}

NodePtr Random::Clone(OpList operands) const {
  return MakeNode<Random>(operands.at(0));
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
