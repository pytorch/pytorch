#include "lazy_tensor_core/csrc/ops/std.h"

#include "absl/strings/str_join.h"
#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/reduction.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Std::Std(const Value& input, std::vector<lazy_tensors::int64> dimensions,
         bool keep_reduced_dimensions, bool unbiased)
    : Node(ir::OpKind(at::aten::std), {input},
           /*num_outputs=*/1,
           lazy_tensors::util::MHash(dimensions, keep_reduced_dimensions,
                                     unbiased)),
      dimensions_(std::move(dimensions)),
      keep_reduced_dimensions_(keep_reduced_dimensions),
      unbiased_(unbiased) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr Std::Clone(OpList operands) const {
  return MakeNode<Std>(operands.at(0), dimensions_, keep_reduced_dimensions_,
                       unbiased_);
}

std::string Std::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dimensions=(" << absl::StrJoin(dimensions_, ", ")
     << "), keep_reduced_dimensions=" << keep_reduced_dimensions_
     << ", unbiased=" << unbiased_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
