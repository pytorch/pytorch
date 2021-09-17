#include "lazy_tensor_core/csrc/ops/std_mean.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/str_join.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

StdMean::StdMean(const Value& input,
                 std::vector<lazy_tensors::int64> dimensions,
                 lazy_tensors::int64 correction, bool keep_reduced_dimensions)
    : Node(ir::OpKind(at::aten::std_mean), {input},
           /*num_outputs=*/2,
           lazy_tensors::util::MHash(dimensions, correction,
                                     keep_reduced_dimensions)),
      dimensions_(std::move(dimensions)),
      correction_(correction),
      keep_reduced_dimensions_(keep_reduced_dimensions) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr StdMean::Clone(OpList operands) const {
  return MakeNode<StdMean>(operands.at(0), dimensions_, correction_,
                           keep_reduced_dimensions_);
}

std::string StdMean::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dimensions=("
     << lazy_tensors::StrJoin(dimensions_, ", ")
     << "), keep_reduced_dimensions=" << keep_reduced_dimensions_
     << ", correction=" << correction_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
