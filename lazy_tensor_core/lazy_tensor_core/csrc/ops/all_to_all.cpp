#include "lazy_tensor_core/csrc/ops/all_to_all.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"
#include "lazy_tensors/str_join.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

AllToAll::AllToAll(const Value& input, const Value& token,
                   lazy_tensors::int64 split_dimension,
                   lazy_tensors::int64 concat_dimension,
                   lazy_tensors::int64 split_count,
                   std::vector<std::vector<lazy_tensors::int64>> groups)
    : Node(ltc_all_to_all, {input, token},
           /*num_outputs=*/2,
           lazy_tensors::util::MHash(split_dimension, concat_dimension,
                                     split_count, groups)),
      split_dimension_(split_dimension),
      concat_dimension_(concat_dimension),
      split_count_(split_count),
      groups_(std::move(groups)) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr AllToAll::Clone(OpList operands) const {
  return MakeNode<AllToAll>(operands.at(0), operands.at(1), split_dimension_,
                            concat_dimension_, split_count_, groups_);
}

std::string AllToAll::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", split_dimension=" << split_dimension_
     << ", concat_dimension=" << concat_dimension_
     << ", split_count=" << split_count_ << ", groups=(";
  for (size_t i = 0; i < groups_.size(); ++i) {
    ss << (i == 0 ? "(" : ",(");
    ss << lazy_tensors::StrJoin(groups_[i], ", ") << ")";
  }
  ss << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
