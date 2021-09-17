#include "lazy_tensor_core/csrc/ops/collective_permute.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"
#include "lazy_tensors/str_join.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

CollectivePermute::CollectivePermute(
    const Value& input, const Value& token,
    std::vector<std::pair<lazy_tensors::int64, lazy_tensors::int64>>
        source_target_pairs)
    : Node(ltc_collective_permute, {input, token},
           /*num_outputs=*/2, lazy_tensors::util::MHash(source_target_pairs)),
      source_target_pairs_(std::move(source_target_pairs)) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr CollectivePermute::Clone(OpList operands) const {
  return MakeNode<CollectivePermute>(operands.at(0), operands.at(1),
                                     source_target_pairs_);
}

std::string CollectivePermute::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", source_target_pairs=(";
  for (size_t i = 0; i < source_target_pairs_.size(); ++i) {
    ss << (i == 0 ? "(" : ", (");
    ss << source_target_pairs_[i].first << ", "
       << source_target_pairs_[i].second << ")";
  }
  ss << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
