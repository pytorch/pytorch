#include "lazy_tensor_core/csrc/ops/all.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/reduction.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/torch_util.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/str_join.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

All::All(const Value& input, std::vector<lazy_tensors::int64> dimensions,
         bool keep_reduced_dimensions)
    : Node(ir::OpKind(at::aten::all), {input},
           /*num_outputs=*/1,
           lazy_tensors::util::MHash(dimensions, keep_reduced_dimensions)),
      dimensions_(std::move(dimensions)),
      keep_reduced_dimensions_(keep_reduced_dimensions) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr All::Clone(OpList operands) const {
  return MakeNode<All>(operands.at(0), dimensions_, keep_reduced_dimensions_);
}

std::string All::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dimensions=("
     << lazy_tensors::StrJoin(dimensions_, ", ")
     << "), keep_reduced_dimensions=" << keep_reduced_dimensions_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
