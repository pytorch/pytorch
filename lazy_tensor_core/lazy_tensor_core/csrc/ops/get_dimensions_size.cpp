#include "lazy_tensor_core/csrc/ops/get_dimensions_size.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"
#include "lazy_tensors/str_join.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

GetDimensionsSize::GetDimensionsSize(
    const torch::lazy::Value& input, std::vector<lazy_tensors::int64> dimensions)
    : TsNode(ltc_get_dimensions_size, {input},
           lazy_tensors::ShapeUtil::MakeShape(
               GetShapeDimensionType(/*device=*/nullptr), {}),
           /*num_outputs=*/1, torch::lazy::MHash(dimensions)),
      dimensions_(std::move(dimensions)) {}

NodePtr GetDimensionsSize::Clone(OpList operands) const {
  return torch::lazy::MakeNode<GetDimensionsSize>(operands.at(0), dimensions_);
}

std::string GetDimensionsSize::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", dimensions=("
     << lazy_tensors::StrJoin(dimensions_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
