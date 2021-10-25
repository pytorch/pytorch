#include "lazy_tensor_core/csrc/ops/cat.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {

// TODO(whc) don't duplicate this
static lazy_tensors::Shape convertShape(
    const std::vector<at::ScalarType>& dtypes,
    const std::vector<std::vector<int64_t>>& shapes) {
  LTC_CHECK_EQ(dtypes.size(), shapes.size());
  if (dtypes.size() == 1) {
    return lazy_tensors::Shape(dtypes[0], shapes[0]);
  }

  std::vector<lazy_tensors::Shape> shape;
  for (int i = 0; i < dtypes.size(); i++) {
    shape.emplace_back(dtypes[i], shapes[i]);
  }

  // Since we are going to remove lazy_tensors::Shape very soon, this
  // copy-by-value is not worth fixing.
  return lazy_tensors::Shape(shape);
}

namespace ops {

Cat::Cat(std::vector<torch::lazy::Value> values, lazy_tensors::int64 dim,
         const std::vector<at::ScalarType>& out_dtypes,
         const std::vector<std::vector<int64_t>>& out_shapes)
    : TsNode(torch::lazy::OpKind(at::aten::cat), values,
             convertShape(out_dtypes, out_shapes),
             /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim),
      at_dtypes_(out_dtypes),
      at_shapes_(out_shapes) {
}

std::string Cat::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
