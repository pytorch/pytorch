#include "lazy_tensor_core/csrc/ops/cat.h"

#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Cat::Cat(std::vector<torch::lazy::Value> values, int64_t dim,
         const std::vector<at::ScalarType>& out_dtypes,
         const std::vector<std::vector<int64_t>>& out_shapes)
    : TsNode(torch::lazy::OpKind(at::aten::cat), values,
             lazy_tensors::convertShapes(out_dtypes, out_shapes),
             /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim),
      at_dtypes_(out_dtypes),
      at_shapes_(out_shapes) {}

std::string Cat::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
