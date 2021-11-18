#include "lazy_tensor_core/csrc/ops/expand.h"

#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Expand::Expand(const torch::lazy::Value& input, std::vector<int64_t> size,
               bool is_scalar_expand)
    : torch::lazy::TsNode(torch::lazy::OpKind(at::aten::expand), {input},
                          /*num_outputs=*/1,
                          torch::lazy::MHash(size, is_scalar_expand)),
      size_(std::move(size)),
      is_scalar_expand_(is_scalar_expand) {
  SetShapeDeferred(
      [&]() { return compiler::InferShape(this); });
}

std::string Expand::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString() << ", size=(" << c10::Join(", ", size_)
     << "), is_scalar_expand=" << is_scalar_expand_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
