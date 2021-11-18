#include "lazy_tensor_core/csrc/ops/repeat.h"

#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Repeat::Repeat(const torch::lazy::Value& input, std::vector<int64_t> repeats)
    : torch::lazy::TsNode(torch::lazy::OpKind(at::aten::repeat), {input},
                          /*num_outputs=*/1, torch::lazy::MHash(repeats)),
      repeats_(std::move(repeats)) {
  SetShapeDeferred(
      [&]() { return compiler::InferShape(this); });
}

std::string Repeat::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString() << ", repeats=("
     << c10::Join(", ", repeats_) << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
