#include "lazy_tensor_core/csrc/view_ops/resize.h"

#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {
namespace {

torch::lazy::Shape NodeOutputShape(const torch::lazy::Value& input,
                                   c10::ArrayRef<int64_t> size) {
  return torch::lazy::Shape(
      torch::lazy::GetShapeFromTsValue(input).scalar_type(), size);
}

}  // namespace

Resize::Resize(const torch::lazy::Value& input, std::vector<int64_t> size)
    : torch::lazy::TsNode(
          torch::lazy::OpKind(at::aten::resize), {input},
          [&]() { return NodeOutputShape(input, size); },
          /*num_outputs=*/1, torch::lazy::MHash(size)),
      size_(std::move(size)) {}

std::string Resize::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString() << ", size=(" << c10::Join(", ", size_)
     << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
