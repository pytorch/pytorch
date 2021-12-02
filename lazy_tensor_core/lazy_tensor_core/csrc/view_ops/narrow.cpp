#include "lazy_tensor_core/csrc/view_ops/narrow.h"

#include "lazy_tensor_core/csrc/ops/ltc_ops.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Narrow::Narrow(const torch::lazy::Value& input,
               c10::ArrayRef<int64_t> base_indices,
               c10::ArrayRef<int64_t> sizes)
    : torch::lazy::TsNode(torch::lazy::OpKind(at::aten::narrow), {input},
                          /*num_outputs=*/1,
                          torch::lazy::MHash(base_indices, sizes)),
      base_indices_(base_indices.begin(), base_indices.end()),
      sizes_(sizes.begin(), sizes.end()) {
  SetShapeDeferred([&]() {
    return torch::lazy::Shape(
        torch::lazy::GetShapeFromTsOutput(operand(0)).scalar_type(), sizes);
  });
}

std::string Narrow::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString() << ", base_indices=("
     << c10::Join(", ", base_indices_) << "), sizes=("
     << c10::Join(", ", sizes_) << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
