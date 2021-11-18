#include "lazy_tensor_core/csrc/ops/constant_pad_nd.h"

#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/ops/scalar.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

ConstantPadNd::ConstantPadNd(const torch::lazy::Value& input,
                             std::vector<int64_t> pad, const at::Scalar& value)
    : torch::lazy::TsNode(
          torch::lazy::OpKind(at::aten::constant_pad_nd), {input},
          /*num_outputs=*/1, torch::lazy::MHash(pad, ScalarHash(value))),
      pad_(std::move(pad)),
      value_(value) {
  SetShapeDeferred(
      [&]() { return compiler::InferShape(this); });
}

std::string ConstantPadNd::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString() << ", pad=(" << c10::Join(", ", pad_)
     << ")"
     << ", value=" << value_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
