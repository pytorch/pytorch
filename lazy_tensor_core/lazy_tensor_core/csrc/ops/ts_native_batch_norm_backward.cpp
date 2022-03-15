#include "lazy_tensor_core/csrc/ops/ts_native_batch_norm_backward.h"

#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

TSNativeBatchNormBackward::TSNativeBatchNormBackward(
    const torch::lazy::Value& grad_out, const torch::lazy::Value& input,
    const torch::lazy::Value& weight, const torch::lazy::Value& running_mean,
    const torch::lazy::Value& running_var, const torch::lazy::Value& save_mean,
    const torch::lazy::Value& save_invstd, bool training, double eps,
    std::array<bool, 3> output_mask)
    : torch::lazy::TsNode(
          torch::lazy::OpKind(at::aten::native_batch_norm_backward),
          {grad_out, input, weight, running_mean, running_var, save_mean,
           save_invstd},
          {input.shape(),
           weight.shape(),
           weight.shape()},
          /*num_outputs=*/3,
          torch::lazy::MHash(training, eps, output_mask[0], output_mask[1],
                             output_mask[2])),
      training_(training),
      eps_(eps),
      output_mask_(output_mask) {}

TSNativeBatchNormBackward::TSNativeBatchNormBackward(
    const torch::lazy::Value& grad_out, const torch::lazy::Value& input,
    const torch::lazy::Value& weight, const torch::lazy::Value& save_mean,
    const torch::lazy::Value& save_invstd, bool training, double eps,
    std::array<bool, 3> output_mask)
    : torch::lazy::TsNode(
          torch::lazy::OpKind(at::aten::native_batch_norm_backward),
          {grad_out, input, weight, save_mean, save_invstd},
          {input.shape(),
           weight.shape(),
           weight.shape()},
          /*num_outputs=*/3,
          torch::lazy::MHash(training, eps, output_mask[0], output_mask[1],
                             output_mask[2])),
      training_(training),
      eps_(eps),
      output_mask_(output_mask) {}

std::string TSNativeBatchNormBackward::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString() << ", training=" << training_
     << ", eps=" << eps_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
