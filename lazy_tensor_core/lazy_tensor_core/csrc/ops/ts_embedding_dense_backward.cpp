#include "lazy_tensor_core/csrc/ops/ts_embedding_dense_backward.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/debug_macros.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

TSEmbeddingDenseBackward::TSEmbeddingDenseBackward(
    const torch::lazy::Value& grad_output, const torch::lazy::Value& indices,
    lazy_tensors::int64 num_weights, lazy_tensors::int64 padding_idx,
    bool scale_grad_by_freq)
    : TsNode(torch::lazy::OpKind(at::aten::embedding_dense_backward),
           {grad_output, indices}, /*num_outputs=*/1,
           torch::lazy::MHash(num_weights, padding_idx,
                                     scale_grad_by_freq)),
      num_weights_(num_weights),
      padding_idx_(padding_idx),
      scale_grad_by_freq_(scale_grad_by_freq) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr TSEmbeddingDenseBackward::Clone(OpList operands) const {
  return torch::lazy::MakeNode<TSEmbeddingDenseBackward>(operands.at(0), operands.at(1),
                                            num_weights_, padding_idx_,
                                            scale_grad_by_freq_);
}

std::string TSEmbeddingDenseBackward::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", num_weights=" << num_weights_
     << ", padding_idx=" << padding_idx_
     << ", scale_grad_by_freq=" << scale_grad_by_freq_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
