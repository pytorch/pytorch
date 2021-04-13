#include "lazy_tensor_core/csrc/ops/svd.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

SVD::SVD(const Value& input, bool some, bool compute_uv)
    : Node(ir::OpKind(at::aten::svd), {input},
           /*num_outputs=*/3, lazy_tensors::util::MHash(some, compute_uv)),
      some_(some),
      compute_uv_(compute_uv) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr SVD::Clone(OpList operands) const {
  return MakeNode<SVD>(operands.at(0), some_, compute_uv_);
}

std::string SVD::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", some=" << some_
     << ", compute_uv=" << compute_uv_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
