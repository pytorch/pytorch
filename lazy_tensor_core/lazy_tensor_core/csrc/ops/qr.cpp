#include "lazy_tensor_core/csrc/ops/qr.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

QR::QR(const Value& input, bool some)
    : Node(ir::OpKind(at::aten::qr), {input},
           /*num_outputs=*/2, lazy_tensors::util::MHash(some)),
      some_(some) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr QR::Clone(OpList operands) const {
  return MakeNode<QR>(operands.at(0), some_);
}

std::string QR::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", some=" << some_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
