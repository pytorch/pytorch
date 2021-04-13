#include "lazy_tensor_core/csrc/ops/symeig.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

SymEig::SymEig(const Value& input, bool eigenvectors, bool lower)
    : Node(ir::OpKind(at::aten::symeig), {input},
           /*num_outputs=*/2, lazy_tensors::util::MHash(eigenvectors, lower)),
      eigenvectors_(eigenvectors),
      lower_(lower) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr SymEig::Clone(OpList operands) const {
  return MakeNode<SymEig>(operands.at(0), eigenvectors_, lower_);
}

std::string SymEig::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", eigenvectors=" << eigenvectors_
     << ", lower=" << lower_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
