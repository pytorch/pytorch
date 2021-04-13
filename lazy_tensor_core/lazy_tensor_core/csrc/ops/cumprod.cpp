#include "lazy_tensor_core/csrc/ops/cumprod.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/reduction.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/torch_util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

CumProd::CumProd(const Value& input, lazy_tensors::int64 dim,
                 c10::optional<at::ScalarType> dtype)
    : Node(ir::OpKind(at::aten::cumprod), {input},
           /*num_outputs=*/1,
           lazy_tensors::util::MHash(dim, OptionalOr<int>(dtype, -1))),
      dim_(dim),
      dtype_(dtype) {
  SetShapeDeferred(
      [&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr CumProd::Clone(OpList operands) const {
  return MakeNode<CumProd>(operands.at(0), dim_, dtype_);
}

std::string CumProd::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  if (dtype_) {
    ss << ", dtype=" << *dtype_;
  }
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
