#include "lazy_tensor_core/csrc/ops/softmax.h"

#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/torch_util.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {
namespace {

lazy_tensors::Shape NodeOutputShape(
    const Value& input, const c10::optional<at::ScalarType>& dtype) {
  if (dtype) {
    return lazy_tensors::ShapeUtil::ChangeElementType(
        input.shape(), MakeLtcPrimitiveType(*dtype, /*device=*/nullptr));
  }
  return input.shape();
}

}  // namespace

Softmax::Softmax(const Value& input, lazy_tensors::int64 dim,
                 c10::optional<at::ScalarType> dtype)
    : Node(ir::OpKind(at::aten::softmax), {input},
           [&]() { return NodeOutputShape(input, dtype); },
           /*num_outputs=*/1,
           lazy_tensors::util::MHash(dim, OptionalOr<int>(dtype, -1))),
      dim_(dim),
      dtype_(dtype) {}

NodePtr Softmax::Clone(OpList operands) const {
  return MakeNode<Softmax>(operands.at(0), dim_, dtype_);
}

std::string Softmax::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_
     << ", dtype=" << OptionalOr<int>(dtype_, -1);
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
