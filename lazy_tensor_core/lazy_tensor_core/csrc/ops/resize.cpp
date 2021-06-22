#include "lazy_tensor_core/csrc/ops/resize.h"

#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {
namespace {

lazy_tensors::Shape NodeOutputShape(
    const Value& input, lazy_tensors::Span<const lazy_tensors::int64> size) {
  return lazy_tensors::ShapeUtil::MakeShape(input.shape().element_type(), size);
}

}  // namespace

Resize::Resize(const Value& input, std::vector<lazy_tensors::int64> size)
    : Node(ir::OpKind(at::aten::resize), {input},
           [&]() { return NodeOutputShape(input, size); },
           /*num_outputs=*/1, lazy_tensors::util::MHash(size)),
      size_(std::move(size)) {}

NodePtr Resize::Clone(OpList operands) const {
  return MakeNode<Resize>(operands.at(0), size_);
}

std::string Resize::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", size=(" << lazy_tensors::StrJoin(size_, ", ")
     << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
