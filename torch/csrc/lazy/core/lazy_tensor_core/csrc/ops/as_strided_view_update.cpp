#include "lazy_tensor_core/csrc/ops/as_strided_view_update.h"

#include "lazy_tensor_core/csrc/ops/as_strided.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/torch_util.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

AsStridedViewUpdate::AsStridedViewUpdate(
    const Value& target, const Value& input,
    std::vector<lazy_tensors::int64> size,
    std::vector<lazy_tensors::int64> stride, lazy_tensors::int64 storage_offset)
    : Node(ltc_as_strided_view_update, {target, input},
           [&]() {
             return lazy_tensors::ShapeUtil::MakeShape(
                 target.shape().element_type(), size);
           },
           /*num_outputs=*/1,
           lazy_tensors::util::MHash(size, stride, storage_offset)),
      size_(std::move(size)),
      stride_(std::move(stride)),
      storage_offset_(storage_offset) {}

std::string AsStridedViewUpdate::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", size=(" << lazy_tensors::StrJoin(size_, ", ")
     << "), stride=(" << lazy_tensors::StrJoin(stride_, ", ")
     << "), storage_offset=" << storage_offset_;
  return ss.str();
}

NodePtr AsStridedViewUpdate::Clone(OpList operands) const {
  return MakeNode<AsStridedViewUpdate>(operands.at(0), operands.at(1), size_,
                                       stride_, storage_offset_);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
