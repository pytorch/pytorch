#include "lazy_tensor_core/csrc/view_ops/as_strided_view_update.h"

#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/view_ops/as_strided.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

AsStridedViewUpdate::AsStridedViewUpdate(const torch::lazy::Value& target,
                                         const torch::lazy::Value& input,
                                         std::vector<int64_t> size,
                                         std::vector<int64_t> stride,
                                         int64_t storage_offset)
    : torch::lazy::TsNode(
          ltc_as_strided_view_update, {target, input},
          [&]() {
            return torch::lazy::Shape(
                torch::lazy::GetShapeFromTsValue(target).scalar_type(), size);
          },
          /*num_outputs=*/1, torch::lazy::MHash(size, stride, storage_offset)),
      size_(std::move(size)),
      stride_(std::move(stride)),
      storage_offset_(storage_offset) {}

std::string AsStridedViewUpdate::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString() << ", size=(" << c10::Join(", ", size_)
     << "), stride=(" << c10::Join(", ", stride_)
     << "), storage_offset=" << storage_offset_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
