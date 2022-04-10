#include <torch/csrc/lazy/core/view_ops/as_strided_view_update.h>

#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>
#include <torch/csrc/lazy/core/tensor_util.h>
#include <torch/csrc/lazy/core/view_ops/as_strided.h>

namespace torch {
namespace lazy {

AsStridedViewUpdate::AsStridedViewUpdate(
    const Value& target,
    const Value& input,
    std::vector<int64_t> size,
    std::vector<int64_t> stride,
    int64_t storage_offset)
    : TsNode(
          ltc_as_strided_view_update,
          {target, input},
          [&]() {
            return Shape(target.shape().scalar_type(), size);
          },
          /*num_outputs=*/1,
          MHash(size, stride, storage_offset)),
      size_(std::move(size)),
      stride_(std::move(stride)),
      storage_offset_(storage_offset) {}

std::string AsStridedViewUpdate::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", size=(" << c10::Join(", ", size_)
     << "), stride=(" << c10::Join(", ", stride_)
     << "), storage_offset=" << storage_offset_;
  return ss.str();
}

} // namespace lazy
} // namespace torch
