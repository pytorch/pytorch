#include "lazy_tensor_core/csrc/ops/as_strided.h"

#include <algorithm>

#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/torch_util.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"
#include "lazy_tensors/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

AsStrided::AsStrided(const Value& input, std::vector<lazy_tensors::int64> size,
                     std::vector<lazy_tensors::int64> stride,
                     lazy_tensors::int64 storage_offset)
    : Node(ir::OpKind(at::aten::as_strided), {input},
           [&]() {
             return lazy_tensors::ShapeUtil::MakeShape(
                 input.shape().element_type(), size);
           },
           /*num_outputs=*/1,
           lazy_tensors::util::MHash(size, stride, storage_offset)),
      size_(std::move(size)),
      stride_(std::move(stride)),
      storage_offset_(storage_offset) {}

std::string AsStrided::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", size=(" << lazy_tensors::StrJoin(size_, ", ")
     << "), stride=(" << lazy_tensors::StrJoin(stride_, ", ")
     << "), storage_offset=" << storage_offset_;
  return ss.str();
}

NodePtr AsStrided::Clone(OpList operands) const {
  return MakeNode<AsStrided>(operands.at(0), size_, stride_, storage_offset_);
}

bool AsStrided::StrideIsSupported(
    const lazy_tensors::Shape& input_shape,
    lazy_tensors::Span<const lazy_tensors::int64> size,
    lazy_tensors::Span<const lazy_tensors::int64> stride,
    lazy_tensors::int64 storage_offset) {
  std::vector<lazy_tensors::int64> sorted_stride(stride.begin(), stride.end());
  std::sort(sorted_stride.begin(), sorted_stride.end());
  return stride.empty() || sorted_stride.front() == 1;
}

std::vector<lazy_tensors::int64> AsStrided::GetArrayStridePermutation(
    lazy_tensors::Span<const lazy_tensors::int64> stride,
    lazy_tensors::Span<const lazy_tensors::int64> size) {
  std::vector<lazy_tensors::int64> permutation =
      lazy_tensors::util::Iota<lazy_tensors::int64>(stride.size());
  std::sort(permutation.begin(), permutation.end(),
            [&](lazy_tensors::int64 a, lazy_tensors::int64 b) {
              return stride[a] > stride[b];
            });
  return permutation;
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
