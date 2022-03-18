#include <torch/csrc/lazy/core/view_ops/as_strided.h>

#include <algorithm>

#include <torch/csrc/lazy/core/tensor_util.h>
#include <torch/csrc/lazy/core/util.h>

namespace torch {
namespace lazy {

AsStrided::AsStrided(
    const Value& input,
    std::vector<int64_t> size,
    std::vector<int64_t> stride,
    int64_t storage_offset)
    : TsNode(
          OpKind(at::aten::as_strided),
          {input},
          [&]() {
            return Shape(input.shape().scalar_type(), size);
          },
          /*num_outputs=*/1,
          MHash(size, stride, storage_offset)),
      size_(std::move(size)),
      stride_(std::move(stride)),
      storage_offset_(storage_offset) {}

std::string AsStrided::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", size=(" << c10::Join(", ", size_)
     << "), stride=(" << c10::Join(", ", stride_)
     << "), storage_offset=" << storage_offset_;
  return ss.str();
}

bool AsStrided::StrideIsSupported(c10::ArrayRef<int64_t> stride) {
  std::vector<int64_t> sorted_stride(stride.begin(), stride.end());
  std::sort(sorted_stride.begin(), sorted_stride.end());
  return stride.empty() || sorted_stride.front() == 1;
}

std::vector<int64_t> AsStrided::GetArrayStridePermutation(
    c10::ArrayRef<int64_t> stride) {
  std::vector<int64_t> permutation = Iota<int64_t>(stride.size());
  std::sort(permutation.begin(), permutation.end(), [&](int64_t a, int64_t b) {
    return stride[a] > stride[b];
  });
  return permutation;
}

} // namespace lazy
} // namespace torch
