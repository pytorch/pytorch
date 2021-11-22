#include "lazy_tensor_core/csrc/view_ops/as_strided.h"

#include <algorithm>

#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

AsStrided::AsStrided(const torch::lazy::Value& input, std::vector<int64_t> size,
                     std::vector<int64_t> stride, int64_t storage_offset)
    : torch::lazy::TsNode(
          torch::lazy::OpKind(at::aten::as_strided), {input},
          [&]() {
            return torch::lazy::Shape(
                torch::lazy::GetShapeFromTsValue(input).scalar_type(), size);
          },
          /*num_outputs=*/1, torch::lazy::MHash(size, stride, storage_offset)),
      size_(std::move(size)),
      stride_(std::move(stride)),
      storage_offset_(storage_offset) {}

std::string AsStrided::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString() << ", size=(" << c10::Join(", ", size_)
     << "), stride=(" << c10::Join(", ", stride_)
     << "), storage_offset=" << storage_offset_;
  return ss.str();
}

bool AsStrided::StrideIsSupported(const torch::lazy::Shape& input_shape,
                                  c10::ArrayRef<int64_t> size,
                                  c10::ArrayRef<int64_t> stride,
                                  int64_t storage_offset) {
  std::vector<int64_t> sorted_stride(stride.begin(), stride.end());
  std::sort(sorted_stride.begin(), sorted_stride.end());
  return stride.empty() || sorted_stride.front() == 1;
}

std::vector<int64_t> AsStrided::GetArrayStridePermutation(
    c10::ArrayRef<int64_t> stride, c10::ArrayRef<int64_t> size) {
  std::vector<int64_t> permutation =
      lazy_tensors::util::Iota<int64_t>(stride.size());
  std::sort(permutation.begin(), permutation.end(),
            [&](int64_t a, int64_t b) { return stride[a] > stride[b]; });
  return permutation;
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
