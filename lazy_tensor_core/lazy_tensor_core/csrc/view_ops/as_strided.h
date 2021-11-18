#pragma once

#include <vector>

#include "lazy_tensor_core/csrc/view_ops/opcode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class AsStrided : public BaseNode {
 public:
  AsStrided(const torch::lazy::Value& input, std::vector<int64_t> size,
            std::vector<int64_t> stride, int64_t storage_offset);

  std::string ToString() const override;

  const std::vector<int64_t>& size() const { return size_; }

  const std::vector<int64_t>& stride() const { return stride_; }

  int64_t storage_offset() const { return storage_offset_; }

  static bool StrideIsSupported(const torch::lazy::Shape& input_shape,
                                c10::ArrayRef<int64_t> size,
                                c10::ArrayRef<int64_t> stride,
                                int64_t storage_offset);

  static std::vector<int64_t> GetArrayStridePermutation(
      c10::ArrayRef<int64_t> stride, c10::ArrayRef<int64_t> size);

 private:
  std::vector<int64_t> size_;
  std::vector<int64_t> stride_;
  int64_t storage_offset_;
};

class AsStridedReverse : public BaseNode {
 public:
  AsStridedReverse(const torch::lazy::Value& target,
                   const torch::lazy::Value& input, std::vector<int64_t> size,
                   std::vector<int64_t> stride, int64_t storage_offset);

  std::string ToString() const override;

  const std::vector<int64_t>& size() const { return size_; }

  const std::vector<int64_t>& stride() const { return stride_; }

  int64_t storage_offset() const { return storage_offset_; }

 private:
  std::vector<int64_t> size_;
  std::vector<int64_t> stride_;
  int64_t storage_offset_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
