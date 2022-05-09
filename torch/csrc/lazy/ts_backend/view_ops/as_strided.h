#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

#include <vector>

namespace torch {
namespace lazy {

class TORCH_API AsStrided : public TsNode {
 public:
  AsStrided(
      const Value& input,
      std::vector<int64_t> size,
      std::vector<int64_t> stride,
      int64_t storage_offset);

  std::string ToString() const override;

  const std::vector<int64_t>& size() const {
    return size_;
  }

  const std::vector<int64_t>& stride() const {
    return stride_;
  }

  int64_t storage_offset() const {
    return storage_offset_;
  }

  static bool StrideIsSupported(c10::ArrayRef<int64_t> stride);

  static std::vector<int64_t> GetArrayStridePermutation(
      c10::ArrayRef<int64_t> stride);

 private:
  std::vector<int64_t> size_;
  std::vector<int64_t> stride_;
  int64_t storage_offset_;
};

} // namespace lazy
} // namespace torch
