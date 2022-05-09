#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

class TORCH_API Narrow : public TsNode {
 public:
  Narrow(
      const Value& input,
      c10::ArrayRef<int64_t> base_indices,
      c10::ArrayRef<int64_t> sizes);

  std::string ToString() const override;

  const std::vector<int64_t>& base_indices() const {
    return base_indices_;
  }

  const std::vector<int64_t>& sizes() const {
    return sizes_;
  }

 private:
  std::vector<int64_t> base_indices_;
  std::vector<int64_t> sizes_;
};

} // namespace lazy
} // namespace torch
