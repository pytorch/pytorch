#pragma once

#include <torch/csrc/lazy/backend/backend_node.h>

namespace torch {
namespace lazy {

class TORCH_API Permute : public BackendNode {
 public:
  Permute(const Value& input, std::vector<int64_t> dims);

  std::string ToString() const override;

  const std::vector<int64_t>& dims() const {
    return dims_;
  }

  static Shape MakePermuteShape(
      const Shape& source_shape,
      c10::ArrayRef<int64_t> permutation);

 private:
  // The permutation of dimensions.
  std::vector<int64_t> dims_;
};

} // namespace lazy
} // namespace torch
