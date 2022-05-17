#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

class TORCH_API Permute : public TsNode {
 public:
  static OpKind ClassOpKind() {
    return OpKind(at::aten::permute);
  }

  Permute(const Value& input, std::vector<int64_t> dims);

  bool CanBeReused(const Value& input, c10::ArrayRef<int64_t> dims) const {
    size_t i = 0;
    return (operand(i++) == input && dims_ == dims);
  }

  std::string ToString() const override;

  const std::vector<int64_t>& dims() const {
    return dims_;
  }

 private:
  // The permutation of dimensions.
  std::vector<int64_t> dims_;
};

} // namespace lazy
} // namespace torch
