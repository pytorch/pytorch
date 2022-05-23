#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

#include <vector>

namespace torch {
namespace lazy {

class TORCH_API View : public TsNode {
 public:
  static OpKind ClassOpKind() {
    return OpKind(at::aten::view);
  }

  View(const Value& input, std::vector<int64_t> output_size);

  std::string ToString() const override;

  const std::vector<int64_t>& output_size() const {
    return output_size_;
  }

 private:
  std::vector<int64_t> output_size_;
};

} // namespace lazy
} // namespace torch
