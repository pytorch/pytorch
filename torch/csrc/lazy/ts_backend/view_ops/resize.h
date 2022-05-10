#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

class TORCH_API Resize : public TsNode {
 public:
  static OpKind ClassOpKind() {
    return OpKind(at::aten::resize);
  }

  Resize(const Value& input, std::vector<int64_t> size);

  std::string ToString() const override;

  const std::vector<int64_t>& size() const {
    return size_;
  }

 private:
  std::vector<int64_t> size_;
};

} // namespace lazy
} // namespace torch
