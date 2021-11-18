#pragma once

#include <vector>

#include "lazy_tensor_core/csrc/view_ops/opcode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class View : public BaseNode {
 public:
  View(const torch::lazy::Value& input, std::vector<int64_t> output_size);

  std::string ToString() const override;

  const std::vector<int64_t>& output_size() const { return output_size_; }

 private:
  std::vector<int64_t> output_size_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
