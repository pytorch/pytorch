#pragma once

#include <torch/csrc/lazy/backend/backend_node.h>

#include <vector>

namespace torch {
namespace lazy {

class TORCH_API View : public BackendNode {
 public:
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
