#pragma once

#include <torch/csrc/lazy/core/ir.h>

#include <string>

namespace torch {
namespace lazy {
    class BackendDevice;
}
}

namespace torch_lazy_tensors {
namespace ir {

class DumpUtil {
 public:
  static std::string ToDot(c10::ArrayRef<torch::lazy::Node *> nodes);

  static std::string PostOrderToDot(
      c10::ArrayRef<torch::lazy::Node *> post_order,
      c10::ArrayRef<torch::lazy::Node *> roots);

  static std::string ToText(c10::ArrayRef<torch::lazy::Node *> nodes);

  static std::string PostOrderToText(
      c10::ArrayRef<torch::lazy::Node *> post_order,
      c10::ArrayRef<torch::lazy::Node *> roots);

  static std::string ToBackend(c10::ArrayRef<torch::lazy::Value> values,
                               const torch::lazy::BackendDevice& device);
};

}  // namespace ir
}  // namespace torch_lazy_tensors
