#pragma once

#include <string>

#include "torch/csrc/lazy/core/ir.h"

namespace torch_lazy_tensors {

struct Device;

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
                               const Device &device);
};

}  // namespace ir
}  // namespace torch_lazy_tensors
