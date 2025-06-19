#pragma once

#include <torch/csrc/lazy/core/ir.h>

#include <string>

namespace torch::lazy {

class BackendDevice;

class TORCH_API DumpUtil {
 public:
  static std::string ToDot(c10::ArrayRef<const Node*> nodes);

  static std::string PostOrderToDot(
      c10::ArrayRef<const Node*> post_order,
      c10::ArrayRef<const Node*> roots);

  static std::string ToText(c10::ArrayRef<const Node*> nodes);

  static std::string PostOrderToText(
      c10::ArrayRef<const Node*> post_order,
      c10::ArrayRef<const Node*> roots);

  static std::string ToBackend(
      c10::ArrayRef<Value> values,
      const BackendDevice& device);
};

} // namespace torch::lazy
