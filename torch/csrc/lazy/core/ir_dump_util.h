#pragma once

#include <torch/csrc/lazy/core/ir.h>

#include <string>

namespace torch {
namespace lazy {

class BackendDevice;

class TORCH_API DumpUtil {
 public:
  static std::string ToDot(c10::ArrayRef<Node*> nodes);

  static std::string PostOrderToDot(
      c10::ArrayRef<Node*> post_order,
      c10::ArrayRef<Node*> roots);

  static std::string ToText(c10::ArrayRef<Node*> nodes);

  static std::string PostOrderToText(
      c10::ArrayRef<Node*> post_order,
      c10::ArrayRef<Node*> roots);

  static std::string ToBackend(
      c10::ArrayRef<Value> values,
      const BackendDevice& device);
};

} // namespace lazy
} // namespace torch
