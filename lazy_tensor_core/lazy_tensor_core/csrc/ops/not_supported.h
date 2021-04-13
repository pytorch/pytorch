#pragma once

#include <string>

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class NotSupported : public Node {
 public:
  NotSupported(std::string description, lazy_tensors::Shape shape);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  const std::string& description() const { return description_; }

 private:
  std::string description_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
