#pragma once

#include "lazy_tensor_core/csrc/ir.h"
#include "lazy_tensors/span.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Repeat : public Node {
 public:
  Repeat(const Value& input, std::vector<lazy_tensors::int64> repeats);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  const std::vector<lazy_tensors::int64>& repeats() const { return repeats_; }

 private:
  // The number of repeats along each dimension.
  std::vector<lazy_tensors::int64> repeats_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
