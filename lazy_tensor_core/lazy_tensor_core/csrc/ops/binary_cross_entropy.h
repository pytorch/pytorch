#pragma once

#include <c10/util/Optional.h>

#include "lazy_tensor_core/csrc/ir.h"
#include "lazy_tensor_core/csrc/reduction.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class BinaryCrossEntropy : public Node {
 public:
  BinaryCrossEntropy(const Value& logits, const Value& labels,
                     const c10::optional<Value>& weight,
                     ReductionMode reduction);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  ReductionMode reduction() const { return reduction_; }

 private:
  ReductionMode reduction_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
