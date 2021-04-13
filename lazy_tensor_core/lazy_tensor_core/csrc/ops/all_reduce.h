#pragma once

#include "lazy_tensor_core/csrc/cross_replica_reduces.h"
#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class AllReduce : public Node {
 public:
  AllReduce(AllReduceType reduce_type, lazy_tensors::Span<const Value> operands,
            const Value& token, double scale,
            std::vector<std::vector<lazy_tensors::int64>> groups);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  AllReduceType reduce_type() const { return reduce_type_; }

  double scale() const { return scale_; }

  const std::vector<std::vector<lazy_tensors::int64>>& groups() const {
    return groups_;
  }

 private:
  AllReduceType reduce_type_;
  double scale_;
  std::vector<std::vector<lazy_tensors::int64>> groups_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
