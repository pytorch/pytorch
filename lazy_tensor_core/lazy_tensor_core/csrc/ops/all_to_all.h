#pragma once

#include "lazy_tensor_core/csrc/cross_replica_reduces.h"
#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class AllToAll : public Node {
 public:
  AllToAll(const Value& input, const Value& token,
           lazy_tensors::int64 split_dimension,
           lazy_tensors::int64 concat_dimension,
           lazy_tensors::int64 split_count,
           std::vector<std::vector<lazy_tensors::int64>> groups);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  lazy_tensors::int64 split_dimension() const { return split_dimension_; }

  lazy_tensors::int64 concat_dimension() const { return concat_dimension_; }

  lazy_tensors::int64 split_count() const { return split_count_; }

  const std::vector<std::vector<lazy_tensors::int64>>& groups() const {
    return groups_;
  }

 private:
  lazy_tensors::int64 split_dimension_;
  lazy_tensors::int64 concat_dimension_;
  lazy_tensors::int64 split_count_;
  std::vector<std::vector<lazy_tensors::int64>> groups_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
