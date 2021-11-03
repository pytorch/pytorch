#pragma once

#include "lazy_tensor_core/csrc/cross_replica_reduces.h"
#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class AllToAll : public TsNode {
 public:
  AllToAll(const torch::lazy::Value& input, const torch::lazy::Value& token,
           int64_t split_dimension, int64_t concat_dimension,
           int64_t split_count, std::vector<std::vector<int64_t>> groups);

  std::string ToString() const override;

  int64_t split_dimension() const { return split_dimension_; }

  int64_t concat_dimension() const { return concat_dimension_; }

  int64_t split_count() const { return split_count_; }

  const std::vector<std::vector<int64_t>>& groups() const { return groups_; }

 private:
  int64_t split_dimension_;
  int64_t concat_dimension_;
  int64_t split_count_;
  std::vector<std::vector<int64_t>> groups_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
