#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

#include "lazy_tensor_core/csrc/cross_replica_reduces.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class AllReduce : public torch::lazy::TsNode {
 public:
  AllReduce(AllReduceType reduce_type, torch::lazy::OpList operands,
            const torch::lazy::Value& token, double scale,
            std::vector<std::vector<int64_t>> groups);

  std::string ToString() const override;

  AllReduceType reduce_type() const { return reduce_type_; }

  double scale() const { return scale_; }

  const std::vector<std::vector<int64_t>>& groups() const { return groups_; }

 private:
  AllReduceType reduce_type_;
  double scale_;
  std::vector<std::vector<int64_t>> groups_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
