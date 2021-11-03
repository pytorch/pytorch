#pragma once

#include "lazy_tensor_core/csrc/cross_replica_reduces.h"
#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class CollectivePermute : public TsNode {
 public:
  CollectivePermute(
      const torch::lazy::Value& input, const torch::lazy::Value& token,
      std::vector<std::pair<int64_t, int64_t>> source_target_pairs);

  std::string ToString() const override;

  const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs() const {
    return source_target_pairs_;
  }

 private:
  std::vector<std::pair<int64_t, int64_t>> source_target_pairs_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
