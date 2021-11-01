#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Random : public TsNode {
 public:
  Random(const torch::lazy::Value& input, const c10::optional<int64_t>& from, const c10::optional<int64_t>& to);

  std::string ToString() const override;
  TSOpVector Lower(std::shared_ptr<torch::jit::GraphFunction> function,
      ts_backend::TSLoweringContext* loctx) const override;

  c10::optional<int64_t> from;
  c10::optional<int64_t> to;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
