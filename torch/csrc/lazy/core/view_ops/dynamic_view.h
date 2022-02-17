#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

#include <vector>

namespace torch {
namespace lazy {

class TORCH_API DynamicView : public TsNode {
 public:
  DynamicView(Value input, OpList dims);

//   std::string ToString() const override;

  virtual TSOpVector Lower(std::shared_ptr<torch::jit::GraphFunction> function,
                          TSLoweringContext* loctx) const override;

};

} // namespace lazy
} // namespace torch
