#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

#include <vector>

namespace torch {
namespace lazy {

class TORCH_API Size : public TsNode {
 public:
  Size(Value input, int64_t dim);

  std::string ToString() const override;

  virtual TSOpVector Lower(std::shared_ptr<torch::jit::GraphFunction> function,
                          TSLoweringContext* loctx) const override;


  private:
  int64_t dim_ = 0;
};

} // namespace lazy
} // namespace torch
