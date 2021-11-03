#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Stack : public TsNode {
 public:
  Stack(OpList values, int64_t dim);

  std::string ToString() const override;

  int64_t dim() const { return dim_; };

 private:
  int64_t dim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
