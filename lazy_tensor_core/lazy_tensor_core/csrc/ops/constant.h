#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

#include "lazy_tensors/literal.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class Constant : public torch::lazy::TsNode {
 public:
  Constant(lazy_tensors::Literal value);

  std::string ToString() const override;

  const lazy_tensors::Literal& value() const { return value_; }

 private:
  lazy_tensors::Literal value_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
