#pragma once

#include <c10/util/ArrayRef.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/backend/lowering_context.h>
#include <torch/csrc/lazy/core/shape.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>

namespace torch {
namespace lazy {

using TSOpVector = std::vector<torch::jit::Value*>;

class TORCH_API TsNode : public lazy::Node {
 public:
  using Node::Node;

  // Lower is a backend-specific method since it returns a backend specific
  // type. hence, it is convenient to define it differently per-backend rather
  // than at Node API
  virtual TSOpVector Lower(std::shared_ptr<torch::jit::GraphFunction> function,
                           TSLoweringContext* loctx) const;
};

}  // namespace lazy
}  // namespace torch
