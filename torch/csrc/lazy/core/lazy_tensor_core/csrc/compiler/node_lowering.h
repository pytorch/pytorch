#pragma once

#include "lazy_tensor_core/csrc/ir.h"
#include "lazy_tensor_core/csrc/lowering_context.h"

namespace torch_lazy_tensors {
namespace compiler {

class NodeLowering {
 public:
  NodeLowering(ir::LoweringContext* loctx) : loctx_(loctx) {}

  virtual ~NodeLowering() = default;

  virtual bool Lower(const ir::Node* node) = 0;

  static std::unique_ptr<NodeLowering> Create(ir::LoweringContext* loctx);

  // TODO(asuhan): this method shouldn't be needed, the core can provide all
  // inference.
  virtual lazy_tensors::Shape Infer(const ir::Node* node) = 0;

  static NodeLowering* Get();

 protected:
  ir::LoweringContext* loctx_;
};

}  // namespace compiler
}  // namespace torch_lazy_tensors
