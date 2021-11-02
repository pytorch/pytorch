#pragma once

#include "lazy_tensor_core/csrc/lowering_context.h"
#include "torch/csrc/lazy/core/ir.h"

namespace torch_lazy_tensors {
namespace compiler {


class NodeLowering {
  /**
   * NodeLowering is an internal interface within a backend.
   * 
   * Its intended use is by backend lowering context impl to perform lowering
   *   created inside TSLoweringContext() using NodeLowering::Create
   *
   * But, it leaked out into Ops/ classes for shape inference.
   * Used from Ops/ classes to do shape inference (to be discontinued..)
   *   NodeLowering::Get() producs the lowering used by Ops/
   *     and is implemented via backend_registrar 
   *      
   * */
 public:
  NodeLowering(ir::LoweringContext* loctx) : loctx_(loctx) {}

  virtual ~NodeLowering() = default;

  virtual bool Lower(const torch::lazy::Node* node) = 0;

  static std::unique_ptr<NodeLowering> Create(ir::LoweringContext* loctx);

  // TODO(asuhan): this method shouldn't be needed, the core can provide all
  // inference.
  virtual lazy_tensors::Shape Infer(const torch::lazy::Node* node) = 0;

  static NodeLowering* Get();

 protected:
  ir::LoweringContext* loctx_;
};


}  // namespace compiler
}  // namespace torch_lazy_tensors
