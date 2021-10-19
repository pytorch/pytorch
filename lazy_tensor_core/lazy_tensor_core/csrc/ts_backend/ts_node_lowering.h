#pragma once

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_lowering_context.h"

namespace torch_lazy_tensors {
namespace compiler {

class TSNodeLoweringInterface : public NodeLowering {
 public:
  TSNodeLoweringInterface(ts_backend::TSLoweringContext* loctx)
      : NodeLowering(loctx) {}

  virtual ~TSNodeLoweringInterface() = default;

  virtual bool Lower(const torch::lazy::Node* node) = 0;

  // TODO(whc) the whole point of this interface class is to let TsNode access
  // LowerNonCodegenOps. maybe there is a simpler way...
  virtual TSOpVector LowerNonCodegenOps(const torch::lazy::Node* node) = 0;
  // ... and LowerBuiltin
  virtual TSOpVector LowerBuiltin(
      c10::Symbol sym, const std::vector<torch::jit::NamedValue>& arguments,
      const std::vector<torch::jit::NamedValue>& kwarguments = {}) = 0;

  static std::unique_ptr<NodeLowering> Create(ir::LoweringContext* loctx);

  virtual lazy_tensors::Shape Infer(const torch::lazy::Node* node) = 0;

  static NodeLowering* Get();
};

TSNodeLoweringInterface* GetTSNodeLowering();
std::unique_ptr<NodeLowering> CreateTSNodeLowering(ir::LoweringContext* loctx);

}  // namespace compiler
}  // namespace torch_lazy_tensors
