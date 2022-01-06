#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/ir_cloner.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Clone an IR node, forwarding the arguments to the IrCloner constructor.
template <class T>
T* IrBuilder::clone(const T* src, IrCloner* ir_cloner) {
  TORCH_INTERNAL_ASSERT(
      ir_cloner != nullptr,
      "Cannot use create when a cloner object is set. Use clone.");

  TORCH_INTERNAL_ASSERT(
      ir_cloner->container() != nullptr,
      "Cloner doesn't have a valid container to store cloned object.");

  T* dest = new T(src, ir_cloner);
  const Statement* src_stmt = dynamic_cast<const Statement*>(src);
  Statement* dest_stmt = dynamic_cast<Statement*>(dest);

  auto dest_container = ir_cloner->container();
  auto src_container = src_stmt->container();

  dest_container->registerStmt(IrBuilderPasskey(dest_container), dest_stmt);

  if (src_container != dest_container) {
    dest_stmt->setName(IrBuilderPasskey(dest_container), src_stmt->name());
  }

  ir_cloner->registerClone(src_stmt, dest_stmt);

  return dest;
}

#define IR_BUILDER_INSTANTIATE(T) \
  template T* IrBuilder::clone(const T* src, IrCloner* ir_cloner);

// Vals
IR_BUILDER_INSTANTIATE(IterDomain)
IR_BUILDER_INSTANTIATE(TensorDomain)
IR_BUILDER_INSTANTIATE(TensorView)
IR_BUILDER_INSTANTIATE(Bool)
IR_BUILDER_INSTANTIATE(Double)
IR_BUILDER_INSTANTIATE(Int)
IR_BUILDER_INSTANTIATE(NamedScalar)

// Exprs
IR_BUILDER_INSTANTIATE(Split)
IR_BUILDER_INSTANTIATE(Merge)
IR_BUILDER_INSTANTIATE(TransposeOp)
IR_BUILDER_INSTANTIATE(ShiftOp)
IR_BUILDER_INSTANTIATE(GatherOp)
IR_BUILDER_INSTANTIATE(ViewOp)
IR_BUILDER_INSTANTIATE(UnaryOp)
IR_BUILDER_INSTANTIATE(BinaryOp)
IR_BUILDER_INSTANTIATE(TernaryOp)
IR_BUILDER_INSTANTIATE(ReductionOp)
IR_BUILDER_INSTANTIATE(WelfordOp)
IR_BUILDER_INSTANTIATE(BroadcastOp)

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
