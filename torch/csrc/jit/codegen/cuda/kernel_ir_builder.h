#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <memory>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace kir {

// Simple classification helpers
bool isLoweredScalar(const Val* val);
bool isLoweredVal(const Val* val);

//! Kernel IR builder interface
//!
//! The only way to create new Kernel IR nodes is through the
//! kir::IrBuilder interface. An IrBuilder instance is attached to a
//! particular Kernel instance and it provides methods for creating
//! single nodes (kir::IrBuilder::create()) or basic composite expressions
//! (ex. kir::IrBuilder::addExpr()).
//!
//! If the Kernel object is readily available, an IrBuilder can be "wrapped"
//! around it directly:
//!
//!   kir::IrBuilder ir_builder(kernel);
//!
//! During lowering, another option is to create an IrBuilder for the
//! kernel that is being created:
//!
//!   kir::IrBuilder ir_builder(GpuLower::current()->kernel());
//!
//! Once we have an IR builder instance, creating nodes looks like:
//!
//!   auto new_node = ir_builder.create<kir::Int>(1));
//!   auto result = ir_builder.mulExpr(lhs, rhs);
//!
class IrBuilder {
 public:
  explicit IrBuilder(Kernel* kernel) : kernel_(kernel) {}

  //! Allocate a new Kernel IR node, forwarding the arguments
  //! to the appropriate constructor
  template <class T, class... Args>
  T* create(Args&&... args) {
    // TODO(kir): switch this to Kernel registration
    return new T(kir::Passkey(), std::forward<Args>(args)...);
  }

  // Binary expressions
  Val* andExpr(Val* lhs, Val* rhs);
  Val* eqExpr(Val* lhs, Val* rhs);
  Val* ltExpr(Val* lhs, Val* rhs);
  Val* addExpr(Val* lhs, Val* rhs);
  Val* subExpr(Val* lhs, Val* rhs);
  Val* mulExpr(Val* lhs, Val* rhs);
  Val* divExpr(Val* lhs, Val* rhs);
  Val* ceilDivExpr(Val* lhs, Val* rhs);
  Val* modExpr(Val* lhs, Val* rhs);

 private:
  Val* newResult(const Val* lhs, const Val* rhs);
  Val* newArithmeticExpr(BinaryOpType op_type, Val* lhs, Val* rhs);
  Val* newLogicExpr(BinaryOpType op_type, Val* lhs, Val* rhs);

 private:
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-private-field"
  // Non-owning pointer to the kernel to be modified
  Kernel* kernel_ = nullptr;
#pragma clang diagnostic pop
};

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
