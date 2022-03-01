#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <memory>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace kir {

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
class TORCH_CUDA_CU_API IrBuilder {
 public:
  explicit IrBuilder(Kernel* kernel) : kernel_(kernel) {}

  //! Allocate a new Kernel IR node, forwarding the arguments
  //! to the appropriate constructor
  template <class T, class... Args>
  T* create(Args&&... args) {
    const kir::Passkey passkey(kernel_);
    const auto node = new T(passkey, std::forward<Args>(args)...);
    kernel_->registerIrNode(passkey, std::unique_ptr<T>(node));
    return node;
  }

  // Unary operations
  Val* negExpr(Val* val);
  Val* notExpr(Val* val);
  Val* setExpr(Val* val);
  Val* setExprNamedScalar(const std::string& name, Val* val);
  Val* addressExprNamedScalar(const std::string& name, Val* val);

  // Binary operations
  Val* andExpr(Val* lhs, Val* rhs);
  Val* eqExpr(Val* lhs, Val* rhs);
  Val* gtExpr(Val* lhs, Val* rhs);
  Val* ltExpr(Val* lhs, Val* rhs);
  Val* leExpr(Val* lhs, Val* rhs);
  Val* geExpr(Val* lhs, Val* rhs);
  Val* addExpr(Val* lhs, Val* rhs);
  Val* subExpr(Val* lhs, Val* rhs);
  Val* mulExpr(Val* lhs, Val* rhs);
  Val* divExpr(Val* lhs, Val* rhs);
  Val* ceilDivExpr(Val* lhs, Val* rhs);
  Val* modExpr(Val* lhs, Val* rhs);
  Val* maxExpr(Val* lhs, Val* rhs);
  Val* minExpr(Val* lhs, Val* rhs);

  // Ternary operations
  Val* whereExpr(Val* pred, Val* lhs, Val* rhs);

  // Shortcuts for frequently used vals
  Int* zeroVal();
  Int* oneVal();
  Bool* falseVal();
  Bool* trueVal();

  NamedScalar* magicZeroVal();

 private:
  Val* newResult(DataType dtype);
  Val* newArithmeticExpr(BinaryOpType op_type, Val* lhs, Val* rhs);
  Val* newLogicExpr(BinaryOpType op_type, Val* lhs, Val* rhs);

 private:
  // Non-owning pointer to the kernel to be modified
  Kernel* kernel_ = nullptr;
  // Frequently used constant vals
  Int* zero_ = nullptr;
  Int* one_ = nullptr;
  Bool* false_ = nullptr;
  Bool* true_ = nullptr;

  // Magic zero corresponds to runtime/helpers.cu magic_zero
  NamedScalar* magic_zero_ = nullptr;
};

//! A wrapper builder with static expression simplification
//!
//! Example:
//! - addExpr(new Int(1), new Int(2)) -> Int(3)
//! - addExpr(new Int(0), new NamedScalar("foo")) -> NamedScalar("foo")
//!
//! Designed to be used to simplify predicate and index expressions in
//! generated code. Also, the shift validation may fail without
//! this simplification.
class TORCH_CUDA_CU_API SimplifyingIrBuilder : public IrBuilder {
 public:
  explicit SimplifyingIrBuilder(Kernel* kernel) : IrBuilder(kernel) {}

  Val* negExpr(Val* val);
  Val* notExpr(Val* val);

  Val* addExpr(Int* lhs, Int::ScalarType rhs);
  Val* addExpr(Int* lhs, Int* rhs);
  Val* addExpr(Val* lhs, Val* rhs);
  Val* subExpr(Val* lhs, Val* rhs);
  Val* andExpr(Val* lhs, Val* rhs);
};

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
