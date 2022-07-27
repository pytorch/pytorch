#pragma once

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_container.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace kir {
class Kernel;
}

class IrCloner;

// Passkey for builder to register properties with statements, and to call
// functions in IrContainer
class TORCH_CUDA_CU_API IrBuilderPasskey {
  friend class IrBuilder;

 public:
  // TODO: Collapse ir_container and Kernel once Kernel inherits from
  // IrContainer
  IrContainer* const ir_container_ = nullptr;

 private:
  explicit IrBuilderPasskey(IrContainer* ir_container);
};

//! IR builder interface
class TORCH_CUDA_CU_API IrBuilder {
 public:
  //! Allocate a new IR node, forwarding the arguments to the appropriate
  //! constructor and registering with the container
  template <class T, class... Args>
  static T* create(Args&&... args) {
    auto container = FusionGuard::getCurFusion();
    // return create<T>(container, std::forward<Args>(args)...);
    TORCH_INTERNAL_ASSERT(
        container != nullptr, "Need an active container to build IR.");
    T* node = new T(IrBuilderPasskey(container), std::forward<Args>(args)...);

    container->registerStmt(IrBuilderPasskey(container), node);

    return node;
  }

  //! Allocate a new IR node, forwarding the arguments to the appropriate
  //! constructor and registering with the container
  template <class T, class... Args>
  static T* create(IrContainer* container, Args&&... args) {
    TORCH_INTERNAL_ASSERT(
        container != nullptr, "Need an active container to build IR.");
    T* node = new T(IrBuilderPasskey(container), std::forward<Args>(args)...);

    container->registerStmt(IrBuilderPasskey(container), node);

    return node;
  }

  //! Clone an IR node, forwarding the arguments to the IrCloner constructor.
  //! Register clones with IrCloner's target container.
  template <class T>
  static T* clone(const T* src, IrCloner* ir_cloner);

  // Unary operations
  static Val* negExpr(Val* val);
  static Val* notExpr(Val* val);
  static Val* setExpr(Val* val);
  static Val* setExprNamedScalar(const std::string& name, Val* val);
  static Val* addressExprNamedScalar(const std::string& name, Val* val);

  // Binary operations
  static Val* andExpr(Val* lhs, Val* rhs);
  static Val* eqExpr(Val* lhs, Val* rhs);
  static Val* gtExpr(Val* lhs, Val* rhs);
  static Val* ltExpr(Val* lhs, Val* rhs);
  static Val* leExpr(Val* lhs, Val* rhs);
  static Val* geExpr(Val* lhs, Val* rhs);
  static Val* addExpr(Val* lhs, Val* rhs);
  static Val* subExpr(Val* lhs, Val* rhs);
  static Val* mulExpr(Val* lhs, Val* rhs);
  static Val* divExpr(Val* lhs, Val* rhs);
  static Val* ceilDivExpr(Val* lhs, Val* rhs);
  static Val* modExpr(Val* lhs, Val* rhs);
  static Val* maxExpr(Val* lhs, Val* rhs);
  static Val* minExpr(Val* lhs, Val* rhs);

  // Ternary operations
  static Val* whereExpr(Val* pred, Val* lhs, Val* rhs);

  // Swizzle operations
  static Val* swizzle2DIntExpr(
      Val* x,
      Val* y,
      Val* extent_x,
      Val* extent_y,
      Swizzle2DType swizzle_type);
  static Val* pairSelectExpr(Val* in, kir::PairSelect::Selection sel);

 private:
  static Val* newResult(DataType dtype);
  static Val* newArithmeticExpr(BinaryOpType op_type, Val* lhs, Val* rhs);
  static Val* newLogicExpr(BinaryOpType op_type, Val* lhs, Val* rhs);
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
  static Val* negExpr(Val* val);
  static Val* notExpr(Val* val);

  static Val* addExpr(Int* lhs, Int::ScalarType rhs);
  static Val* addExpr(Val* lhs, Int::ScalarType rhs);
  static Val* addExpr(Int* lhs, Int* rhs);
  static Val* addExpr(Val* lhs, Val* rhs);
  static Val* subExpr(Val* lhs, Val* rhs);
  static Val* mulExpr(Int* lhs, Int::ScalarType rhs);
  static Val* mulExpr(Val* lhs, Int::ScalarType rhs);
  static Val* mulExpr(Int* lhs, Int* rhs);
  static Val* mulExpr(Val* lhs, Val* rhs);
  static Val* andExpr(Val* lhs, Val* rhs);
  static Val* maxExpr(Val* lhs, Val* rhs);
  static Val* minExpr(Val* lhs, Val* rhs);
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
