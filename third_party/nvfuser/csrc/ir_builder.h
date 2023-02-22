#pragma once

#include <ir_all_nodes.h>
#include <ir_builder_passkey.h>

namespace nvfuser {

namespace kir {
class Kernel;
}

class IrCloner;
class IrContainer;

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
  static Bool* andExpr(Val* lhs, Val* rhs);
  static Bool* orExpr(Val* lhs, Val* rhs);
  static Bool* eqExpr(Val* lhs, Val* rhs);
  static Bool* neExpr(Val* lhs, Val* rhs);
  static Bool* gtExpr(Val* lhs, Val* rhs);
  static Bool* ltExpr(Val* lhs, Val* rhs);
  static Bool* leExpr(Val* lhs, Val* rhs);
  static Bool* geExpr(Val* lhs, Val* rhs);
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

  static Val* newScalar(DataType dtype);

  template <typename T>
  static Val* newConstant(T value, DataType dtype);

 private:
  static Val* newArithmeticExpr(BinaryOpType op_type, Val* lhs, Val* rhs);
  static Bool* newLogicExpr(BinaryOpType op_type, Val* lhs, Val* rhs);
};

template <typename T>
Val* IrBuilder::newConstant(T value, DataType dtype) {
  switch (std::get<PrimDataType>(dtype.type)) {
    case DataType::Bool:
      return IrBuilder::create<Bool>((bool)value);
    case DataType::Float:
    case DataType::Double:
      return IrBuilder::create<Double>((double)value, dtype);
    case DataType::Int:
    case DataType::Int32:
    case DataType::Index:
      return IrBuilder::create<Int>((int64_t)value, dtype);
    case DataType::ComplexFloat:
    case DataType::ComplexDouble:
      return IrBuilder::create<ComplexDouble>(
          (std::complex<double>)value, dtype);
    default:
      TORCH_CHECK(false, "Unexpected data type: ", dtype);
  }
}

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

  static Val* divExpr(Val* lhs, Val* rhs);

  static Val* ceilDivExpr(Int* lhs, Int* rhs);
  static Val* ceilDivExpr(Val* lhs, Val* rhs);

  static Val* modExpr(Val* lhs, Val* rhs);
  static Bool* andExpr(Val* lhs, Val* rhs);
  static Val* maxExpr(Val* lhs, Val* rhs);
  static Val* minExpr(Val* lhs, Val* rhs);

  static Val* whereExpr(Val* pred, Val* lhs, Val* rhs);
};

template <typename T>
NVFUSER_DEFINE_CLONE(Scalar<T>)

} // namespace nvfuser
