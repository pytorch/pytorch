#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/type.h>

#include <cstdint>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// TODO: Add more types (int32, int64)
// TODO: sameAs should have better logic to check against any type and return
// gracefully

/*
 * This file defines the base IR structure. Any IR node in this system will
 * inherit from one of the following classes: Statement, Expr, Val,
 * IRInputOutput IR is any information that the code generation stack may need
 * for analysis. By analysis we're refering to anything done in response to a
 * user facing call of this stack. This could be careful tracking of user calls,
 * and any transformation including optimizing transformations, user declared
 * transformations, and lowering the IR.
 */

namespace torch {
namespace jit {
namespace fuser {

using StmtNameType = unsigned int;
constexpr StmtNameType UNINITIALIZED_STMTNAMETYPE =
    std::numeric_limits<unsigned int>::max();

struct Fusion;
struct FusionGuard;
struct Expr;
struct Val;
struct UnaryOp;
struct BinaryOp;
struct IterDomain;

/*
 * Statement is the highest level node representation. Everything that is
 * considered "IR" will be derived from this class at some point. Both Values
 * and Expr's are a Statement. If there will ever be any more fundamental types,
 * they will also derive from Statement.
 *
 * We use Statements to pass around nodes of unknown compile type. Therefore it
 * is also important for the design to have a dispatch system for a Statment.
 * Basically beinng able to succienctly traverse down the inhereitance stack of
 * a Statment at runtime. This is currently implemented in dispatch.h
 */
struct TORCH_CUDA_API Statement {
  virtual ~Statement() = default;

  // Dispatch functions, definitions in dispatch.cpp
  template <typename T>
  static void dispatch(T handler, Statement*);

  template <typename T>
  static void constDispatch(T handler, const Statement* const);

  template <typename T>
  static Statement* mutatorDispatch(T mutator, Statement*);

  // Accessor functions to types. Vals always have a DataType, Exprs never do
  virtual c10::optional<ValType> getValType() const noexcept {
    return c10::nullopt;
  }
  virtual c10::optional<DataType> getDataType() const {
    return c10::nullopt;
  }
  virtual c10::optional<ExprType> getExprType() const noexcept {
    return c10::nullopt;
  }

  // Short cut to figure out if it is a value/expression
  bool isVal() const noexcept {
    return getValType() != c10::nullopt;
  }
  bool isExpr() const noexcept {
    return getExprType() != c10::nullopt;
  }

  // Make sure this is a Val and return it as a Val*
  Val* asVal();

  // Make sure this is an Expr and return it as an Expr*
  Expr* asExpr();

  // Replacement for static_cast<T*>(ptr): ptr->as<T>()
  template <class T>
  T* as() {
#ifdef NDEBUG
    auto downcast_ptr = static_cast<T*>(this);
#else
    auto downcast_ptr = dynamic_cast<T*>(this);
    TORCH_INTERNAL_ASSERT(downcast_ptr != nullptr);
#endif
    return downcast_ptr;
  }

  template <class T>
  const T* as() const {
#ifdef NDEBUG
    auto downcast_ptr = static_cast<const T*>(this);
#else
    auto downcast_ptr = dynamic_cast<const T*>(this);
    TORCH_INTERNAL_ASSERT(downcast_ptr != nullptr);
#endif
    return downcast_ptr;
  }

  // Return the fusion this statement belongs to
  Fusion* fusion() const noexcept {
    return fusion_;
  }

  // Return the int that represents its name
  StmtNameType name() const noexcept {
    return name_;
  }

  virtual bool sameType(const Statement* const other) {
    if (isVal() && other->isVal())
      return getValType().value() == other->getValType().value();
    if (isExpr() && other->isExpr())
      return getExprType().value() == other->getExprType().value();
    return false;
  }

  // Return if this statement is the same as another statement
  // TODO: should this run through dispatch on this and other?
  bool sameAs(const Statement* const other) const {
    return this == other;
  }

 protected:
  StmtNameType name_ = UNINITIALIZED_STMTNAMETYPE;
  Fusion* fusion_ = nullptr;
};

/*
 * A Val represents a "value." These are objects, like tensors, scalars, and
 * memory locations, that are inputs and outputs of computations (represented
 * by Exprs, below). Vals are constant and unique and should always be passed
 * around as a pointer. Val can generally be thought of as representing any type
 * of data. Some examples: a constant size like convolution filter width a
 * runtime constant like batch normalizations momentum a "symbolic" tensor like
 * one passed down from the JIT a memory buffer used in device code
 *
 * Adding a Val:
 * Right now adding a Val is quite involved. Val's can be defined in ir.h or in
 * their own header file. The following is what is currently needed to add a new
 * Val:
 * 1) Definition inheriting from Val
 *     - Members must be private or protected
 *     - Accessor functions for members
 *     - Must call Val constructor, Val constructor registers with fusion
 *     - Implementation of bool sameAs(...)
 * 2) dispatch.h/.cpp must be updated to include dispatch of the new Val
 * 3) Default mutator function should be added to mutator.cpp
 * 4) Printing functions should be added to ir_iostream.h/.cpp
 * 5) An enum value must be added to ValType in type.h
 * 6) A string entry must be added in val_type_string_map
 */
struct TORCH_CUDA_API Val : public Statement {
 public:
  virtual ~Val() = default;

  Val() = delete;

  // We may not want to register this value during Val's constructor. The reason
  // for this is that if we register the val, then ina derived constructor try
  // to throw, fusion's destructor will get called, but the pointer to this Val
  // will be invalid. When fusion tries to delete this value it will cause a seg
  // fault, instead of showing the thrown error.
  Val(ValType _vtype,
      DataType _dtype = DataType::Null,
      bool register_val = true);

  // TODO: Values are unique and not copyable
  Val(const Val& other) = delete;
  Val& operator=(const Val& other) = delete;

  Val(Val&& other) = delete;
  Val& operator=(Val&& other) = delete;

  c10::optional<ValType> getValType() const noexcept override {
    return vtype_;
  }

  // Throws if no DataType is found. Vals must have a DataType
  c10::optional<DataType> getDataType() const override;

  bool isScalar() const {
    return vtype_ == ValType::Scalar || vtype_ == ValType::NamedScalar;
  }

  bool isConstScalar() const;

  bool isAnInt() const {
    return isScalar() && dtype_ == DataType::Int;
  }

  bool isZeroInt() const;
  bool isOneInt() const;

  // Returns the Expr that this value is an output of, returns nullptr if none
  // was found
  Expr* getOrigin();

  virtual bool sameType(const Statement* other) {
    return Statement::sameType(other) &&
        getDataType() == static_cast<const Val*>(other)->getDataType();
  }

  // TODO: Make this more sophisticated. A value being the same as another value
  // should be evaluated based on the DAG that created it, and that DAGs leaf
  // nodes
  bool sameAs(const Val* const other) const {
    return this == other;
  }

  // Dispatch functions, definitions in dispatch.cpp
  template <typename T>
  static void dispatch(T handler, Val*);

  template <typename T>
  static void constDispatch(T handler, const Val* const);

  template <typename T>
  static Statement* mutatorDispatch(T mutator, Val*);

 protected:
  const ValType vtype_;
  const DataType dtype_;
};

// TODO: We should use this for the following:
//    Fusion
//    IfThenElse
//    ForLoop
struct TORCH_CUDA_API Scope {
 public:
  const std::vector<Expr*>& exprs() const noexcept {
    return exprs_;
  }

  void push_back(Expr* e) {
    exprs_.push_back(e);
  }

  void insert(std::vector<Expr*>::iterator it, Expr* expr) {
    exprs_.insert(it, expr);
  }

  void erase(std::vector<Expr*>::iterator it) {
    exprs_.erase(it);
  }

  bool empty() const {
    return exprs_.empty();
  }

  auto size() const {
    return exprs_.size();
  }

  // Insert expr before ref
  void insert_before(Expr* ref, Expr* expr);

  // Insert expr after ref
  void insert_after(Expr* ref, Expr* expr);

  bool contains(Expr* expr) const;

  void erase(Expr* ref);

  bool sameAs(const Scope& other) const;

  void clear();

 private:
  std::vector<Expr*> exprs_;
};

/*
 * IRInputOutput is a function on Vals. Has inputs and outputs that are all
 * Vals. It is used for anything that connects values and therefore would be
 * used during dependency analysis. Typically classes that inherit from
 * IRInputOutput will do so by inheriting from Exprs. Expr's are expected for
 * most dependency based operations like IterVisitor, or DependencyCheck.
 *
 * Examples:
 *   binary operations on tensors, scalar values, or a combination, a thread all
 *   reduce, for loops
 */
struct TORCH_CUDA_API IRInputOutput {
  virtual ~IRInputOutput() = default;

  // Returns if Val is an input or output of this IRInputOutput instance
  bool hasInput(const Val* const input) const;
  bool hasOutput(const Val* const output) const;

  // Input/output accessors
  void addInputAt(std::deque<Val*>::size_type pos, Val* input) {
    inputs_.insert(inputs_.begin() + pos, input);
  }

  void addOutputAt(std::deque<Val*>::size_type pos, Val* output) {
    outputs_.insert(outputs_.begin() + pos, output);
  }

  const std::deque<Val*>& inputs() const noexcept {
    return inputs_;
  }
  const std::deque<Val*>& outputs() const noexcept {
    return outputs_;
  }

  Val* input(std::deque<Val*>::size_type idx) const {
    return inputs_[idx];
  }
  Val* output(std::deque<Val*>::size_type idx) const {
    return outputs_[idx];
  }

  void addInput(Val* input) {
    inputs_.push_back(input);
  }
  void addOutput(Val* output) {
    outputs_.push_back(output);
  }

  void replaceInput(Val* replace, Val* with);
  void replaceOutput(Val* replace, Val* with);

  void removeInput(Val* val);
  void removeOutput(Val* val);

  std::deque<Val*>::size_type nInputs() const noexcept {
    return inputs_.size();
  }
  std::deque<Val*>::size_type nOutputs() const noexcept {
    return outputs_.size();
  }

 protected:
  std::deque<Val*> inputs_;
  std::deque<Val*> outputs_;
};

/*
 * A Expr represents a "computation." These are functions that takes inputs
 * and produce outputs, inputs and outputs all being Vals. There are
 * specializations of BinaryOp which takes 2 inputs and produces 1 output, and
 * UnaryOp which takes 1 input and produces 1 output. Exprs are unique and
 * immutable. Conceptually, Exprs could always be manipulated using unique
 * pointers, and we could add this later. However, for now Exprs can be replaced
 * in a fusion, but they cannot be modified in place.
 *
 * The IR is static single assignment (SSA). Values can only be defined as an
 * output of an Expr once. If they are re-defined the original definition is
 * deleted from the program, as opposed to an ordered redefinition of the value
 * in the program.
 *
 * Note: Registering an Expr with a Fusion is actually 2 parts, one part is done
 * in the Expr constructor, so that should be called on anything that inherits
 * Expr. The issue with having registration in Expr's constructor, is that the
 * constructor of an Expr will set ouputs and inputs. This information is
 * important for registration with Fuser, so it can track the dependency chain.
 *
 * Adding an Expr:
 * Right now adding an Expr is quite involved. Expr's can be defined in ir.h or
 * in their own header file. The following is what is currently needed for Expr
 * definitions:
 * 1) Definition inheriting from Expr.
 *     - Members must be private or protected
 *     - Accessor functions for members
 *     - Constructors need to register with the Fusion after inputs/outputs are
 *        defined
 *     - Implementation of bool sameAs(...)
 * 2) dispatch.h/.cpp must be updated to include dispatch of the new Val
 * 3) Default mutator function should be added to mutator.h/.cpp
 * 4) Printing functions should be added to ir_iostream.h/.cpp
 * 5) Lower case convenience functions should be added to arith.h/.cpp (If user
 *  facing)
 * 6) An enum value must be added to ExprType in type.h 7) A string
 *  entry must be added in expr_type_string_map
 */
struct TORCH_CUDA_API Expr : public Statement, IRInputOutput {
 public:
  virtual ~Expr() = default;
  Expr() = delete;
  Expr(ExprType _type);

  Expr(const Expr& other) = delete;
  Expr& operator=(const Expr& other) = delete;

  Expr(Expr&& other) = delete;
  Expr& operator=(Expr&& other) = delete;

  c10::optional<ExprType> getExprType() const noexcept override {
    return type_;
  }
  ExprType type() const noexcept {
    return type_;
  }

  bool sameAs(const Expr* const other) const {
    if (getExprType() != other->getExprType())
      return false;
    if (inputs().size() != other->inputs().size() ||
        outputs().size() != other->outputs().size())
      return false;
    for (size_t i = 0; i < inputs().size(); i++) {
      if (!input(i)->sameAs(other->input(i)))
        return false;
    }
    return true;
  }

  // Dispatch functions, definitions in dispatch.cpp
  template <typename T>
  static void dispatch(T handler, Expr*);

  template <typename T>
  static void constDispatch(T handler, const Expr* const);

  template <typename T>
  static Statement* mutatorDispatch(T mutator, Expr*);

 private:
  ExprType type_;
};

} // namespace fuser
} // namespace jit
} // namespace torch
