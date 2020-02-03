 #pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <c10/util/Optional.h>
#include <c10/util/Exception.h>

#include <torch/csrc/jit/fuser/common/type.h>
#include <torch/csrc/jit/fuser/common/visitor.h>

#include <cstdint>
#include <unordered_map>
#include <stdexcept>
#include <vector>
#include <limits>
#include <deque>
#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

// TODO: add comment explaining structure
// TODO: automatic casting (check type for binary ops, cast inputs to common type)
// TODO: Add casting function
// TODO: Add more binary ops (div, mul, mod, sub, LT)
// TODO: Add more types (int32, int64)

using StmtNameType = unsigned int;
constexpr StmtNameType UNINITIALIZED_STMTNAMETYPE = std::numeric_limits<unsigned int>::max();

struct Fusion;
struct Expr;
struct Add;

struct TORCH_API Statement {
  virtual ~Statement() = 0;

  //dispatch is used to take a handler, and call 
  template <typename T>
  int dispatch(T handler) const;

  template <typename T>
  const Statement* dispatch_mutator(T mutator) const;

  virtual c10::optional<ValType> getValType() const noexcept { return c10::nullopt; }
  virtual c10::optional<ExprType> getExprType() const noexcept { return c10::nullopt; }

  bool isVal() const noexcept { return getValType() != c10::nullopt; }
  bool isExpr() const noexcept{ return getExprType() != c10::nullopt; }

  Fusion* fusion() const noexcept { return fusion_; }
  StmtNameType name() const noexcept { return name_; }

protected:
  StmtNameType name_ = UNINITIALIZED_STMTNAMETYPE;
  Fusion* fusion_ = nullptr;
};

TORCH_API std::ostream& operator<<(std::ostream& out, const Statement* const stmt);

/*
* A Val represents a "value." These are objects, like tensors, scalars, and
* memory locations, that are inputs and outputs of computations (represented
* by Exprs, below). They also represent the flow of data through a program.
*
* Vals are constant and unique. Vals are always passed around as a pointer.
*/
struct TORCH_API Val : public Statement {

public:
  virtual ~Val() = 0;

  Val() = delete;
  Val(const ValType _type);

  //TODO: Values are unique and not copyable
  Val(const Val& other) = delete;
  Val& operator=(const Val& other) = delete;

  Val(Val&& other) = delete;
  Val& operator=(Val&& other) = delete;

  c10::optional<ValType> getValType() const noexcept override { return type_; }
  ValType type() const noexcept { return type_; }

  bool isScalar(){
    static_assert(((int)ValType::Float) == 1); //Depend on ordering to know if Val is a scalar.
    return type() >= ValType::Float;
  }
  
protected:
  const ValType type_;
};

struct TORCH_API Tensor : public Val {
  ~Tensor() = default;

  Tensor()
  : Val(ValType::Tensor){}

};

// TODO: do we want this to be a separate class (FloatImm?)
struct TORCH_API Float : public Val {
  ~Float() = default;

  Float()
  : Val(ValType::Float)
  , maybe_value_{c10::nullopt} { }

  Float(
    const float _value)
  : Val(ValType::Float)
  , maybe_value_{_value} { }

  Float(const Float& other) = delete;
  Float& operator=(const Float& other) = delete;

  Float(Float&& other) = delete;
  Float& operator=(Float&& other) = delete;

  bool isSymbolic() const { return !(maybe_value_.has_value()); }
  bool isConst() const { return maybe_value_.has_value(); }
  c10::optional<float> value() const noexcept { return maybe_value_; }

private:
  c10::optional<float> maybe_value_;
};

struct TORCH_API Int : public Val {
  ~Int() = default;

  Int()
  : Val(ValType::Int)
  , maybe_value_{c10::nullopt} { }

  Int(
    const float _value)
  : Val(ValType::Int)
  , maybe_value_{_value} { }

  Int(const Int& other) = delete;
  Int& operator=(const Int& other) = delete;

  Int(Int&& other) = delete;
  Int& operator=(Int&& other) = delete;

  bool isSymbolic() const { return !(maybe_value_.has_value()); }
  bool isConst() const { return maybe_value_.has_value(); }
  c10::optional<int> value() const noexcept { return maybe_value_; }

private:
  c10::optional<int> maybe_value_;
};


// TODO: comment
struct TORCH_API IRInputOutput {
  virtual ~IRInputOutput() = 0;

  const std::deque<const Val*>& inputs() const noexcept { return inputs_; }
  const std::deque<const Val*>& outputs() const noexcept { return outputs_; }

  const Val* getInput(const std::deque<const Val*>::size_type idx) const {
    return inputs_[idx];
  }
  const Val* getOutput(const std::deque<const Val*>::size_type idx) const {
    return outputs_[idx];
  }

  void addInput(const Val* input) {
    
    inputs_.push_back(input);
  }
  void addOutput(const Val* output) {
    outputs_.push_back(output);
  }

  void addInputAt(const std::deque<const Val*>::size_type pos, const Val* input) {
    inputs_.insert(inputs_.begin() + pos, input);
  }
  void addOutputAt(const std::deque<const Val*>::size_type pos, const Val* output) {
    outputs_.insert(outputs_.begin() + pos, output);
  }

  std::deque<const Val*>::size_type nInputs() const noexcept { return inputs_.size(); }
  std::deque<const Val*>::size_type nOutputs() const noexcept { return outputs_.size(); }

protected:
  std::deque<const Val*> inputs_;
  std::deque<const Val*> outputs_;

};


// TODO: improve input/output model to track dataflow
// TODO: add regions (e.g. loop exprs have bodies)
/*
* A Expr represents a "computation." These are functions that may take inputs
* and produce outputs.
*
* Exprs are unique and mutable. Conceptually, Exprs could always be manipulated
* using unique pointers.
*/
struct TORCH_API Expr : public Statement, IRInputOutput {
public:
  virtual ~Expr() = 0;
  Expr() = delete;
  Expr(const ExprType _type);

  c10::optional<ExprType> getExprType() const noexcept override { return type_; }
  ExprType type() const noexcept { return type_; }

  // std::vector<Region*>& regions() noexcept { return regions_; }
  // const std::vector<Region*>& regions() const noexcept { return regions_; }

  // void addRegion(Region* region);

private:
  ExprType type_;
  // std::vector<Region*> regions_;

  // void register_callback(Statement* stmt) override;
};

// TODO: comment
struct TORCH_API Add : public Expr {
  ~Add() = default;
  Add(
    const Val* _out
  , const Val* _lhs
  , const Val* _rhs)
  : Expr(ExprType::Add)
  , out_{_out}
  , lhs_{_lhs}
  , rhs_{_rhs} {
    // addOutput(_out);
    // addInput(_lhs);
    // addInput(_rhs);
  }

  Add(const Add& other) = delete;
  Add& operator=(const Add& other) = delete;

  Add(Add&& other) = delete;
  Add& operator=(Add&& other) = delete;

  const Val* out() const noexcept { return out_; }
  const Val* lhs() const noexcept { return lhs_; }
  const Val* rhs() const noexcept { return rhs_; }

private:
  const Val* out_;
  const Val* lhs_;
  const Val* rhs_;
};

}}} //torch::jit::fuser
