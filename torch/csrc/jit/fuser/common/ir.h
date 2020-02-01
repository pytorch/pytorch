 #pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <c10/util/Optional.h>
#include <c10/util/Exception.h>

#include <torch/csrc/jit/fuser/common/type.h>

#include <cstdint>
#include <unordered_map>
#include <stdexcept>
#include <vector>
#include <limits>
#include <deque>

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
struct Region;
struct Expr;
struct Add;

struct TORCH_API Statement {
  virtual ~Statement() = 0;

  template <typename T>
  int dispatch(T handler) const;

  virtual c10::optional<ValType> getValType() const noexcept { return c10::nullopt; }
  virtual c10::optional<ExprType> getExprType() const noexcept { return c10::nullopt; }

  bool isVal() const noexcept { return getValType() != c10::nullopt; }
  bool isExpr() const noexcept{ return getExprType() != c10::nullopt; }

  Fusion* fusion() const noexcept { return fusion_; }
  Region* region() const noexcept { return region_; }
  StmtNameType name() const noexcept { return name_; }

  void setFusion(Fusion* fusion) {
    if(fusion_!=nullptr)
      std::runtime_error("Fusion group cannot be changed once set, must make a new statment to set Fusion.");
    // TODO: we need to know if fusion has this Satement, if not should we add it or error out?
    fusion_ = fusion;
  }

  void setRegion(Region* region) {
    if(isVal())
      std::runtime_error("Values cannot have regions.");
    region_ = region;
  }
  void setName(const StmtNameType name) {
    name_ = name;
  }

protected:
  Fusion* fusion_ = nullptr;
  Region* region_ = nullptr;
  StmtNameType name_ = UNINITIALIZED_STMTNAMETYPE;
};

TORCH_API std::ostream& operator<<(std::ostream& out, const Statement* const stmt);

/*
* A Val represents a "value." These are objects, like tensors, scalars, and
* memory locations, that are inputs and outputs of computations (represented
* by Exprs, below). They also represent the flow of data through a program.
*
* Vals are constant and not unique. Conceptually, Vals could always be
* manipulated using shared pointers to const.
*/
struct TORCH_API Val : public Statement {

public:
  virtual ~Val() = 0;

  Val() = delete;
  Val(const ValType _type);

  //TODO: we need a way to prevent users from copying values as we need
  //any values that are the "same" to reference the same object or we need
  //to have proper implementations and correct == overload
  Val(const Val& other) = default;
  Val& operator=(const Val& other) = default;

  Val(Val&& other) = default;
  Val& operator=(Val&& other) = default;

  c10::optional<ValType> getValType() const noexcept override { return type_; }
  ValType type() const noexcept { return type_; }

  bool is_scalar(){
    static_assert(((int)ValType::Float) == 1); //Depend on ordering to know if Val is a scalar.
    return type() >= ValType::Float;
  }
  
protected:
  const ValType type_;
};

// TODO: support symbolic floats vs literal (const) floats (make value an optional)
// likely want this to be a separate class (FloatImm?)
struct TORCH_API Tensor : public Val {
  ~Tensor() = default;

  Tensor()
  : Val(ValType::Tensor){}

  //Not copyable
  Tensor(const Tensor& other) = delete;
  Tensor& operator=(const Tensor& other) = delete;

  Tensor(Tensor&& other) = delete;
  Tensor& operator=(Tensor&& other) = delete;

};

// TODO: support symbolic floats vs literal (const) floats (make value an optional)
// likely want this to be a separate class (FloatImm?)
struct TORCH_API Float : public Val {
  ~Float() = default;

  Float()
  : Val(ValType::Float)
  , maybe_value_{c10::nullopt} { }

  Float(
    const float _value)
  : Val(ValType::Float)
  , maybe_value_{_value} { }

  Float(const Float& other) = default;
  Float& operator=(const Float& other) = default;

  Float(Float&& other) = default;
  Float& operator=(Float&& other) = default;

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

  Int(const Int& other) = default;
  Int& operator=(const Int& other) = default;

  Int(Int&& other) = default;
  Int& operator=(Int&& other) = default;

  bool isSymbolic() const { return !(maybe_value_.has_value()); }
  bool isConst() const { return maybe_value_.has_value(); }
  c10::optional<int> value() const noexcept { return maybe_value_; }

private:
  c10::optional<int> maybe_value_;
};


// TODO: comment
struct TORCH_API IRInputOutput {
  virtual ~IRInputOutput() = 0;

  std::deque<Val*>& inputs() noexcept { return inputs_; }
  std::deque<Val*>& outputs() noexcept { return outputs_; }

  const std::deque<Val*>& inputs() const noexcept { return inputs_; }
  const std::deque<Val*>& outputs() const noexcept { return outputs_; }

  Val* getInput(const std::deque<Val*>::size_type idx) {
    return inputs_[idx];
  }
  Val* getOutput(const std::deque<Val*>::size_type idx) {
    return outputs_[idx];
  }

  const Val* getInput(const std::deque<Val*>::size_type idx) const {
    return inputs_[idx];
  }
  const Val* getOutput(const std::deque<Val*>::size_type idx) const {
    return outputs_[idx];
  }

  void addInput(Val* input) {
    register_callback(input);
    inputs_.push_back(input);
  }
  void addOutput(Val* output) {
    register_callback(output);
    outputs_.push_back(output);
  }

  void addInputAt(const std::deque<Val*>::size_type pos, Val* input) {
    register_callback(input);
    inputs_.insert(inputs_.begin() + pos, input);
  }
  void addOutputAt(const std::deque<Val*>::size_type pos, Val* output) {
    register_callback(output);
    outputs_.insert(outputs_.begin() + pos, output);
  }

  std::deque<Val*>::size_type nInputs() const noexcept { return inputs_.size(); }
  std::deque<Val*>::size_type nOutputs() const noexcept { return outputs_.size(); }

protected:
  std::deque<Val*> inputs_;
  std::deque<Val*> outputs_;

  virtual void register_callback(Statement* stmt) { }
};

// TOOD: comment
struct TORCH_API Region : public IRInputOutput {
  ~Region() = default;

  std::deque<Expr*>& exprs() noexcept { return exprs_; }
  const std::deque<Expr*>& exprs() const noexcept { return exprs_; }

  Fusion* fusion() const noexcept { return fusion_; }
  Expr* parent() const noexcept {return parent_; }

  void setFusion(Fusion* fusion) {
    TORCH_CHECK(fusion_ == nullptr);
    fusion_ = fusion;
  }
  void setParent(Expr* parent) {
    TORCH_CHECK(parent_ == nullptr);
    parent_ = parent;
  }

  bool inRegion(const Statement* stmt) {
    return (stmt->region() == this);
  }

  // TODO: Lets put some safety into these 2 functions. Run through a quick dependency check
  // on the expr's inputs.
  void insertAtStart(Expr* expr) {
    registerExpr(expr);
    exprs_.push_front(expr);
  }
  void insertAtEnd(Expr* expr) {
    registerExpr(expr);
    exprs_.push_back(expr);
  };

  void insertLeftBeforeRight(Expr* left, Expr* right);
  void insertLeftAfterRight(Expr* left, Expr* right);

  std::deque<Expr*>::size_type indexOf(const Expr* expr) {
    for (auto i = decltype(exprs_.size()){0}; i < exprs_.size(); ++i) {
      if (expr == exprs_[i]) {
        return i;
      }
    }

    return -1; // TODO: return a named marker value or throw an error?
  }

  StmtNameType registerStatement(Statement* stmt);

private:
  Fusion* fusion_ = nullptr;
  Expr* parent_ = nullptr;
  std::deque<Expr*> exprs_;

  void register_callback(Statement* stmt) override;
  StmtNameType registerVal(Val* val);
  StmtNameType registerExpr(Expr* expr);
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

  std::vector<Region*>& regions() noexcept { return regions_; }
  const std::vector<Region*>& regions() const noexcept { return regions_; }

  void addRegion(Region* region);

private:
  ExprType type_;
  std::vector<Region*> regions_;

  void register_callback(Statement* stmt) override;
};

// TODO: comment
struct TORCH_API Add : public Expr {
  ~Add() = default;
  Add(
    Val* _out
  , Val* _lhs
  , Val* _rhs)
  : Expr(ExprType::Add)
  , out_{_out}
  , lhs_{_lhs}
  , rhs_{_rhs} {
    addOutput(_out);
    addInput(_lhs);
    addInput(_rhs);
  }

  Add(const Add& other) = default;
  Add& operator=(const Add& other) = default;

  Add(Add&& other) = default;
  Add& operator=(Add&& other) = default;

  Val* out() noexcept { return out_; }
  Val* lhs() noexcept { return lhs_; }
  Val* rhs() noexcept { return rhs_; }

  const Val* out() const noexcept { return out_; }
  const Val* lhs() const noexcept { return lhs_; }
  const Val* rhs() const noexcept { return rhs_; }

private:
  Val* out_;
  Val* lhs_;
  Val* rhs_;
};

}}} //torch::jit::fuser
