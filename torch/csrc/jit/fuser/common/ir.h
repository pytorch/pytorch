 #pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <c10/util/Optional.h>
#include <c10/util/Exception.h>
#include <c10/core/ScalarType.h>

#include <torch/csrc/jit/fuser/common/type.h>
#include <torch/csrc/jit/fuser/common/visitor.h>

#include <cstdint>
#include <unordered_map>
#include <stdexcept>
#include <vector>
#include <limits>
#include <deque>
#include <iostream>
#include <memory>

namespace c10 {
struct TensorType;
} // namespace c10

namespace torch {
namespace jit {

struct Value; 

namespace fuser {

/*
 * TODO: Add more types (int32, int64)
 * // TODO: add regions (e.g. loop exprs have bodies)
 */

/* 
 * This file defines the basic IR structure.
 * IR is any information that the code generation stack may need for analysis. By analysis
 * we're refering to anything done in response to a user facing call of this stack. Analysis
 * is the first step in any IR modifying calls. This could be careful tracking of user calls,
 * and any transformation including optimizing transformations, user declared transformations,
 * and lowering the IR.
 * For now the IR has 4 major classes:

 * Statement:
 * Statement should be inhereited at some point by every IR node. It may be better to call Statement
 * a node. We use Statements to pass around nodes of unknown compile type. Therefore it is also
 * important for the design to have easy to use dispatch of a Statment. Basically beinng able to
 * succienctly traverse down the inhereitance stack of a Statment at runtime.
 *
 * Val:
 * Val can generally be thought of as any data. This could be a float that is either a constant
 * (constants in this context could be compile time or run time known from the perspective of a 
 * pytorch end user). 
 * Some examples:
 *     a constant size like convolution filter width
 *     a runtime constant like batch normalizations momentum
 *     a "symbolic" tensor like one passed down from the JIT
 *     a memory buffer for device code
 * 
 * Adding a Val:
 * Right now adding a Val is quite involved. Val's can be defined in ir.h or in their own header file.
 * Val's classes must be uppercase. The following is what is currently needed for Val definitions:
 * 1) Definition inheriting from Val
 *     - Members must be at minimum private, often they should be const and private.
 *     - Accessor functions for members
 *     - Must cal Val constructor, Val constructor registers with fusion
 * 2) Statement::dispatch and Statement::dispatch_mutator in ir.cpp must be updated
 * 3) Virtual handle functions must be added to iter_visitor.h/.cpp
 * 4) Mutator fucntions must be added to mutator.h/.cpp
 * 5) Printing functions should be added to iriostream.h/.cpp
 * 6) An enum value should be added to ValType in type.h
 * 7) A string entry should be added in val_type_string_map
 *
 * IRInputOutput:
 * A function on Vals. Has inputs and outputs that are all Vals. Anything that connects
 * values and therefore would be used during dependency analysis.
 * Examples:
 *     binary operations on combinations of tensors, scalar values, or combinations of such
 *     a thread all reduce
 *     for loops
 * 
 * Expr
 * Expr is an IRInputOutput node. It takes multiple inputs and does *an* operation. There are specializations
 * of BinaryOp which takes 2 inputs and produces 1 output, and UnaryOp which takes 1 input and produces 1 output.
 *
 * The IR is static single assignment (SSA). Values can only be defined once. If they are re-defined
 * the original definition is deleted from the program, as opposed to an ordered redefinition of
 * the value in the program.
 * 
 * Adding an Expr:
 * Right now adding an Expr is quite involved. Expr's can be defined in ir.h or in their own header file.
 * Expr's classes must be uppercase. The following is what is currently needed for Expr definitions:
 * 1) Definition inheriting from Expr.
 *    - Members must at minimum be private/protected, and often const if they must never be changed
 *    - Accessor functions for members
 *    - Constructors need to register with the Fusion after inputs/outputs are defined
 *    - Implementation of bool same_as(...)
 * 2) Statement::dispatch and Statement::dispatch_mutator in ir.cpp must be updated to include
 *         dispatch on the added Expr.
 * 3) Virtual handle functions must be added to iter_visitor.h/.cpp
 * 4) Mutator fucntions must be added to mutator.h/.cpp
 * 5) Lower case convenience functions can be added to arith.h/.cpp
 * 6) Printing functions should be added to iriostream.h/.cpp
 * 7) An enum value should be added to ExprType in type.h
 * 8) A string entry should be added in expr_type_string_map
 * 
 */


using StmtNameType = unsigned int;
constexpr StmtNameType UNINITIALIZED_STMTNAMETYPE = std::numeric_limits<unsigned int>::max();

struct Fusion;
struct FusionGuard;
struct Expr;
struct UnaryOp;
struct BinaryOp;


/*
 * Statement is the highest level node representation. Everything that is considered "IR" will
 * be derived from this class eventually. Both Values and Expr's are a Statement. If there will
 * ever be any more fundamental types, they will also derive from Statement.
 */
struct TORCH_API Statement {
  virtual ~Statement() = 0;

  //dispatch is used to take a handler, and call 
  template <typename T>
  void dispatch(T handler) const;

  template <typename T>
  const Statement* dispatch_mutator(T mutator) const;

  virtual c10::optional<ValType> getValType() const noexcept { return c10::nullopt; }
  virtual c10::optional<DataType> getDataType() const noexcept { return c10::nullopt; }
  virtual c10::optional<ExprType> getExprType() const noexcept { return c10::nullopt; }

  bool isVal() const noexcept { return getValType() != c10::nullopt; }
  bool isExpr() const noexcept{ return getExprType() != c10::nullopt; }

  Fusion* fusion() const noexcept { return fusion_; }
  StmtNameType name() const noexcept { return name_; }

  virtual bool same_as(const Statement* other) const {
    return this == other;
  }

protected:
  StmtNameType name_ = UNINITIALIZED_STMTNAMETYPE;
  Fusion* fusion_ = nullptr;
};

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
  Val(const ValType _vtype, const DataType _dtype = DataType::Null);

  //TODO: Values are unique and not copyable
  Val(const Val& other) = delete;
  Val& operator=(const Val& other) = delete;

  Val(Val&& other) = delete;
  Val& operator=(Val&& other) = delete;

  c10::optional<ValType> getValType() const noexcept override { return vtype_; }
  c10::optional<DataType> getDataType() const noexcept override {
    if(dtype_ == DataType::Null)
      return c10::nullopt;
    return dtype_;
  }

  bool isScalar(){ return vtype_ == ValType::Scalar; }

  const Expr* getOrigin();

  //TODO: We want to make this more sophisticated. A value being the same as another
  //value should be evaluated based on the DAG that created it, and that DAGs leaf nodes
  bool same_as(const Val* other) const { return this == other;}

  template <typename T>
  void dispatch(T handler) const;

  template <typename T>
  const Statement* dispatch_mutator(T mutator) const;

protected:
  const ValType vtype_;
  const DataType dtype_;
};

/*
 * IRInputOutput is any type of node that has values as inputes and outputs. Anything that we may want
 * to do dependency analysis on should derive IRInputOutput.
 * TODO: Uncertain if we want to specialize this to enforce nodes that strictly only support 1 output.
 */
struct TORCH_API IRInputOutput {
  virtual ~IRInputOutput() = 0;

  const std::deque<const Val*>& inputs() const noexcept { return inputs_; }
  const std::deque<const Val*>& outputs() const noexcept{ return outputs_; }

  const Val* input(const std::deque<const Val*>::size_type idx) const {
    return inputs_[idx];
  }
  const Val* output(const std::deque<const Val*>::size_type idx) const {
    return outputs_[idx];
  }

  void addInput(const Val* input) {
    inputs_.push_back(input);
  }
  void addOutput(const Val* output) {
    outputs_.push_back(output);
  }

  bool hasInput(const Val* input) const {
    for(auto val : inputs_)
      if(val == input)
        return true;
    return false;
  }

  bool hasOutput(const Val* output) const {
    for(auto val : outputs_)
      if(val == output)
        return true;
    return false;
  }

  void addInputAt(const std::deque<const Val*>::size_type pos, const Val* input) {
    inputs_.insert(inputs_.begin() + pos, input);
  }

  void addOutputAt(const std::deque<const Val*>::size_type pos, Val* output) {
    outputs_.insert(outputs_.begin() + pos, output);
  }

  void removeOutput(const Val* val){
    auto it = outputs_.begin();
    for(; it != outputs_.end(); ++it){
      if((*it) == val)
        break;
    }
    assert(it!=outputs_.end());
    outputs_.erase(it);    
  }

  std::deque<const Val*>::size_type nInputs() const noexcept { return inputs_.size(); }
  std::deque<const Val*>::size_type nOutputs() const noexcept { return outputs_.size(); }

protected:
  std::deque<const Val*> inputs_;
  std::deque<const Val*> outputs_;

};

// TODO: do we want this to be a separate class (FloatImm?)
struct TORCH_API Float : public Val {
  ~Float() = default;

  Float()
  : Val(ValType::Scalar, DataType::Float)
  , maybe_value_{c10::nullopt} { }

  Float(
    const float _value)
  : Val(ValType::Scalar, DataType::Float)
  , maybe_value_{_value} { }

  Float(const Float& other) = delete;
  Float& operator=(const Float& other) = delete;

  Float(Float&& other) = delete;
  Float& operator=(Float&& other) = delete;

  bool isSymbolic() const { return !(maybe_value_.has_value()); }
  bool isConst() const { return maybe_value_.has_value(); }
  c10::optional<float> value() const noexcept { return maybe_value_; }

  virtual bool same_as(const Float* other) const {
    if(isConst() && other->isConst())
      return *value() == *(other->value());
    return this == other;
  }

private:
  c10::optional<float> maybe_value_;
};

struct TORCH_API Int : public Val {
  ~Int() = default;

  Int()
  : Val(ValType::Scalar, DataType::Int)
  , maybe_value_{c10::nullopt} { }

  Int(
    const int _value)
  : Val(ValType::Scalar, DataType::Int)
  , maybe_value_{_value} { }

  Int(const Int& other) = delete;
  Int& operator=(const Int& other) = delete;

  Int(Int&& other) = delete;
  Int& operator=(Int&& other) = delete;

  bool isSymbolic() const { return !(maybe_value_.has_value()); }
  bool isConst() const { return maybe_value_.has_value(); }
  c10::optional<int> value() const noexcept { return maybe_value_; }

  virtual bool same_as(const Int* other) const {
    if(isConst() && other->isConst())
      return *value() == *(other->value());
    return this == other;
  }

private:
  c10::optional<int> maybe_value_;
};

/*
* A Expr represents a "computation." These are functions that may take inputs
* and produce outputs.
*
* Exprs are unique and immutable. Conceptually, Exprs could always be manipulated
* using unique pointers, and we could add this later. However, for now Exprs can be
* replaced in a fusion, but they cannot be modified in place.
*
* Note: Registering an Expr with a Fusion is actually 2 parts, one part is done in
* the Expr constructor, so that should be called on anything that inherits Expr.
* The issue with having registration in Expr's constructor, is that the constructor
* of an Expr will set ouputs and inputs. This information is important for registration
* with Fuser, so it can track the dependency chain.
*/
struct TORCH_API Expr : public Statement, IRInputOutput {
public:
  virtual ~Expr() = 0;
  Expr() = delete;
  Expr(const ExprType _type);

  Expr(const Expr& other) = delete;
  Expr& operator=(const Expr& other) = delete;

  Expr(Expr&& other) = delete;
  Expr& operator=(Expr&& other) = delete;

  c10::optional<ExprType> getExprType() const noexcept override { return type_; }
  ExprType type() const noexcept { return type_; }

  virtual bool same_as(const Expr* other) const {
    if(getExprType() != other->getExprType())
      return false;
    if(inputs().size() != other->inputs().size()
    || outputs().size() != other->outputs().size())
      return false;
    for(int i=0; i<inputs().size(); i++){
      if(!input(i)->same_as(other->input(i)))
        return false;
    }
    return true;
  }

  template <typename T>
  void dispatch(T handler) const;

  template <typename T>
  const Statement* dispatch_mutator(T mutator) const;

private:
  ExprType type_;
};

// TODO: comment
struct TORCH_API UnaryOp : public Expr {
  ~UnaryOp() = default;
  UnaryOp(
	const UnaryOpType _type
  , const Val* _out
  , const Val* _in);

  UnaryOp(const UnaryOp& other) = delete;
  UnaryOp& operator=(const UnaryOp& other) = delete;

  UnaryOp(UnaryOp&& other) = delete;
  UnaryOp& operator=(UnaryOp&& other) = delete;

  const Val* out() const noexcept { return out_; }
  const Val* in()  const noexcept { return in_; }
  
  UnaryOpType type() const noexcept { return unary_op_type_; }

  bool same_as(const UnaryOp* other) const {
    if(this->type() != other->type())
      return false;
    return static_cast<const Expr*>(this)->same_as(other);
  }

private:
  const UnaryOpType unary_op_type_;
  const Val* out_;
  const Val* in_;
};

// TODO: comment
struct TORCH_API BinaryOp : public Expr {
  ~BinaryOp() = default;
  BinaryOp(
	const BinaryOpType _type
  , const Val* _out
  , const Val* _lhs
  , const Val* _rhs);

  BinaryOp(const BinaryOp& other) = delete;
  BinaryOp& operator=(const BinaryOp& other) = delete;

  BinaryOp(BinaryOp&& other) = delete;
  BinaryOp& operator=(BinaryOp&& other) = delete;

  const Val* out() const noexcept { return out_; }
  const Val* lhs() const noexcept { return lhs_; }
  const Val* rhs() const noexcept { return rhs_; }
  
  BinaryOpType type() const noexcept { return binary_op_type_; }

  bool same_as(const BinaryOp* other) const {
    if(type() != other->type())
      return false;
    return static_cast<const Expr*>(this)->same_as(other);
  }

private:
  const BinaryOpType binary_op_type_;
  const Val* out_;
  const Val* lhs_;
  const Val* rhs_;
};

}}} //torch::jit::fuser
