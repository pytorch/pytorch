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
// TODO: Add casting function
// TODO: Add more binary ops, maybe reduce to single class (div, mul, mod, sub, LT)
// TODO: Add unary ops, maybe reduce to single class (casting)
// TODO: Add more types (int32, int64)

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
 * IRInputOutput:
 * A function on Vals. Has inputs and outputs that are all Vals. Anything that connects
 * values and therefore would be used during dependency analysis.
 * Examples:
 *     binary operations on combinations of tensors, scalar values, or combinations of such
 *     a thread all reduce
 *     for loops
 * 
 * Expr:
 * Expr should be simple IRInputOutput nodes. We may want to even specialize them to be limited
 * to maximum 2 inputs and a single output. For now we're using it for things like binary and
 * unary operations.
 *
 * For now this IR is static single assignment. Values can only be defined once. If they are re-defined
 * the original definition should be deleted from the program, as opposed to an ordered redefinition of
 * the value in the program. Since for now the IR will be provided to us as a graph and we will translate
 * it, this should be easier of a framework to work in. We in theory could support a non SSA interface
 * and translate to SSA, but that's outside the scope of this work for now.
 */


using StmtNameType = unsigned int;
constexpr StmtNameType UNINITIALIZED_STMTNAMETYPE = std::numeric_limits<unsigned int>::max();

struct Fusion;
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

  const Expr* getOrigin(){return origin_;}

protected:
  const ValType type_;
  Expr *volatile origin_ = nullptr;
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

// TODO: improve input/output model to track dataflow
// TODO: add regions (e.g. loop exprs have bodies)
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

private:
  const BinaryOpType binary_op_type_;
  const Val* out_;
  const Val* lhs_;
  const Val* rhs_;
};

}}} //torch::jit::fuser
