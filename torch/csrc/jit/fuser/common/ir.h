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

// IR Requirements & Goals:
// clear, easy, extensible way to add new vals, new exprs
// nice pretty printing
// validating by lowering TorchScript IR and pretty printing
// supports visitor pattern
// allows for easy ~manipulation/~ querying
// Immutable, when a node changes it must be recreated

using StmtNameType = unsigned int;
constexpr StmtNameType UNINITIALIZED_STMTNAMETYPE = std::numeric_limits<unsigned int>::max();

struct Fusion;
struct Region;

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
    fusion_ = fusion;
  }
  void setRegion(Region* region) {
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
  Val(const ValType _type) : type_{_type} { }

  Val(const Val& other) = default;
  Val& operator=(const Val& other) = default;

  Val(Val&& other) = default;
  Val& operator=(Val&& other) = default;

  c10::optional<ValType> getValType() const noexcept override { return type_; }
  ValType type() const noexcept { return type_; }

protected:
  const ValType type_;
};

// TODO: support symbolic floats vs literal (const) floats (make value an optional)
struct TORCH_API Float : public Val {
  ~Float() = default;
  Float() = delete;

  Float(
    const float _value)
  : Val(ValType::Float)
  , value_{_value} { }


  Float(const Float& other) = default;
  Float& operator=(const Float& other) = default;

  Float(Float&& other) = default;
  Float& operator=(Float&& other) = default;

  float value() const noexcept { return value_; }

private:
  const float value_;
};

struct TORCH_API IRInputOutput {
  virtual ~IRInputOutput() = 0;

  std::deque<Statement*>& inputs() noexcept { return inputs_; }
  std::deque<Statement*>& outputs() noexcept { return outputs_; }

  const std::deque<Statement*>& inputs() const noexcept { return inputs_; }
  const std::deque<Statement*>& outputs() const noexcept { return outputs_; }

  Statement* getInput(const std::deque<Statement*>::size_type idx) {
    return inputs_[idx];
  }
  Statement* getOutput(const std::deque<Statement*>::size_type idx) {
    return outputs_[idx];
  }

  const Statement* getInput(const std::deque<Statement*>::size_type idx) const {
    return inputs_[idx];
  }
  const Statement* getOutput(const std::deque<Statement*>::size_type idx) const {
    return outputs_[idx];
  }

  void addInput(Statement* input) {
    register_callback(input);
    inputs_.push_back(input);
  }
  void addOutput(Statement* output) {
    register_callback(output);
    outputs_.push_back(output);
  }

  void addInputAt(const std::deque<Statement*>::size_type pos, Statement* input) {
    register_callback(input);
    inputs_.insert(inputs_.begin() + pos, input);
  }
  void addOutputAt(const std::deque<Statement*>::size_type pos, Statement* output) {
    register_callback(output);
    outputs_.insert(outputs_.begin() + pos, output);
  }

  std::deque<Statement*>::size_type nInputs() const noexcept { return inputs_.size(); }
  std::deque<Statement*>::size_type nOutputs() const noexcept { return outputs_.size(); }

protected:
  std::deque<Statement*> inputs_;
  std::deque<Statement*> outputs_;

  virtual void register_callback(Statement* stmt) { }
};

struct Expr;

struct TORCH_API Region : public IRInputOutput {
  ~Region() = default;

  std::deque<Expr*>& exprs() noexcept { return exprs_; }
  const std::deque<Expr*>& exprs() const noexcept { return exprs_; }

  Fusion* fusion() const noexcept { return fusion_; }
  Expr* parent() const noexcept {return parent_; }

  void setFusion(Fusion* fusion) noexcept {
    TORCH_CHECK(fusion_ == nullptr);
    fusion_ = fusion;
  }
  void setParent(Expr* parent) noexcept {
    TORCH_CHECK(parent_ == nullptr);
    parent_ = parent;
  }

  bool inRegion(const Statement* stmt) {
    return (stmt->region() == this);
  }

  void insertAtStart(Expr* expr) {
    exprs_.push_front(expr);
  }
  void insertAtEnd(Expr* expr) { exprs_.push_back(expr); };

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
  Expr(
    const ExprType _type)
  : type_{_type} { }

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

struct TORCH_API Add : public Expr {
  ~Add() = default;
  Add(
    Statement* _lhs
  , Statement* _rhs)
  : Expr(ExprType::Add)
  , lhs_{_lhs}
  , rhs_{_rhs} {
    addInput(_lhs);
    addInput(_rhs);
  }

  Add(const Add& other) = default;
  Add& operator=(const Add& other) = default;

  Add(Add&& other) = default;
  Add& operator=(Add&& other) = default;

  Statement* lhs() noexcept { return lhs_; }
  Statement* rhs() noexcept { return rhs_; }

  const Statement* lhs() const noexcept { return lhs_; }
  const Statement* rhs() const noexcept { return rhs_; }

private:
  Statement* lhs_;
  Statement* rhs_;
};

} // namespace fuser
} // namespace jit
} // namespace torch


// torch::jit::fuser

// #include <string>
// #include <vector>

// #include <torch/csrc/jit/fuser/common/expr.h>
// #include <torch/csrc/jit/fuser/common/types.h>


// enum IRNodeType {
//   kAdd,
//   kSub,
//   kMul,
//   kDiv,
// };

// class Cast : public ExprNode<Cast> {
//  public:
//   const Expr& src_value() const { return src_value_; }
//   const DType& dst_type() const {return dst_type_;}

//   Cast(const DType& dst_type, const Expr& src_value) : ExprNode<Cast>(dst_type), dst_type_(dst_type), src_value_(src_value){}

//   static Expr make(const DType& dst_type, const Expr& src_value) {
//     return Expr(new Cast(dst_type, src_value));
//   }

//  private:
//   const DType dst_type_;
//   const Expr src_value_;
// };

// // Represent the expression node for binary operators.
// // A CRTP pattern to share common code among the operators.
// template <typename Op>
// class BinaryOpNode : public ExprNode<Op> {
//  public:
//   const Expr& lhs() const { return this->lhs_; }
//   const Expr& rhs() const { return this->rhs_; }
//   IRNodeType expr_type() const { return expr_type_; }

//   static Expr make(const Expr& lhs, const Expr& rhs) { return Expr(new Op(lhs, rhs)); }

//  protected:
//   BinaryOpNode(const Expr& lhs_v, const Expr& rhs_v, IRNodeType expr_type)
//       : ExprNode<Op>(promote(lhs_v.dtype(), rhs_v.dtype())),
//         lhs_(CastIfNeeded(lhs_v, ExprNode<Op>::dtype())),
//         rhs_(CastIfNeeded(rhs_v, ExprNode<Op>::dtype())),
//         expr_type_(expr_type) {}

//  private:
//   static Expr CastIfNeeded(const Expr& expr, DType dst_dtype) {
//     if (expr.dtype() == dst_dtype) {
//       return expr;
//     }
//     return Cast::make(dst_dtype, expr);
//   }

//   Expr lhs_;
//   Expr rhs_;
//   IRNodeType expr_type_;
// };

// class Add : public BinaryOpNode<Add> {
//  private:
//   Add(const Expr& lhs, const Expr& rhs) : BinaryOpNode(lhs, rhs, IRNodeType::kAdd) {}
//   friend class BinaryOpNode<Add>;
// };

// class Sub : public BinaryOpNode<Sub> {
//  private:
//   Sub(const Expr& lhs, const Expr& rhs) : BinaryOpNode(lhs, rhs, IRNodeType::kSub) {}
//   friend class BinaryOpNode<Sub>;
// };

// class Mul : public BinaryOpNode<Mul> {
//  private:
//   Mul(const Expr& lhs, const Expr& rhs) : BinaryOpNode(lhs, rhs, IRNodeType::kMul) {}
//   friend class BinaryOpNode<Mul>;
// };

// class Div : public BinaryOpNode<Div> {
//  private:
//   Div(const Expr& lhs, const Expr& rhs) : BinaryOpNode(lhs, rhs, IRNodeType::kDiv) {}
//   friend class BinaryOpNode<Div>;
// };

// // Encode an integer immediate value.
// class IntImm : public ExprNode<IntImm> {
//  public:
//   int value() const { return value_; }
//   static Expr make(int value) { return Expr(new IntImm(value)); }

//  private:
//   IntImm(int value) : ExprNodeBase(DType(CType::kInt32, 1)), value_(value) {}
//   int value_;
// };

// // Encode an fp32 immediate value.
// class FloatImm : public ExprNode<FloatImm> {
//  public:
//   float value() const { return value_; }
//   static Expr make(float value) { return Expr(new FloatImm(value)); }

//  private:
//   FloatImm(float value) : ExprNodeBase(DType(CType::kFloat32, 1)), value_(value) {}
//   float value_;
// };

// // The underlying representation node to a Variable.
// // Currently, each Variable object represents a unique variable, even though the names
// // might be the same. We should consider add a unique_name as well.
// class Variable : public ExprNode<Variable> {
//  public:
//   static Expr make(const std::string& name_hint, DType dtype) {
//     return Expr(new Variable(name_hint, dtype));
//   }
//   static Expr make(DType dtype) { return Expr(new Variable("", dtype)); }

//   // TODO: unique_name
//   const std::string& name_hint() const {
//     return name_hint_;
//   }

//  private:
//   Variable(const std::string& name_hint, DType dtype)
//       : ExprNodeBase(dtype), name_hint_(name_hint) {}
//   std::string name_hint_;
// };

// class Block : public ExprNode<Block> {
//  public:
//   static Expr make(const std::vector<Expr>& exprs) { return Expr(new Block(exprs)); }
//   int nexprs() const { return exprs_.size(); }
//   const Expr& expr(int index) const { return exprs_[index]; }

//  private:
//   explicit Block(const std::vector<Expr>& exprs) : ExprNodeBase(DType(CType::kNull)), exprs_(exprs) {}
//   std::vector<Expr> exprs_;
// };

// class For : public ExprNode<For> {
//  public:
//   const Expr& var() const { return var_; }
//   const Expr& start() const { return start_; }
//   const Expr& stop() const { return stop_; }
//   const Expr& body() const { return body_; }
//   static Expr make(const Expr& var, const Expr& start, const Expr& stop, const Expr& body) {
//     return Expr(new For(var, start, stop, body));
//   }

//  private:
//   For(const Expr& var, const Expr& start, const Expr& stop, const Expr& body)
//       : ExprNodeBase(DType(CType::kNull)), var_(var), start_(start), stop_(stop), body_(body) {
//         //TODO! make sure var is of type Variable
//       }
//   Expr var_;
//   Expr start_;
//   Expr stop_;
//   Expr body_;
// };

// //Dummy expr for testing
// class EmptyExpr : public  ExprNode<EmptyExpr> {
//   public:
//     static Expr make (){ return Expr(new EmptyExpr()); }
//   private:
//     EmptyExpr():ExprNodeBase(DType(CType::kNull)){}
// };

// } // namespace fuser
// } // namespace jit
// } // namespace torch
