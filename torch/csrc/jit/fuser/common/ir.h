#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <c10/util/Optional.h>

#include <torch/csrc/jit/fuser/common/type.h>

#include <cstdint>
#include <unordered_map>
#include <stdexcept>
#include <vector>

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

using StmtNameType = std::uint32_t;
struct Fusion;

struct TORCH_API Statement {
  friend struct Fusion;

  template <typename T>
  int dispatch(T* handler) const;

  virtual c10::optional<ValType> getValType() const noexcept { return c10::nullopt; }
  virtual c10::optional<ExprType> getExprType() const noexcept { return c10::nullopt; }

  bool isVal() const noexcept { return getValType() != c10::nullopt; }
  bool isExpr() const noexcept{ return getExprType() != c10::nullopt; }

  Fusion* fusion() const noexcept { return fusion_; }
  StmtNameType name() const noexcept { return name_; }

  void setFusion(Fusion* fusion) {
    fusion_ = fusion;
  }
  void setName(const StmtNameType name) {
    name_ = name;
  }

protected:
  virtual ~Statement() = 0;

  Fusion* fusion_;
  StmtNameType name_;
};

struct TORCH_API Val : public Statement {

public:
  virtual ~Val() = 0;

  Val() = delete;
  Val(
    const ValType _type,
    Fusion& fusion);
  
  c10::optional<ValType> getValType() const noexcept override { return type_; }

  ValType type() const noexcept { return type_; }

protected:
  ValType type_;

private:
    Val(const Val& other) = delete;
  Val& operator=(const Val& other) = delete;

  Val(Val&& other) = delete;
  Val& operator=(Val&& other) = delete;

};

// TODO: support symbolic floats vs literal (const) floats (make value an optional)
struct TORCH_API Float : public Val {
  Float() = delete;

  Float(
    const float _value,
    Fusion& fusion)
  : Val(ValType::Float, fusion)
  , value_{_value}
  { }


  Float(const Float& other) = delete;
  Float& operator=(const Float& other) = delete;

  Float(Float&& other) = delete;
  Float& operator=(Float&& other) = delete;

  float value() const noexcept { return value_; }

protected:
  ~Float() = default;
  
private:
  float value_;
};

// TODO: improve input/output model to track dataflow
// TODO: add regions (e.g. loop exprs have bodies)
struct TORCH_API Expr : public Statement {
public:
  Expr() = delete;

  c10::optional<ExprType> getExprType() const noexcept override { return type_; }

  ExprType type() const noexcept { return type_; }

  const Statement* getInput(const std::vector<Statement*>::size_type idx) const {
    return inputs_[idx];
  }

  const Statement* getOutput(const std::vector<Statement*>::size_type idx) const {
    return outputs_[idx];
  }

  std::vector<Statement*>::size_type n_inputs() const {return inputs_.size();}
  std::vector<Statement*>::size_type n_outputs() const {return outputs_.size();}

protected:

  Expr(
    const ExprType _type,
    Fusion& fusion);

  void addInput(const Statement* input) {
    inputs_.push_back(input);
  }

  void addOutput(const Statement* output) {
    outputs_.push_back(output);
  }

private:

  ExprType type_;

  std::vector<const Statement*> inputs_;
  std::vector<const Statement*> outputs_;
};

struct TORCH_API Add : public Expr {
  Add(const Statement* _lhs, const Statement* _rhs, Fusion& fusion )
  : Expr(ExprType::Add, fusion)
  , lhs_(_lhs)
  , rhs_(_rhs) 
  {
    addInput(_lhs);
    addInput(_rhs);
  }

  const Statement* const lhs_;
  const Statement* const rhs_;

  Add(const Add& other) = delete;
  Add& operator=(const Add& other) = delete;

  Add(Add&& other) = delete;
  Add& operator=(Add&& other) = delete;

  protected:
  virtual ~Add() = default;
};

/*
struct TORCH_API Region : public Expr {
  Region(const Statement* _lhs, Fusion& fusion )
  : Expr(ExprType::Add, fusion)
  , lhs_(_lhs)
  , rhs_(_rhs) 
  {
    addInput(_lhs);
    addInput(_rhs);
  }

  const Statement* const lhs_;
  const Statement* const rhs_;

  Region(const Region& other) = default;
  Region& operator=(const Region& other) = delete;

  Add(Add&& other) = default;
  Add& operator=(Add&& other) = default;

  virtual ~Add() = default;
};
*/

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
