#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstdint>
#include <unordered_map>
#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

using ValNameType = std::uint32_t;

enum class TORCH_API ValType {
  Expr
, TensorLike
, Addr
, Scalar
, Range
};

enum class TORCH_API ExprType {
  Loop  // swap, merge, split
, Index
, Add
};

TORCH_API std::ostream& operator<<(std::ostream& out, const ValType valtype);

// IR Requirements & Goals:
// clear, easy, extensible way to add new vals, new exprs
// nice pretty printing
// validating by lowering TorchScript IR and pretty printing
// supports visitor pattern
// allows for easy manipulation/querying

// Val Requirements & Goals
  // support for dynamic dispatch
  // symbol to determine dispatch / true type
  // allow dispatch / visitors to handle a subset a vals with
  //  a generic fallback
  // allowing naming
  // allow queries on fusion for what values are / what they're doing
  // constness support

struct TORCH_API Val {
  Val() = delete;
  Val(
    const ValNameType _name
  , const ValType _type)
  : name_{_name}
  , type_{_type} { }

  Val(const Val& other) = default;
  Val& operator=(const Val& other) = default;

  Val(Val&& other) = default;
  Val& operator=(Val&& other) = default;

  ~Val() = default;

  template <typename T>
  int dispatch(T& handler);

  ValNameType name() const noexcept { return name_; }
  ValType type() const noexcept { return type_; }

private:
  ValNameType name_;
  ValType type_;
  void* contained_ = nullptr;
};

struct Expr { // inputs, outputs, RAII, getType, cast/expect, dispatch method
  // getFusion, getRegion
  // implement IRVisitor for dispatch? want callable on one node vs. visit all
};

struct TORCH_API SampleValHandler {
  template <typename T>
  int handle(Val* val, T* contained);

  int handle (Val* val, Expr* expr);
};



struct Region { // list of nodes (expressions)
};

// Owns the exprs and vals
// val name -> val map
//
struct Fusion {
    // fusion: removeNode, addNodeBefore, addNodeAfter, addNodeAt
    // name generator
    // contains region
private:
  std::unordered_map<ValNameType, Val*> val_map_;
};

}}} // torch::jit::fuser

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
