#pragma once


// #include <torch/csrc/jit/fuser/common/refcount.h>
// #include <torch/csrc/jit/fuser/common/types.h>
// #include <torch/csrc/jit/fuser/common/ir_visitor.h>

namespace torch {
namespace jit {
namespace fuser {



}}} // torch::jit::fuser

// // The commomn class between all IR nodes.
// class IRNode : public RefCounted {
//  public:
//   virtual void accept(IRVisitor* visitor) const { }
//   virtual ~IRNode() {}
// };

// // The common base between all expression node.
// class BaseExprNode : public IRNode {
//  public:
//   explicit BaseExprNode(DType dtype) : dtype_(dtype) {}

//   DType dtype() const { return dtype_; }

//  private:
//   DType dtype_;
// };

// // A CRTP pattern to accept visitors for children class,
// // and dispatch back to the children.
// template <class Op>
// class ExprNode : public BaseExprNode {
//  public:
//   using ExprNodeBase = ExprNode<Op>;
//   void accept(IRVisitor* visitor) const override {
//     visitor->visit(static_cast<const Op*>(this));
//   }
//   explicit ExprNode(DType dtype) : BaseExprNode(dtype) {}
// };

// // A refcounted pointer to the underlying ExprNode.
// // Also serves the primary way to build and operate on other expressions.
// class Expr : public RefHandle<BaseExprNode> {
//  public:
//   using BaseHandle = RefHandle<BaseExprNode>;
//   explicit Expr(BaseExprNode* node) : BaseHandle(node) {}

//   void accept(IRVisitor* visitor) const {
//     // TODO: Consider implement this without using recursion. Otherwise,
//     // if the expression tree is degenerate and too long, it could cause a
//     // stack overflow.
//     node()->accept(visitor);
//   }

//   Expr(int v);
//   Expr(float v);

//   template <class Op>
//   const Op* AsNode() const {
//     //TODO! fix unsafe casting
//     return dynamic_cast<const Op*>(this->node());
//   }

//   DType dtype() const { return node()->dtype(); }

//   // Handling the math operators.
//   Expr operator+(const Expr& other) const;
//   Expr operator-(const Expr& other) const;
//   Expr operator*(const Expr& other) const;
//   Expr operator/(const Expr& other) const;
// };

// } // namespace fuser
// } // namespace jit
// } // namespace torch
