#ifndef NNC_TESTS_TEST_UTILS_H_INCLUDED__
#define NNC_TESTS_TEST_UTILS_H_INCLUDED__

#include <gtest/gtest.h>
#include <unordered_map>

#include "ir.h"

namespace nnc {

template <typename T>
class SimpleExprEvaluator : public IRVisitor {
 public:
  void visit(const Add* v) override { visit_binary_op(v); }

  void visit(const Sub* v) override { visit_binary_op(v); }

  void visit(const Mul* v) override { visit_binary_op(v); }

  void visit(const Div* v) override { visit_binary_op(v); }

  template <typename Op>
  void visit_binary_op(const BinaryOpNode<Op>* v) {
    v->lhs().accept(this);
    T lhs_v = this->value_;
    v->rhs().accept(this);
    T rhs_v = this->value_;
    switch (v->expr_type()) {
      case ExprNodeType::kAdd:
        this->value_ = lhs_v + rhs_v;
        break;
      case ExprNodeType::kSub:
        this->value_ = lhs_v - rhs_v;
        break;
      case ExprNodeType::kMul:
        this->value_ = lhs_v * rhs_v;
        break;
      case ExprNodeType::kDiv:
        this->value_ = lhs_v / rhs_v;
        break;
      default:
        // TODO: change to a proper error report
        throw std::runtime_error("invalid operator type");
    }
  }

  void visit(const IntImm* v) override { value_ = (T)(v->value()); }
  void visit(const FloatImm* v) override { value_ = (T)(v->value()); }

  void visit(const Let* v) override {
    const Variable* var = v->var().AsNode<Variable>();
    ASSERT_NE(var, nullptr);
    v->value().accept(this);
    T value = value_;
    auto iter = eval_context_.find(var);
    ASSERT_EQ(iter, eval_context_.end());
    eval_context_[var] = value_;

    v->body().accept(this);

    eval_context_.erase(var);
  }

  void visit(const Variable* v) override {
    auto iter = eval_context_.find(v);
    ASSERT_NE(iter, eval_context_.end());
    value_ = iter->second;
  }

  T value() const { return value_; }

 private:
  T value_ = T();
  std::unordered_map<const BaseExprNode*, T> eval_context_;
};

}  // namespace nnc

#endif  // NNC_TESTS_TEST_UTILS_H_INCLUDED__
