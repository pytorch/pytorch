#ifndef NNC_TESTS_TEST_UTILS_H_INCLUDED__
#define NNC_TESTS_TEST_UTILS_H_INCLUDED__

#include <gtest/gtest.h>
#include <unordered_map>

#include "function.h"
#include "ir.h"
#include "tensor.h"
#include "types.h"

namespace nnc {

class SimpleExprEvaluator : public IRVisitor {
 public:
  void visit(const Add* v) override { visit_binary_op(v); }
  void visit(const Sub* v) override { visit_binary_op(v); }
  void visit(const Mul* v) override { visit_binary_op(v); }
  void visit(const Div* v) override { visit_binary_op(v); }

  template <typename T>
  Scalar binary_op(const Scalar& lhs, const Scalar& rhs, IRNodeType op_type) {
    T lhs_v = lhs.as<T>();
    T rhs_v = rhs.as<T>();
    T result_v = T();
    switch (op_type) {
      case IRNodeType::kAdd:
        result_v = lhs_v + rhs_v;
        break;
      case IRNodeType::kSub:
        result_v = lhs_v - rhs_v;
        break;
      case IRNodeType::kMul:
        result_v = lhs_v * rhs_v;
        break;
      case IRNodeType::kDiv:
        result_v = lhs_v / rhs_v;
        break;
      default:
        // TODO: change to a proper error report
        throw std::runtime_error("invalid operator type");
    }
    return Scalar(result_v);
  }

  template <typename Op>
  void visit_binary_op(const BinaryOpNode<Op>* v) {
    v->lhs().accept(this);
    Scalar lhs_v = value_;
    v->rhs().accept(this);
    Scalar rhs_v = value_;
    CHECK_EQ(lhs_v.dtype(), rhs_v.dtype());
    IRNodeType expr_type = v->expr_type();
    if (lhs_v.dtype() == kFloat32) {
      value_ = binary_op<float>(lhs_v, rhs_v, expr_type);
    } else if (lhs_v.dtype() == kInt32) {
      value_ = binary_op<int>(lhs_v, rhs_v, expr_type);
    } else {
      LOG(FATAL) << "invalid dtype: " << lhs_v.dtype();
    }
  }

  void visit(const IntImm* v) override { value_ = Scalar(v->value()); }
  void visit(const FloatImm* v) override { value_ = Scalar(v->value()); }

  void visit(const Let* v) override {
    const Variable* var = v->var().AsNode<Variable>();
    ASSERT_NE(var, nullptr);
    v->value().accept(this);
    Scalar value = value_;
    auto iter = eval_context_.find(var);
    // TODO: make the same value settable multiple times.
    CHECK(iter == eval_context_.end()) << "var must not exist in the context before";
    eval_context_[var] = value_;

    v->body().accept(this);

    eval_context_.erase(var);
  }

  void visit(const Variable* v) override {
    auto iter = eval_context_.find(v);
    CHECK(iter != eval_context_.end()) << "var must be defined in the context before";
    value_ = iter->second;
  }

  void visit(const Cast* v) override {
    const Expr& src_value = v->src_value();
    src_value.accept(this);
    Dtype dst_dtype = v->dtype();
    Dtype src_dtype = src_value.dtype();
    if (src_dtype != dst_dtype) {
      if (src_dtype == kFloat32 && dst_dtype == kInt32) {
        int v = static_cast<int>(value_.as<float>());
        value_ = Scalar(v);
      } else if (src_dtype == kInt32 && dst_dtype == kFloat32) {
        float v = static_cast<float>(value_.as<int>());
        value_ = Scalar(v);
      }
    }
  }

  Scalar value() const { return value_; }

 private:
  Scalar value_;
  std::unordered_map<const BaseExprNode*, Scalar> eval_context_;
};

template <class T>
class SimpleTensorEvaluator {
 public:
  void evaluate(const Tensor& t, std::vector<T>* output) {
    int ndim = t.ndim();
    std::vector<int> dims;
    int size = 1;
    for (int i = 0; i < ndim; i++) {
      t.dim(i).accept(&expr_eval_);
      int dim = expr_eval_.value().template as<int>();
      dims.push_back(dim);
      size *= dim;
    }
    const Function& func = t.function();
    const Expr& body = func.body();
    eval_func(dims, func, 0, output, body);
  }

 private:
  void eval_func(const std::vector<int>& dims, const Function& func, int level,
                 std::vector<T>* output, const Expr& body) {
    if (level >= dims.size()) {
      body.accept(&expr_eval_);
      output->push_back(expr_eval_.value().template as<T>());
      return;
    }
    for (int i = 0; i < dims[level]; i++) {
      Expr wrapped_body = Let::make(func.arg(level), Expr(i), body);
      eval_func(dims, func, level + 1, output, wrapped_body);
    }
  }

  SimpleExprEvaluator expr_eval_;
};

}  // namespace nnc

#endif  // NNC_TESTS_TEST_UTILS_H_INCLUDED__
