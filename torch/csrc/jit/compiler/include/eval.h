#pragma once

#include <unordered_map>
#include <vector>

#include "torch/csrc/jit/compiler/include/function.h"
#include "torch/csrc/jit/compiler/include/ir.h"
#include "torch/csrc/jit/compiler/include/logging.h"
#include "torch/csrc/jit/compiler/include/tensor.h"
#include "torch/csrc/jit/compiler/include/types.h"

namespace torch {
namespace jit {
namespace compiler {

class Value {
 public:
  Value() : dtype_(kInt32) {
    i32_values.push_back(0);
  }
  Value(int v) : dtype_(kInt32) {
    i32_values.push_back(v);
  }
  Value(float v) : dtype_(kFloat32) {
    f32_values.push_back(v);
  }
  Value(const std::vector<int>& v)
      : dtype_(Dtype(kInt32, v.size())), i32_values(v) {}
  Value(const std::vector<float>& v)
      : dtype_(Dtype(kFloat32, v.size())), f32_values(v) {}

  template <typename T>
  T as() const;

  template <typename T>
  const std::vector<T>& as_vec() const;

  Dtype dtype() const {
    return dtype_;
  }

 private:
  Dtype dtype_;
  std::vector<int32> i32_values;
  std::vector<float> f32_values;
  void* ptr;
};

template <>
inline int Value::as<int>() const {
  CHECK_EQ(dtype_, kInt32) << "invalid dtype";
  return i32_values[0];
}

template <>
inline float Value::as<float>() const {
  CHECK_EQ(dtype_, kFloat32) << "invalid dtype";
  return f32_values[0];
}

template <>
inline const std::vector<float>& Value::as_vec<float>() const {
  CHECK_EQ(dtype_.scalar_type(), kFloat32) << "invalid dtype";
  return f32_values;
}

template <>
inline const std::vector<int>& Value::as_vec<int>() const {
  CHECK_EQ(dtype_.scalar_type(), kInt32) << "invalid dtype";
  return i32_values;
}

class SimpleIREvaluator : public IRVisitor {
 public:
  void visit(const Add* v) override {
    visit_binary_op(v);
  }
  void visit(const Sub* v) override {
    visit_binary_op(v);
  }
  void visit(const Mul* v) override {
    visit_binary_op(v);
  }
  void visit(const Div* v) override {
    visit_binary_op(v);
  }

  template <typename T>
  Value binary_op(const Value& lhs, const Value& rhs, IRNodeType op_type) {
    std::vector<T> lhs_v = lhs.as_vec<T>();
    std::vector<T> rhs_v = rhs.as_vec<T>();
    std::vector<T> result_v(lhs_v.size());
    for (int i = 0; i < lhs_v.size(); i++) {
      switch (op_type) {
        case IRNodeType::kAdd:
          result_v[i] = lhs_v[i] + rhs_v[i];
          break;
        case IRNodeType::kSub:
          result_v[i] = lhs_v[i] - rhs_v[i];
          break;
        case IRNodeType::kMul:
          result_v[i] = lhs_v[i] * rhs_v[i];
          break;
        case IRNodeType::kDiv:
          result_v[i] = lhs_v[i] / rhs_v[i];
          break;
        default:
          // TODO: change to a proper error report
          throw std::runtime_error("invalid operator type");
      }
    }
    return Value(result_v);
  }

  template <typename Op>
  void visit_binary_op(const BinaryOpNode<Op>* v) {
    v->lhs().accept(this);
    Value lhs_v = value_;
    v->rhs().accept(this);
    Value rhs_v = value_;
    CHECK_EQ(lhs_v.dtype(), rhs_v.dtype());
    IRNodeType expr_type = v->expr_type();
    if (lhs_v.dtype().scalar_type() == kFloat32) {
      value_ = binary_op<float>(lhs_v, rhs_v, expr_type);
    } else if (lhs_v.dtype().scalar_type() == kInt32) {
      value_ = binary_op<int>(lhs_v, rhs_v, expr_type);
    } else {
      LOG(FATAL) << "invalid dtype: " << lhs_v.dtype();
    }
  }

  void visit(const IntImm* v) override {
    value_ = Value(v->value());
  }
  void visit(const FloatImm* v) override {
    value_ = Value(v->value());
  }

  void visit(const Let* v) override {
    const Variable* var = v->var().AsNode<Variable>();
    CHECK(var != nullptr);
    v->value().accept(this);
    Value value = value_;
    auto iter = eval_context_.find(var);
    // TODO: make the same value settable multiple times.
    CHECK(iter == eval_context_.end())
        << "var must not exist in the context before";
    eval_context_[var] = value_;

    v->body().accept(this);

    eval_context_.erase(var);
  }

  void visit(const Variable* v) override {
    auto iter = eval_context_.find(v);
    CHECK(iter != eval_context_.end())
        << "var must be defined in the context before";
    value_ = iter->second;
  }

  void visit(const Cast* v) override {
    const Expr& src_value = v->src_value();
    src_value.accept(this);
    Dtype dst_dtype = v->dtype();
    Dtype src_dtype = src_value.dtype();
    CHECK_EQ(src_dtype.lanes(), dst_dtype.lanes());
    if (src_dtype != dst_dtype) {
      if (src_dtype == kFloat32 && dst_dtype == kInt32) {
        const std::vector<float>& src_values = value_.as_vec<float>();
        std::vector<int> dst_values(src_values.size());
        for (int i = 0; i < src_dtype.lanes(); ++i) {
          dst_values[i] = static_cast<int>(src_values[i]);
        }
        this->value_ = Value(dst_values);
      } else if (src_dtype == kInt32 && dst_dtype == kFloat32) {
        const std::vector<int>& src_values = value_.as_vec<int>();
        std::vector<float> dst_values(src_values.size());
        for (int i = 0; i < src_dtype.lanes(); ++i) {
          dst_values[i] = static_cast<float>(src_values[i]);
        }
        this->value_ = Value(dst_values);
      }
    }
  }

  void visit(const For* v) override {
    const BaseExprNode* var_node = v->var().node();
    v->start().accept(this);
    int start = value_.as<int>();
    v->stop().accept(this);
    int stop = value_.as<int>();
    auto iter = eval_context_.find(var_node);
    CHECK(iter == eval_context_.end())
        << "var in For must not exist in eval context";
    for (int i = start; i < stop; i++) {
      eval_context_[var_node] = Value(i);
      v->body().accept(this);
    }
    eval_context_.erase(var_node);
  }

  void visit(const Ramp* v) override {
    v->base().accept(this);
    int base = value().as<int>();
    v->stride().accept(this);
    int stride = value().as<int>();
    int lanes = v->lanes();

    std::vector<int> values(lanes);
    for (int i = 0; i < lanes; i++) {
      values[i] = base + i * stride;
    }

    value_ = Value(values);
  }

  void visit(const Broadcast* v) override {
    v->value().accept(this);
    Value value = this->value();
    int lanes = v->lanes();
    if (value.dtype() == kInt32) {
      std::vector<int> v(lanes, value.as<int>());
      value_ = Value(v);
    } else if (value.dtype() == kFloat32) {
      std::vector<float> v(lanes, value.as<float>());
      value_ = Value(v);
    } else {
      LOG(FATAL) << "invalid dtype: " << value.dtype();
    }
  }

  void visit(const Load* v) override {
    const Variable* base_node = v->base_handle().node();
    auto iter = buffer_mapping_.find(base_node);
    CHECK(iter != buffer_mapping_.end());
    void* ptr = iter->second;

    v->index().accept(this);
    std::vector<int> index = value().as_vec<int>();
    v->mask().accept(this);
    std::vector<int> mask = value().as_vec<int>();
    Dtype v_sdtype = v->dtype().scalar_type();
    if (v_sdtype == kFloat32) {
      float* ptr_f = static_cast<float*>(ptr);
      std::vector<float> v(index.size());
      for (int i = 0; i < index.size(); i++) {
        if (mask[i]) {
          v[i] = ptr_f[index[i]];
        }
      }
      value_ = Value(v);
    } else if (v_sdtype == kInt32) {
      int* ptr_i = static_cast<int*>(ptr);
      std::vector<int> v(index.size());
      for (int i = 0; i < index.size(); i++) {
        if (mask[i]) {
          v[i] = ptr_i[index[i]];
        }
      }
      value_ = Value(v);
    } else {
      LOG(FATAL) << "Invalid dtype: " << v_sdtype;
    }
  }

  void visit(const Store* v) override {
    const Variable* base_node = v->base_handle().node();
    auto iter = buffer_mapping_.find(base_node);
    CHECK(iter != buffer_mapping_.end());
    void* ptr = iter->second;

    v->index().accept(this);
    std::vector<int> index = value().as_vec<int>();
    v->mask().accept(this);
    std::vector<int> mask = value().as_vec<int>();
    CHECK_EQ(index.size(), mask.size());
    Dtype v_sdtype = v->value().dtype().scalar_type();
    if (v_sdtype == kFloat32) {
      v->value().accept(this);
      std::vector<float> value = this->value().as_vec<float>();
      CHECK_EQ(index.size(), value.size());
      float* ptr_f = static_cast<float*>(ptr);
      for (int i = 0; i < index.size(); i++) {
        if (mask[i]) {
          ptr_f[index[i]] = value[i];
        }
      }
    } else if (v_sdtype == kInt32) {
      v->value().accept(this);
      std::vector<int> value = this->value().as_vec<int>();
      CHECK_EQ(index.size(), value.size());
      int* ptr_i = static_cast<int*>(ptr);
      for (int i = 0; i < index.size(); i++) {
        if (mask[i]) {
          ptr_i[index[i]] = value[i];
        }
      }
    } else {
      LOG(FATAL) << "Invalid dtype: " << v_sdtype;
    }
  }

  using BufferMapping = std::unordered_map<const BaseExprNode*, void*>;
  void SetBufferMapping(const BufferMapping& buffer_mapping) {
    buffer_mapping_ = buffer_mapping;
  }

  Value value() const {
    return value_;
  }

 private:
  Value value_;
  std::unordered_map<const BaseExprNode*, Value> eval_context_;
  BufferMapping buffer_mapping_;
};

using VarMapping = std::vector<std::pair<Expr, Expr>>;

class VarSubMutator : public IRMutator {
 public:
  VarSubMutator(const VarMapping& var_mapping) {
    for (const auto& entry : var_mapping) {
      const Expr& key = entry.first;
      const Expr& value = entry.second;
      const Variable* key_var = key.AsNode<Variable>();
      CHECK(key_var != nullptr);
      var_mapping_[key_var] = value;
    }
  }

  Expr mutate(const Variable* var) override {
    auto iter = var_mapping_.find(var);
    if (iter == var_mapping_.end()) {
      return Expr(const_cast<Variable*>(var));
    }
    return iter->second;
  }

 private:
  std::unordered_map<const Variable*, Expr> var_mapping_;
};

inline Expr Substitute(Expr* expr, const VarMapping& var_mapping) {
  VarSubMutator var_sub(var_mapping);
  return expr->accept_mutator(&var_sub);
}

inline Stmt Substitute(Stmt* stmt, const VarMapping& var_mapping) {
  VarSubMutator var_sub(var_mapping);
  return stmt->accept_mutator(&var_sub);
}

} // namespace compiler
} // namespace jit
} // namespace torch
