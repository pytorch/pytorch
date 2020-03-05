#pragma once

#include <cmath>
#include <unordered_map>
#include <vector>

#include <c10/util/Logging.h>
#include "torch/csrc/jit/tensorexpr/buffer.h"
#include "torch/csrc/jit/tensorexpr/codegen.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/types.h"

namespace torch {
namespace jit {
namespace tensorexpr {

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
  void* ptr{};
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

inline int mod_value(int lhs, int rhs) {
  return lhs % rhs;
}

inline float mod_value(float lhs, float rhs) {
  return std::fmod(lhs, rhs);
}

class SimpleIREvaluator : public CodeGen, public IRVisitor {
 public:
  using CodeGen::CodeGen;

  ~SimpleIREvaluator() override {}

  TORCH_API void call(const std::vector<CallArg>& args) override {
    CHECK_EQ(args.size(), buffer_args().size());
    for (size_t i = 0; i < args.size(); i++) {
      bind(buffer_args()[i], args[i]);
    }
    stmt().accept(this);
    eval_context_.clear();
    buffer_mapping_.clear();
    internal_buffers_.clear();
  }

  void bind(const BufferArg& buf, const CallArg& data) {
    if (buf.isVar()) {
      if (buf.dtype() == kInt32) {
        eval_context_[buf.var().node()] = data.intData();
      } else if (buf.dtype() == kFloat32) {
        eval_context_[buf.var().node()] = data.floatData();
      } else {
        LOG(FATAL) << "Unhandled dtype for argument " << buf.var().name_hint()
                   << ": " << buf.dtype();
      }
    } else {
      buffer_mapping_[buf.var().node()] = data.data();
    }
  }

  template <typename... Ts>
  void operator()(const Ts&... ts) {
    std::vector<CallArg> args({CallArg(ts)...});
    call(args);
  }

  TORCH_API void visit(const Add* v) override {
    visit_binary_op(v);
  }
  TORCH_API void visit(const Sub* v) override {
    visit_binary_op(v);
  }
  TORCH_API void visit(const Mul* v) override {
    visit_binary_op(v);
  }
  TORCH_API void visit(const Div* v) override {
    visit_binary_op(v);
  }
  TORCH_API void visit(const Mod* v) override {
    visit_binary_op(v);
  }
  TORCH_API void visit(const Max* v) override {
    visit_binary_op(v, v->propagate_nans());
  }
  TORCH_API void visit(const Min* v) override {
    visit_binary_op(v, v->propagate_nans());
  }

  void visit(const CompareSelect* v) override {
    visit_compare_select_op(v, v->compare_select_op());
  }

  template <typename T>
  Value binary_op(
      const Value& lhs,
      const Value& rhs,
      IRNodeType op_type,
      bool option = false) {
    std::vector<T> lhs_v = lhs.as_vec<T>();
    std::vector<T> rhs_v = rhs.as_vec<T>();
    std::vector<T> result_v(lhs_v.size());
    for (size_t i = 0; i < lhs_v.size(); i++) {
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
        case IRNodeType::kMod:
          result_v[i] = mod_value(lhs_v[i], rhs_v[i]);
          break;
        case IRNodeType::kMax:
          if (lhs.dtype() == kFloat32 && rhs.dtype() == kFloat32 && option) {
            // Propagate NaNs
            if (std::isnan((float)lhs_v[i])) {
              result_v[i] = lhs_v[i];
            } else if (std::isnan((float)rhs_v[i])) {
              result_v[i] = rhs_v[i];
            }
          } else {
            result_v[i] = lhs_v[i] > rhs_v[i] ? lhs_v[i] : rhs_v[i];
          }
          break;
        case IRNodeType::kMin:
          if (lhs.dtype() == kFloat32 && rhs.dtype() == kFloat32 && option) {
            // Propagate NaNs
            if (std::isnan((float)lhs_v[i])) {
              result_v[i] = lhs_v[i];
            } else if (std::isnan((float)rhs_v[i])) {
              result_v[i] = rhs_v[i];
            }
          } else {
            result_v[i] = lhs_v[i] < rhs_v[i] ? lhs_v[i] : rhs_v[i];
          }
          break;
        default:
          // TODO: change to a proper error report
          throw std::runtime_error("invalid operator type");
      }
    }
    return Value(result_v);
  }

  template <typename T>
  Value compare_select_op(
      const Value& lhs,
      const Value& rhs,
      CompareSelectOperation cmp_op) {
    std::vector<T> lhs_v = lhs.as_vec<T>();
    std::vector<T> rhs_v = rhs.as_vec<T>();
    std::vector<int> result_v(lhs_v.size());
    for (size_t i = 0; i < lhs_v.size(); i++) {
      switch (cmp_op) {
        case CompareSelectOperation::kEQ:
          result_v[i] = (lhs_v[i] == rhs_v[i]) ? 1 : 0;
          break;
        case CompareSelectOperation::kNE:
          result_v[i] = (lhs_v[i] != rhs_v[i]) ? 1 : 0;
          break;
        case CompareSelectOperation::kGT:
          result_v[i] = (lhs_v[i] > rhs_v[i]) ? 1 : 0;
          break;
        case CompareSelectOperation::kGE:
          result_v[i] = (lhs_v[i] >= rhs_v[i]) ? 1 : 0;
          break;
        case CompareSelectOperation::kLT:
          result_v[i] = (lhs_v[i] < rhs_v[i]) ? 1 : 0;
          break;
        case CompareSelectOperation::kLE:
          result_v[i] = (lhs_v[i] <= rhs_v[i]) ? 1 : 0;
          break;
        default:
          // TODO: change to a proper error report
          throw std::runtime_error("invalid operator type");
      }
    }
    return Value(result_v);
  }

  template <typename Op>
  void visit_binary_op(const BinaryOpNode<Op>* v, bool option = false) {
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

  void visit_compare_select_op(
      const CompareSelect* v,
      CompareSelectOperation cmp_op) {
    v->lhs().accept(this);
    Value lhs_v = value_;
    v->rhs().accept(this);
    Value rhs_v = value_;
    CHECK_EQ(lhs_v.dtype(), rhs_v.dtype());
    if (lhs_v.dtype().scalar_type() == kFloat32) {
      value_ = compare_select_op<float>(lhs_v, rhs_v, cmp_op);
    } else if (lhs_v.dtype().scalar_type() == kInt32) {
      value_ = compare_select_op<int>(lhs_v, rhs_v, cmp_op);
    } else {
      LOG(FATAL) << "invalid dtype: " << lhs_v.dtype();
    }
  }

  TORCH_API void visit(const IntImm* v) override {
    value_ = Value(v->value());
  }
  TORCH_API void visit(const FloatImm* v) override {
    value_ = Value(v->value());
  }

  TORCH_API void visit(const Let* v) override {
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

  TORCH_API void visit(const Variable* v) override {
    auto iter = eval_context_.find(v);
    CHECK(iter != eval_context_.end())
        << "var must be defined in the context before";
    value_ = iter->second;
  }

  TORCH_API void visit(const Cast* v) override {
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

  TORCH_API void visit(const For* v) override {
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

  TORCH_API void visit(const Ramp* v) override {
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

  TORCH_API void visit(const Broadcast* v) override {
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

  TORCH_API void visit(const IfThenElse* v) override {
    v->condition().accept(this);
    if (value_.as<int>()) {
      v->true_value().accept(this);
    } else {
      v->false_value().accept(this);
    }
  }

  TORCH_API void visit(const Load* v) override {
    const Variable* base_node = v->base_handle().node();
    auto iter = buffer_mapping_.find(base_node);
    CHECK(iter != buffer_mapping_.end())
        << "missing buffer binding: " << base_node->name_hint();
    void* ptr = iter->second;

    v->index().accept(this);
    std::vector<int> index = value().as_vec<int>();
    v->mask().accept(this);
    std::vector<int> mask = value().as_vec<int>();
    Dtype v_sdtype = v->dtype().scalar_type();
    if (v_sdtype == kFloat32) {
      float* ptr_f = static_cast<float*>(ptr);
      std::vector<float> v(index.size());
      for (size_t i = 0; i < index.size(); i++) {
        if (mask[i]) {
          v[i] = ptr_f[index[i]];
        }
      }
      value_ = Value(v);
    } else if (v_sdtype == kInt32) {
      int* ptr_i = static_cast<int*>(ptr);
      std::vector<int> v(index.size());
      for (size_t i = 0; i < index.size(); i++) {
        if (mask[i]) {
          v[i] = ptr_i[index[i]];
        }
      }
      value_ = Value(v);
    } else {
      LOG(FATAL) << "Invalid dtype: " << v_sdtype;
    }
  }

  TORCH_API void visit(const Store* v) override {
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
      for (size_t i = 0; i < index.size(); i++) {
        if (mask[i]) {
          ptr_f[index[i]] = value[i];
        }
      }
    } else if (v_sdtype == kInt32) {
      v->value().accept(this);
      std::vector<int> value = this->value().as_vec<int>();
      CHECK_EQ(index.size(), value.size());
      int* ptr_i = static_cast<int*>(ptr);
      for (size_t i = 0; i < index.size(); i++) {
        if (mask[i]) {
          ptr_i[index[i]] = value[i];
        }
      }
    } else {
      LOG(FATAL) << "Invalid dtype: " << v_sdtype;
    }
  }

  void visit(const Allocate* v) override {
    const Variable* buffer_var = v->buffer_var().AsNode<Variable>();
    std::vector<Expr> dims = v->dims();
    int total_byte_size = v->dtype().byte_size();
    for (size_t i = 0; i < dims.size(); i++) {
      dims[i].accept(this);
      total_byte_size *= value_.as<int>();
    }
    int int_count = (total_byte_size + sizeof(int) - 1) / sizeof(int);
    std::unique_ptr<std::vector<int>> buffer(new std::vector<int>(int_count));
    auto iter = buffer_mapping_.find(buffer_var);
    if (iter != buffer_mapping_.end() && iter->second != nullptr) {
      throw std::runtime_error(
          "Allocate a buffer that has already been allocated: " +
          buffer_var->name_hint());
    }
    buffer_mapping_[buffer_var] = buffer->data();
    internal_buffers_.insert(std::make_pair(buffer_var, std::move(buffer)));
  }

  void visit(const Free* v) override {
    const Variable* buffer_var = v->buffer_var().AsNode<Variable>();
    int count = internal_buffers_.erase(buffer_var);
    if (count == 0) {
      throw std::runtime_error(
          "Free a buffer that is not currently bound: " +
          buffer_var->name_hint());
    }
  }

  void visit(const Cond* v) override {
    v->condition().accept(this);
    if (value().as<int>()) {
      v->true_stmt().accept(this);
    } else {
      v->false_stmt().accept(this);
    }
  }

  Value value() const {
    return value_;
  }

 private:
  Value value_;
  std::unordered_map<const BaseExprNode*, Value> eval_context_;
  std::unordered_map<const BaseExprNode*, void*> buffer_mapping_;
  std::unordered_map<const Variable*, std::unique_ptr<std::vector<int>>>
      internal_buffers_;
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

template <class CodeGenType>
class ExprEval {
 public:
  using BufferArg = CodeGen::BufferArg;
  using CallArg = CodeGen::CallArg;

  template <typename... Ts>
  ExprEval(const Expr& expr, Ts... ts) : ExprEval(expr, {BufferArg(ts)...}) {}

  ExprEval(const Expr& expr, const std::vector<BufferArg>& buffer_args)
      : dtype_(expr.dtype()) {
    std::vector<BufferArg> buffer_args_extended = buffer_args;
    Buffer ret_buf("ret_val", dtype_, {1});
    Stmt store_stmt = Store::make(ret_buf.data(), 0, expr);
    buffer_args_extended.push_back(ret_buf);
    codegen_.reset(new CodeGenType(store_stmt, buffer_args_extended));
  }

  template <typename... Ts>
  void operator()(Ts... ts) {
    call(ts...);
  }

  void operator()(const std::vector<CallArg>& call_args) {
    call(call_args);
  }

  template <typename... Ts>
  void call(Ts... ts) {
    call({CallArg(ts)...});
  }

  void call(const std::vector<CallArg>& call_args) {
    std::vector<CallArg> call_args_extended = call_args;
    if (dtype_ == kFloat32) {
      std::vector<float> ret_val_arg(1);
      call_args_extended.push_back(CallArg(ret_val_arg));
      codegen_->call(call_args_extended);
      ret_value_ = Value(ret_val_arg[0]);
    } else if (dtype_ == kInt32) {
      std::vector<int> ret_val_arg(1);
      call_args_extended.push_back(CallArg(ret_val_arg));
      codegen_->call(call_args_extended);
      ret_value_ = Value(ret_val_arg[0]);
    } else {
      throw std::runtime_error("Invalid dtype");
    }
  }

  template <typename T, typename... Ts>
  T value(Ts... ts) {
    call(std::forward<Ts>(ts)...);
    return ret_value_.as<T>();
  }

 private:
  Dtype dtype_;
  std::unique_ptr<CodeGenType> codegen_;
  Value ret_value_;
};

inline Expr Substitute(Expr* expr, const VarMapping& var_mapping) {
  VarSubMutator var_sub(var_mapping);
  return expr->accept_mutator(&var_sub);
}

inline Stmt Substitute(Stmt* stmt, const VarMapping& var_mapping) {
  VarSubMutator var_sub(var_mapping);
  return stmt->accept_mutator(&var_sub);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
