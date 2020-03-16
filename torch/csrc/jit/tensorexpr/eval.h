#pragma once

#include <cmath>
#include <unordered_map>
#include <vector>

#include <c10/util/Logging.h>
#include "torch/csrc/jit/tensorexpr/buffer.h"
#include "torch/csrc/jit/tensorexpr/codegen.h"
#include "torch/csrc/jit/tensorexpr/execution_counter.h"
#include "torch/csrc/jit/tensorexpr/function.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/ir_printer.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"
#include "torch/csrc/jit/tensorexpr/types.h"

namespace torch {
namespace jit {
namespace tensorexpr {

DECLARE_TRIGGER(simple_ir_eval_executed);

class Value {
 public:
  Value() : dtype_(kInt) {
    Intvalues.push_back(0);
  }

#define VALUE_CTOR(Type, Name)      \
  Value(Type v) : dtype_(k##Name) { \
    Name##values.push_back(v);      \
  }
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, VALUE_CTOR);
#undef VALUE_CTOR

#define VALUE_VEC_CTOR(Type, Name)  \
  Value(const std::vector<Type>& v) \
      : dtype_(Dtype(k##Name, v.size())), Name##values(v) {}
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, VALUE_VEC_CTOR);
#undef VALUE_VEC_CTOR

  template <typename T>
  T as() const;

  template <typename T>
  const std::vector<T>& as_vec() const;

  Dtype dtype() const {
    return dtype_;
  }

 private:
  Dtype dtype_;

#define VALUE_STORAGE(Type, Name) std::vector<Type> Name##values;
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, VALUE_STORAGE);
#undef VALUE_STORAGE
  void* ptr;
};

#define VALUE_AS_DISPATCH(Type, Name)             \
  template <>                                     \
  inline Type Value::as<Type>() const {           \
    CHECK_EQ(dtype_, k##Name) << "invalid dtype"; \
    return Name##values[0];                       \
  }
AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, VALUE_AS_DISPATCH);
#undef VALUE_AS_DISPATCH

#define VALUE_AS_VEC_DISPATCH(Type, Name)                                \
  template <>                                                            \
  inline const std::vector<Type>& Value::as_vec<Type>() const {          \
    CHECK_EQ(dtype_.scalar_type(), ScalarType::Name) << "invalid dtype"; \
    return Name##values;                                                 \
  }
AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, VALUE_AS_VEC_DISPATCH);
#undef VALUE_AS_VEC_DISPATCH

template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type mod_value(
    T lhs,
    T rhs) {
  return lhs % rhs;
}

template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
mod_value(T lhs, T rhs) {
  return std::fmod(lhs, rhs);
}

inline bool mod_value(bool lhs, bool rhs) {
  LOG(FATAL) << "Attempted modulus of bool";
  return false;
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
    stmt()->accept(this);
    eval_context_.clear();
    buffer_mapping_.clear();
    internal_buffers_.clear();
    USE_TRIGGER(simple_ir_eval_executed);
  }

  void bind(const BufferArg& buf, const CallArg& data) {
    if (!buf.isVar()) {
      buffer_mapping_[buf.var()] = data.data();
      return;
    }

    switch (buf.dtype().scalar_type()) {
#define TYPE_CASE(Type, Name)                     \
  case ScalarType::Name:                          \
    eval_context_[buf.var()] = data.Name##Data(); \
    break;
      AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, TYPE_CASE);
#undef TYPE_CASE
      default:
        LOG(FATAL) << "Unhandled dtype for argument " << buf.var()->name_hint()
                   << ": " << buf.dtype();
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

  TORCH_API void visit(const And* v) override {
    visit_binary_op(v);
  }
  TORCH_API void visit(const Or* v) override {
    visit_binary_op(v);
  }
  TORCH_API void visit(const Xor* v) override {
    visit_binary_op(v);
  }
  TORCH_API void visit(const Lshift* v) override {
    visit_binary_op(v);
  }
  TORCH_API void visit(const Rshift* v) override {
    visit_binary_op(v);
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
          if (option) {
            // Propagate NaNs
            if (is_floating_point(lhs.dtype().scalar_type()) &&
                is_floating_point(rhs.dtype().scalar_type())) {
              result_v[i] = lhs_v[i];
            } else if (std::isnan((float)rhs_v[i])) {
              result_v[i] = rhs_v[i];
            }
          } else {
            result_v[i] = lhs_v[i] > rhs_v[i] ? lhs_v[i] : rhs_v[i];
          }
          break;
        case IRNodeType::kMin:
          if (option) {
            // Propagate NaNs
            if (is_floating_point(lhs.dtype().scalar_type()) &&
                is_floating_point(rhs.dtype().scalar_type())) {
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

  Value bitwise_binary_op(
      const Value& lhs,
      const Value& rhs,
      IRNodeType op_type) {
    std::vector<int> lhs_v = lhs.as_vec<int>();
    std::vector<int> rhs_v = rhs.as_vec<int>();
    std::vector<int> result_v(lhs_v.size());
    for (size_t i = 0; i < lhs_v.size(); i++) {
      switch (op_type) {
        case IRNodeType::kAnd:
          result_v[i] = lhs_v[i] & rhs_v[i];
          break;
        case IRNodeType::kOr:
          result_v[i] = lhs_v[i] | rhs_v[i];
          break;
        case IRNodeType::kXor:
          result_v[i] = lhs_v[i] ^ rhs_v[i];
          break;
        case IRNodeType::kLshift:
          result_v[i] = lhs_v[i] << rhs_v[i];
          break;
        case IRNodeType::kRshift:
          result_v[i] = lhs_v[i] >> rhs_v[i];
          break;
        default:
          // TODO: change to a proper error report
          throw std::runtime_error("invalid operator type");
      }
    }
    return Value(result_v);
  }

  template <typename T, typename R>
  Value compare_select_op(
      const Value& lhs,
      const Value& rhs,
      const Value& retval1,
      const Value& retval2,
      CompareSelectOperation cmp_op) {
    std::vector<T> lhs_v = lhs.as_vec<T>();
    std::vector<T> rhs_v = rhs.as_vec<T>();
    std::vector<R> ret_val1_v = retval1.as_vec<R>();
    std::vector<R> ret_val2_v = retval2.as_vec<R>();
    std::vector<R> result_v(lhs_v.size());
    for (size_t i = 0; i < lhs_v.size(); i++) {
      switch (cmp_op) {
        case CompareSelectOperation::kEQ:
          result_v[i] = (lhs_v[i] == rhs_v[i]) ? ret_val1_v[i] : ret_val2_v[i];
          break;
        case CompareSelectOperation::kNE:
          result_v[i] = (lhs_v[i] != rhs_v[i]) ? ret_val1_v[i] : ret_val2_v[i];
          break;
        case CompareSelectOperation::kGT:
          result_v[i] = (lhs_v[i] > rhs_v[i]) ? ret_val1_v[i] : ret_val2_v[i];
          break;
        case CompareSelectOperation::kGE:
          result_v[i] = (lhs_v[i] >= rhs_v[i]) ? ret_val1_v[i] : ret_val2_v[i];
          break;
        case CompareSelectOperation::kLT:
          result_v[i] = (lhs_v[i] < rhs_v[i]) ? ret_val1_v[i] : ret_val2_v[i];
          break;
        case CompareSelectOperation::kLE:
          result_v[i] = (lhs_v[i] <= rhs_v[i]) ? ret_val1_v[i] : ret_val2_v[i];
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
    v->lhs()->accept(this);
    Value lhs_v = value_;
    v->rhs()->accept(this);
    Value rhs_v = value_;
    CHECK_EQ(lhs_v.dtype(), rhs_v.dtype());
    IRNodeType expr_type = v->expr_type();
    if (expr_type == IRNodeType::kAnd || expr_type == IRNodeType::kOr ||
        expr_type == IRNodeType::kXor || expr_type == IRNodeType::kLshift ||
        expr_type == IRNodeType::kRshift) {
      value_ = bitwise_binary_op(lhs_v, rhs_v, expr_type);
      return;
    }

    switch (lhs_v.dtype().scalar_type()) {
#define TYPE_CASE(Type, Name)                          \
  case ScalarType::Name:                               \
    value_ = binary_op<Type>(lhs_v, rhs_v, expr_type); \
    break;
      AT_FORALL_SCALAR_TYPES_AND(Half, TYPE_CASE);
#undef TYPE_CASE
      case ScalarType::Bool:
        value_ = binary_op<unsigned char>(lhs_v, rhs_v, expr_type);
        break;
      default:
        LOG(FATAL) << "invalid dtype: " << lhs_v.dtype();
    }
  }

  void visit_compare_select_op(
      const CompareSelect* v,
      CompareSelectOperation cmp_op) {
    v->lhs()->accept(this);
    Value lhs_v = value_;
    v->rhs()->accept(this);
    Value rhs_v = value_;
    v->ret_val1()->accept(this);
    Value ret_val1_v = value_;
    v->ret_val2()->accept(this);
    Value ret_val2_v = value_;

    CHECK_EQ(lhs_v.dtype(), rhs_v.dtype());
    CHECK_EQ(ret_val1_v.dtype(), ret_val2_v.dtype());

    switch (lhs_v.dtype().scalar_type()) {
#define TYPE_CASE(Type, Name)                          \
  case ScalarType::Name:                               \
    value_ = compare_select_op<Type, int>(             \
        lhs_v, rhs_v, ret_val1_v, ret_val2_v, cmp_op); \
    break;
      AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, TYPE_CASE);
#undef TYPE_CASE
      default:
        LOG(FATAL) << "invalid dtype: " << lhs_v.dtype();
    }
  }

#define IMM_VISIT(Type, Name)                         \
  TORCH_API void visit(const Name##Imm* v) override { \
    value_ = Value(v->value());                       \
  }
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_VISIT);
#undef IMM_VISIT

  TORCH_API void visit(const Let* v) override {
    const Var* var = dynamic_cast<const Var*>(v->var());
    CHECK(var != nullptr);
    v->value()->accept(this);
    Value value = value_;
    auto iter = eval_context_.find(var);
    // TODO: make the same value settable multiple times.
    CHECK(iter == eval_context_.end())
        << "var must not exist in the context before";
    eval_context_[var] = value_;

    v->body()->accept(this);

    eval_context_.erase(var);
  }

  TORCH_API void visit(const LetStmt* v) override {
    const Var* var = v->var();
    CHECK(var != nullptr);
    v->value()->accept(this);
    Value value = value_;
    auto iter = eval_context_.find(var);
    // TODO: make the same value settable multiple times.
    CHECK(iter == eval_context_.end())
        << "var must not exist in the context before";
    eval_context_[var] = value_;

    v->body()->accept(this);

    eval_context_.erase(var);
  }

  TORCH_API void visit(const Var* v) override {
    auto iter = eval_context_.find(v);
    CHECK(iter != eval_context_.end())
        << "var must be defined in the context before";
    value_ = iter->second;
  }

  template <typename SrcType, typename DstType>
  std::vector<DstType> castValues(const Dtype& src_dtype, const Value& v) {
    const std::vector<SrcType>& src_values = v.as_vec<SrcType>();
    std::vector<DstType> dst_values(src_values.size());
    for (int i = 0; i < src_dtype.lanes(); ++i) {
      dst_values[i] = static_cast<DstType>(src_values[i]);
    }
    return dst_values;
  }

  template <typename SrcType>
  void doCastFromSrc(
      const Dtype& src_dtype,
      const Dtype& dst_dtype,
      const Value& v) {
    switch (dst_dtype.scalar_type()) {
#define DST_TYPE_CASE(Type, Name)                                  \
  case ScalarType::Name:                                           \
    this->value_ = Value(castValues<SrcType, Type>(src_dtype, v)); \
    break;
      AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, DST_TYPE_CASE);
#undef DST_TYPE_CASE
      default:
        LOG(FATAL) << "Cast invalid dst type " << dst_dtype << "\n";
    }
  }

  TORCH_API void visit(const Cast* v) override {
    const Expr* src_value = v->src_value();
    src_value->accept(this);
    Dtype dst_dtype = v->dtype();
    Dtype src_dtype = src_value->dtype();
    CHECK_EQ(src_dtype.lanes(), dst_dtype.lanes());

    if (src_dtype != dst_dtype) {
      switch (src_dtype.scalar_type()) {
#define SRC_TYPE_CASE(Type, Name)                      \
  case ScalarType::Name:                               \
    doCastFromSrc<Type>(src_dtype, dst_dtype, value_); \
    break;
        AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, SRC_TYPE_CASE);
#undef SRC_TYPE_CASE
        default:
          LOG(FATAL) << "Cast invalid src type " << src_dtype << "\n";
      }
    }
  }

  TORCH_API void visit(const For* v) override {
    const Expr* var_node = v->var();
    v->start()->accept(this);
    int start = value_.as<int>();
    v->stop()->accept(this);
    int stop = value_.as<int>();
    auto iter = eval_context_.find(var_node);
    CHECK(iter == eval_context_.end())
        << "var in For must not exist in eval context";
    for (int i = start; i < stop; i++) {
      eval_context_[var_node] = Value(i);
      if (v->body()) {
        v->body()->accept(this);
      }
    }
    eval_context_.erase(var_node);
  }

  TORCH_API void visit(const Ramp* v) override {
    v->base()->accept(this);
    int base = value().as<int>();
    v->stride()->accept(this);
    int stride = value().as<int>();
    int lanes = v->lanes();

    std::vector<int> values(lanes);
    for (int i = 0; i < lanes; i++) {
      values[i] = base + i * stride;
    }

    value_ = Value(values);
  }

  TORCH_API void visit(const Broadcast* v) override {
    v->value()->accept(this);
    Value value = this->value();
    int lanes = v->lanes();
    switch (value.dtype().scalar_type()) {
#define TYPE_CASE(Type, Name)                     \
  case ScalarType::Name: {                        \
    std::vector<Type> v(lanes, value.as<Type>()); \
    value_ = Value(v);                            \
  } break;
      AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, TYPE_CASE);
#undef TYPE_CASE
      default:
        LOG(FATAL) << "invalid dtype: " << value.dtype();
    }
  }

  TORCH_API void visit(const IfThenElse* v) override {
    v->condition()->accept(this);
    if (value_.as<int>()) {
      v->true_value()->accept(this);
    } else {
      v->false_value()->accept(this);
    }
  }

  TORCH_API void visit(const Load* v) override {
    const Var* base_node = v->base_handle();
    auto iter = buffer_mapping_.find(base_node);
    CHECK(iter != buffer_mapping_.end())
        << "missing buffer binding: " << base_node->name_hint();
    void* ptr = iter->second;

    v->index()->accept(this);
    std::vector<int> index = value().as_vec<int>();
    v->mask()->accept(this);
    std::vector<int> mask = value().as_vec<int>();
    ScalarType v_sdtype = v->dtype().scalar_type();
    switch (v_sdtype) {
#define TYPE_CASE(Type, Name)                   \
  case ScalarType::Name: {                      \
    Type* ptr##Name = static_cast<Type*>(ptr);  \
    std::vector<Type> v(index.size());          \
    for (size_t i = 0; i < index.size(); i++) { \
      if (mask[i]) {                            \
        v[i] = ptr##Name[index[i]];             \
      }                                         \
    }                                           \
    value_ = Value(v);                          \
  } break;
      AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, TYPE_CASE);
#undef TYPE_CASE
      default:
        LOG(FATAL) << "Invalid dtype: " << v_sdtype;
    }
  }

  TORCH_API void visit(const Store* v) override {
    const Var* base_node = v->base_handle();
    auto iter = buffer_mapping_.find(base_node);
    CHECK(iter != buffer_mapping_.end());
    void* ptr = iter->second;

    v->index()->accept(this);
    std::vector<int> index = value().as_vec<int>();
    v->mask()->accept(this);
    std::vector<int> mask = value().as_vec<int>();
    CHECK_EQ(index.size(), mask.size());
    ScalarType v_sdtype = v->value()->dtype().scalar_type();

    switch (v_sdtype) {
#define TYPE_CASE(Type, Name)                               \
  case ScalarType::Name: {                                  \
    v->value()->accept(this);                               \
    std::vector<Type> value = this->value().as_vec<Type>(); \
    CHECK_EQ(index.size(), value.size());                   \
    Type* ptr##Name = static_cast<Type*>(ptr);              \
    for (size_t i = 0; i < index.size(); i++) {             \
      if (mask[i]) {                                        \
        ptr##Name[index[i]] = value[i];                     \
      }                                                     \
    }                                                       \
  } break;
      AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, TYPE_CASE);
#undef TYPE_CASE
      default:
        LOG(FATAL) << "Invalid dtype: " << v_sdtype;
    }
  }

  TORCH_API void visit(const BaseCallNode* v) override {
    LOG(FATAL) << "unsupported visit to BaseCallNode";
  }

  TORCH_API void visit(const Intrinsics* v) override {
    std::vector<Value> values(v->nparams());
    for (int i = 0; i < v->nparams(); i++) {
      v->param(i)->accept(this);
      values[i] = this->value();
    }
    std::vector<float> v1;
    if (values.size() >= 1ULL) {
      v1 = values[0].as_vec<float>();
    }
    std::vector<float> v2;
    if (values.size() >= 2ULL) {
      v2 = values[1].as_vec<float>();
      CHECK_EQ(v1.size(), v2.size()) << "mismatch vectorize sizes";
    }
    CHECK_LE(values.size(), 2ULL)
        << "no support for intrinsics for more than two operand yet";
    std::vector<float> result(v1.size(), -1);
    if (values.size() == 1ULL) {
      for (size_t i = 0; i < v1.size(); i++) {
        result[i] = compute_intrinsics(v->op_type(), v1[i]);
      }
    } else {
      for (size_t i = 0; i < v1.size(); i++) {
        result[i] = compute_intrinsics(v->op_type(), v1[i], v2[i]);
      }
    }
    value_ = Value(result);
  }

  void visit(const Allocate* v) override {
    const Var* buffer_var = v->buffer_var();
    std::vector<const Expr*> dims = v->dims();
    int total_byte_size = v->dtype().byte_size();
    for (size_t i = 0; i < dims.size(); i++) {
      dims[i]->accept(this);
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
    const Var* buffer_var = v->buffer_var();
    int count = internal_buffers_.erase(buffer_var);
    if (count == 0) {
      throw std::runtime_error(
          "Free a buffer that is not currently bound: " +
          buffer_var->name_hint());
    }
  }

  void visit(const Cond* v) override {
    v->condition()->accept(this);
    if (value().as<int>()) {
      if (v->true_stmt()) {
        v->true_stmt()->accept(this);
      }
    } else {
      if (v->false_stmt()) {
        v->false_stmt()->accept(this);
      }
    }
  }

  Value value() const {
    return value_;
  }

 private:
  static float compute_intrinsics(IntrinsicsOp op_type, float v) {
    switch (op_type) {
      case kSin:
        return std::sin(v);
      case kCos:
        return std::cos(v);
      case kTan:
        return std::tan(v);
      case kAsin:
        return std::asin(v);
      case kAcos:
        return std::acos(v);
      case kAtan:
        return std::atan(v);
      case kSinh:
        return std::sinh(v);
      case kCosh:
        return std::cosh(v);
      case kTanh:
        return std::tanh(v);
      case kExp:
        return std::exp(v);
      case kFabs:
        return std::fabs(v);
      case kExpm1:
        return std::expm1(v);
      case kLog:
        return std::log(v);
      case kLog2:
        return std::log2(v);
      case kLog10:
        return std::log10(v);
      case kLog1p:
        return std::log1p(v);
      case kErf:
        return std::erf(v);
      case kErfc:
        return std::erfc(v);
      case kSqrt:
        return std::sqrt(v);
      case kRsqrt:
        return 1.0f / std::sqrt(v);
      case kCeil:
        return std::ceil(v);
      case kFloor:
        return std::floor(v);
      case kRound:
        return std::round(v);
      case kTrunc:
        return std::trunc(v);
      case kLgamma:
        return std::lgamma(v);
      case kFrac:
        float intpart;
        return std::modf(v, &intpart);
      default:
        throw std::runtime_error("invalid op_type: " + std::to_string(op_type));
    }
  }

  static float compute_intrinsics(IntrinsicsOp op_type, float v1, float v2) {
    switch (op_type) {
      case kPow:
        return std::pow(v1, v2);
      case kFmod:
        return std::fmod(v1, v2);
      case kRemainder:
        return std::remainderf(v1, v2);
      case kAtan2:
        return std::atan2(v1, v2);
      default:
        throw std::runtime_error("nvalid op_type: " + std::to_string(op_type));
    }
  }

  Value value_;
  std::unordered_map<const Expr*, Value> eval_context_;
  std::unordered_map<const Var*, void*> buffer_mapping_;
  std::unordered_map<const Var*, std::unique_ptr<std::vector<int>>>
      internal_buffers_;
};

using VarMapping = std::vector<std::pair<ExprHandle, ExprHandle>>;

class VarSubMutator : public IRMutator {
 public:
  VarSubMutator(const VarMapping& var_mapping) {
    for (const auto& entry : var_mapping) {
      const ExprHandle& key = entry.first;
      const ExprHandle& value = entry.second;
      const Var* key_var = key.AsNode<Var>();
      CHECK(key_var != nullptr);
      var_mapping_[key_var] = value;
    }
  }

  const Expr* mutate(const Var* var) override {
    auto iter = var_mapping_.find(var);
    if (iter == var_mapping_.end()) {
      return const_cast<Var*>(var);
    }
    return iter->second.node();
  }

 private:
  std::unordered_map<const Var*, ExprHandle> var_mapping_;
};

template <class CodeGenType>
class ExprEval {
 public:
  using BufferArg = CodeGen::BufferArg;
  using CallArg = CodeGen::CallArg;

  template <typename... Ts>
  ExprEval(const ExprHandle& expr, Ts... ts)
      : ExprEval(expr, {BufferArg(ts)...}) {}

  ExprEval(const ExprHandle& expr, const std::vector<BufferArg>& buffer_args)
      : dtype_(expr.dtype()) {
    std::vector<BufferArg> buffer_args_extended = buffer_args;
    Buffer ret_buf("ret_val", dtype_, {1});
    Stmt* store_stmt = Store::make(VarHandle(ret_buf.data()), 0, expr);
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
    switch (dtype_.scalar_type()) {
#define TYPE_CASE(Type, Name)                           \
  case ScalarType::Name: {                              \
    std::vector<Type> ret_val_arg(1);                   \
    call_args_extended.push_back(CallArg(ret_val_arg)); \
    codegen_->call(call_args_extended);                 \
    ret_value_ = Value(ret_val_arg[0]);                 \
  } break;
      AT_FORALL_SCALAR_TYPES_AND(Half, TYPE_CASE);
#undef TYPE_CASE
      case ScalarType::Bool: {
        std::vector<unsigned char> ret_val_arg(1);
        call_args_extended.push_back(CallArg(ret_val_arg.data()));
        codegen_->call(call_args_extended);
        ret_value_ = Value((bool)ret_val_arg[0]);
      } break;
      default:
        LOG(FATAL) << "Invalid Dtype " << dtype_ << "\n";
    }
  }

  template <typename T, typename... Ts>
  T value(Ts... ts) {
    call(std::forward<Ts>(ts)...);
    return ret_value_.as<T>();
  }

  Dtype dtype() {
    return dtype_;
  }

 private:
  Dtype dtype_;
  std::unique_ptr<CodeGenType> codegen_;
  Value ret_value_;
};

inline ExprHandle Substitute(ExprHandle* expr, const VarMapping& var_mapping) {
  VarSubMutator var_sub(var_mapping);
  return ExprHandle(expr->node()->accept_mutator(&var_sub));
}

inline Stmt* Substitute(Stmt* stmt, const VarMapping& var_mapping) {
  VarSubMutator var_sub(var_mapping);
  return stmt->accept_mutator(&var_sub);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
