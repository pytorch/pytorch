#include <torch/csrc/jit/tensorexpr/eval.h>

#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>

namespace torch {
namespace jit {
namespace tensorexpr {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_TRIGGER(simple_ir_eval_executed);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
RegisterCodeGen<SimpleIREvaluator> ir_eval_codegen_reg("simple_ir_eval");

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
  throw std::runtime_error("Attempted modulus of bool");
}

template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type div_value(
    T lhs,
    T rhs) {
  TORCH_CHECK(rhs != 0, "Division by zero");
  return lhs / rhs;
}

template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value, T>::
    type __ubsan_ignore_float_divide_by_zero__
    div_value(T lhs, T rhs) {
  return lhs / rhs;
}

inline bool div_value(bool lhs, bool rhs) {
  LOG(FATAL) << "Attempted division of bool";
  return false;
}

inline c10::Half div_value(c10::Half lhs, c10::Half rhs) {
  return lhs / rhs;
}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class SimpleIREvaluatorImpl : public IRVisitor {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  SimpleIREvaluatorImpl() = default;

  ~SimpleIREvaluatorImpl() override = default;

  void bindBuf(const Buf* buf, void* ptr) {
    buffer_mapping_[buf] = ptr;
  }
  void bindVar(const Var* var, const Value& val) {
    eval_context_[var] = val;
  }

  Value evaluateExpr(const Expr* e) {
    e->accept(this);
    return value_;
  }

  Value value() const {
    return value_;
  }

  void clear() {
    eval_context_.clear();
    buffer_mapping_.clear();
    internal_buffers_.clear();
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
  typename std::enable_if_t<std::is_floating_point<T>::value, T> max_value(
      T a,
      T b) {
    return std::isnan(a) ? a : (std::isnan(b) ? b : (a < b ? b : a));
  }

  template <typename T>
  typename std::enable_if_t<!std::is_floating_point<T>::value, T> max_value(
      T a,
      T b) {
    return a < b ? b : a;
  }

  template <typename T>
  typename std::enable_if_t<std::is_floating_point<T>::value, T> min_value(
      T a,
      T b) {
    return std::isnan(a) ? a : (std::isnan(b) ? b : (a < b ? a : b));
  }

  template <typename T>
  typename std::enable_if_t<!std::is_floating_point<T>::value, T> min_value(
      T a,
      T b) {
    return a < b ? a : b;
  }

  template <typename T>
  Value binary_op(const Value& lhs, const Value& rhs, IRNodeType op_type) {
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
          result_v[i] = div_value(lhs_v[i], rhs_v[i]);
          break;
        case IRNodeType::kMod:
          result_v[i] = mod_value(lhs_v[i], rhs_v[i]);
          break;
        case IRNodeType::kMax:
          result_v[i] = max_value(lhs_v[i], rhs_v[i]);
          break;
        case IRNodeType::kMin:
          result_v[i] = min_value(lhs_v[i], rhs_v[i]);
          break;
        default:
          // TODO: change to a proper error report
          throw std::runtime_error("invalid operator type");
      }
    }
    return Value(result_v);
  }

  template <typename T>
  Value bitwise_binary_op(
      const Value& lhs,
      const Value& rhs,
      IRNodeType op_type) {
    std::vector<T> lhs_v = lhs.as_vec<T>();
    std::vector<T> rhs_v = rhs.as_vec<T>();
    std::vector<T> result_v(lhs_v.size());
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
        default:
          // TODO: change to a proper error report
          throw std::runtime_error("invalid operator type");
      }
    }
    return Value(result_v);
  }

  template <typename T>
  Value shift_binary_op(
      const Value& lhs,
      const Value& rhs,
      IRNodeType op_type) {
    std::vector<T> lhs_v = lhs.as_vec<T>();
    std::vector<T> rhs_v = rhs.as_vec<T>();
    std::vector<T> result_v(lhs_v.size());
    for (size_t i = 0; i < lhs_v.size(); i++) {
      switch (op_type) {
        case IRNodeType::kLshift: {
          typename std::make_unsigned<T>::type a =
              static_cast<typename std::make_unsigned<T>::type>(lhs_v[i]);
          result_v[i] = a << rhs_v[i];
          break;
        }
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
    if (lhs_v.dtype() != rhs_v.dtype()) {
      throw malformed_input("bad dtype in binary op", v);
    }

    IRNodeType expr_type = v->expr_type();
    if (expr_type == IRNodeType::kAnd || expr_type == IRNodeType::kOr ||
        expr_type == IRNodeType::kXor) {
      switch (lhs_v.dtype().scalar_type()) {
#define TYPE_CASE(Type, Name)                                  \
  case ScalarType::Name:                                       \
    value_ = bitwise_binary_op<Type>(lhs_v, rhs_v, expr_type); \
    break;
        AT_FORALL_INT_TYPES(TYPE_CASE);
#undef TYPE_CASE
        case ScalarType::Bool:
          value_ = bitwise_binary_op<unsigned char>(lhs_v, rhs_v, expr_type);
          break;
        default:
          throw unsupported_dtype();
      }
      return;
    }

    if (expr_type == IRNodeType::kLshift || expr_type == IRNodeType::kRshift) {
      switch (lhs_v.dtype().scalar_type()) {
#define TYPE_CASE(Type, Name)                                \
  case ScalarType::Name:                                     \
    value_ = shift_binary_op<Type>(lhs_v, rhs_v, expr_type); \
    break;
        AT_FORALL_INT_TYPES(TYPE_CASE);
#undef TYPE_CASE
        case ScalarType::Bool:
          value_ = shift_binary_op<unsigned char>(lhs_v, rhs_v, expr_type);
          break;
        default:
          throw unsupported_dtype();
      }
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
        throw unsupported_dtype();
    }
  }

  template <typename T>
  Value compare_select_op_helper(
      const Value& lhs,
      const Value& rhs,
      const Value& retval1,
      const Value& retval2,
      CompareSelectOperation cmp_op) {
    Value value;
    switch (retval1.dtype().scalar_type()) {
#define TYPE_CASE(Type, Name)                                               \
  case ScalarType::Name:                                                    \
    value = compare_select_op<T, Type>(lhs, rhs, retval1, retval2, cmp_op); \
    break;
      AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, TYPE_CASE);
#undef TYPE_CASE
      default:
        throw unsupported_dtype();
    }

    return value;
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

    if (lhs_v.dtype() != rhs_v.dtype() ||
        ret_val1_v.dtype() != ret_val2_v.dtype()) {
      throw malformed_input("bad dtype in CompareSelect", v);
    }

    switch (lhs_v.dtype().scalar_type()) {
#define TYPE_CASE(Type, Name)                          \
  case ScalarType::Name:                               \
    value_ = compare_select_op_helper<Type>(           \
        lhs_v, rhs_v, ret_val1_v, ret_val2_v, cmp_op); \
    break;
      AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, TYPE_CASE);
#undef TYPE_CASE
      default:
        throw unsupported_dtype();
    }
  }

#define IMM_VISIT(Type, Name)                         \
  TORCH_API void visit(const Name##Imm* v) override { \
    value_ = Value(v->value());                       \
  }
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_VISIT);
#undef IMM_VISIT

  TORCH_API void visit(const Block* v) override {
    const Block* last = scope_;
    scope_ = v;
    for (Stmt* s : v->stmts()) {
      s->accept(this);
    }

    auto it = var_by_scope_.find(v);
    if (it != var_by_scope_.end()) {
      for (const Expr* v : it->second) {
        eval_context_.erase(v);
      }
      var_by_scope_.erase(it);
    }

    scope_ = last;
  }

  TORCH_API void visit(const Var* v) override {
    auto iter = eval_context_.find(v);
    if (iter == eval_context_.end()) {
      throw malformed_input("could not find Var in context", v);
    }

    value_ = iter->second;
  }

  template <typename SrcType, typename DstType>
  std::vector<DstType> castValues(const Dtype& src_dtype, const Value& v) {
    const std::vector<SrcType>& src_values = v.as_vec<SrcType>();
    std::vector<DstType> dst_values(src_values.size());
    for (int i = 0; i < src_dtype.lanes(); ++i) {
      // NOLINTNEXTLINE(bugprone-signed-char-misuse)
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
        throw unsupported_dtype();
    }
  }

  TORCH_API void visit(const Cast* v) override {
    const Expr* src_value = v->src_value();
    src_value->accept(this);
    Dtype dst_dtype = v->dtype();
    Dtype src_dtype = src_value->dtype();
    if (src_dtype.lanes() != dst_dtype.lanes()) {
      throw malformed_input("lane mismatch in Cast", v);
    }

    if (src_dtype != dst_dtype) {
      switch (src_dtype.scalar_type()) {
#define SRC_TYPE_CASE(Type, Name)                      \
  case ScalarType::Name:                               \
    doCastFromSrc<Type>(src_dtype, dst_dtype, value_); \
    break;
        AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, SRC_TYPE_CASE);
#undef SRC_TYPE_CASE
        default:
          throw unsupported_dtype();
      }
    }
  }

  template <typename SrcType, typename DstType>
  std::vector<DstType> bitcastValues(const Dtype& src_dtype, const Value& v) {
    const std::vector<SrcType>& src_values = v.as_vec<SrcType>();
    std::vector<DstType> dst_values(src_values.size());
    for (int i = 0; i < src_dtype.lanes(); ++i) {
      dst_values[i] = raw_bitcast<DstType>(src_values[i]);
    }
    return dst_values;
  }

  template <typename SrcType>
  void doBitCastFromSrc(
      const Dtype& src_dtype,
      const Dtype& dst_dtype,
      const Value& v) {
    switch (dst_dtype.scalar_type()) {
#define DST_TYPE_CASE(Type, Name)                                     \
  case ScalarType::Name:                                              \
    this->value_ = Value(bitcastValues<SrcType, Type>(src_dtype, v)); \
    break;
      // bool/half not supported
      AT_FORALL_SCALAR_TYPES(DST_TYPE_CASE);
#undef DST_TYPE_CASE
      default:
        throw unsupported_dtype();
    }
  }

  TORCH_API void visit(const BitCast* v) override {
    const Expr* src_value = v->src_value();
    src_value->accept(this);
    Dtype dst_dtype = v->dtype();
    Dtype src_dtype = src_value->dtype();
    if (src_dtype.byte_size() != dst_dtype.byte_size()) {
      throw malformed_input("lane mismatch in Cast", v);
    }
    if (src_dtype != dst_dtype) {
      switch (src_dtype.scalar_type()) {
#define SRC_TYPE_CASE(Type, Name)                         \
  case ScalarType::Name:                                  \
    doBitCastFromSrc<Type>(src_dtype, dst_dtype, value_); \
    break;
        // bool/half not supported
        AT_FORALL_SCALAR_TYPES(SRC_TYPE_CASE);
#undef SRC_TYPE_CASE
        default:
          throw unsupported_dtype();
      }
    }
  }

  TORCH_API void visit(const For* v) override {
    const Expr* var_node = v->var();
    v->start()->accept(this);
    int start = value_.as<int>();
    v->stop()->accept(this);
    int stop = value_.as<int>();
    if (eval_context_.count(var_node)) {
      throw malformed_input("could not find var_node in For context", v);
    }

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
        throw unsupported_dtype();
    }
  }

  TORCH_API void visit(const IfThenElse* v) override {
    v->condition()->accept(this);
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    bool cond_v;
    switch (value_.dtype().scalar_type()) {
#define TYPE_CASE(Type, Name)   \
  case ScalarType::Name: {      \
    cond_v = value_.as<Type>(); \
  } break;
      AT_FORALL_SCALAR_TYPES_AND(Bool, TYPE_CASE);
#undef TYPE_CASE
      case ScalarType::Half:
        throw unsupported_dtype("IfThenElse condition can't have Half dtype");
      default:
        throw unsupported_dtype();
    }

    if (cond_v) {
      v->true_value()->accept(this);
    } else {
      v->false_value()->accept(this);
    }
  }

  TORCH_API void visit(const Load* v) override {
    auto iter = buffer_mapping_.find(v->buf());
    if (iter == buffer_mapping_.end()) {
      throw malformed_input("could not find base node in Load", v);
    }
    void* ptr = iter->second;

    const Expr* flat_idx = flatten_index(v->buf()->dims(), v->indices());
    flat_idx->accept(this);
    std::vector<int> index = value().as_vec<int>();
    ScalarType v_sdtype = v->dtype().scalar_type();
    switch (v_sdtype) {
#define TYPE_CASE(Type, Name)                   \
  case ScalarType::Name: {                      \
    Type* ptr##Name = static_cast<Type*>(ptr);  \
    std::vector<Type> v(index.size());          \
    for (size_t i = 0; i < index.size(); i++) { \
      v[i] = ptr##Name[index[i]];               \
    }                                           \
    value_ = Value(v);                          \
  } break;
      AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, TYPE_CASE);
#undef TYPE_CASE
      default:
        throw unsupported_dtype();
    }
  }

  TORCH_API void visit(const Store* v) override {
    auto iter = buffer_mapping_.find(v->buf());
    if (iter == buffer_mapping_.end()) {
      throw malformed_input("could not find base node in Store", v);
    }

    void* ptr = iter->second;

    const Expr* flat_idx = flatten_index(v->buf()->dims(), v->indices());
    flat_idx->accept(this);
    std::vector<int> index = value().as_vec<int>();
    ScalarType v_sdtype = v->value()->dtype().scalar_type();

    switch (v_sdtype) {
#define TYPE_CASE(Type, Name)                                   \
  case ScalarType::Name: {                                      \
    v->value()->accept(this);                                   \
    std::vector<Type> value = this->value().as_vec<Type>();     \
    if (index.size() != value.size()) {                         \
      throw malformed_input("value size mismatch in Store", v); \
    }                                                           \
    Type* ptr##Name = static_cast<Type*>(ptr);                  \
    for (size_t i = 0; i < index.size(); i++) {                 \
      ptr##Name[index[i]] = value[i];                           \
    }                                                           \
  } break;
      AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, TYPE_CASE);
#undef TYPE_CASE
      default:
        throw unsupported_dtype();
    }
  }

  void visit(const ExternalCall* v) override {
    auto& func_registry = getNNCFunctionRegistry();
    if (!func_registry.count(v->func_name())) {
      throw unimplemented_lowering(v);
    }

    std::vector<const Buf*> bufs(v->buf_args());
    bufs.insert(bufs.begin(), v->buf());

    std::vector<void*> buf_ptrs;
    std::vector<int64_t> buf_ranks;
    std::vector<int64_t> buf_dims;
    std::vector<int8_t> buf_dtypes;
    std::vector<int64_t> extra_args;

    for (const Buf* b : bufs) {
      auto iter = buffer_mapping_.find(b);
      if (iter == buffer_mapping_.end()) {
        throw malformed_input("could not find buf", v);
      }

      buf_ptrs.push_back(iter->second);
      buf_ranks.push_back(b->dims().size());
      buf_dtypes.push_back((int8_t)b->dtype().scalar_type());
      for (const Expr* dim_expr : b->dims()) {
        dim_expr->accept(this);
        buf_dims.push_back(value().as<int>());
      }
    }
    for (const Expr* a : v->args()) {
      a->accept(this);
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int64_t val;
      if (value().dtype() == kLong) {
        val = value().as<int64_t>();
      } else if (value().dtype() == kInt) {
        val = value().as<int>();
      } else {
        throw malformed_input(
            "extra_args in ExternalCalls must have int64 dtype", v);
      }
      extra_args.push_back(val);
    }

    auto fn_ptr = func_registry.at(v->func_name());
    (*fn_ptr)(
        bufs.size(),
        buf_ptrs.data(),
        buf_ranks.data(),
        buf_dims.data(),
        buf_dtypes.data(),
        extra_args.size(),
        extra_args.data());
  }

  template <typename TReturn, typename TInput>
  void visit_intrinsics_helper(const Intrinsics* v) {
    std::vector<Value> values(v->nparams());
    for (int i = 0; i < v->nparams(); i++) {
      v->param(i)->accept(this);
      values[i] = this->value();
    }
    std::vector<TInput> v1;
    if (values.size() >= 1ULL) {
      v1 = values[0].as_vec<TInput>();
    }
    std::vector<TInput> v2;
    if (values.size() >= 2ULL) {
      v2 = values[1].as_vec<TInput>();
      if (v1.size() != v2.size()) {
        throw malformed_input("value size mismatch in Intrinsics", v);
      }
    }

    if (values.size() > 2) {
      throw unimplemented_lowering(v);
    }

    std::vector<TReturn> result(v1.size(), -1);
    if (values.size() == 1ULL) {
      for (size_t i = 0; i < v1.size(); i++) {
        result[i] = compute_intrinsics<TReturn>(v->op_type(), v1[i]);
      }
    } else {
      for (size_t i = 0; i < v1.size(); i++) {
        result[i] = compute_intrinsics<TReturn>(v->op_type(), v1[i], v2[i]);
      }
    }
    value_ = Value(result);
  }

  TORCH_API void visit(const Intrinsics* v) override {
    auto ty = v->dtype().scalar_type();
    if (v->op_type() == kIsNan) {
      auto inp_dtype = v->params().at(0)->dtype().scalar_type();
      if (inp_dtype == ScalarType::Float) {
        visit_intrinsics_helper<int, float>(v);
      } else if (inp_dtype == ScalarType::Double) {
        visit_intrinsics_helper<int, double>(v);
      } else if (inp_dtype == ScalarType::Half) {
        throw unsupported_dtype(); // TODO
      }
    } else {
      switch (ty) {
#define TYPE_CASE(Type, Name)               \
  case ScalarType::Name:                    \
    visit_intrinsics_helper<Type, Type>(v); \
    break;
        AT_FORALL_SCALAR_TYPES(TYPE_CASE);
#undef TYPE_CASE
        default:
          throw unsupported_dtype();
      }
    }
  }

  void visit(const Allocate* v) override {
    const Buf* b = v->buf();
    std::vector<const Expr*> dims = b->dims();
    int total_byte_size = b->dtype().byte_size();
    for (auto& dim : dims) {
      dim->accept(this);
      total_byte_size *= value_.as<int>();
    }
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    int int_count = (total_byte_size + sizeof(int) - 1) / sizeof(int);
    std::unique_ptr<std::vector<int>> buffer(new std::vector<int>(int_count));
    auto iter = buffer_mapping_.find(b);
    if (iter != buffer_mapping_.end() && iter->second != nullptr) {
      throw std::runtime_error(
          "Allocate a buffer that has already been allocated: " +
          v->buffer_var()->name_hint());
    }
    buffer_mapping_[b] = buffer->data();
    internal_buffers_.insert(std::make_pair(b, std::move(buffer)));
  }

  void visit(const Free* v) override {
    const Buf* b = v->buf();
    int count = internal_buffers_.erase(b);
    if (count == 0) {
      throw std::runtime_error(
          "Free a buffer that is not currently bound: " +
          v->buffer_var()->name_hint());
    }
    buffer_mapping_.erase(b);
  }

  void visit(const Let* v) override {
    var_by_scope_[scope_].push_back(v->var());
    bindVar(v->var(), evaluateExpr(v->value()));
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

 private:
  template <
      typename TReturn,
      typename TInput,
      typename std::enable_if<std::is_floating_point<TInput>::value, int>::
          type = 0>
  static TReturn compute_intrinsics(IntrinsicsOp op_type, TInput v) {
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
      case kAbs:
        return std::abs(v);
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
      case kRsqrt: {
        auto rsqrt = [](TInput v) __ubsan_ignore_float_divide_by_zero__ {
          return 1.0f / std::sqrt(v);
        };
        return rsqrt(v);
      }
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
        TInput intpart;
        return std::modf(v, &intpart);
      case kIsNan:
        return std::isnan(v);
      default:
        throw std::runtime_error("Invalid op_type: " + c10::to_string(op_type));
    }
  }

  template <
      typename TReturn,
      typename TInput,
      typename std::enable_if<std::is_integral<TInput>::value, int>::type = 0>
  static TReturn compute_intrinsics(IntrinsicsOp op_type, TInput v) {
    switch (op_type) {
      case kAbs: {
        // internal tool complains about calling `abs` on unsigned, the
        // following makes the tool happy
        using X =
            std::conditional_t<std::is_unsigned<TInput>::value, int, TInput>;
        return std::is_unsigned<TInput>::value ? v
                                               : std::abs(static_cast<X>(v));
      }
      default:
        throw std::runtime_error(
            "Invalid integral op_type: " + c10::to_string(op_type));
    }
  }

  // specialization for float -> int ops (just kIsNan currently)
  int compute_intrinsics(IntrinsicsOp op_type, float v) {
    switch (op_type) {
      case kIsNan:
        return std::isnan(v);
      default:
        throw std::runtime_error("Invalid op_type: " + c10::to_string(op_type));
    }
  }

  template <typename TReturn, typename TInput>
  TReturn compute_intrinsics(IntrinsicsOp op_type, TInput v1, TInput v2) {
    switch (op_type) {
      case kPow:
        return std::pow(v1, v2);
      case kFmod:
        return std::fmod(v1, v2);
      case kRemainder:
        return std::remainder(v1, v2);
      case kAtan2:
        return std::atan2(v1, v2);
      default:
        throw std::runtime_error("Invalid op_type: " + c10::to_string(op_type));
    }
  }

  Value value_;
  const Block* scope_;
  std::unordered_map<const Expr*, Value> eval_context_;
  std::unordered_map<const Block*, std::vector<const Expr*>> var_by_scope_;
  std::unordered_map<const Buf*, void*> buffer_mapping_;
  std::unordered_map<const Buf*, std::unique_ptr<std::vector<int>>>
      internal_buffers_;
};

SimpleIREvaluator::SimpleIREvaluator(
    Stmt* stmt,
    const std::vector<BufferArg>& buffer_args,
    at::Device device,
    const std::string& kernel_func_name)
    : CodeGen(stmt, buffer_args, device, kernel_func_name) {
  impl_ = std::make_unique<SimpleIREvaluatorImpl>();
  expand_intrinsics();
}

SimpleIREvaluator::~SimpleIREvaluator() = default;

void SimpleIREvaluator::call(const std::vector<CallArg>& args) {
  std::vector<void*> raw_args(args.size());
  for (size_t i = 0; i < args.size(); i++) {
    auto const& bufferArg = buffer_args()[i];
    auto const& callArg = args[i];
    raw_args[i] = argToPtr(bufferArg, callArg);
  }
  call_raw(raw_args);
}

void SimpleIREvaluator::call_raw(const std::vector<void*>& args) {
  if (args.size() != buffer_args().size()) {
    throw malformed_input("bad args in IREvaluator call");
  }
  for (size_t i = 0; i < args.size(); i++) {
    bindArg(buffer_args()[i], args[i]);
  }
  stmt()->accept(&*impl_);
  impl_->clear();
  USE_TRIGGER(simple_ir_eval_executed);
}

void SimpleIREvaluator::bindArg(const BufferArg& bufArg, void* data) {
  if (!bufArg.isVar()) {
    impl_->bindBuf(bufArg.buf(), data);
    return;
  }

  switch (bufArg.dtype().scalar_type()) {
#define TYPE_CASE(Type, Name)                 \
  case ScalarType::Name: {                    \
    Type typed_data;                          \
    memcpy(&typed_data, data, sizeof(Type));  \
    impl_->bindVar(bufArg.var(), typed_data); \
    break;                                    \
  }
    AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, TYPE_CASE);
#undef TYPE_CASE
    default:
      throw unsupported_dtype();
  }
}

void SimpleIREvaluator::bindVar(const Var* v, const Expr* e) {
  impl_->bindVar(v, impl_->evaluateExpr(e));
}

Value SimpleIREvaluator::value() const {
  return impl_->value();
}
} // namespace tensorexpr
} // namespace jit
} // namespace torch
