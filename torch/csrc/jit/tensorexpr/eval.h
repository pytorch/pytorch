#pragma once

#include <cmath>
#include <cstring>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <c10/macros/Macros.h>
#include <c10/util/Logging.h>
#include <c10/util/string_utils.h>
#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/exceptions.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/csrc/jit/tensorexpr/types.h>
#include <torch/csrc/jit/tensorexpr/var_substitutor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

class InterpValue {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  InterpValue() : dtype_(kInt) {
    Intvalues.push_back(0);
  }

  template <typename T>
  InterpValue(Dtype dtype, T v) : dtype_(dtype) {
#define TYPE_CASE(Type, Name)  \
  if (dtype == k##Name) {      \
    Name##values.push_back(v); \
    return;                    \
  }
    AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TYPE_CASE);
#undef TYPE_CASE
    throw unsupported_dtype();
  }

#define VALUE_CTOR(Type, Name)            \
  InterpValue(Type v) : dtype_(k##Name) { \
    Name##values.push_back(v);            \
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, VALUE_CTOR);
#undef VALUE_CTOR

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  explicit InterpValue(c10::quint8 v) : dtype_(kQUInt8) {
    QUInt8values.emplace_back(v.val_);
  }

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  explicit InterpValue(c10::qint8 v) : dtype_(kQInt8) {
    QInt8values.emplace_back(v.val_);
  }

#define VALUE_VEC_CTOR(Type, Name)        \
  InterpValue(const std::vector<Type>& v) \
      : dtype_(Dtype(k##Name, v.size())), Name##values(v) {}
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, VALUE_VEC_CTOR);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  VALUE_VEC_CTOR(c10::quint8, QUInt8)
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  VALUE_VEC_CTOR(c10::qint8, QInt8)
#undef VALUE_VEC_CTOR

  template <typename T>
  T as() const;

  template <typename T>
  const std::vector<T>& as_vec() const;

  int64_t intValue() const;

  Dtype dtype() const {
    return dtype_;
  }

 private:
  Dtype dtype_;

#define VALUE_STORAGE(Type, Name) std::vector<Type> Name##values;
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, VALUE_STORAGE);
  VALUE_STORAGE(c10::qint8, QInt8);
  VALUE_STORAGE(c10::quint8, QUInt8);
#undef VALUE_STORAGE
  void* ptr;
};

#define VALUE_AS_DISPATCH(Type, Name)         \
  template <>                                 \
  inline Type InterpValue::as<Type>() const { \
    if (dtype_ != k##Name) {                  \
      throw unsupported_dtype();              \
    }                                         \
    return Name##values[0];                   \
  }
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, VALUE_AS_DISPATCH);
VALUE_AS_DISPATCH(c10::quint8, QUInt8);
VALUE_AS_DISPATCH(c10::qint8, QInt8);
#undef VALUE_AS_DISPATCH

#define VALUE_AS_VEC_DISPATCH(Type, Name)                             \
  template <>                                                         \
  inline const std::vector<Type>& InterpValue::as_vec<Type>() const { \
    if (dtype_.scalar_type() != ScalarType::Name) {                   \
      throw unsupported_dtype();                                      \
    }                                                                 \
    return Name##values;                                              \
  }
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, VALUE_AS_VEC_DISPATCH);
VALUE_AS_VEC_DISPATCH(c10::quint8, QUInt8);
VALUE_AS_VEC_DISPATCH(c10::qint8, QInt8);
#undef VALUE_AS_VEC_DISPATCH

template <typename Type>
auto underlyingValue(Type x) {
  return x;
}

template <>
inline auto underlyingValue<c10::quint8>(c10::quint8 x) {
  return x.val_;
}

template <>
inline auto underlyingValue<c10::qint8>(c10::qint8 x) {
  return x.val_;
}

template <typename To, typename From>
To raw_bitcast(const From& src) {
  TORCH_CHECK(sizeof(To) == sizeof(From), "Invalid bitcast invocation");
  To storage;
  std::memcpy(&storage, &src, sizeof(To));
  return reinterpret_cast<To&>(storage);
}

class SimpleIREvaluatorImpl;
class TORCH_API SimpleIREvaluator : public CodeGen {
 public:
  SimpleIREvaluator(
      StmtPtr stmt,
      const std::vector<BufferArg>& buffer_args,
      at::Device device = at::kCPU,
      const std::string& kernel_func_name = "func");

  ~SimpleIREvaluator() override;

  void call(const std::vector<CallArg>& args) override;
  void call_raw(const std::vector<void*>& args) override;

  template <typename... Ts>
  void operator()(const Ts&... ts) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<CallArg> args({CallArg(ts)...});
    call(args);
  }

  void bindVar(VarPtr v, ExprPtr e);
  InterpValue value() const;

 private:
  void bindArg(const BufferArg& buf, void* data);
  void expand_intrinsics() {
    GenericIntrinsicsExpander intrinsics_expander;
    apply_mutator(&intrinsics_expander);
  }

  std::unique_ptr<SimpleIREvaluatorImpl> impl_;
};

template <class CodeGenType>
class ExprEval {
 public:
  using BufferArg = CodeGen::BufferArg;
  using CallArg = CodeGen::CallArg;

  template <typename... Ts>
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  ExprEval(const ExprHandle& expr, Ts... ts)
      : ExprEval(expr, {BufferArg(ts)...}) {}

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  ExprEval(const ExprHandle& expr, const std::vector<BufferArg>& buffer_args)
      : dtype_(expr.dtype()) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<BufferArg> buffer_args_extended = buffer_args;
    BufHandle ret_buf("ret_val", {1}, dtype_);
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<ExprHandle> indices;
    ExprHandle zero = IntImm::make(0);
    for (size_t i = 0; i < ret_buf.ndim(); i++) {
      indices.push_back(zero);
    }
    StmtPtr store_stmt = Store::make(ret_buf, indices, expr);
    buffer_args_extended.emplace_back(ret_buf);
    codegen_.reset(new CodeGenType(store_stmt, buffer_args_extended));
  }

  template <typename... Ts>
  void operator()(Ts... ts) {
    call(ts...);
  }

  void operator()(const std::vector<CallArg>& call_args) {
    call(call_args);
  }

  void bindVar(VarPtr v, ExprPtr e) {
    codegen_->bindVar(v, e);
  }

  void bindVar(const VarHandle& v, const ExprHandle& e) {
    codegen_->bindVar(v.node(), e.node());
  }

  template <typename... Ts>
  void call(Ts... ts) {
    call({CallArg(ts)...});
  }

  void call(const std::vector<CallArg>& call_args) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<CallArg> call_args_extended = call_args;
    switch (dtype_.scalar_type()) {
#define TYPE_CASE(Type, Name)                           \
  case ScalarType::Name: {                              \
    std::vector<Type> ret_val_arg(1);                   \
    call_args_extended.push_back(CallArg(ret_val_arg)); \
    codegen_->call(call_args_extended);                 \
    ret_value_ = InterpValue(ret_val_arg[0]);           \
  } break;
      // NOLINTNEXTLINE(modernize-use-emplace)
      AT_FORALL_SCALAR_TYPES_AND2(Half, BFloat16, TYPE_CASE);
      // NOLINTNEXTLINE(modernize-use-emplace)
      TYPE_CASE(c10::quint8, QUInt8);
      // NOLINTNEXTLINE(modernize-use-emplace)
      TYPE_CASE(c10::qint8, QInt8);
#undef TYPE_CASE
      case ScalarType::Bool: {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        std::vector<unsigned char> ret_val_arg(1);
        call_args_extended.emplace_back(ret_val_arg.data());
        codegen_->call(call_args_extended);
        ret_value_ = InterpValue((bool)ret_val_arg[0]);
      } break;
      default:
        throw unsupported_dtype();
    }
  }

  void call_raw(const std::vector<void*>& args) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<void*> args_extended = args;
    switch (dtype_.scalar_type()) {
#define TYPE_CASE(Type, Name)                    \
  case ScalarType::Name: {                       \
    std::vector<Type> ret_val_arg(1);            \
    args_extended.push_back(ret_val_arg.data()); \
    codegen_->call_raw(args_extended);           \
    ret_value_ = InterpValue(ret_val_arg[0]);    \
  } break;
      AT_FORALL_SCALAR_TYPES_AND2(Half, BFloat16, TYPE_CASE);
      TYPE_CASE(c10::quint8, QUInt8);
      TYPE_CASE(c10::qint8, QInt8);
#undef TYPE_CASE
      case ScalarType::Bool: {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        std::vector<unsigned char> ret_val_arg(1);
        args_extended.push_back(ret_val_arg.data());
        codegen_->call_raw(args_extended);
        ret_value_ = InterpValue((bool)ret_val_arg[0]);
      } break;
      default:
        throw unsupported_dtype();
    }
  }

  template <typename T>
  T value(const std::vector<void*>& args) {
    call_raw(args);
    return ret_value_.as<T>();
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
  InterpValue ret_value_;
};

// Evaluates the given expression and returns an int64_t value if the result of
// the given expression is int64_t.
c10::optional<int64_t> evalInt(ExprPtr e);

// Substitutes the given vars with their corresponding expressions in the input
// expression.
inline ExprPtr Substitute(ExprPtr expr, const VarMapping& var_mapping) {
  VarSubMutator var_sub(var_mapping);
  return expr->accept_mutator(&var_sub);
}

// Substitutes the given vars with their corresponding expressions in the input
// statement.
inline StmtPtr Substitute(StmtPtr stmt, const VarMapping& var_mapping) {
  VarSubMutator var_sub(var_mapping);
  return stmt->accept_mutator(&var_sub);
}

// Creates a clone of the input expression and substitutes the given vars with
// their corresponding expressions in the clone.
// NOTE: This works because cloning reuses variables and does not create new
// ones, and `VarMapping` input has variables as the key.
inline ExprPtr SubstituteInClone(ExprPtr expr, const VarMapping& var_mapping) {
  VarSubMutator var_sub(var_mapping);
  return Expr::clone(std::move(expr))->accept_mutator(&var_sub);
}

// Creates a clone of the input statement and substitutes the given vars with
// their corresponding expressions in the clone.
// NOTE: This works because cloning reuses variables and does not create new
// ones, and `VarMapping` input has variables as the key.
inline StmtPtr SubstituteInClone(StmtPtr stmt, const VarMapping& var_mapping) {
  VarSubMutator var_sub(var_mapping);
  return Stmt::clone(std::move(stmt))->accept_mutator(&var_sub);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
