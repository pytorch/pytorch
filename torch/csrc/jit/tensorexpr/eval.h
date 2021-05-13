#pragma once

#include <cmath>
#include <cstring>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <c10/macros/Macros.h>
#include <c10/util/Logging.h>
#include <c10/util/math_compat.h>
#include <c10/util/string_utils.h>
#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/exceptions.h>
#include <torch/csrc/jit/tensorexpr/execution_counter.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/csrc/jit/tensorexpr/types.h>
#include <torch/csrc/jit/tensorexpr/var_substitutor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DECLARE_TRIGGER(simple_ir_eval_executed);

class Value {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  Value() : dtype_(kInt) {
    Intvalues.push_back(0);
  }

#define VALUE_CTOR(Type, Name)      \
  Value(Type v) : dtype_(k##Name) { \
    Name##values.push_back(v);      \
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, VALUE_CTOR);
#undef VALUE_CTOR

#define VALUE_VEC_CTOR(Type, Name)  \
  Value(const std::vector<Type>& v) \
      : dtype_(Dtype(k##Name, v.size())), Name##values(v) {}
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
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

#define VALUE_AS_DISPATCH(Type, Name)   \
  template <>                           \
  inline Type Value::as<Type>() const { \
    if (dtype_ != k##Name) {            \
      throw unsupported_dtype();        \
    }                                   \
    return Name##values[0];             \
  }
AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, VALUE_AS_DISPATCH);
#undef VALUE_AS_DISPATCH

#define VALUE_AS_VEC_DISPATCH(Type, Name)                       \
  template <>                                                   \
  inline const std::vector<Type>& Value::as_vec<Type>() const { \
    if (dtype_.scalar_type() != ScalarType::Name) {             \
      throw unsupported_dtype();                                \
    }                                                           \
    return Name##values;                                        \
  }
AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, VALUE_AS_VEC_DISPATCH);
#undef VALUE_AS_VEC_DISPATCH

template <typename To, typename From>
To raw_bitcast(const From& src) {
  TORCH_CHECK(sizeof(To) == sizeof(From), "Invalid bitcast invocation");
  To storage;
  std::memcpy(&storage, &src, sizeof(From));
  return reinterpret_cast<To&>(storage);
}

class SimpleIREvaluatorImpl;
class TORCH_API SimpleIREvaluator : public CodeGen {
 public:
  SimpleIREvaluator(
      Stmt* stmt,
      const std::vector<BufferArg>& buffer_args,
      at::Device device = at::kCPU,
      const std::string& kernel_func_name = "func");

  ~SimpleIREvaluator() override;

  void call(const std::vector<CallArg>& args) override;
  void call_raw(const std::vector<void*>& args) override;

  template <typename... Ts>
  void operator()(const Ts&... ts) {
    std::vector<CallArg> args({CallArg(ts)...});
    call(args);
  }

  void bindVar(const Var* v, const Expr* e);
  Value value() const;

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

  ExprEval(const ExprHandle& expr, const std::vector<BufferArg>& buffer_args)
      : dtype_(expr.dtype()) {
    std::vector<BufferArg> buffer_args_extended = buffer_args;
    Placeholder ret_buf("ret_val", dtype_, {1});
    std::vector<const Expr*> indices;
    const Expr* zero = new IntImm(0);
    for (size_t i = 0; i < ret_buf.data()->ndim(); i++) {
      indices.push_back(zero);
    }
    Stmt* store_stmt =
        // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
        new Store(ret_buf.data(), indices, expr.node());
    // NOLINTNEXTLINE(modernize-use-emplace)
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

  void bindVar(const Var* v, const Expr* e) {
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
    std::vector<CallArg> call_args_extended = call_args;
    switch (dtype_.scalar_type()) {
#define TYPE_CASE(Type, Name)                           \
  case ScalarType::Name: {                              \
    std::vector<Type> ret_val_arg(1);                   \
    call_args_extended.push_back(CallArg(ret_val_arg)); \
    codegen_->call(call_args_extended);                 \
    ret_value_ = Value(ret_val_arg[0]);                 \
  } break;
      // NOLINTNEXTLINE(modernize-use-emplace)
      AT_FORALL_SCALAR_TYPES_AND(Half, TYPE_CASE);
#undef TYPE_CASE
      case ScalarType::Bool: {
        std::vector<unsigned char> ret_val_arg(1);
        // NOLINTNEXTLINE(modernize-use-emplace)
        call_args_extended.push_back(CallArg(ret_val_arg.data()));
        codegen_->call(call_args_extended);
        ret_value_ = Value((bool)ret_val_arg[0]);
      } break;
      default:
        throw unsupported_dtype();
    }
  }

  void call_raw(const std::vector<void*>& args) {
    std::vector<void*> args_extended = args;
    switch (dtype_.scalar_type()) {
#define TYPE_CASE(Type, Name)                    \
  case ScalarType::Name: {                       \
    std::vector<Type> ret_val_arg(1);            \
    args_extended.push_back(ret_val_arg.data()); \
    codegen_->call_raw(args_extended);           \
    ret_value_ = Value(ret_val_arg[0]);          \
  } break;
      AT_FORALL_SCALAR_TYPES_AND(Half, TYPE_CASE);
#undef TYPE_CASE
      case ScalarType::Bool: {
        std::vector<unsigned char> ret_val_arg(1);
        args_extended.push_back(ret_val_arg.data());
        codegen_->call_raw(args_extended);
        ret_value_ = Value((bool)ret_val_arg[0]);
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
  Value ret_value_;
};

inline const Expr* Substitute(const Expr* expr, const VarMapping& var_mapping) {
  VarSubMutator var_sub(var_mapping);
  return expr->accept_mutator(&var_sub);
}

inline Stmt* Substitute(Stmt* stmt, const VarMapping& var_mapping) {
  VarSubMutator var_sub(var_mapping);
  return stmt->accept_mutator(&var_sub);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
