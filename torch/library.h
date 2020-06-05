#pragma once

#include <c10/core/DispatchKey.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/op_registration/infer_schema.h>
#if defined(EXPOSE_C2_OPS) || !defined(CAFFE2_IS_XPLAT_BUILD)
#include <torch/csrc/jit/frontend/function_schema_parser.h>
#endif

// Just for inferFunctionSchemaFromFunctor
#include <ATen/core/op_registration/op_registration.h>

namespace torch {

template <class CurClass>
class class_;

// A quick tour of a few usage examples:
//
//  // Define a library whose operators live in the namespace 'aten'.
//  // You must define all of the operators for this library in
//  // this namespace.
//  TORCH_LIBRARY(aten, m) {
//    // Define a schema for an operator, but provide no implementation
//    m.def("mul(Tensor self, Tensor other) -> Tensor");
//
//    // Define a operator with exactly one implementation for all backends.
//    m.def("add(Tensor self, Tensor other) -> Tensor", &add_impl);
//
//    // Provide an implementation for a defined operator (you can
//    // provide multiple; one per backend).  We'll take care of calling
//    // the correct implementation depending on if we get a CPU
//    // tensor or a CUDA tensor
//    m.impl("mul", torch::kCPU, &mul_cpu_impl);
//    m.impl("mul", torch::kCUDA, &mul_cuda_impl);
//  }
//
//  // Define implementations for operators for a non-standard backend,
//  // e.g., XLA (valid values are entries of DispatchKey).  These
//  // operator names are not namespaced; you can define implementations
//  // for any namespace.
//  TORCH_LIBRARY_IMPL(aten, XLA, m) {
//    m.impl("mul", &mul_xla_impl);
//  }


// Represents a C++ function that implements an operator.  Most users won't
// interact directly with this class, except via error messages: the
// constructors this function define the set of permissible "function"-like
// things you can bind via the interface.
//
// This class erases the type of the passed in function, but durably records
// the type via an inferred schema for the function.
//
// TODO: This is morally the same thing as KernelRegistrationConfig, but it's
// opaque to the user.
class CAFFE2_API CppFunction final {
public:
  // This overload accepts function pointers, e.g., CppFunction(&add_impl)
  template <typename Func>
  explicit CppFunction(Func* f, std::enable_if_t<c10::guts::is_function_type<Func>::value, std::nullptr_t> = nullptr)
    : func_(c10::KernelFunction::makeFromUnboxedRuntimeFunction(f))
    // TODO: Don't go through WrapRuntimeKernelFunctor
    , schema_(c10::detail::inferFunctionSchemaFromFunctor<c10::impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<Func>>>())
    , debug_()
    {}

  // This overload accepts lambdas, e.g., CppFunction([](const Tensor& self) { ... })
  template <typename Lambda>
  explicit CppFunction(Lambda&& f, std::enable_if_t<c10::guts::is_functor<std::decay_t<Lambda>>::value, std::nullptr_t> = nullptr)
    : func_(c10::KernelFunction::makeFromUnboxedLambda(std::forward<Lambda>(f)))
    // TODO: Don't go through WrapRuntimeKernelFunctor
    , schema_(c10::detail::inferFunctionSchemaFromFunctor<c10::impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<Lambda>>>())
    , debug_()
    {}

  // This static factory lets you create CppFunctions that (1) don't have boxing
  // wrappers (because we don't support it yet) and (2) don't have schema
  // inference (because some ops don't support it).
  //
  // TODO: Eliminate the necessity for this function entirely.
  template <typename Func>
  static CppFunction makeUnboxedOnly(Func* f) {
    return CppFunction(
      c10::KernelFunction::makeFromUnboxedOnlyRuntimeFunction(f),
      /* schema */ nullptr
    );
  }

  // TODO: more user friendly API
  static CppFunction makeFallthrough() {
    return CppFunction(
      c10::KernelFunction::makeFallthrough(),
      /* schema */ nullptr
    );
  }

  static CppFunction makeNamedNotSupported() {
    return CppFunction(
      c10::KernelFunction::makeNamedNotSupported(),
      /* schema */ nullptr
    );
  }

  // TODO: more user friendly API
  template<c10::KernelFunction::BoxedKernelFunction* func>
  static CppFunction makeFromBoxedFunction() {
    return CppFunction(
      c10::KernelFunction::makeFromBoxedFunction<func>(),
      /* schema */ nullptr
    );
  }

  CppFunction&& debug(std::string d) && {
    debug_ = std::move(d);
    return std::move(*this);
  }

private:
  c10::optional<c10::DispatchKey> dispatch_key_;
  c10::KernelFunction func_;
  std::unique_ptr<c10::FunctionSchema> schema_;
  std::string debug_;

  // The "setter" for dispatch_key_
  template <typename Func>
  friend CppFunction dispatch(c10::DispatchKey, Func&&);

  // The only class which actually pulls out values from CppFunction (does so
  // destructively, felt too lazy to write accessors that I don't even
  // want users to use)
  friend class Library;

  CppFunction(c10::KernelFunction func, std::unique_ptr<c10::FunctionSchema> schema);
};

// Create a CppFunction which is associated with a specific dispatch key.
// CppFunctions that are tagged with a DispatchKey don't get invoked /unless/
// the dispatcher determines that the DispatchKey is the best choice for
// a function
template <typename Func>
inline CppFunction dispatch(c10::DispatchKey k, Func&& raw_f) {
  CppFunction f(std::forward<Func>(raw_f));
  if (k == c10::DispatchKey::CatchAll) {
    f.dispatch_key_ = c10::nullopt;
  } else {
    f.dispatch_key_ = k;
  }
  return f;
}

// Convenience overload of dispatch which accepts DeviceType
template <typename Func>
inline CppFunction dispatch(c10::DeviceType type, Func&& raw_f) {
  auto deviceTypeToDispatchKey = [](c10::DeviceType t){
    switch (t) {
      // This list is synchronized with the k-constants in c10/core/DeviceType.h
      case c10::DeviceType::CPU:
        return c10::DispatchKey::CPU;
      case c10::DeviceType::CUDA:
        return c10::DispatchKey::CUDA;
      case c10::DeviceType::XLA:
        return c10::DispatchKey::XLA;
      case c10::DeviceType::HIP:
        return c10::DispatchKey::HIP;
      case c10::DeviceType::MSNPU:
        return c10::DispatchKey::MSNPU;
      default:
        TORCH_CHECK(false,
          "Device type ", t, " cannot be overloaded at dispatch time, "
          "please file a bug report explaining what you were trying to do.");
    }
  };
  return dispatch(deviceTypeToDispatchKey(type), std::forward<Func>(raw_f));
}

inline c10::FunctionSchema schema(const char* str, c10::AliasAnalysisKind k) {
  c10::FunctionSchema s = torch::jit::parseSchema(str);
  s.setAliasAnalysis(k);
  return s;
}
inline c10::FunctionSchema schema(const char* s) {
  return schema(s, c10::AliasAnalysisKind::FROM_SCHEMA);
}
inline c10::FunctionSchema&& schema(c10::FunctionSchema&& s) { return std::move(s); }

namespace detail {

  inline c10::either<c10::OperatorName, c10::FunctionSchema> constructSchemaOrName(c10::FunctionSchema&& s) {
    return c10::make_right<c10::OperatorName, c10::FunctionSchema>(std::move(s));
  }
  inline c10::either<c10::OperatorName, c10::FunctionSchema> constructSchemaOrName(c10::OperatorName&& n) {
    return c10::make_left<c10::OperatorName, c10::FunctionSchema>(std::move(n));
  }
  inline c10::either<c10::OperatorName, c10::FunctionSchema> constructSchemaOrName(const char* str) {
    auto s = torch::jit::parseSchemaOrName(str);
    if (s.is_right()) {
      s.right().setAliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA);
    }
    return s;
  }

  class TorchLibraryInit;

}

// This is the "handle" by which functions defined in TORCH_LIBRARY
// and TORCH_LIBRARY_IMPL can define operators and override implementations
// at certain backends.
//
// Conventionally, you get access to it using those two macros:
//
// TORCH_LIBRARY(torchvision, m) {
//    // m is a torch::Library
//    m.def("roi_align", ...);
//    ...
// }
//
// TORCH_LIBRARY_IMPL(aten, XLA, m) {
//    // m is a torch::Library
//    m.impl("add", ...);
//    ...
// }
//
// In some cases, you need to define something that applies to all namespaces,
// not just one namespace (usually a fallback).  In that case, use the reserved
// namespace _, e.g.,
//
// TORCH_LIBRARY_IMPL(_, XLA, m) {
//    m.fallback(xla_fallback);
// }
//
class CAFFE2_API Library final {
public:
  // Which type of macro produced this Library
  enum Kind {
    DEF, // from TORCH_LIBRARY (no qualifier)
    IMPL,
    FRAGMENT,
  };

  // Use TORCH_LIBRARY/TORCH_LIBRARY_IMPL instead of these constructors directly
  Library(Kind kind, std::string ns, c10::optional<c10::DispatchKey> k, const char* file, uint32_t line);

  Library(const Library&) = delete;
  Library& operator=(const Library&) = delete;
  Library(Library&&) = default;
  Library& operator=(Library&&) = default;

  // Some notes about the API design here.  We had the following constraints:
  //
  //  - We need to support multiple "types" of arguments for schema and
  //    functions (e.g., unnamed lambda types, regular functions, const char*,
  //    fully instantiated schemas)
  //  - We don't want to write exponentially many overloads
  //  - We don't want to rely on implicit conversion to a common type,
  //    because the C++ compiler will only be willing to do a single
  //    implicit conversion (reducing the set of valid types which you
  //    can invoke with); also error messages are worse when an implicit
  //    conversion is not selected (as the compiler will not explain
  //    why it didn't select an implicit conversion; this is different
  //    from overloads where it will explain each candidate overload and
  //    why it didn't apply)
  //
  // To solve all of these constraints at the same time, we use a trick taken
  // from the pybind11 library: template over the argument in the user visible
  // API, and inside of the templated function explicitly call an overloaded
  // function to resolve the argument to a real type.  You get the good error
  // messages from overloads, but at the same time you only need to write the
  // overload for any given argument type once.

  // Declare an operator with a schema, but don't provide any implementations
  // for it.  You're expected to then provide implementations using the
  // impl() method.
  template <typename Schema>
  Library& def(Schema&& raw_schema) & {
    c10::FunctionSchema s = schema(std::forward<Schema>(raw_schema));
    return _def(std::move(s));
  }

  // Convenience method to define an operator for a schema and then register
  // an implementation for it.  def(n, f) is almost equivalent to def(n).impl(f),
  // except that if n is not a schema, then the schema is inferred from the
  // static type of f.
  template <typename NameOrSchema, typename Func>
  Library& def(NameOrSchema&& raw_name_or_schema, Func&& raw_f) & {
    CppFunction f(std::forward<Func>(raw_f));
    auto name_or_schema = detail::constructSchemaOrName(std::forward<NameOrSchema>(raw_name_or_schema));
    return _def(std::move(name_or_schema), std::move(f));
  }

  // Register an implementation for an operator.  You may register multiple
  // implementations for a single operator at different dispatch keys
  // (see torch::dispatch).  Implementations must have a corresponding
  // declaration (from def), otherwise they are invalid.
  template <typename Func>
  Library& impl(const char* name, Func&& raw_f) & {
    CppFunction f(std::forward<Func>(raw_f));
    return _impl(name, std::move(f));
  }

  // Convenience overload for directly specifying the dispatch key.  Dispatch
  // can validly be either DeviceType or DispatchKey; check torch::dispatch for
  // the canonical list of accepted overloads.
  template <typename Dispatch, typename Func>
  Library& impl(const char* name, Dispatch&& key, Func&& raw_f) & {
    return impl(name, dispatch(std::forward<Dispatch>(key), std::forward<Func>(raw_f)));
  }

  // Convenience overload for unboxed only kernels.  These are quite common
  // but will be eventually eliminated; this function makes it easy to grep for
  // them.
  //
  // TODO: Remove this overload once the makeUnboxedOnly incidence rate
  // goes way down
  template <typename Func>
  Library& impl_UNBOXED(const char* name, Func* raw_f) & {
    return impl(name, CppFunction::makeUnboxedOnly(raw_f));
  }

  // Register a fallback implementation for all operators which will be used
  // if there is not a specific implementation for an operator available.
  // Providing a DispatchKey is MANDATORY for fallback at the moment; e.g.,
  // only call this from TORCH_LIBRARY_IMPL
  template <typename Func>
  Library& fallback(Func&& raw_f) & {
    CppFunction f((std::forward<Func>(raw_f)));
    return _fallback(std::move(f));
  }

  template <class CurClass>
  inline class_<CurClass> class_(const std::string& className);

private:
  Kind kind_;
  c10::optional<std::string> ns_;
  c10::optional<c10::DispatchKey> dispatch_key_;
  const char* file_;
  uint32_t line_;

  std::vector<c10::RegistrationHandleRAII> registrars_;

  friend class detail::TorchLibraryInit;

  // Non-user visible actual implementations of functions.  These aren't
  // public because we only implement & qualifier and not && qualifier
  Library& _def(c10::FunctionSchema&& schema, c10::OperatorName* out_name = nullptr) &;
  Library& _def(c10::either<c10::OperatorName, c10::FunctionSchema>&&, CppFunction&& f) &;
  Library& _impl(const char* name, CppFunction&& f) &;
  Library& _fallback(CppFunction&& f) &;
};

namespace detail {

class TorchLibraryInit final {
private:
  using InitFn = void(Library&);
  Library lib_;
public:
  TorchLibraryInit(Library::Kind kind, InitFn* fn, const char* ns, c10::optional<c10::DispatchKey> k, const char* file, uint32_t line)
    : lib_(kind, ns, k, file, line) {
    fn(lib_);
  }
};

} // namespace detail

} // namespace torch

// NB: The EXACT NAMING of the initializer functions (e.g.,
// TORCH_LIBRARY_init_aten) matters for the code analyzer;
// see the regexes at tools/code_analyzer/run_analyzer.sh

#define TORCH_LIBRARY(ns, m) \
  static void TORCH_LIBRARY_init_ ## ns (torch::Library&); \
  static torch::detail::TorchLibraryInit TORCH_LIBRARY_static_init_ ## ns ( \
    torch::Library::DEF, \
    &TORCH_LIBRARY_init_ ## ns, \
    #ns, c10::nullopt, __FILE__, __LINE__ \
  ); \
  void TORCH_LIBRARY_init_ ## ns (torch::Library& m)

// This macro is a version of TORCH_LIBRARY that doesn't enforce that there
// is only one library (it is a "fragment").  This should ONLY be used
// with PerOpRegistration (as its name suggests).
#define TORCH_LIBRARY_FRAGMENT_THIS_API_IS_FOR_PER_OP_REGISTRATION_ONLY(ns, m) \
  static void TORCH_LIBRARY_FRAGMENT_init_ ## ns ## _ ## k (torch::Library&); \
  static torch::detail::TorchLibraryInit TORCH_LIBRARY_FRAGMENT_static_init_ ## ns ## _ ## k ( \
    torch::Library::FRAGMENT, \
    &TORCH_LIBRARY_FRAGMENT_init_ ## ns ## _ ## k, \
    #ns, c10::nullopt, __FILE__, __LINE__ \
  ); \
  void TORCH_LIBRARY_FRAGMENT_init_ ## ns ## _ ## k (torch::Library& m)

#define TORCH_LIBRARY_IMPL(ns, k, m) \
  static void TORCH_LIBRARY_IMPL_init_ ## ns ## _ ## k (torch::Library&); \
  static torch::detail::TorchLibraryInit TORCH_LIBRARY_IMPL_static_init_ ## ns ## _ ## k ( \
    torch::Library::IMPL, \
    & TORCH_LIBRARY_IMPL_init_ ## ns ## _ ## k, \
    #ns, c10::make_optional(c10::DispatchKey::k), __FILE__, __LINE__ \
  ); \
  void TORCH_LIBRARY_IMPL_init_ ## ns ## _ ## k (torch::Library& m)

// These are variants of the macros above which are to be used for testing (they
// don't setup the static initializer, so you can control the visibility of
// the allocated library yourself).
//
// DO NOT use these in production code, they are NOT understood by the
// code analyzer and will be incorrectly analyzed in those situations.
#define MAKE_TORCH_LIBRARY(ns) torch::Library(torch::Library::DEF, #ns, c10::nullopt, __FILE__, __LINE__)
#define MAKE_TORCH_LIBRARY_IMPL(ns, k) torch::Library(torch::Library::IMPL, #ns, c10::make_optional(c10::DispatchKey::k), __FILE__, __LINE__)

#include <torch/custom_class.h>
