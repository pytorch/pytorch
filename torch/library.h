#pragma once

/// \file
///
/// This header provides an API for extending PyTorch's core library
/// of operators with user defined operators and data types.  This
/// API can be used in a few ways:
///
/// * You can define new custom operators and classes with TORCH_LIBRARY(),
///   making them available for use in both eager Python as well as in TorchScript.
///   This API is modeled off of pybind11's `PYBIND11_MODULE` macro, as
///   the provided functionality is similar (pybind11 lets you bind C++
///   to Python only; `torch/library.h` lets you bind C++ simultaneously
///   to Python and TorchScript).
///
/// * You can override existing operators with TORCH_LIBRARY_IMPL(),
///   providing a new implementation for these operators for a custom
///   backend (e.g., XLA).  When you pass operators with tensors of your custom
///   backend, your overridden implementations will be called instead
///   of the standard implementations.
///
/// * You can use both capabilities at the same time, allowing you
///   to write custom operators that register CPU/CUDA/Autograd
///   implementations without having to write the boilerplate
///   conditionals yourself.
///
/// For a tutorial style introduction to the library API, check
/// out the [Extending TorchScript with Custom C++
/// Operators](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html)
/// tutorial.
///
/// ```
/// // Define a library whose operators live in the namespace 'myops'.
/// // You must define all of the operators for this library in
/// // this namespace.
/// TORCH_LIBRARY(myops, m) {
///   // Define a operator with exactly one implementation for all backends.
///   m.def("add(Tensor self, Tensor other) -> Tensor", &add_impl);
///
///   // Define a schema for an operator, but provide no implementation
///   // (use this syntax if you want to use the dispatcher)
///   m.def("mul(Tensor self, Tensor other) -> Tensor");
///
///   // Provide an implementation for a defined operator (you can
///   // provide multiple; one per backend).  The dispatcher takes care of
///   // calling the correct implementation depending on if we get a CPU
///   // tensor or a CUDA tensor
///   m.impl("mul", torch::kCPU, &mul_cpu_impl);
///   m.impl("mul", torch::kCUDA, &mul_cuda_impl);
/// }
///
/// // Define implementations for operators for a non-standard backend,
/// // e.g., XLA (valid values are entries of DispatchKey).  This can
/// // be used to define operators in a different file than the initial
/// // TORCH_LIBRARY definition (e.g., if it is in an external library)
/// TORCH_LIBRARY_IMPL(myops, XLA, m) {
///   m.impl("mul", &mul_xla_impl);
/// }
/// ```

#include <c10/core/DispatchKey.h>
#include <ATen/core/op_registration/op_whitelist.h>
#include <ATen/core/op_registration/infer_schema.h>
#if defined(EXPOSE_C2_OPS) || !defined(CAFFE2_IS_XPLAT_BUILD)
#include <torch/csrc/jit/frontend/function_schema_parser.h>
#endif

// Just for inferFunctionSchemaFromFunctor
#include <ATen/core/op_registration/op_registration.h>

namespace torch {

#if defined C10_MOBILE
/**
 * The NoInferSchemaTag is a type name used to indicate that this call to the
 * CppFunction constructor should not trigger schema inference from functor.
 * Schema inference from functor utilizes template meta-programming, and is
 * costly from a size perspective. Ideally, one would expect that the schema
 * inference would require very little binary size since most of the
 * computation can be done by the compiler at build time, but that isn't
 * necessarily the case.
 *
 * Schema inference is elided only for mobile use-cases where we don't need
 * the additional runtime cost or size overhead on client devices.
 *
 */
struct NoInferSchemaTag {};
#endif

template <class CurClass>
class class_;

/// Represents a C++ function that implements an operator.  Most users won't
/// interact directly with this class, except via error messages: the
/// constructors this function define the set of permissible "function"-like
/// things you can bind via the interface.
///
/// This class erases the type of the passed in function, but durably records
/// the type via an inferred schema for the function.
class TORCH_API CppFunction final {
  // TODO: This is morally the same thing as KernelRegistrationConfig, but it's
  // opaque to the user.

public:
  /// This overload accepts function pointers, e.g., `CppFunction(&add_impl)`
  template <typename Func>
  explicit CppFunction(Func* f, std::enable_if_t<c10::guts::is_function_type<Func>::value, std::nullptr_t> = nullptr)
    : func_(c10::KernelFunction::makeFromUnboxedRuntimeFunction(f))
    , cpp_signature_(c10::impl::CppSignature::make<Func>())
    // TODO: Don't go through WrapRuntimeKernelFunctor
    , schema_(c10::detail::inferFunctionSchemaFromFunctor<c10::impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<Func>>>())
    , debug_()
    {}

  /// This overload accepts compile time function pointers, e.g., `CppFunction(TORCH_FN(add_impl))`
  template <typename FuncPtr>
  explicit CppFunction(FuncPtr f, std::enable_if_t<c10::is_compile_time_function_pointer<FuncPtr>::value, std::nullptr_t> = nullptr)
    : func_(c10::KernelFunction::makeFromUnboxedFunction(f))
    , cpp_signature_(c10::impl::CppSignature::make<typename FuncPtr::FuncType>())
    // TODO: Don't go through WrapRuntimeKernelFunctor
    , schema_(c10::detail::inferFunctionSchemaFromFunctor<c10::impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<typename FuncPtr::FuncType>>>())
    , debug_()
    {}

  /// This overload accepts lambdas, e.g., `CppFunction([](const Tensor& self) { ... })`
  template <typename Lambda>
  explicit CppFunction(Lambda&& f, std::enable_if_t<c10::guts::is_functor<std::decay_t<Lambda>>::value, std::nullptr_t> = nullptr)
    : func_(c10::KernelFunction::makeFromUnboxedLambda(std::forward<Lambda>(f)))
    , cpp_signature_(c10::impl::CppSignature::make<Lambda>())
    // TODO: Don't go through WrapRuntimeKernelFunctor
    , schema_(c10::detail::inferFunctionSchemaFromFunctor<c10::impl::WrapFunctionIntoRuntimeFunctor<std::decay_t<Lambda>>>())
    , debug_()
    {}

#if defined C10_MOBILE
  /// This overload accepts function pointers, e.g., `CppFunction(&add_impl, NoInferSchemaTag())`
  template <typename Func>
  explicit CppFunction(Func* f, NoInferSchemaTag, std::enable_if_t<c10::guts::is_function_type<Func>::value, std::nullptr_t> = nullptr)
    : func_(c10::KernelFunction::makeFromUnboxedRuntimeFunction(f))
    , cpp_signature_(c10::impl::CppSignature::make<Func>())
    // TODO: Don't go through WrapRuntimeKernelFunctor
    , schema_(nullptr)
    , debug_()
    {}

  /// This overload accepts compile time function pointers, e.g., `CppFunction(TORCH_FN(add_impl), NoInferSchemaTag())`
  template <typename FuncPtr>
  explicit CppFunction(FuncPtr f, NoInferSchemaTag, std::enable_if_t<c10::is_compile_time_function_pointer<FuncPtr>::value, std::nullptr_t> = nullptr)
    : func_(c10::KernelFunction::makeFromUnboxedFunction(f))
    , cpp_signature_(c10::impl::CppSignature::make<typename FuncPtr::FuncType>())
    // TODO: Don't go through WrapRuntimeKernelFunctor
    , schema_(nullptr)
    , debug_()
    {}

  /// This overload accepts lambdas, e.g., `CppFunction([](const Tensor& self) { ... }. NoInferSchemaTag())`
  template <typename Lambda>
  explicit CppFunction(Lambda&& f, NoInferSchemaTag, std::enable_if_t<c10::guts::is_functor<std::decay_t<Lambda>>::value, std::nullptr_t> = nullptr)
    : func_(c10::KernelFunction::makeFromUnboxedLambda(std::forward<Lambda>(f)))
    , cpp_signature_(c10::impl::CppSignature::make<Lambda>())
    // TODO: Don't go through WrapRuntimeKernelFunctor
    , schema_(nullptr)
    , debug_()
    {}
#endif

  /// This creates a fallthrough function.  Fallthrough functions
  /// immediately redispatch to the next available dispatch key,
  /// but are implemented more efficiently than a hand written
  /// function done in the same way.
  static CppFunction makeFallthrough() {
    // TODO: more user friendly API
    return CppFunction(
      c10::KernelFunction::makeFallthrough(),
      /* cpp_signature */ c10::nullopt, // not known for fallthroughs
      /* schema */ nullptr
    );
  }

  /// \private
  ///
  /// Creates a function that raises an error saying that named tensors
  /// are not supported when called.
  static CppFunction makeNamedNotSupported() {
    return CppFunction(
      c10::KernelFunction::makeNamedNotSupported(),
      /* cpp_signature */ c10::nullopt, // not known for fallthroughs
      /* schema */ nullptr
    );
  }

  /// Create a function from a boxed kernel function with signature
  /// `void(const OperatorHandle&, Stack*)`; i.e., they receive a
  /// stack of arguments in a boxed calling convention, rather than
  /// in the native C++ calling convention.  Boxed functions are
  /// typically only used to register backend fallbacks via
  /// torch::Library::fallback().
  template<c10::KernelFunction::BoxedKernelFunction* func>
  static CppFunction makeFromBoxedFunction() {
    // TODO: more user friendly API
    return CppFunction(
      c10::KernelFunction::makeFromBoxedFunction<func>(),
      /* cpp_signature */ c10::nullopt, // not known for boxed functions
      /* schema */ nullptr
    );
  }

  // Variant that takes in a boxed kernel function with a plumbed DispatchKeySet.
  // See Note [Plumbing Keys Through The Dispatcher] for details.
  template<c10::KernelFunction::BoxedKernelFunction_withDispatchKeys* func>
  static CppFunction makeFromBoxedFunction() {
    // TODO: more user friendly API
    return CppFunction(
      c10::KernelFunction::makeFromBoxedFunction<func>(),
      /* cpp_signature */ c10::nullopt, // not known for boxed functions
      /* schema */ nullptr
    );
  }

  /// Create a function from an unboxed kernel function.
  /// This is typically used to register common operators.
  template<typename FuncPtr, std::enable_if_t<c10::guts::is_function_type<FuncPtr>::value, std::nullptr_t> = nullptr>
  static CppFunction makeFromUnboxedFunction(FuncPtr* f) {
    return CppFunction(f);
  }

  /// Create a function from a compile time unboxed kernel function pointer.
  /// This is typically used to register common operators.
  /// Compile time function pointers can be used to allow the compiler
  /// to optimize (e.g. inline) calls to it.
  template<typename FuncPtr, std::enable_if_t<c10::is_compile_time_function_pointer<FuncPtr>::value, std::nullptr_t> = nullptr>
  static CppFunction makeFromUnboxedFunction(FuncPtr f) {
    return CppFunction(f);
  }

  CppFunction&& debug(std::string d) && {
    debug_ = std::move(d);
    return std::move(*this);
  }

private:
  c10::optional<c10::DispatchKey> dispatch_key_;
  c10::KernelFunction func_;
  c10::optional<c10::impl::CppSignature> cpp_signature_;
  std::unique_ptr<c10::FunctionSchema> schema_;
  std::string debug_;

  // The "setter" for dispatch_key_
  template <typename Func>
  friend CppFunction dispatch(c10::DispatchKey, Func&&);

  // The only class which actually pulls out values from CppFunction (does so
  // destructively, felt too lazy to write accessors that I don't even
  // want users to use)
  friend class Library;

  CppFunction(c10::KernelFunction func, c10::optional<c10::impl::CppSignature> cpp_signature, std::unique_ptr<c10::FunctionSchema> schema);
};

/// \defgroup torch-dispatch-overloads torch::dispatch overloads

/// Create a torch::CppFunction which is associated with a specific
/// dispatch key.  torch::CppFunctions that are tagged with a
/// c10::DispatchKey don't get invoked unless the dispatcher determines
/// that this particular c10::DispatchKey is the one that should be
/// dispatched to.
///
/// This function is generally not used directly, instead, prefer using
/// TORCH_LIBRARY_IMPL(), which will implicitly set the c10::DispatchKey
/// for all registration calls inside of its body.
///
/// \ingroup torch-dispatch-overloads
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

/// Convenience overload of dispatch() which accepts c10::DeviceType
///
/// \ingroup torch-dispatch-overloads
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

/// \defgroup torch-schema-overloads torch::schema overloads

/// Construct a c10::FunctionSchema from a string, with an explicitly
/// specified c10::AliasAnalysisKind.  Ordinarily, schemas are simply
/// passed in as strings, but if you need to specify a custom alias
/// analysis, you can replace the string with a call to this function.
///
/// ```
/// // Default alias analysis (FROM_SCHEMA)
/// m.def("def3(Tensor self) -> Tensor");
/// // Pure function alias analysis
/// m.def(torch::schema("def3(Tensor self) -> Tensor", c10::AliasAnalysisKind::PURE_FUNCTION));
/// ```
///
/// \ingroup torch-schema-overloads
inline c10::FunctionSchema schema(const char* str, c10::AliasAnalysisKind k) {
  c10::FunctionSchema s = torch::jit::parseSchema(str);
  s.setAliasAnalysis(k);
  return s;
}

/// Function schemas can be directly constructed from string literals.
///
/// \ingroup torch-schema-overloads
inline c10::FunctionSchema schema(const char* s) {
  return schema(s, c10::AliasAnalysisKind::FROM_SCHEMA);
}

/// \private
///
/// Already constructed function schemas are accepted if they are
/// rvalues.
///
/// \ingroup torch-schema-overloads
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

} // namespace detail

// Note [Selective build]
// ~~~~~~~~~~~~~~~~~~~~~~
// In some settings, especially mobile, it is important to avoid compiling any
// references to functions that you aren't actually going to use, so that they
// can be eliminated by the linker.  We call this capability "selective build".
//
// A very easy way to implement selective build which results in a lot of
// boilerplate is to just add ifdef's around every registration call, but this
// means you have to write a lot of extra lines of code at every registration
// site, and it also means you have to define some munging scheme to map
// operators to macros.
//
// Instead of doing this, we have a different mechanism centered around the
// concept of a SelectiveStr.  A selective name is like a const char* string,
// except it also carries at compile time a boolean saying whether or not a
// registration should actually happen or not.  We then have extra overloads which
// bypass registration entirely if a selective name is disabled.  We do a
// constexpr test to see if a operator should be enabled or not; this is
// currently implemented in ATen/core/op_registration/op_whitelist.h

namespace detail {

  // A SelectiveStr is like a const char*, except that it also comes
  // with a type brand that says whether or not the name is enabled or
  // not.  If the string is disabled, then (at compile time) we DON'T generate
  // a registration call for it.  This class is not intended to be called
  // directly; use TORCH_SELECTIVE_NAME or TORCH_SELECTIVE_SCHEMA macros below
  // to create it.
  template <bool enabled>
  class SelectiveStr {
  public:
    constexpr SelectiveStr(const char* name) : name_(name) {}
    constexpr operator const char*() { return name_; }
  private:
    const char* name_;
  };

#define TORCH_SELECTIVE_NAME(n) torch::detail::SelectiveStr<c10::impl::op_whitelist_check(n)>(n)
#define TORCH_SELECTIVE_SCHEMA(n) torch::detail::SelectiveStr<c10::impl::schema_whitelist_check(n)>(n)

}

/// This object provides the API for defining operators and providing
/// implementations at dispatch keys.  Typically, a torch::Library
/// is not allocated directly; instead it is created by the
/// TORCH_LIBRARY() or TORCH_LIBRARY_IMPL() macros.
///
/// Most methods on torch::Library return a reference to itself,
/// supporting method chaining.
///
/// ```
/// // Examples:
///
/// TORCH_LIBRARY(torchvision, m) {
///    // m is a torch::Library
///    m.def("roi_align", ...);
///    ...
/// }
///
/// TORCH_LIBRARY_IMPL(aten, XLA, m) {
///    // m is a torch::Library
///    m.impl("add", ...);
///    ...
/// }
/// ```
///
class TORCH_API Library final {
public:
  /// \private
  ///
  /// Which type of macro produced this Library
  enum Kind {
    DEF, // from TORCH_LIBRARY (no qualifier)
    IMPL,
    FRAGMENT,
  };

  /// \private
  ///
  /// Use TORCH_LIBRARY() or TORCH_LIBRARY_IMPL() instead of using these
  /// constructors directly
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

  /// Declare an operator with a schema, but don't provide any implementations
  /// for it.  You're expected to then provide implementations using the
  /// impl() method.  All template arguments are inferred.
  ///
  /// \param raw_schema The schema of the operator to be defined.
  ///     Typically, this is a `const char*` string literal, but any type
  ///     accepted by torch::schema() is accepted here.
  ///
  /// ```
  /// // Example:
  /// TORCH_LIBRARY(myops, m) {
  ///   m.def("add(Tensor self, Tensor other) -> Tensor");
  /// }
  /// ```
  template <typename Schema>
  Library& def(Schema&& raw_schema) & {
    c10::FunctionSchema s = schema(std::forward<Schema>(raw_schema));
    return _def(std::move(s));
  }

  /// Define an operator for a schema and then register an implementation for
  /// it.  This is typically what you would use if you aren't planning
  /// on making use of the dispatcher to structure your operator
  /// implementation.  It's roughly equivalent to calling def() and
  /// then impl(), but if you omit the schema of the operator, we will
  /// infer it from the type of your C++ function.  All template
  /// arguments are inferred.
  ///
  /// \param raw_name_or_schema The schema of the operator to be
  ///   defined, or just the name of the operator if the schema is to be
  ///   inferred from `raw_f`.  Typically a `const char*` literal.
  /// \param raw_f The C++ function that implements this operator.
  ///   Any valid constructor of torch::CppFunction is accepted here;
  ///   typically you provide a function pointer or lambda.
  ///
  /// ```
  /// // Example:
  /// TORCH_LIBRARY(myops, m) {
  ///   m.def("add", add_fn);
  /// }
  /// ```
  template <typename NameOrSchema, typename Func>
  Library& def(NameOrSchema&& raw_name_or_schema, Func&& raw_f) & {
    CppFunction f(std::forward<Func>(raw_f));
    auto name_or_schema = detail::constructSchemaOrName(std::forward<NameOrSchema>(raw_name_or_schema));
    return _def(std::move(name_or_schema), std::move(f));
  }

  /// Register an implementation for an operator.  You may register multiple
  /// implementations for a single operator at different dispatch keys
  /// (see torch::dispatch()).  Implementations must have a corresponding
  /// declaration (from def()), otherwise they are invalid.  If you plan
  /// to register multiple implementations, DO NOT provide a function
  /// implementation when you def() the operator.
  ///
  /// \param name The name of the operator to implement.  Do NOT provide
  ///   schema here.
  /// \param raw_f The C++ function that implements this operator.  Any
  ///   valid constructor of torch::CppFunction is accepted here;
  ///   typically you provide a function pointer or lambda.
  ///
  /// ```
  /// // Example:
  /// TORCH_LIBRARY_IMPL(myops, CUDA, m) {
  ///   m.impl("add", add_cuda);
  /// }
  /// ```
  template <typename Name, typename Func>
  Library& impl(Name name, Func&& raw_f) & {
    // TODO: need to raise an error when you impl a function that has a
    // catch all def
  #if defined C10_MOBILE
    CppFunction f(std::forward<Func>(raw_f), NoInferSchemaTag());
  #else
    CppFunction f(std::forward<Func>(raw_f));
  #endif
      return _impl(name, std::move(f));
  }

#if defined C10_MOBILE
  // Note: This overload is needed only for C10_MOBILE, since the automatically
  // defined copy constructor for the CppFunction doesn't have the additional
  // NoInferSchemaTag argument. We define the overload for the impl() function
  // to accept a CppFunction&& argument. The already constructed CppFunction
  // object may or may not have the inferred schema, but it doesn't matter
  // for our purposes since if it already has the inferred schema, then we
  // might as well just pass it through directly.
  //
  template <typename Name>
  Library& impl(Name name, CppFunction&& raw_f) & {
    // TODO: need to raise an error when you impl a function that has a
    // catch all def
    CppFunction f(std::forward<CppFunction>(raw_f));
    return _impl(name, std::move(f));
  }
#endif

  /// \private
  ///
  /// Convenience overload for directly specifying the dispatch key when
  /// impl().  You probably don't need this; instead, prefer specifying
  /// the dispatch key for the entire block in TORCH_LIBRARY_IMPL()
  template <typename Name, typename Dispatch, typename Func>
  Library& impl(Name name, Dispatch&& key, Func&& raw_f) & {
    return impl(name, dispatch(std::forward<Dispatch>(key), std::forward<Func>(raw_f)));
  }

  template <typename Name, typename Func>
  Library& impl_UNBOXED(Name name, Func* raw_f) & {
    static_assert(c10::guts::false_t<Func>(), ".impl_UNBOXED(...) was removed. Please use .impl(...) instead.");
    return *this;
  }

  // These overloads cover cases when a SelectiveStr (see Note [Selective build])
  // has been disabled at compile time.  In that case, don't generate any code
  // referencing the passed in functions at all.
  Library& def(detail::SelectiveStr<false>) & { return *this; }
  Library& def(detail::SelectiveStr<true> raw_schema) & {
    return def(raw_schema.operator const char *());
  }
  template <typename Func>
  Library& def(detail::SelectiveStr<false>, Func&& raw_f) & { return *this; }
  template <typename Func>
  Library& def(detail::SelectiveStr<true> raw_name_or_schema, Func&& raw_f) & {
    return def(raw_name_or_schema.operator const char *(), std::forward<Func>(raw_f));
  }

  template <typename Func>
  Library& impl(detail::SelectiveStr<false>, Func&& raw_f) & { return *this; }
  template <typename Dispatch, typename Func>
  Library& impl(detail::SelectiveStr<false>, Dispatch&& key, Func&& raw_f) & { return *this; }
  template <typename Func>
  Library& impl_UNBOXED(detail::SelectiveStr<false> name, Func* raw_f) & {
    static_assert(c10::guts::false_t<Func>(), ".impl_UNBOXED(...) was removed. Please use .impl(...) instead.");
    return *this;
  }

  template <typename Func>
  Library& impl(detail::SelectiveStr<true> name, Func&& raw_f) & {
    return impl(name.operator const char*(), std::forward<Func>(raw_f));
  }
  template <typename Dispatch, typename Func>
  Library& impl(detail::SelectiveStr<true> name, Dispatch&& key, Func&& raw_f) & {
    return impl(name.operator const char*(), std::forward<Dispatch>(key), std::forward<Func>(raw_f));
  }
  template <typename Func>
  Library& impl_UNBOXED(detail::SelectiveStr<true> name, Func* raw_f) & {
    static_assert(c10::guts::false_t<Func>(), ".impl_UNBOXED(...) was removed. Please use .impl(...) instead.");
    return *this;
  }

  /// Register a fallback implementation for all operators which will be used
  /// if there is not a specific implementation for an operator available.
  /// There MUST be a DispatchKey associated with a fallback; e.g.,
  /// only call this from TORCH_LIBRARY_IMPL() with namespace `_`.
  ///
  /// \param raw_f The function that implements the fallback.  Unboxed
  ///   functions typically do not work as fallback functions, as
  ///   fallback functions must work for every operator (even though
  ///   they have varying type signatures).  Typical arguments are
  ///   CppFunction::makeFallthrough() or
  ///   CppFunction::makeFromBoxedFunction()
  ///
  /// ```
  /// // Example:
  ///
  /// TORCH_LIBRARY_IMPL(_, AutogradXLA, m) {
  ///   // If there is not a kernel explicitly registered
  ///   // for AutogradXLA, fallthrough to the next
  ///   // available kernel
  ///   m.fallback(torch::CppFunction::makeFallthrough());
  /// }
  ///
  /// // See aten/src/ATen/core/dispatch/backend_fallback_test.cpp
  /// // for a full example of boxed fallback
  /// ```
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


/// Macro for defining a function that will be run at static
/// initialization time to define a library of operators in the
/// namespace `ns` (must be a valid C++ identifier, no quotes).
/// Use this macro when you want to define a new set of custom operators
/// that do not already exist in PyTorch.
///
/// Example usage:
///
/// ```
/// TORCH_LIBRARY(myops, m) {
///   // m is a torch::Library; methods on it will define
///   // operators in the myops namespace
///   m.def("add", add_impl);
/// }
/// ```
///
/// The `m` argument is bound to a torch::Library that is used to
/// register operators.  There may only be one TORCH_LIBRARY()
/// for any given namespace.
#define TORCH_LIBRARY(ns, m) \
  static void TORCH_LIBRARY_init_ ## ns (torch::Library&); \
  static const torch::detail::TorchLibraryInit TORCH_LIBRARY_static_init_ ## ns ( \
    torch::Library::DEF, \
    &TORCH_LIBRARY_init_ ## ns, \
    #ns, c10::nullopt, __FILE__, __LINE__ \
  ); \
  void TORCH_LIBRARY_init_ ## ns (torch::Library& m)

/// \private
///
/// This macro is a version of TORCH_LIBRARY() that doesn't enforce that there
/// is only one library (it is a "fragment").  This is used inside the
/// PerOpRegistration.cpp file, as well as in places where all op registrations
/// within the same namespace cannot be easily put into one macro block
/// (this is mostly the case for custom ops in fbcode that were ported from
/// the old API)
#define TORCH_LIBRARY_FRAGMENT(ns,  m) _TORCH_LIBRARY_FRAGMENT(ns, m, C10_UID)

/// \private
///
/// The above macro requires an extra unique identifier (uid) to prevent variable name collisions
/// This can happen if TORCH_LIBRARY_FRAGMENT is called multiple times with the same namespace
/// in the same translation unit.
/// Note that the TORCH_LIBRARY variant doesn't run into this problem, because it enforces
/// that it can only be called once for a given namespace.
#define _TORCH_LIBRARY_FRAGMENT(ns, m, uid) \
  static void C10_CONCATENATE(TORCH_LIBRARY_FRAGMENT_init_ ## ns ## _, uid) (torch::Library&); \
  static const torch::detail::TorchLibraryInit C10_CONCATENATE(TORCH_LIBRARY_FRAGMENT_static_init_ ## ns ## _, uid) ( \
    torch::Library::FRAGMENT, \
    &C10_CONCATENATE(TORCH_LIBRARY_FRAGMENT_init_ ## ns ## _, uid), \
    #ns, c10::nullopt, __FILE__, __LINE__ \
  ); \
  void C10_CONCATENATE(TORCH_LIBRARY_FRAGMENT_init_ ## ns ## _, uid) (torch::Library& m)

/// Macro for defining a function that will be run at static
/// initialization time to define operator overrides for dispatch key
/// `k` (must be an unqualified enum member of c10::DispatchKey) in
/// namespace `ns` (must be a valid C++ identifer, no quotes).  Use this
/// macro when you want to implement a preexisting set of custom
/// operators on a new dispatch key (e.g., you want to provide CUDA
/// implementations of already existing operators).  One common usage
/// pattern is to use TORCH_LIBRARY() to define schema for all new
/// operators you want to define, and then use several
/// TORCH_LIBRARY_IMPL() blocks to provide implementations of the
/// operator for CPU, CUDA and Autograd.
///
/// In some cases, you need to define something that applies to all namespaces,
/// not just one namespace (usually a fallback).  In that case, use the reserved
/// namespace _, e.g.,
///
/// ```
/// TORCH_LIBRARY_IMPL(_, XLA, m) {
///    m.fallback(xla_fallback);
/// }
/// ```
///
/// Example usage:
///
/// ```
/// TORCH_LIBRARY_IMPL(myops, CPU, m) {
///   // m is a torch::Library; methods on it will define
///   // CPU implementations of operators in the myops namespace.
///   // It is NOT valid to call torch::Library::def()
///   // in this context.
///   m.impl("add", add_cpu_impl);
/// }
/// ```
///
/// If ``add_cpu_impl`` is an overloaded function, use a
/// ``static_cast`` to specify which overload you want
/// (by providing the full type).
///
// NB: if the dispatch key is not whitelisted, we simply omit the Library
// call entirely
#define TORCH_LIBRARY_IMPL(ns, k, m) _TORCH_LIBRARY_IMPL(ns, k, m, C10_UID)

/// \private
///
/// The above macro requires an extra unique identifier (uid) to prevent variable name collisions.
/// This can happen if TORCH_LIBRARY_IMPL is called multiple times with the same namespace
/// and dispatch key in the same translation unit.
#define _TORCH_LIBRARY_IMPL(ns, k, m, uid) \
  static void C10_CONCATENATE(TORCH_LIBRARY_IMPL_init_ ## ns ## _ ## k ## _, uid) (torch::Library&); \
  static const torch::detail::TorchLibraryInit C10_CONCATENATE(TORCH_LIBRARY_IMPL_static_init_ ## ns ## _ ## k ## _, uid) ( \
    torch::Library::IMPL, \
    c10::guts::if_constexpr<c10::impl::dispatch_key_whitelist_check(c10::DispatchKey::k)>( \
      []() { return & C10_CONCATENATE(TORCH_LIBRARY_IMPL_init_ ## ns ## _ ## k ## _, uid); }, \
      []() { return [](torch::Library&) -> void {}; } \
    ), \
    #ns, c10::make_optional(c10::DispatchKey::k), \
    __FILE__, __LINE__ \
  ); \
  void C10_CONCATENATE(TORCH_LIBRARY_IMPL_init_ ## ns ## _ ## k ## _, uid) (torch::Library& m)


// These are variants of the macros above which are to be used for testing (they
// don't setup the static initializer, so you can control the visibility of
// the allocated library yourself).
//
// DO NOT use these in production code, they are NOT understood by the
// code analyzer and will be incorrectly analyzed in those situations.

/// \private
#define MAKE_TORCH_LIBRARY(ns) torch::Library(torch::Library::DEF, #ns, c10::nullopt, __FILE__, __LINE__)
/// \private
#define MAKE_TORCH_LIBRARY_IMPL(ns, k) torch::Library(torch::Library::IMPL, #ns, c10::make_optional(c10::DispatchKey::k), __FILE__, __LINE__)

// Make the custom class API visible, so it is available from
// torch::Library.

#include <torch/custom_class.h>
