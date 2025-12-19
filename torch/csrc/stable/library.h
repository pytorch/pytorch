#pragma once
// this file can only have stable stuff! Akin to shim.h
// but unlike shim.h, this file can contain header-only C++
// code for better UX.

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/Metaprogramming.h>

// Technically, this file doesn't use anything from stableivalue_conversions.h,
// but we need to include it here as the contents of stableivalue_conversions.h
// used to live here and so we need to expose them for backwards compatibility.
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/version.h>

HIDDEN_NAMESPACE_BEGIN(torch, stable, detail)

class StableLibrary final {
 private:
  TorchLibraryHandle lib_;

 public:
  enum class Kind {
    DEF,
    IMPL,
    FRAGMENT,
  };

  // constructor
  /// \private
  ///
  /// Use STABLE_TORCH_LIBRARY or STABLE_TORCH_LIBRARY_IMPL() instead of using
  /// these constructors directly
  StableLibrary(
      Kind kind,
      const char* ns,
      const char* k,
      const char* file,
      uint32_t line) {
    if (kind == Kind::IMPL) {
      aoti_torch_library_init_impl(ns, k, file, line, &lib_);
    } else if (kind == Kind::DEF) {
      aoti_torch_library_init_def(ns, file, line, &lib_);
    } else { // kind == FRAGMENT
      aoti_torch_library_init_fragment(ns, file, line, &lib_);
    }
  }

  // do not permit copy
  StableLibrary(const StableLibrary&) = delete;
  StableLibrary& operator=(const StableLibrary&) = delete;

  // do not permit move
  StableLibrary(StableLibrary&& other) = delete;
  StableLibrary& operator=(StableLibrary&& other) = delete;

  ~StableLibrary() {
    aoti_torch_delete_library_object(lib_);
  }

  // corresponds to a limited, stable version of torch::library::impl()
  // Inputs:
  //   name: the name of the function to implement
  //   fn: a boxed function with schema
  //       (StableIValue* stack, uint64_t num_inputs, uint64_t num_outputs) ->
  //       void
  // fn should follow the calling convention of our boxed kernels that convert
  // to IValues. fn will be called with a StableIValue* array of length
  // max(num_inputs, num_outputs), where the first num_inputs entries are
  // populated with inputs. fn is responsible for stealing the memory of the
  // inputs, in effect "popping" them off the stack, and then populating the
  // stack with StableIValue outputs. Concretely, fn should:
  //    1. read StableIValue inputs from the given stack
  //    2. convert the inputs to the proper types
  //    3. call the function corresponding to name with the inputs
  //    4. convert the outputs to StableIValues
  //    5. populate the now empty stack with StableIValue outputs
  // If the operation corresponding to name takes in 4 inputs and returns 2
  // outputs, fn should expect stack to contain 4 StableIValues:
  //    [stable_arg1, stable_arg2, stable_arg3, stable_arg4]
  // to end, fn should fill the stack with 2 StableIValues representing outputs:
  //    [stable_ret1, stable_ret2, -, -]
  StableLibrary& impl(
      const char* name,
      void (*fn)(StableIValue*, uint64_t, uint64_t)) {
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
    torch_library_impl(lib_, name, fn, TORCH_ABI_VERSION);
#else
    aoti_torch_library_impl(lib_, name, fn);
#endif
    return *this;
  }

  // corresponds to a limited, stable version of torch::library::def()
  StableLibrary& def(const char* schema) {
    aoti_torch_library_def(lib_, schema);
    return *this;
  }
};

class StableTorchLibraryInit final {
 private:
  using InitFn = void(StableLibrary&);
  StableLibrary lib_;

 public:
  StableTorchLibraryInit(
      StableLibrary::Kind kind,
      InitFn* fn,
      const char* ns,
      const char* k,
      const char* file,
      uint32_t line)
      : lib_(kind, ns, k, file, line) {
    fn(lib_);
  }
};

// type mapper: since to<HeaderOnlyArrayRef<T>> cannot exist,
// we map that to to<std::vector<T>> to preserve ownership semantics.
// note that unbox_type_t is used to convert ParamTypes, so that
// the tuple holding the arguments will have proper ownership too.
template <typename T>
struct UnboxType {
  using type = T;
};

template <typename T>
struct UnboxType<torch::headeronly::HeaderOnlyArrayRef<T>> {
  using type = std::vector<T>;
};

template <typename T>
struct UnboxType<std::optional<torch::headeronly::HeaderOnlyArrayRef<T>>> {
  using type = std::optional<std::vector<T>>;
};

template <>
struct UnboxType<std::string_view> {
  using type = std::string;
};

// const and reference are stripped before UnboxType is applied
// in order to avoid ambiguous template matches
template <typename T>
using unbox_type_t =
    typename UnboxType<std::remove_cv_t<std::remove_reference_t<T>>>::type;

template <class... T, std::size_t... I>
std::tuple<T...> unbox_to_tuple_impl(
    StableIValue* stack,
    std::index_sequence<I...>) {
  return std::make_tuple(to<T>(stack[I])...);
}

template <class... T>
std::tuple<T...> unbox_to_tuple(StableIValue* stack) {
  return unbox_to_tuple_impl<T...>(
      stack, std::make_index_sequence<sizeof...(T)>());
}

template <class... T, std::size_t... I>
void box_from_tuple_impl(
    StableIValue* stack,
    std::tuple<T...> vals,
    std::index_sequence<I...>) {
  ((stack[I] = from<T>(std::get<I>(vals))), ...);
}

template <class... T>
void box_from_tuple(StableIValue* stack, std::tuple<T...> vals) {
  box_from_tuple_impl<T...>(
      stack, vals, std::make_index_sequence<sizeof...(T)>());
}

template <
    typename ReturnType,
    typename ParameterTypeList,
    typename FuncT,
    FuncT* func>
struct boxer_impl {
  static_assert(
      torch::headeronly::guts::false_t<ReturnType>::value,
      "Unsupported function schema for TORCH_BOX.");
};

// Multiple returns
template <
    typename... ReturnTypes,
    typename... ParameterTypes,
    typename FuncT,
    FuncT* func>
struct boxer_impl<
    std::tuple<ReturnTypes...>,
    torch::headeronly::guts::typelist::typelist<ParameterTypes...>,
    FuncT,
    func> {
  static void boxed_fn(
      StableIValue* stack,
      uint64_t num_args,
      uint64_t num_outputs) {
    STD_TORCH_CHECK(
        num_args == sizeof...(ParameterTypes),
        "Registered schema has ",
        num_args,
        " args, but the kernel to box has ",
        sizeof...(ParameterTypes));
    STD_TORCH_CHECK(
        num_outputs == sizeof...(ReturnTypes),
        "Registered schema has ",
        num_outputs,
        " outputs, but the kernel to box has ",
        sizeof...(ReturnTypes));
    std::tuple<unbox_type_t<ParameterTypes>...> args =
        unbox_to_tuple<unbox_type_t<ParameterTypes>...>(stack);
    auto res = std::apply(func, args);
    box_from_tuple<ReturnTypes...>(stack, res);
  }
};

// Single return
template <
    typename ReturnType,
    typename... ParameterTypes,
    typename FuncT,
    FuncT* func>
struct boxer_impl<
    ReturnType,
    torch::headeronly::guts::typelist::typelist<ParameterTypes...>,
    FuncT,
    func> {
  static void boxed_fn(
      StableIValue* stack,
      uint64_t num_args,
      uint64_t num_outputs) {
    STD_TORCH_CHECK(
        num_args == sizeof...(ParameterTypes),
        "Registered schema has ",
        num_args,
        " args, but the kernel to box has ",
        sizeof...(ParameterTypes));
    STD_TORCH_CHECK(
        num_outputs == 1,
        "Registered schema has ",
        num_outputs,
        " outputs, but the kernel to box has ",
        1);
    std::tuple<unbox_type_t<ParameterTypes>...> args =
        unbox_to_tuple<unbox_type_t<ParameterTypes>...>(stack);
    auto res = std::apply(func, args);
    stack[0] = from<ReturnType>(res);
  }
};

// No/void return
template <typename... ParameterTypes, typename FuncT, FuncT* func>
struct boxer_impl<
    void,
    torch::headeronly::guts::typelist::typelist<ParameterTypes...>,
    FuncT,
    func> {
  static void boxed_fn(
      StableIValue* stack,
      uint64_t num_args,
      uint64_t num_outputs) {
    STD_TORCH_CHECK(
        num_args == sizeof...(ParameterTypes),
        "Registered schema has ",
        num_args,
        " args, but the kernel to box has ",
        sizeof...(ParameterTypes));
    STD_TORCH_CHECK(
        num_outputs == 0,
        "Registered schema has ",
        num_outputs,
        " outputs, but the kernel to box has ",
        0);
    std::tuple<unbox_type_t<ParameterTypes>...> args =
        unbox_to_tuple<unbox_type_t<ParameterTypes>...>(stack);
    std::apply(func, args);
  }
};

template <typename FuncT, FuncT* func>
struct boxer {
  using FunctionTraits =
      torch::headeronly::guts::infer_function_traits_t<FuncT>;

  static void boxed_fn(
      StableIValue* stack,
      uint64_t num_args,
      uint64_t num_outputs) {
    boxer_impl<
        typename FunctionTraits::return_type,
        typename FunctionTraits::parameter_types,
        FuncT,
        func>::boxed_fn(stack, num_args, num_outputs);
  }
};

HIDDEN_NAMESPACE_END(torch, stable, detail)

/**
 * @brief Wraps a function to conform to the stable boxed kernel calling
 * convention.
 *
 * This macro takes an unboxed kernel function and generates a boxed wrapper
 * that can be registered with the stable library API. The boxed wrapper handles
 * conversion between StableIValue representations and native C++ types.
 *
 * @param func The unboxed kernel function to wrap. Must be a function pointer
 *             or a reference to a function.
 *
 * @return A pointer to the boxed function with signature:
 *         `void(StableIValue* stack, uint64_t num_inputs, uint64_t
 * num_outputs)`
 *
 * Example usage:
 * @code
 * Tensor my_kernel(const Tensor& input, int64_t size) {
 *     return input.reshape({size});
 * }
 *
 * STABLE_TORCH_LIBRARY_IMPL(my_namespace, CPU, m) {
 *     m.impl("my_op", TORCH_BOX(my_kernel));
 * }
 * @endcode
 */
#define TORCH_BOX(func)                                               \
  torch::stable::detail::boxer<                                       \
      std::remove_pointer_t<std::remove_reference_t<decltype(func)>>, \
      (func)>::boxed_fn

// macros copied from c10/macros/Macros.h
#ifdef __COUNTER__
#define STABLE_UID __COUNTER__
#else
#define STABLE_UID __LINE__
#endif

#define STABLE_CONCATENATE_IMPL(s1, s2) s1##s2
#define STABLE_CONCATENATE(s1, s2) STABLE_CONCATENATE_IMPL(s1, s2)
// end of macros copied from c10/macros/Macros.h

/**
 * @brief Registers operator implementations for a specific dispatch key using
 * the stable ABI.
 *
 * This is the stable ABI equivalent of `TORCH_LIBRARY_IMPL`. Use this macro
 * to provide implementations of operators for a specific dispatch key (e.g.,
 * CPU, CUDA) while maintaining binary compatibility across PyTorch versions.
 *
 * @note All kernel functions registered with this macro must be boxed using
 *       the `TORCH_BOX` macro. The boxed calling convention is required for
 *       stable ABI compatibility.
 *
 * @param ns The namespace in which the operators are defined (e.g., `myops`).
 * @param k The dispatch key for which implementations are being registered
 *          (e.g., `CPU`, `CUDA`).
 * @param m The name of the StableLibrary variable that will be available in the
 *          block for registering implementations.
 *
 * Example usage:
 * @code
 * STABLE_TORCH_LIBRARY_IMPL(myops, CPU, m) {
 *     m.impl("my_op", TORCH_BOX(my_cpu_kernel));
 * }
 *
 * STABLE_TORCH_LIBRARY_IMPL(myops, CUDA, m) {
 *     m.impl("my_op", TORCH_BOX(my_cuda_kernel));
 * }
 * @endcode
 *
 * @see STABLE_TORCH_LIBRARY for defining operator schemas
 * @see TORCH_BOX for wrapping kernel functions
 */
#define STABLE_TORCH_LIBRARY_IMPL(ns, k, m) \
  _STABLE_TORCH_LIBRARY_IMPL(ns, k, m, STABLE_UID)

#define _STABLE_TORCH_LIBRARY_IMPL(ns, k, m, uid)                             \
  static void STABLE_CONCATENATE(                                             \
      STABLE_TORCH_LIBRARY_IMPL_init_##ns##_##k##_,                           \
      uid)(torch::stable::detail::StableLibrary&);                            \
  static const torch::stable::detail::StableTorchLibraryInit                  \
      STABLE_CONCATENATE(                                                     \
          STABLE_TORCH_LIBRARY_IMPL_static_init_##ns##_##k##_, uid)(          \
          torch::stable::detail::StableLibrary::Kind::IMPL,                   \
          &STABLE_CONCATENATE(                                                \
              STABLE_TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid),             \
          #ns,                                                                \
          #k,                                                                 \
          __FILE__,                                                           \
          __LINE__);                                                          \
  void STABLE_CONCATENATE(STABLE_TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)( \
      torch::stable::detail::StableLibrary & m)

/**
 * @brief Defines a library of operators in a namespace using the stable ABI.
 *
 * This is the stable ABI equivalent of `TORCH_LIBRARY`. Use this macro to
 * define operator schemas that will maintain binary compatibility across
 * PyTorch versions. Only one `STABLE_TORCH_LIBRARY` block can exist per
 * namespace; use `STABLE_TORCH_LIBRARY_FRAGMENT` for additional definitions
 * in the same namespace from different translation units.
 *
 * @param ns The namespace in which to define operators (e.g., `myops`).
 *           This should be a valid C++ identifier.
 * @param m The name of the StableLibrary variable that will be available in the
 *          block for defining operator schemas.
 *
 * Example usage:
 * @code
 * STABLE_TORCH_LIBRARY(myops, m) {
 *     m.def("my_op(Tensor input, int size) -> Tensor");
 *     m.def("another_op(Tensor a, Tensor b) -> Tensor");
 * }
 * @endcode
 *
 * @see STABLE_TORCH_LIBRARY_IMPL for registering implementations
 * @see STABLE_TORCH_LIBRARY_FRAGMENT for extending a namespace
 */
#define STABLE_TORCH_LIBRARY(ns, m)                          \
  static void STABLE_TORCH_LIBRARY_init_##ns(                \
      torch::stable::detail::StableLibrary&);                \
  static const torch::stable::detail::StableTorchLibraryInit \
      STABLE_TORCH_LIBRARY_static_init_##ns(                 \
          torch::stable::detail::StableLibrary::Kind::DEF,   \
          &STABLE_TORCH_LIBRARY_init_##ns,                   \
          #ns,                                               \
          nullptr,                                           \
          __FILE__,                                          \
          __LINE__);                                         \
  void STABLE_TORCH_LIBRARY_init_##ns(torch::stable::detail::StableLibrary& m)

/**
 * @brief Extends operator definitions in an existing namespace using the stable
 * ABI.
 *
 * This is the stable ABI equivalent of `TORCH_LIBRARY_FRAGMENT`. Use this macro
 * to add additional operator definitions to a namespace that was already
 * created with `STABLE_TORCH_LIBRARY`. This is useful when operator definitions
 * need to be split across multiple translation units or files.
 *
 * @param ns The namespace to extend (must match a namespace previously defined
 *           with `STABLE_TORCH_LIBRARY`).
 * @param m The name of the StableLibrary variable that will be available in the
 *          block for defining operator schemas.
 *
 * Example usage:
 * @code
 * // In file1.cpp
 * STABLE_TORCH_LIBRARY(myops, m) {
 *     m.def("op1(Tensor input) -> Tensor");
 * }
 *
 * // In file2.cpp
 * STABLE_TORCH_LIBRARY_FRAGMENT(myops, m) {
 *     m.def("op2(Tensor input) -> Tensor");
 * }
 * @endcode
 *
 * @see STABLE_TORCH_LIBRARY for initial namespace definition
 * @see STABLE_TORCH_LIBRARY_IMPL for registering implementations
 */
#define STABLE_TORCH_LIBRARY_FRAGMENT(ns, m) \
  _STABLE_TORCH_LIBRARY_FRAGMENT(ns, m, STABLE_UID)

#define _STABLE_TORCH_LIBRARY_FRAGMENT(ns, m, uid)                          \
  static void STABLE_CONCATENATE(                                           \
      STABLE_TORCH_LIBRARY_FRAGMENT_init_##ns##_,                           \
      uid)(torch::stable::detail::StableLibrary&);                          \
  static const torch::stable::detail::StableTorchLibraryInit                \
      STABLE_CONCATENATE(                                                   \
          STABLE_TORCH_LIBRARY_FRAGMENT_static_init_##ns##_, uid)(          \
          torch::stable::detail::StableLibrary::Kind::FRAGMENT,             \
          &STABLE_CONCATENATE(                                              \
              STABLE_TORCH_LIBRARY_FRAGMENT_init_##ns##_, uid),             \
          #ns,                                                              \
          nullptr,                                                          \
          __FILE__,                                                         \
          __LINE__);                                                        \
  void STABLE_CONCATENATE(STABLE_TORCH_LIBRARY_FRAGMENT_init_##ns##_, uid)( \
      torch::stable::detail::StableLibrary & m)
