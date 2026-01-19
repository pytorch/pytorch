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
    std::index_sequence<I...> /*unused*/) {
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
    std::index_sequence<I...> /*unused*/) {
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
