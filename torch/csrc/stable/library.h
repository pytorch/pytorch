#pragma once
// this file can only have stable stuff! Akin to shim.h
// but unlike shim.h, this file can contain header-only C++
// code for better UX.

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/Metaprogramming.h>

// Technically, this file doesn't use anything from stableivalue_conversions.h,
// but we need to include it here as the contents of stableivalue_conversions.h
// used to live here and so we need to expose them for backwards compatibility.
#include <torch/csrc/stable/stableivalue_conversions.h>

// use anonymous namespace to avoid collisions between differing
// versions of this file that may be included by different sources
namespace {

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
    aoti_torch_library_impl(lib_, name, fn);
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

} // namespace

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
      STABLE_TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(StableLibrary&);     \
  static const StableTorchLibraryInit STABLE_CONCATENATE(                     \
      STABLE_TORCH_LIBRARY_IMPL_static_init_##ns##_##k##_, uid)(              \
      StableLibrary::Kind::IMPL,                                              \
      &STABLE_CONCATENATE(STABLE_TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid), \
      #ns,                                                                    \
      #k,                                                                     \
      __FILE__,                                                               \
      __LINE__);                                                              \
  void STABLE_CONCATENATE(                                                    \
      STABLE_TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(StableLibrary & m)

#define STABLE_TORCH_LIBRARY(ns, m)                                          \
  static void STABLE_TORCH_LIBRARY_init_##ns(StableLibrary&);                \
  static const StableTorchLibraryInit STABLE_TORCH_LIBRARY_static_init_##ns( \
      StableLibrary::Kind::DEF,                                              \
      &STABLE_TORCH_LIBRARY_init_##ns,                                       \
      #ns,                                                                   \
      nullptr,                                                               \
      __FILE__,                                                              \
      __LINE__);                                                             \
  void STABLE_TORCH_LIBRARY_init_##ns(StableLibrary& m)

#define STABLE_TORCH_LIBRARY_FRAGMENT(ns, m) \
  _STABLE_TORCH_LIBRARY_FRAGMENT(ns, m, STABLE_UID)

#define _STABLE_TORCH_LIBRARY_FRAGMENT(ns, m, uid)                          \
  static void STABLE_CONCATENATE(                                           \
      STABLE_TORCH_LIBRARY_FRAGMENT_init_##ns##_, uid)(StableLibrary&);     \
  static const StableTorchLibraryInit STABLE_CONCATENATE(                   \
      STABLE_TORCH_LIBRARY_FRAGMENT_static_init_##ns##_, uid)(              \
      StableLibrary::Kind::FRAGMENT,                                        \
      &STABLE_CONCATENATE(STABLE_TORCH_LIBRARY_FRAGMENT_init_##ns##_, uid), \
      #ns,                                                                  \
      nullptr,                                                              \
      __FILE__,                                                             \
      __LINE__);                                                            \
  void STABLE_CONCATENATE(                                                  \
      STABLE_TORCH_LIBRARY_FRAGMENT_init_##ns##_, uid)(StableLibrary & m)

namespace {

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
  static_assert(c10::guts::false_t<ReturnType>::value, "Unsupported");
};

template <
    typename... ReturnTypes,
    typename... ParameterTypes,
    typename FuncT,
    FuncT* func>
struct boxer_impl<
    std::tuple<ReturnTypes...>,
    c10::guts::typelist::typelist<ParameterTypes...>,
    FuncT,
    func> {
  static void boxed_fn(
      StableIValue* stack,
      uint64_t num_args,
      uint64_t num_outputs) {
    STD_TORCH_CHECK(
        num_args == sizeof...(ParameterTypes),
        "Expected ",
        num_args,
        " args, got ",
        sizeof...(ParameterTypes));
    STD_TORCH_CHECK(
        num_outputs == sizeof...(ReturnTypes),
        "Expected ",
        num_outputs,
        " outputs, got ",
        sizeof...(ReturnTypes));
    std::tuple<ParameterTypes...> args =
        unbox_to_tuple<ParameterTypes...>(stack);
    auto res = std::apply(func, args);
    box_from_tuple<ReturnTypes...>(stack, res);
  }
};

template <
    typename ReturnType,
    typename... ParameterTypes,
    typename FuncT,
    FuncT* func>
struct boxer_impl<
    ReturnType,
    c10::guts::typelist::typelist<ParameterTypes...>,
    FuncT,
    func> {
  static void boxed_fn(
      StableIValue* stack,
      uint64_t num_args,
      uint64_t num_outputs) {
    STD_TORCH_CHECK(
        num_args == sizeof...(ParameterTypes),
        "Expected ",
        num_args,
        " args, got ",
        sizeof...(ParameterTypes));
    STD_TORCH_CHECK(
        num_outputs == 1, "Expected ", num_outputs, " outputs, got ", 1);
    std::tuple<ParameterTypes...> args =
        unbox_to_tuple<ParameterTypes...>(stack);
    auto res = std::apply(func, args);
    stack[0] = from<ReturnType>(res);
  }
};

template <typename FuncT, FuncT* func>
struct boxer {
  using FunctionTraits = c10::guts::infer_function_traits_t<FuncT>;

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

} // namespace

#define TORCH_BOX(func)                                               \
  boxer<                                                              \
      std::remove_pointer_t<std::remove_reference_t<decltype(func)>>, \
      (func)>::boxed_fn
