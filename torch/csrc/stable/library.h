// this file can only have stable stuff! Akin to shim.h
// but unlike shim.h, this file can contain header-only C++
// code for better UX.

#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <optional>

// use anonymous namespace to avoid collisions between differing
// versions of this file that may be included by different sources
namespace {

namespace detail {
// utility functions to detect optional
template <typename V>
struct is_optional : std::false_type {};
template <typename V>
struct is_optional<std::optional<V>> : std::true_type {};
} // namespace detail

template <
    typename T,
    std::enable_if_t<!detail::is_optional<T>::value, bool> = true>
StableIValue from(T val) {
  static_assert(
      sizeof(T) <= sizeof(StableIValue),
      "StableLibrary stack does not support parameter types larger than 64 bits.");
  return *reinterpret_cast<StableIValue*>(&val);
}

// Specialization for std::nullopt_t
template <>
StableIValue from(std::nullopt_t val) {
  return from(nullptr);
}

// Specialization for std::optional
// [Handling std::optional]
// When the schema is represented by an optional type, say int?, then we
// expect the custom extension representation to be a std::optional<int>
// (critically NOT int!). In order for all parameters to be stably parsed and
// handled by our dispatcher, we liaison custom extension parameters through
// boxed kernels, meaning that every value will make its way to be an IValue:
//
// custom extension value --(from)-> StableIValue --(to_ivalue)-> IValue
//
// When the custom extension value is a literal that can be trivially
// casted to StableIValue, e.g., an int, a float, a pointer, this route is
// ...trivial. The below specialization is for a case when the custom
// extension value would NOT fit within a StableIValue: a std::optional.
//
// If the std::optional has no value, it is treated as std::nullopt,
// whose StableIValue representation is from(nullptr). Otherwise, we:
// 1. unwrap the std::optional<T>
// 2. recursively convert its value of type T to a StableIValue
// 3. allocate heap space for said StableIValue
// 4. convert the resulting StableIValue* into a StableIValue
//
// note that this allocates heap memory! which we expect to be cleaned
// up in the to_ivalue() function defined in shim_common.cpp. We
// purposefully hide this implementation detail from the user so that
// all the user needs to know is:
//
// The schema requests an optional (T?) so I must call `from` on a
// std::optional<T> or a std::nullopt.
template <typename T>
StableIValue from(std::optional<T> val) {
  if (!val.has_value()) {
    return from(std::nullopt);
  }
  StableIValue* heap_val = new StableIValue(from(val.value()));
  return from(heap_val);
}

template <
    typename T,
    std::enable_if_t<!detail::is_optional<T>::value, bool> = true>
T to(StableIValue val) {
  return *reinterpret_cast<T*>(&val);
}

template <
    typename T,
    std::enable_if_t<std::is_same_v<T, std::nullopt_t>, bool> = true>
T to(StableIValue val) {
  // val should be equivalent to from(nullptr)
  return std::nullopt;
}

// Specialization for std::optional, see [Handling std::optional] above
// as the semantic is the same but in reverse direction as we go from
// IValue --(from_ivalue)-> StableIValue --(to<T>)-> T in custom extension
template <
    typename T,
    std::enable_if_t<detail::is_optional<T>::value, bool> = true>
T to(StableIValue val) {
  using V = typename T::value_type;
  auto sivp = to<StableIValue*>(val);

  // sivp is either nullptr or a pointer to a StableIValue
  if (sivp == nullptr) {
    return {};
  }
  auto inner_val = to<V>(*sivp);

  // free the memory associated with StableIValue* sivp
  delete sivp;

  return std::make_optional(inner_val);
}
// end to helpers for converting between StableIValue and actual IValues

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
