#pragma once
// this file can only have stable stuff! Akin to shim.h
// but unlike shim.h, this file can contain header-only C++
// code for better UX.

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/tensor.h>

#include <optional>

// use anonymous namespace to avoid collisions between differing
// versions of this file that may be included by different sources
namespace {

namespace detail {
/// Utility to detect optional types
template <typename T>
struct is_optional : std::false_type {};
template <typename T>
struct is_optional<std::optional<T>> : std::true_type {};
template <typename T>
inline constexpr bool is_optional_v = is_optional<T>::value;

// Utility to detect if type is torch::stable::Tensor
template <typename T>
struct is_stable_tensor : std::false_type {};
template <>
struct is_stable_tensor<torch::stable::Tensor> : std::true_type {};
template <typename T>
inline constexpr bool is_stable_tensor_v = is_stable_tensor<T>::value;

// Utility to detect nullopt_t
template <typename T>
struct is_nullopt : std::false_type {};
template <>
struct is_nullopt<std::nullopt_t> : std::true_type {};
template <typename T>
inline constexpr bool is_nullopt_v = is_nullopt<T>::value;

// Combined check for all non-standard types except nullopt_t
template <typename T>
inline constexpr bool is_special_type_v =
    is_stable_tensor_v<T> || is_optional_v<T> || is_nullopt_v<T>;
} // namespace detail

// =============================================================================
// FROM CONVERSIONS (T -> StableIValue)
// =============================================================================

// Specialization for general copyable types (catch-all)
template <typename T>
std::enable_if_t<!detail::is_special_type_v<T>, StableIValue> from(T val) {
  static_assert(
      sizeof(T) <= sizeof(StableIValue),
      "StableLibrary stack does not support parameter types larger than 64 bits.");
  static_assert(std::is_trivially_copyable_v<T>);
  // Initialization should be cheap enough; let's give people well-specified
  // reproducible behavior.
  StableIValue result = 0;
  // NOTE [-Wclass-memaccess ]: reinterpret_cast to suppress
  // overzealous -Wclass-memaccess. (see
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=107361) We have a
  // static_assert above that T is trivially copyable, which should be
  // enough.
  std::memcpy(&result, reinterpret_cast<void*>(&val), sizeof(val));
  return result;
}

// Specialization for std::nullopt_t
template <typename T>
std::enable_if_t<std::is_same_v<T, std::nullopt_t>, StableIValue> from(T val) {
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
std::enable_if_t<detail::is_optional_v<T>, StableIValue> from(T val) {
  if (!val.has_value()) {
    return from(std::nullopt);
  }
  StableIValue* heap_val = new StableIValue(from(val.value()));
  return from(heap_val);
}

// Specialization for torch::stable::Tensor
// I NEED TO REDOCUMENT EVERYTHING BUT SHOULD SLEEP RN SO I'LL COME BACK TO THIS
// I'M JUST GLAD MY TEMPLATE SPECIALIZATIONS FINALLY COMPILE AND PASS TESTS
// The following is our way of incrementing the refcount of the underlying
// Tensor that we point to. Why do we want this supposedly weird behavior?
// Because! We expect users to only need a StableIValue when they are trying
// to pass the Tensor into a stack-based API, e,g.,
// aoti_torch_call_dispatcher.
//
// A stack-based API is one that expects a stack of inputs converted to
// StableIValues. Our contract with any stack-based API is that the stack
// has ownership of its Tensor arguments. Since this torch::stable::Tensor
// object will likely go out of scope by the end of the user extension's
// local function and will thus delete its reference on the at::Tensor,
// we create a new AtenTensorHandle for that use case.
template <typename T>
std::enable_if_t<detail::is_stable_tensor_v<T>, StableIValue> from(T val) {
  AtenTensorHandle new_ath;
  aoti_torch_new_tensor_handle(val.get(), &new_ath);
  return from(new_ath);
}

// =============================================================================
// TO CONVERSIONS (StableIValue -> T)
// =============================================================================

// Specialization for general copyable types (catch-all)
template <typename T>
std::enable_if_t<!detail::is_special_type_v<T>, T> to(StableIValue val) {
  static_assert(std::is_trivially_copyable_v<T>);
  // T may not have a default constructor. (For example, it might be
  // c10::Device.) However, std::memcpy implicitly creates a T at the
  // destination. So, we can use a union to work around this lack of
  // default constructor.
  union Result {
    Result() {}
    T t;
  };
  Result result;
  // See NOTE[ -Wclass-memaccess ] above.
  std::memcpy(reinterpret_cast<void*>(&result.t), &val, sizeof(result));
  return result.t;
}

// Specialization for std::nullopt_t
template <typename T>
std::enable_if_t<detail::is_nullopt_v<T>, T> to(StableIValue val) {
  // val should be equivalent to from(nullptr)
  return std::nullopt;
}

// Specialization for std::optional, see [Handling std::optional] above
// as the semantic is the same but in reverse direction as we go from
// IValue --(from_ivalue)-> StableIValue --(to<T>)-> T in custom extension
template <typename T>
std::enable_if_t<detail::is_optional_v<T>, T> to(StableIValue val) {
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

// Specialization for torch::stable::Tensor
template <typename T>
std::enable_if_t<detail::is_stable_tensor_v<T>, T> to(StableIValue val) {
  return torch::stable::Tensor(to<AtenTensorHandle>(val));
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
