#pragma once

#include <c10/util/Metaprogramming.h>
#include <c10/util/TypeTraits.h>

namespace torch {
namespace jit {

namespace detail {

// Argument type utilities
template <class R, class...>
struct types {
  using type = types;
};

template <typename Method>
struct WrapMethod;

template <typename R, typename CurrClass, typename... Args>
struct WrapMethod<R (CurrClass::*)(Args...)> {
  WrapMethod(R (CurrClass::*m)(Args...)) : m(std::move(m)) {}

  R operator()(c10::intrusive_ptr<CurrClass> cur, Args... args) {
    return c10::guts::invoke(m, *cur, args...);
  }

  R (CurrClass::*m)(Args...);
};

template <typename R, typename CurrClass, typename... Args>
struct WrapMethod<R (CurrClass::*)(Args...) const> {
  WrapMethod(R (CurrClass::*m)(Args...) const) : m(std::move(m)) {}

  R operator()(c10::intrusive_ptr<CurrClass> cur, Args... args) {
    return c10::guts::invoke(m, *cur, args...);
  }

  R (CurrClass::*m)(Args...) const;
};

// Adapter for different callable types
template <
    typename CurClass,
    typename Func,
    std::enable_if_t<
        std::is_member_function_pointer<std::decay_t<Func>>::value,
        bool> = false>
WrapMethod<Func> wrap_func(Func f) {
  return WrapMethod<Func>(std::move(f));
}

template <
    typename CurClass,
    typename Func,
    std::enable_if_t<
        !std::is_member_function_pointer<std::decay_t<Func>>::value,
        bool> = false>
Func wrap_func(Func f) {
  return f;
}

struct RegisteredClassRecord {
  std::string qualClassName;
  std::string classTypeidName_intrusive_ptr;
  std::string classTypeidName_tagged_capsule;

  // Store registered methods. This maps the method name
  // (e.g. "forward", "__getstate__") to the qualified
  // name for the registered op.
  std::unordered_map<std::string, std::string> registeredMethods;
};

} // namespace detail

TORCH_API std::vector<c10::RegisterOperators>& registeredOps();
TORCH_API std::unordered_map<std::string, detail::RegisteredClassRecord>& registeredClasses();
TORCH_API void invokeClassRegistrationCallbacks(const detail::RegisteredClassRecord& class_record);
TORCH_API void invokeMethodRegistrationCallbacks(const detail::RegisteredClassRecord& class_record, const std::string& method_name);

} // namespace jit
} // namespace torch
