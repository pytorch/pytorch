#pragma once

#include <type_traits>

#include <c10/macros/Macros.h>

namespace torch {

template <typename To, typename From>
To unsafe_cast_function(From func) {
  static_assert(std::is_pointer_v<To>);
  static_assert(std::is_function_v<std::remove_pointer_t<To>>);
  static_assert(std::is_pointer_v<From>);
  static_assert(std::is_function_v<std::remove_pointer_t<From>>);

  C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wcast-function-type")
  C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wcast-function-type-strict")
  return reinterpret_cast<To>(func);
  C10_DIAGNOSTIC_POP()
  C10_DIAGNOSTIC_POP()
}

} // namespace torch
