#pragma once

#include <torch/headeronly/core/ScalarType.h>

#define AT_PRIVATE_CASE_TYPE_USING_HINT_TMPL(PRELUDE, enum_type, HINT, ...)   \
  case enum_type: {                                                           \
    PRELUDE(enum_type);                                                       \
    using HINT [[maybe_unused]] = c10::impl::ScalarTypeToCPPTypeT<enum_type>; \
    return __VA_ARGS__();                                                     \
  }

#define AT_DISPATCH_CASE_TMPL(CASE_TYPE_USING_HINT, enum_type, ...) \
  CASE_TYPE_USING_HINT(enum_type, scalar_t, __VA_ARGS__)

namespace detail {
inline c10::ScalarType scalar_type(c10::ScalarType s) {
  return s;
}
} // namespace detail

#define AT_DISPATCH_SWITCH_TMPL(                                            \
    PRELUDE, CHECK_NOT_IMPLEMENTED, TYPE, NAME, ...)                        \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    constexpr const char* at_dispatch_name = NAME;                          \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    c10::ScalarType _st = ::detail::scalar_type(the_type);                  \
    PRELUDE(at_dispatch_name, _st);                                         \
    switch (_st) {                                                          \
      __VA_ARGS__                                                           \
      default:                                                              \
        CHECK_NOT_IMPLEMENTED(                                              \
            false,                                                          \
            '"',                                                            \
            at_dispatch_name,                                               \
            "\" not implemented for '",                                     \
            c10::toString(_st),                                             \
            "'");                                                           \
    }                                                                       \
  }()

// THO_EMPTY is a helper macro that discards its arguments.
#define THO_EMPTY(...)

// THO_PRIVATE_CASE_TYPE_USING_HINT is same as
// AT_PRIVATE_CASE_TYPE_USING_HINT with call to macro
// AT_PRIVATE_CHECK_SELECTIVE_BUILD removed.
#define THO_PRIVATE_CASE_TYPE_USING_HINT(enum_type, HINT, ...) \
  AT_PRIVATE_CASE_TYPE_USING_HINT_TMPL(THO_EMPTY, enum_type, HINT, __VA_ARGS__)

// THO_DISPATCH_SWITCH is same as AT_DISPATCH_SWITCH with call to
// macro RECORD_KERNEL_FUNCTION_DTYPE removed and using
// STD_TORCH_CHECK instead of TORCH_CHECK_NOT_IMPLEMENTED.
#define THO_DISPATCH_SWITCH(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH_TMPL(THO_EMPTY, STD_TORCH_CHECK, TYPE, NAME, __VA_ARGS__)

// THO_DISPATCH_CASE is same as AT_DISPATCH_CASE but using
// THO_PRIVATE_CASE_TYPE_USING_HINT instead of
// AT_PRIVATE_CASE_TYPE_USING_HINT.
#define THO_DISPATCH_CASE(enum_type, ...) \
  AT_DISPATCH_CASE_TMPL(                  \
      THO_PRIVATE_CASE_TYPE_USING_HINT, enum_type, __VA_ARGS__)
