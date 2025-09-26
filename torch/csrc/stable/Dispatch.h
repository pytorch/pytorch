#pragma once
/*
  This file defines STABLE_DISPATCH_... macros for torch stable ABI
  that are expected to implement the same interface and semantics as
  the corresponding AT_DISPATCH_... macros in ATen/Dispatch.h.

  Note that only a subset of AT_DISPATCH_... macros have the
  corresponding STABLE_DISPATCH_... macros defined here (in
  particular, the ..._AND? macro variants are skipped) but the set may
  be extended in the future based on the actual need.
*/

#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

namespace torch::stable {

using torch::headeronly::ScalarType;

namespace impl {

inline const char* toString(ScalarType t) {
#define DEFINE_CASE(_, name) \
  case ScalarType::name:     \
    return #name;

  switch (t) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CASE)
    default:
      return "UNKNOWN_SCALAR";
  }
#undef DEFINE_CASE
}

template <ScalarType N>
struct ScalarTypeToCPPType;

#define SPECIALIZE_ScalarTypeToCPPType(cpp_type, scalar_type) \
  template <>                                                 \
  struct ScalarTypeToCPPType<ScalarType::scalar_type> {       \
    using type = cpp_type;                                    \
  };

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(SPECIALIZE_ScalarTypeToCPPType)

#undef SPECIALIZE_ScalarTypeToCPPType

template <ScalarType N>
using ScalarTypeToCPPTypeT = typename ScalarTypeToCPPType<N>::type;

} // namespace impl

} // namespace torch::stable

#define STABLE_DISPATCH_CASE(enum_type, HINT, ...)            \
  case enum_type: {                                           \
    using HINT [[maybe_unused]] =                             \
        torch::stable::impl::ScalarTypeToCPPTypeT<enum_type>; \
    return __VA_ARGS__();                                     \
  }

#define STABLE_DISPATCH_SWITCH(TYPE, NAME, ...)        \
  [&] {                                                \
    const auto& the_type = TYPE;                       \
    constexpr const char* stable_dispatch_name = NAME; \
    switch (the_type) {                                \
      __VA_ARGS__                                      \
      default:                                         \
        STD_TORCH_CHECK(                               \
            false,                                     \
            '"',                                       \
            stable_dispatch_name,                      \
            "\" not implemented for '",                \
            torch::stable::impl::toString(the_type),   \
            "'");                                      \
    }                                                  \
  }()

#define STABLE_DISPATCH_CASE_FLOATING_TYPES(...)                           \
  STABLE_DISPATCH_CASE(torch::headeronly::ScalarType::Double, __VA_ARGS__) \
  STABLE_DISPATCH_CASE(torch::headeronly::ScalarType::Float, __VA_ARGS__)

#define STABLE_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  STABLE_DISPATCH_SWITCH(                               \
      TYPE, NAME, STABLE_DISPATCH_CASE_FLOATING_TYPES(scalar_t, __VA_ARGS__))

#define STABLE_DISPATCH_CASE_FLOATING_TYPES_AND_HALF(...)                  \
  STABLE_DISPATCH_CASE(torch::headeronly::ScalarType::Double, __VA_ARGS__) \
  STABLE_DISPATCH_CASE(torch::headeronly::ScalarType::Float, __VA_ARGS__)  \
  STABLE_DISPATCH_CASE(torch::headeronly::ScalarType::Half, __VA_ARGS__)

#define STABLE_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...) \
  STABLE_DISPATCH_SWITCH(                                        \
      TYPE,                                                      \
      NAME,                                                      \
      STABLE_DISPATCH_CASE_FLOATING_TYPES_AND_HALF(scalar_t, __VA_ARGS__))

#define STABLE_DISPATCH_CASE_REDUCED_FLOATING_TYPES(...)                 \
  STABLE_DISPATCH_CASE(torch::headeronly::ScalarType::Half, __VA_ARGS__) \
  STABLE_DISPATCH_CASE(torch::headeronly::ScalarType::BFloat16, __VA_ARGS__)

#define STABLE_DISPATCH_REDUCED_FLOATING_TYPES(TYPE, NAME, ...) \
  STABLE_DISPATCH_SWITCH(                                       \
      TYPE,                                                     \
      NAME,                                                     \
      STABLE_DISPATCH_CASE_REDUCED_FLOATING_TYPES(scalar_t, __VA_ARGS__))

#define STABLE_DISPATCH_CASE_COMPLEX_TYPES(...)                  \
  STABLE_DISPATCH_CASE(                                          \
      torch::headeronly::ScalarType::ComplexDouble, __VA_ARGS__) \
  STABLE_DISPATCH_CASE(torch::headeronly::ScalarType::ComplexFloat, __VA_ARGS__)

#define STABLE_DISPATCH_COMPLEX_TYPES(TYPE, NAME, ...) \
  STABLE_DISPATCH_SWITCH(                              \
      TYPE, NAME, STABLE_DISPATCH_CASE_COMPLEX_TYPES(scalar_t, __VA_ARGS__))

#define STABLE_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES(...) \
  STABLE_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__)           \
  STABLE_DISPATCH_CASE_COMPLEX_TYPES(__VA_ARGS__)

#define STABLE_DISPATCH_FLOATING_AND_COMPLEX_TYPES(TYPE, NAME, ...) \
  STABLE_DISPATCH_SWITCH(                                           \
      TYPE,                                                         \
      NAME,                                                         \
      STABLE_DISPATCH_CASE_FLOATING_AND_COMPLEX_TYPES(scalar_t, __VA_ARGS__))

#define STABLE_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(...) \
  STABLE_DISPATCH_CASE_ALL_TYPES(__VA_ARGS__)           \
  STABLE_DISPATCH_CASE_COMPLEX_TYPES(__VA_ARGS__)

#define STABLE_DISPATCH_ALL_TYPES_AND_COMPLEX(TYPE, NAME, ...) \
  STABLE_DISPATCH_SWITCH(                                      \
      TYPE,                                                    \
      NAME,                                                    \
      STABLE_DISPATCH_CASE_ALL_TYPES_AND_COMPLEX(scalar_t, __VA_ARGS__))

#define STABLE_DISPATCH_CASE_INTEGRAL_TYPES(...)                         \
  STABLE_DISPATCH_CASE(torch::headeronly::ScalarType::Byte, __VA_ARGS__) \
  STABLE_DISPATCH_CASE(torch::headeronly::ScalarType::Char, __VA_ARGS__) \
  STABLE_DISPATCH_CASE(torch::headeronly::ScalarType::Int, __VA_ARGS__)  \
  STABLE_DISPATCH_CASE(torch::headeronly::ScalarType::Long, __VA_ARGS__) \
  STABLE_DISPATCH_CASE(torch::headeronly::ScalarType::Short, __VA_ARGS__)

#define STABLE_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...) \
  STABLE_DISPATCH_SWITCH(                               \
      TYPE, NAME, STABLE_DISPATCH_CASE_INTEGRAL_TYPES(scalar_t, __VA_ARGS__))

#define STABLE_DISPATCH_INDEX_TYPES(TYPE, NAME, ...)                \
  STABLE_DISPATCH_SWITCH(                                           \
      TYPE,                                                         \
      NAME,                                                         \
      STABLE_DISPATCH_CASE(                                         \
          torch::headeronly::ScalarType::Int, index_t, __VA_ARGS__) \
          STABLE_DISPATCH_CASE(                                     \
              torch::headeronly::ScalarType::Long, index_t, __VA_ARGS__))

#define STABLE_DISPATCH_CASE_ALL_TYPES(...)        \
  STABLE_DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__) \
  STABLE_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__)

#define STABLE_DISPATCH_ALL_TYPES(TYPE, NAME, ...) \
  STABLE_DISPATCH_SWITCH(                          \
      TYPE, NAME, STABLE_DISPATCH_CASE_ALL_TYPES(scalar_t, __VA_ARGS__))

#define STABLE_DISPATCH_CASE_ALL_TYPES_AND(SCALARTYPE, ...) \
  STABLE_DISPATCH_CASE_ALL_TYPES(__VA_ARGS__)               \
  STABLE_DISPATCH_CASE(SCALARTYPE, __VA_ARGS__)

#define STABLE_DISPATCH_ALL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  STABLE_DISPATCH_SWITCH(                                          \
      TYPE,                                                        \
      NAME,                                                        \
      STABLE_DISPATCH_CASE_ALL_TYPES_AND(SCALARTYPE, scalar_t, __VA_ARGS__))
