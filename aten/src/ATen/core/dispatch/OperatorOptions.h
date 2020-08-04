#pragma once

#include <cstdint>

namespace c10 {

// AliasAnalysisKind is a hint for the TorchScript JIT. It allows the JIT to
// reason about the return values and arguments passes to registered custom
// operations.
//
// If aliasAnalysisKind is not specified then default is CONSERVATIVE.
//
// CONSERVATIVE:TorchScript assumes there are side effects in every input
// arguments.
//   "foo(Tensor x, Tensor y, Tensor z) -> Tensor"
//  there are side effects in x , y and z and return value can alias any
//  input argument.
//
// FROM_SCHEMA: TorchScript uses the SCHEMA to identify which input
// argument is mutated.
//     "bar(Tensor(a) x, Tensor y, Tensor z) -> Tensor(a)"
// x is mutated but not y and z.
//
//     "bar(Tensor(a!) x, Tensor y, Tensor z) -> ()"
// x is inplace mutated.
//
// PURE_FUNCTION: TorchScript assumes there are no side effect in any
// input argument and return value does not alias them.
//
// INTERNAL_SPECIAL_CASE: do not use
enum class AliasAnalysisKind : uint8_t {
  INTERNAL_SPECIAL_CASE,
  CONSERVATIVE, // The most conservative alias analysis type, assumes
                // side-effects. This is the default analysis.
  FROM_SCHEMA,
  PURE_FUNCTION
};

#if !defined(_MSC_VER)
constexpr // Our current MSVC version has a bug that doesn't allow this to be constexpr.
#endif
inline const char* toString(AliasAnalysisKind aliasAnalysisKind) {
  return (aliasAnalysisKind == AliasAnalysisKind::CONSERVATIVE)
      ? "CONSERVATIVE"
      : (aliasAnalysisKind == AliasAnalysisKind::FROM_SCHEMA)
          ? "FROM_SCHEMA"
          : (aliasAnalysisKind == AliasAnalysisKind::PURE_FUNCTION)
              ? "PURE_FUNCTION"
              : (aliasAnalysisKind == AliasAnalysisKind::INTERNAL_SPECIAL_CASE)
                  ? "INTERNAL_SPECIAL_CASE"
                  : "UNKNOWN";
}

} // namespace c10
