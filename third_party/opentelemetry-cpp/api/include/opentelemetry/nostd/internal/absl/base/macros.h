//
// Copyright 2017 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// -----------------------------------------------------------------------------
// File: macros.h
// -----------------------------------------------------------------------------
//
// This header file defines the set of language macros used within Abseil code.
// For the set of macros used to determine supported compilers and platforms,
// see absl/base/config.h instead.
//
// This code is compiled directly on many platforms, including client
// platforms like Windows, Mac, and embedded systems.  Before making
// any changes here, make sure that you're not breaking any platforms.

#ifndef OTABSL_BASE_MACROS_H_
#define OTABSL_BASE_MACROS_H_

#include <cassert>
#include <cstddef>

#include "attributes.h"
#include "optimization.h"
#include "port.h"

// OTABSL_ARRAYSIZE()
//
// Returns the number of elements in an array as a compile-time constant, which
// can be used in defining new arrays. If you use this macro on a pointer by
// mistake, you will get a compile-time error.
#define OTABSL_ARRAYSIZE(array) \
  (sizeof(::absl::macros_internal::ArraySizeHelper(array)))

namespace absl {
OTABSL_NAMESPACE_BEGIN
namespace macros_internal {
// Note: this internal template function declaration is used by OTABSL_ARRAYSIZE.
// The function doesn't need a definition, as we only use its type.
template <typename T, size_t N>
auto ArraySizeHelper(const T (&array)[N]) -> char (&)[N];
}  // namespace macros_internal
OTABSL_NAMESPACE_END
}  // namespace absl

// kLinkerInitialized
//
// An enum used only as a constructor argument to indicate that a variable has
// static storage duration, and that the constructor should do nothing to its
// state. Use of this macro indicates to the reader that it is legal to
// declare a static instance of the class, provided the constructor is given
// the absl::base_internal::kLinkerInitialized argument.
//
// Normally, it is unsafe to declare a static variable that has a constructor or
// a destructor because invocation order is undefined. However, if the type can
// be zero-initialized (which the loader does for static variables) into a valid
// state and the type's destructor does not affect storage, then a constructor
// for static initialization can be declared.
//
// Example:
//       // Declaration
//       explicit MyClass(absl::base_internal:LinkerInitialized x) {}
//
//       // Invocation
//       static MyClass my_global(absl::base_internal::kLinkerInitialized);
namespace absl {
OTABSL_NAMESPACE_BEGIN
namespace base_internal {
enum LinkerInitialized {
  kLinkerInitialized = 0,
};
}  // namespace base_internal
OTABSL_NAMESPACE_END
}  // namespace absl

// OTABSL_FALLTHROUGH_INTENDED
//
// Annotates implicit fall-through between switch labels, allowing a case to
// indicate intentional fallthrough and turn off warnings about any lack of a
// `break` statement. The OTABSL_FALLTHROUGH_INTENDED macro should be followed by
// a semicolon and can be used in most places where `break` can, provided that
// no statements exist between it and the next switch label.
//
// Example:
//
//  switch (x) {
//    case 40:
//    case 41:
//      if (truth_is_out_there) {
//        ++x;
//        OTABSL_FALLTHROUGH_INTENDED;  // Use instead of/along with annotations
//                                    // in comments
//      } else {
//        return x;
//      }
//    case 42:
//      ...
//
// Notes: when compiled with clang in C++11 mode, the OTABSL_FALLTHROUGH_INTENDED
// macro is expanded to the [[clang::fallthrough]] attribute, which is analysed
// when  performing switch labels fall-through diagnostic
// (`-Wimplicit-fallthrough`). See clang documentation on language extensions
// for details:
// https://clang.llvm.org/docs/AttributeReference.html#fallthrough-clang-fallthrough
//
// When used with unsupported compilers, the OTABSL_FALLTHROUGH_INTENDED macro
// has no effect on diagnostics. In any case this macro has no effect on runtime
// behavior and performance of code.
#ifdef OTABSL_FALLTHROUGH_INTENDED
#error "OTABSL_FALLTHROUGH_INTENDED should not be defined."
#endif

// TODO(zhangxy): Use c++17 standard [[fallthrough]] macro, when supported.
#if defined(__clang__) && defined(__has_warning)
#if __has_feature(cxx_attributes) && __has_warning("-Wimplicit-fallthrough")
#define OTABSL_FALLTHROUGH_INTENDED [[clang::fallthrough]]
#endif
#elif defined(__GNUC__) && __GNUC__ >= 7
#define OTABSL_FALLTHROUGH_INTENDED [[gnu::fallthrough]]
#endif

#ifndef OTABSL_FALLTHROUGH_INTENDED
#define OTABSL_FALLTHROUGH_INTENDED \
  do {                            \
  } while (0)
#endif

// OTABSL_DEPRECATED()
//
// Marks a deprecated class, struct, enum, function, method and variable
// declarations. The macro argument is used as a custom diagnostic message (e.g.
// suggestion of a better alternative).
//
// Examples:
//
//   class OTABSL_DEPRECATED("Use Bar instead") Foo {...};
//
//   OTABSL_DEPRECATED("Use Baz() instead") void Bar() {...}
//
//   template <typename T>
//   OTABSL_DEPRECATED("Use DoThat() instead")
//   void DoThis();
//
// Every usage of a deprecated entity will trigger a warning when compiled with
// clang's `-Wdeprecated-declarations` option. This option is turned off by
// default, but the warnings will be reported by clang-tidy.
#if defined(__clang__) && __cplusplus >= 201103L
#define OTABSL_DEPRECATED(message) __attribute__((deprecated(message)))
#endif

#ifndef OTABSL_DEPRECATED
#define OTABSL_DEPRECATED(message)
#endif

// OTABSL_BAD_CALL_IF()
//
// Used on a function overload to trap bad calls: any call that matches the
// overload will cause a compile-time error. This macro uses a clang-specific
// "enable_if" attribute, as described at
// https://clang.llvm.org/docs/AttributeReference.html#enable-if
//
// Overloads which use this macro should be bracketed by
// `#ifdef OTABSL_BAD_CALL_IF`.
//
// Example:
//
//   int isdigit(int c);
//   #ifdef OTABSL_BAD_CALL_IF
//   int isdigit(int c)
//     OTABSL_BAD_CALL_IF(c <= -1 || c > 255,
//                       "'c' must have the value of an unsigned char or EOF");
//   #endif // OTABSL_BAD_CALL_IF
#if OTABSL_HAVE_ATTRIBUTE(enable_if)
#define OTABSL_BAD_CALL_IF(expr, msg) \
  __attribute__((enable_if(expr, "Bad call trap"), unavailable(msg)))
#endif

// OTABSL_ASSERT()
//
// In C++11, `assert` can't be used portably within constexpr functions.
// OTABSL_ASSERT functions as a runtime assert but works in C++11 constexpr
// functions.  Example:
//
// constexpr double Divide(double a, double b) {
//   return OTABSL_ASSERT(b != 0), a / b;
// }
//
// This macro is inspired by
// https://akrzemi1.wordpress.com/2017/05/18/asserts-in-constexpr-functions/
#if defined(NDEBUG)
#define OTABSL_ASSERT(expr) \
  (false ? static_cast<void>(expr) : static_cast<void>(0))
#else
#define OTABSL_ASSERT(expr)                           \
  (OTABSL_PREDICT_TRUE((expr)) ? static_cast<void>(0) \
                             : [] { assert(false && #expr); }())  // NOLINT
#endif

#ifdef OTABSL_HAVE_EXCEPTIONS
#define OTABSL_INTERNAL_TRY try
#define OTABSL_INTERNAL_CATCH_ANY catch (...)
#define OTABSL_INTERNAL_RETHROW do { throw; } while (false)
#else  // OTABSL_HAVE_EXCEPTIONS
#define OTABSL_INTERNAL_TRY if (true)
#define OTABSL_INTERNAL_CATCH_ANY else if (false)
#define OTABSL_INTERNAL_RETHROW do {} while (false)
#endif  // OTABSL_HAVE_EXCEPTIONS

#endif  // OTABSL_BASE_MACROS_H_
