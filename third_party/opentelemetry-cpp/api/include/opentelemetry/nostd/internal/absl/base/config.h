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
// File: config.h
// -----------------------------------------------------------------------------
//
// This header file defines a set of macros for checking the presence of
// important compiler and platform features. Such macros can be used to
// produce portable code by parameterizing compilation based on the presence or
// lack of a given feature.
//
// We define a "feature" as some interface we wish to program to: for example,
// a library function or system call. A value of `1` indicates support for
// that feature; any other value indicates the feature support is undefined.
//
// Example:
//
// Suppose a programmer wants to write a program that uses the 'mmap()' system
// call. The Abseil macro for that feature (`OTABSL_HAVE_MMAP`) allows you to
// selectively include the `mmap.h` header and bracket code using that feature
// in the macro:
//
//   #include "absl/base/config.h"
//
//   #ifdef OTABSL_HAVE_MMAP
//   #include "sys/mman.h"
//   #endif  //OTABSL_HAVE_MMAP
//
//   ...
//   #ifdef OTABSL_HAVE_MMAP
//   void *ptr = mmap(...);
//   ...
//   #endif  // OTABSL_HAVE_MMAP

#ifndef OTABSL_BASE_CONFIG_H_
#define OTABSL_BASE_CONFIG_H_

// Included for the __GLIBC__ macro (or similar macros on other systems).
#include <limits.h>

#ifdef __cplusplus
// Included for __GLIBCXX__, _LIBCPP_VERSION
#include <cstddef>
#endif  // __cplusplus

#if defined(__APPLE__)
// Included for TARGET_OS_IPHONE, __IPHONE_OS_VERSION_MIN_REQUIRED,
// __IPHONE_8_0.
#include <Availability.h>
#include <TargetConditionals.h>
#endif

#include "options.h"
#include "policy_checks.h"

// Helper macro to convert a CPP variable to a string literal.
#define OTABSL_INTERNAL_DO_TOKEN_STR(x) #x
#define OTABSL_INTERNAL_TOKEN_STR(x) OTABSL_INTERNAL_DO_TOKEN_STR(x)

// -----------------------------------------------------------------------------
// Abseil namespace annotations
// -----------------------------------------------------------------------------

// OTABSL_NAMESPACE_BEGIN/OTABSL_NAMESPACE_END
//
// An annotation placed at the beginning/end of each `namespace absl` scope.
// This is used to inject an inline namespace.
//
// The proper way to write Abseil code in the `absl` namespace is:
//
// namespace absl {
// OTABSL_NAMESPACE_BEGIN
//
// void Foo();  // absl::Foo().
//
// OTABSL_NAMESPACE_END
// }  // namespace absl
//
// Users of Abseil should not use these macros, because users of Abseil should
// not write `namespace absl {` in their own code for any reason.  (Abseil does
// not support forward declarations of its own types, nor does it support
// user-provided specialization of Abseil templates.  Code that violates these
// rules may be broken without warning.)
#if !defined(OTABSL_OPTION_USE_INLINE_NAMESPACE) || \
    !defined(OTABSL_OPTION_INLINE_NAMESPACE_NAME)
#error options.h is misconfigured.
#endif

// Check that OTABSL_OPTION_INLINE_NAMESPACE_NAME is neither "head" nor ""
#if defined(__cplusplus) && OTABSL_OPTION_USE_INLINE_NAMESPACE == 1

#define OTABSL_INTERNAL_INLINE_NAMESPACE_STR \
  OTABSL_INTERNAL_TOKEN_STR(OTABSL_OPTION_INLINE_NAMESPACE_NAME)

static_assert(OTABSL_INTERNAL_INLINE_NAMESPACE_STR[0] != '\0',
              "options.h misconfigured: OTABSL_OPTION_INLINE_NAMESPACE_NAME must "
              "not be empty.");
static_assert(OTABSL_INTERNAL_INLINE_NAMESPACE_STR[0] != 'h' ||
                  OTABSL_INTERNAL_INLINE_NAMESPACE_STR[1] != 'e' ||
                  OTABSL_INTERNAL_INLINE_NAMESPACE_STR[2] != 'a' ||
                  OTABSL_INTERNAL_INLINE_NAMESPACE_STR[3] != 'd' ||
                  OTABSL_INTERNAL_INLINE_NAMESPACE_STR[4] != '\0',
              "options.h misconfigured: OTABSL_OPTION_INLINE_NAMESPACE_NAME must "
              "be changed to a new, unique identifier name.");

#endif

#if OTABSL_OPTION_USE_INLINE_NAMESPACE == 0
#define OTABSL_NAMESPACE_BEGIN
#define OTABSL_NAMESPACE_END
#elif OTABSL_OPTION_USE_INLINE_NAMESPACE == 1
#define OTABSL_NAMESPACE_BEGIN \
  inline namespace OTABSL_OPTION_INLINE_NAMESPACE_NAME {
#define OTABSL_NAMESPACE_END }
#else
#error options.h is misconfigured.
#endif

// -----------------------------------------------------------------------------
// Compiler Feature Checks
// -----------------------------------------------------------------------------

// OTABSL_HAVE_BUILTIN()
//
// Checks whether the compiler supports a Clang Feature Checking Macro, and if
// so, checks whether it supports the provided builtin function "x" where x
// is one of the functions noted in
// https://clang.llvm.org/docs/LanguageExtensions.html
//
// Note: Use this macro to avoid an extra level of #ifdef __has_builtin check.
// http://releases.llvm.org/3.3/tools/clang/docs/LanguageExtensions.html
#ifdef __has_builtin
#define OTABSL_HAVE_BUILTIN(x) __has_builtin(x)
#else
#define OTABSL_HAVE_BUILTIN(x) 0
#endif

#if defined(__is_identifier)
#define OTABSL_INTERNAL_HAS_KEYWORD(x) !(__is_identifier(x))
#else
#define OTABSL_INTERNAL_HAS_KEYWORD(x) 0
#endif

// OTABSL_HAVE_TLS is defined to 1 when __thread should be supported.
// We assume __thread is supported on Linux when compiled with Clang or compiled
// against libstdc++ with _GLIBCXX_HAVE_TLS defined.
#ifdef OTABSL_HAVE_TLS
#error OTABSL_HAVE_TLS cannot be directly set
#elif defined(__linux__) && (defined(__clang__) || defined(_GLIBCXX_HAVE_TLS))
#define OTABSL_HAVE_TLS 1
#endif

// OTABSL_HAVE_STD_IS_TRIVIALLY_DESTRUCTIBLE
//
// Checks whether `std::is_trivially_destructible<T>` is supported.
//
// Notes: All supported compilers using libc++ support this feature, as does
// gcc >= 4.8.1 using libstdc++, and Visual Studio.
#ifdef OTABSL_HAVE_STD_IS_TRIVIALLY_DESTRUCTIBLE
#error OTABSL_HAVE_STD_IS_TRIVIALLY_DESTRUCTIBLE cannot be directly set
#elif defined(_LIBCPP_VERSION) ||                                        \
    (defined(__clang__) && __clang_major__ >= 15) ||                     \
    (!defined(__clang__) && defined(__GNUC__) && defined(__GLIBCXX__) && \
     (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8))) ||        \
    defined(_MSC_VER)
#define OTABSL_HAVE_STD_IS_TRIVIALLY_DESTRUCTIBLE 1
#endif

// OTABSL_HAVE_STD_IS_TRIVIALLY_CONSTRUCTIBLE
//
// Checks whether `std::is_trivially_default_constructible<T>` and
// `std::is_trivially_copy_constructible<T>` are supported.

// OTABSL_HAVE_STD_IS_TRIVIALLY_ASSIGNABLE
//
// Checks whether `std::is_trivially_copy_assignable<T>` is supported.

// Notes: Clang with libc++ supports these features, as does gcc >= 5.1 with
// either libc++ or libstdc++, and Visual Studio (but not NVCC).
#if defined(OTABSL_HAVE_STD_IS_TRIVIALLY_CONSTRUCTIBLE)
#error OTABSL_HAVE_STD_IS_TRIVIALLY_CONSTRUCTIBLE cannot be directly set
#elif defined(OTABSL_HAVE_STD_IS_TRIVIALLY_ASSIGNABLE)
#error OTABSL_HAVE_STD_IS_TRIVIALLY_ASSIGNABLE cannot directly set
#elif (defined(__clang__) && defined(_LIBCPP_VERSION)) ||        \
    (defined(__clang__) && __clang_major__ >= 15) ||             \
    (!defined(__clang__) && defined(__GNUC__) &&                 \
     (__GNUC__ > 7 || (__GNUC__ == 7 && __GNUC_MINOR__ >= 4)) && \
     (defined(_LIBCPP_VERSION) || defined(__GLIBCXX__))) ||      \
    (defined(_MSC_VER) && !defined(__NVCC__))
#define OTABSL_HAVE_STD_IS_TRIVIALLY_CONSTRUCTIBLE 1
#define OTABSL_HAVE_STD_IS_TRIVIALLY_ASSIGNABLE 1
#endif

// OTABSL_HAVE_STD_IS_TRIVIALLY_COPYABLE
//
// Checks whether `std::is_trivially_copyable<T>` is supported.
//
// Notes: Clang 15+ with libc++ supports these features, GCC hasn't been tested.
#if defined(OTABSL_HAVE_STD_IS_TRIVIALLY_COPYABLE)
#error OTABSL_HAVE_STD_IS_TRIVIALLY_COPYABLE cannot be directly set
#elif defined(__clang__) && (__clang_major__ >= 15)
#define OTABSL_HAVE_STD_IS_TRIVIALLY_COPYABLE 1
#endif

// OTABSL_HAVE_SOURCE_LOCATION_CURRENT
//
// Indicates whether `absl::SourceLocation::current()` will return useful
// information in some contexts.
#ifndef OTABSL_HAVE_SOURCE_LOCATION_CURRENT
#if OTABSL_INTERNAL_HAS_KEYWORD(__builtin_LINE) && \
    OTABSL_INTERNAL_HAS_KEYWORD(__builtin_FILE)
#define OTABSL_HAVE_SOURCE_LOCATION_CURRENT 1
#endif
#endif

// OTABSL_HAVE_THREAD_LOCAL
//
// Checks whether C++11's `thread_local` storage duration specifier is
// supported.
#ifdef OTABSL_HAVE_THREAD_LOCAL
#error OTABSL_HAVE_THREAD_LOCAL cannot be directly set
#elif defined(__APPLE__)
// Notes:
// * Xcode's clang did not support `thread_local` until version 8, and
//   even then not for all iOS < 9.0.
// * Xcode 9.3 started disallowing `thread_local` for 32-bit iOS simulator
//   targeting iOS 9.x.
// * Xcode 10 moves the deployment target check for iOS < 9.0 to link time
//   making __has_feature unreliable there.
//
// Otherwise, `__has_feature` is only supported by Clang so it has be inside
// `defined(__APPLE__)` check.
#if __has_feature(cxx_thread_local) && \
    !(TARGET_OS_IPHONE && __IPHONE_OS_VERSION_MIN_REQUIRED < __IPHONE_9_0)
#define OTABSL_HAVE_THREAD_LOCAL 1
#endif
#else  // !defined(__APPLE__)
#define OTABSL_HAVE_THREAD_LOCAL 1
#endif

// There are platforms for which TLS should not be used even though the compiler
// makes it seem like it's supported (Android NDK < r12b for example).
// This is primarily because of linker problems and toolchain misconfiguration:
// Abseil does not intend to support this indefinitely. Currently, the newest
// toolchain that we intend to support that requires this behavior is the
// r11 NDK - allowing for a 5 year support window on that means this option
// is likely to be removed around June of 2021.
// TLS isn't supported until NDK r12b per
// https://developer.android.com/ndk/downloads/revision_history.html
// Since NDK r16, `__NDK_MAJOR__` and `__NDK_MINOR__` are defined in
// <android/ndk-version.h>. For NDK < r16, users should define these macros,
// e.g. `-D__NDK_MAJOR__=11 -D__NKD_MINOR__=0` for NDK r11.
#if defined(__ANDROID__) && defined(__clang__)
#if __has_include(<android/ndk-version.h>)
#include <android/ndk-version.h>
#endif  // __has_include(<android/ndk-version.h>)
#if defined(__ANDROID__) && defined(__clang__) && defined(__NDK_MAJOR__) && \
    defined(__NDK_MINOR__) &&                                               \
    ((__NDK_MAJOR__ < 12) || ((__NDK_MAJOR__ == 12) && (__NDK_MINOR__ < 1)))
#undef OTABSL_HAVE_TLS
#undef OTABSL_HAVE_THREAD_LOCAL
#endif
#endif  // defined(__ANDROID__) && defined(__clang__)

// Emscripten doesn't yet support `thread_local` or `__thread`.
// https://github.com/emscripten-core/emscripten/issues/3502
#if defined(__EMSCRIPTEN__)
#undef OTABSL_HAVE_TLS
#undef OTABSL_HAVE_THREAD_LOCAL
#endif  // defined(__EMSCRIPTEN__)

// OTABSL_HAVE_INTRINSIC_INT128
//
// Checks whether the __int128 compiler extension for a 128-bit integral type is
// supported.
//
// Note: __SIZEOF_INT128__ is defined by Clang and GCC when __int128 is
// supported, but we avoid using it in certain cases:
// * On Clang:
//   * Building using Clang for Windows, where the Clang runtime library has
//     128-bit support only on LP64 architectures, but Windows is LLP64.
// * On Nvidia's nvcc:
//   * nvcc also defines __GNUC__ and __SIZEOF_INT128__, but not all versions
//     actually support __int128.
#ifdef OTABSL_HAVE_INTRINSIC_INT128
#error OTABSL_HAVE_INTRINSIC_INT128 cannot be directly set
#elif defined(__SIZEOF_INT128__)
#if (defined(__clang__) && !defined(_WIN32)) || \
    (defined(__CUDACC__) && __CUDACC_VER_MAJOR__ >= 9) ||                \
    (defined(__GNUC__) && !defined(__clang__) && !defined(__CUDACC__))
#define OTABSL_HAVE_INTRINSIC_INT128 1
#elif defined(__CUDACC__)
// __CUDACC_VER__ is a full version number before CUDA 9, and is defined to a
// string explaining that it has been removed starting with CUDA 9. We use
// nested #ifs because there is no short-circuiting in the preprocessor.
// NOTE: `__CUDACC__` could be undefined while `__CUDACC_VER__` is defined.
#if __CUDACC_VER__ >= 70000
#define OTABSL_HAVE_INTRINSIC_INT128 1
#endif  // __CUDACC_VER__ >= 70000
#endif  // defined(__CUDACC__)
#endif  // OTABSL_HAVE_INTRINSIC_INT128

// OTABSL_HAVE_EXCEPTIONS
//
// Checks whether the compiler both supports and enables exceptions. Many
// compilers support a "no exceptions" mode that disables exceptions.
//
// Generally, when OTABSL_HAVE_EXCEPTIONS is not defined:
//
// * Code using `throw` and `try` may not compile.
// * The `noexcept` specifier will still compile and behave as normal.
// * The `noexcept` operator may still return `false`.
//
// For further details, consult the compiler's documentation.
#ifdef OTABSL_HAVE_EXCEPTIONS
#error OTABSL_HAVE_EXCEPTIONS cannot be directly set.

#elif defined(__clang__)

#if __clang_major__ > 3 || (__clang_major__ == 3 && __clang_minor__ >= 6)
// Clang >= 3.6
#if __has_feature(cxx_exceptions)
#define OTABSL_HAVE_EXCEPTIONS 1
#endif  // __has_feature(cxx_exceptions)
#else
// Clang < 3.6
// http://releases.llvm.org/3.6.0/tools/clang/docs/ReleaseNotes.html#the-exceptions-macro
#if defined(__EXCEPTIONS) && __has_feature(cxx_exceptions)
#define OTABSL_HAVE_EXCEPTIONS 1
#endif  // defined(__EXCEPTIONS) && __has_feature(cxx_exceptions)
#endif  // __clang_major__ > 3 || (__clang_major__ == 3 && __clang_minor__ >= 6)

// Handle remaining special cases and default to exceptions being supported.
#elif !(defined(__GNUC__) && (__GNUC__ < 5) && !defined(__EXCEPTIONS)) &&    \
    !(defined(__GNUC__) && (__GNUC__ >= 5) && !defined(__cpp_exceptions)) && \
    !(defined(_MSC_VER) && !defined(_CPPUNWIND))
#define OTABSL_HAVE_EXCEPTIONS 1
#endif

// -----------------------------------------------------------------------------
// Platform Feature Checks
// -----------------------------------------------------------------------------

// Currently supported operating systems and associated preprocessor
// symbols:
//
//   Linux and Linux-derived           __linux__
//   Android                           __ANDROID__ (implies __linux__)
//   Linux (non-Android)               __linux__ && !__ANDROID__
//   Darwin (macOS and iOS)            __APPLE__
//   Akaros (http://akaros.org)        __ros__
//   Windows                           _WIN32
//   NaCL                              __native_client__
//   AsmJS                             __asmjs__
//   WebAssembly                       __wasm__
//   Fuchsia                           __Fuchsia__
//
// Note that since Android defines both __ANDROID__ and __linux__, one
// may probe for either Linux or Android by simply testing for __linux__.

// OTABSL_HAVE_MMAP
//
// Checks whether the platform has an mmap(2) implementation as defined in
// POSIX.1-2001.
#ifdef OTABSL_HAVE_MMAP
#error OTABSL_HAVE_MMAP cannot be directly set
#elif defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__) ||   \
    defined(__ros__) || defined(__native_client__) || defined(__asmjs__) || \
    defined(__wasm__) || defined(__Fuchsia__) || defined(__sun) || \
    defined(__ASYLO__)
#define OTABSL_HAVE_MMAP 1
#endif

// OTABSL_HAVE_PTHREAD_GETSCHEDPARAM
//
// Checks whether the platform implements the pthread_(get|set)schedparam(3)
// functions as defined in POSIX.1-2001.
#ifdef OTABSL_HAVE_PTHREAD_GETSCHEDPARAM
#error OTABSL_HAVE_PTHREAD_GETSCHEDPARAM cannot be directly set
#elif defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__) || \
    defined(__ros__)
#define OTABSL_HAVE_PTHREAD_GETSCHEDPARAM 1
#endif

// OTABSL_HAVE_SCHED_YIELD
//
// Checks whether the platform implements sched_yield(2) as defined in
// POSIX.1-2001.
#ifdef OTABSL_HAVE_SCHED_YIELD
#error OTABSL_HAVE_SCHED_YIELD cannot be directly set
#elif defined(__linux__) || defined(__ros__) || defined(__native_client__)
#define OTABSL_HAVE_SCHED_YIELD 1
#endif

// OTABSL_HAVE_SEMAPHORE_H
//
// Checks whether the platform supports the <semaphore.h> header and sem_init(3)
// family of functions as standardized in POSIX.1-2001.
//
// Note: While Apple provides <semaphore.h> for both iOS and macOS, it is
// explicitly deprecated and will cause build failures if enabled for those
// platforms.  We side-step the issue by not defining it here for Apple
// platforms.
#ifdef OTABSL_HAVE_SEMAPHORE_H
#error OTABSL_HAVE_SEMAPHORE_H cannot be directly set
#elif defined(__linux__) || defined(__ros__)
#define OTABSL_HAVE_SEMAPHORE_H 1
#endif

// OTABSL_HAVE_ALARM
//
// Checks whether the platform supports the <signal.h> header and alarm(2)
// function as standardized in POSIX.1-2001.
#ifdef OTABSL_HAVE_ALARM
#error OTABSL_HAVE_ALARM cannot be directly set
#elif defined(__GOOGLE_GRTE_VERSION__)
// feature tests for Google's GRTE
#define OTABSL_HAVE_ALARM 1
#elif defined(__GLIBC__)
// feature test for glibc
#define OTABSL_HAVE_ALARM 1
#elif defined(_MSC_VER)
// feature tests for Microsoft's library
#elif defined(__MINGW32__)
// mingw32 doesn't provide alarm(2):
// https://osdn.net/projects/mingw/scm/git/mingw-org-wsl/blobs/5.2-trunk/mingwrt/include/unistd.h
// mingw-w64 provides a no-op implementation:
// https://sourceforge.net/p/mingw-w64/mingw-w64/ci/master/tree/mingw-w64-crt/misc/alarm.c
#elif defined(__EMSCRIPTEN__)
// emscripten doesn't support signals
#elif defined(__Fuchsia__)
// Signals don't exist on fuchsia.
#elif defined(__native_client__)
#else
// other standard libraries
#define OTABSL_HAVE_ALARM 1
#endif

// OTABSL_IS_LITTLE_ENDIAN
// OTABSL_IS_BIG_ENDIAN
//
// Checks the endianness of the platform.
//
// Notes: uses the built in endian macros provided by GCC (since 4.6) and
// Clang (since 3.2); see
// https://gcc.gnu.org/onlinedocs/cpp/Common-Predefined-Macros.html.
// Otherwise, if _WIN32, assume little endian. Otherwise, bail with an error.
#if defined(OTABSL_IS_BIG_ENDIAN)
#error "OTABSL_IS_BIG_ENDIAN cannot be directly set."
#endif
#if defined(OTABSL_IS_LITTLE_ENDIAN)
#error "OTABSL_IS_LITTLE_ENDIAN cannot be directly set."
#endif

#if (defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
     __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define OTABSL_IS_LITTLE_ENDIAN 1
#elif defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define OTABSL_IS_BIG_ENDIAN 1
#elif defined(_WIN32)
#define OTABSL_IS_LITTLE_ENDIAN 1
#else
#error "absl endian detection needs to be set up for your compiler"
#endif

// macOS 10.13 and iOS 10.11 don't let you use <any>, <optional>, or <variant>
// even though the headers exist and are publicly noted to work.  See
// https://github.com/abseil/abseil-cpp/issues/207 and
// https://developer.apple.com/documentation/xcode_release_notes/xcode_10_release_notes
// libc++ spells out the availability requirements in the file
// llvm-project/libcxx/include/__config via the #define
// _LIBCPP_AVAILABILITY_BAD_OPTIONAL_ACCESS.
#if defined(__APPLE__) && defined(_LIBCPP_VERSION) && \
  ((defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) && \
   __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ < 101400) || \
  (defined(__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__) && \
   __ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__ < 120000) || \
  (defined(__ENVIRONMENT_WATCH_OS_VERSION_MIN_REQUIRED__) && \
   __ENVIRONMENT_WATCH_OS_VERSION_MIN_REQUIRED__ < 120000) || \
  (defined(__ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__) && \
   __ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__ < 50000))
#define OTABSL_INTERNAL_APPLE_CXX17_TYPES_UNAVAILABLE 1
#else
#define OTABSL_INTERNAL_APPLE_CXX17_TYPES_UNAVAILABLE 0
#endif

// OTABSL_HAVE_STD_ANY
//
// Checks whether C++17 std::any is available by checking whether <any> exists.
#ifdef OTABSL_HAVE_STD_ANY
#error "OTABSL_HAVE_STD_ANY cannot be directly set."
#endif

#ifdef __has_include
#if __has_include(<any>) && __cplusplus >= 201703L && \
    !OTABSL_INTERNAL_APPLE_CXX17_TYPES_UNAVAILABLE
#define OTABSL_HAVE_STD_ANY 1
#endif
#endif

// OTABSL_HAVE_STD_OPTIONAL
//
// Checks whether C++17 std::optional is available.
#ifdef OTABSL_HAVE_STD_OPTIONAL
#error "OTABSL_HAVE_STD_OPTIONAL cannot be directly set."
#endif

#ifdef __has_include
#if __has_include(<optional>) && __cplusplus >= 201703L && \
    !OTABSL_INTERNAL_APPLE_CXX17_TYPES_UNAVAILABLE
#define OTABSL_HAVE_STD_OPTIONAL 1
#endif
#endif

// OTABSL_HAVE_STD_VARIANT
//
// Checks whether C++17 std::variant is available.
#ifdef OTABSL_HAVE_STD_VARIANT
#error "OTABSL_HAVE_STD_VARIANT cannot be directly set."
#endif

#ifdef __has_include
#if __has_include(<variant>) && __cplusplus >= 201703L && \
    !OTABSL_INTERNAL_APPLE_CXX17_TYPES_UNAVAILABLE
#define OTABSL_HAVE_STD_VARIANT 1
#endif
#endif

// OTABSL_HAVE_STD_STRING_VIEW
//
// Checks whether C++17 std::string_view is available.
#ifdef OTABSL_HAVE_STD_STRING_VIEW
#error "OTABSL_HAVE_STD_STRING_VIEW cannot be directly set."
#endif

#ifdef __has_include
#if __has_include(<string_view>) && __cplusplus >= 201703L
#define OTABSL_HAVE_STD_STRING_VIEW 1
#endif
#endif

// For MSVC, `__has_include` is supported in VS 2017 15.3, which is later than
// the support for <optional>, <any>, <string_view>, <variant>. So we use
// _MSC_VER to check whether we have VS 2017 RTM (when <optional>, <any>,
// <string_view>, <variant> is implemented) or higher. Also, `__cplusplus` is
// not correctly set by MSVC, so we use `_MSVC_LANG` to check the language
// version.
// TODO(zhangxy): fix tests before enabling aliasing for `std::any`.
#if defined(_MSC_VER) && _MSC_VER >= 1910 && \
    ((defined(_MSVC_LANG) && _MSVC_LANG > 201402) || __cplusplus > 201402)
// #define OTABSL_HAVE_STD_ANY 1
#define OTABSL_HAVE_STD_OPTIONAL 1
#define OTABSL_HAVE_STD_VARIANT 1
#define OTABSL_HAVE_STD_STRING_VIEW 1
#endif

// OTABSL_USES_STD_ANY
//
// Indicates whether absl::any is an alias for std::any.
#if !defined(OTABSL_OPTION_USE_STD_ANY)
#error options.h is misconfigured.
#elif OTABSL_OPTION_USE_STD_ANY == 0 || \
    (OTABSL_OPTION_USE_STD_ANY == 2 && !defined(OTABSL_HAVE_STD_ANY))
#undef OTABSL_USES_STD_ANY
#elif OTABSL_OPTION_USE_STD_ANY == 1 || \
    (OTABSL_OPTION_USE_STD_ANY == 2 && defined(OTABSL_HAVE_STD_ANY))
#define OTABSL_USES_STD_ANY 1
#else
#error options.h is misconfigured.
#endif

// OTABSL_USES_STD_OPTIONAL
//
// Indicates whether absl::optional is an alias for std::optional.
#if !defined(OTABSL_OPTION_USE_STD_OPTIONAL)
#error options.h is misconfigured.
#elif OTABSL_OPTION_USE_STD_OPTIONAL == 0 || \
    (OTABSL_OPTION_USE_STD_OPTIONAL == 2 && !defined(OTABSL_HAVE_STD_OPTIONAL))
#undef OTABSL_USES_STD_OPTIONAL
#elif OTABSL_OPTION_USE_STD_OPTIONAL == 1 || \
    (OTABSL_OPTION_USE_STD_OPTIONAL == 2 && defined(OTABSL_HAVE_STD_OPTIONAL))
#define OTABSL_USES_STD_OPTIONAL 1
#else
#error options.h is misconfigured.
#endif

// OTABSL_USES_STD_VARIANT
//
// Indicates whether absl::variant is an alias for std::variant.
#if !defined(OTABSL_OPTION_USE_STD_VARIANT)
#error options.h is misconfigured.
#elif OTABSL_OPTION_USE_STD_VARIANT == 0 || \
    (OTABSL_OPTION_USE_STD_VARIANT == 2 && !defined(OTABSL_HAVE_STD_VARIANT))
#undef OTABSL_USES_STD_VARIANT
#elif OTABSL_OPTION_USE_STD_VARIANT == 1 || \
    (OTABSL_OPTION_USE_STD_VARIANT == 2 && defined(OTABSL_HAVE_STD_VARIANT))
#define OTABSL_USES_STD_VARIANT 1
#else
#error options.h is misconfigured.
#endif

// OTABSL_USES_STD_STRING_VIEW
//
// Indicates whether absl::string_view is an alias for std::string_view.
#if !defined(OTABSL_OPTION_USE_STD_STRING_VIEW)
#error options.h is misconfigured.
#elif OTABSL_OPTION_USE_STD_STRING_VIEW == 0 || \
    (OTABSL_OPTION_USE_STD_STRING_VIEW == 2 &&  \
     !defined(OTABSL_HAVE_STD_STRING_VIEW))
#undef OTABSL_USES_STD_STRING_VIEW
#elif OTABSL_OPTION_USE_STD_STRING_VIEW == 1 || \
    (OTABSL_OPTION_USE_STD_STRING_VIEW == 2 &&  \
     defined(OTABSL_HAVE_STD_STRING_VIEW))
#define OTABSL_USES_STD_STRING_VIEW 1
#else
#error options.h is misconfigured.
#endif

// In debug mode, MSVC 2017's std::variant throws a EXCEPTION_ACCESS_VIOLATION
// SEH exception from emplace for variant<SomeStruct> when constructing the
// struct can throw. This defeats some of variant_test and
// variant_exception_safety_test.
#if defined(_MSC_VER) && _MSC_VER >= 1700 && defined(_DEBUG)
#define OTABSL_INTERNAL_MSVC_2017_DBG_MODE
#endif

// OTABSL_INTERNAL_MANGLED_NS
// OTABSL_INTERNAL_MANGLED_BACKREFERENCE
//
// Internal macros for building up mangled names in our internal fork of CCTZ.
// This implementation detail is only needed and provided for the MSVC build.
//
// These macros both expand to string literals.  OTABSL_INTERNAL_MANGLED_NS is
// the mangled spelling of the `absl` namespace, and
// OTABSL_INTERNAL_MANGLED_BACKREFERENCE is a back-reference integer representing
// the proper count to skip past the CCTZ fork namespace names.  (This number
// is one larger when there is an inline namespace name to skip.)
#if defined(_MSC_VER)
#if OTABSL_OPTION_USE_INLINE_NAMESPACE == 0
#define OTABSL_INTERNAL_MANGLED_NS "absl"
#define OTABSL_INTERNAL_MANGLED_BACKREFERENCE "5"
#else
#define OTABSL_INTERNAL_MANGLED_NS \
  OTABSL_INTERNAL_TOKEN_STR(OTABSL_OPTION_INLINE_NAMESPACE_NAME) "@absl"
#define OTABSL_INTERNAL_MANGLED_BACKREFERENCE "6"
#endif
#endif

#undef OTABSL_INTERNAL_HAS_KEYWORD

// OTABSL_DLL
//
// When building Abseil as a DLL, this macro expands to `__declspec(dllexport)`
// so we can annotate symbols appropriately as being exported. When used in
// headers consuming a DLL, this macro expands to `__declspec(dllimport)` so
// that consumers know the symbol is defined inside the DLL. In all other cases,
// the macro expands to nothing.
#if defined(_MSC_VER)
#if defined(OTABSL_BUILD_DLL)
#define OTABSL_DLL __declspec(dllexport)
#elif 1
#define OTABSL_DLL __declspec(dllimport)
#else
#define OTABSL_DLL
#endif
#else
#define OTABSL_DLL
#endif  // defined(_MSC_VER)

#endif  // OTABSL_BASE_CONFIG_H_
