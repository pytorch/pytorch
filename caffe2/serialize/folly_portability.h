/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// MSCV 2017 __cplusplus definition by default does not track the C++ version.
// https://devblogs.microsoft.com/cppblog/msvc-now-correctly-reports-__cplusplus/
#if !defined(_MSC_VER) || _MSC_VER >= 2000
static_assert(__cplusplus >= 201402L, "__cplusplus >= 201402L");
#endif

#if defined(__GNUC__) && !defined(__clang__)
static_assert(__GNUC__ >= 5, "__GNUC__ >= 5");
#endif

#include <cstddef>

#pragma once

#if __has_include(<features.h>)
#include <features.h> // @manual
#endif

#if __has_include(<bits/c++config.h>)
#include <bits/c++config.h> // @manual
#endif

#ifdef __ANDROID__
#include <android/api-level.h> // @manual
#endif

#ifdef __APPLE__
#include <Availability.h> // @manual
#endif

/**
 * Portable version check.
 */
#ifndef __GNUC_PREREQ
#if defined __GNUC__ && defined __GNUC_MINOR__
/* nolint */
#define __GNUC_PREREQ(maj, min) \
  ((__GNUC__ << 16) + __GNUC_MINOR__ >= ((maj) << 16) + (min))
#else
/* nolint */
#define __GNUC_PREREQ(maj, min) 0
#endif
#endif

// portable version check for clang
#ifndef __CLANG_PREREQ
#if defined __clang__ && defined __clang_major__ && defined __clang_minor__
/* nolint */
#define __CLANG_PREREQ(maj, min) \
  ((__clang_major__ << 16) + __clang_minor__ >= ((maj) << 16) + (min))
#else
/* nolint */
#define __CLANG_PREREQ(maj, min) 0
#endif
#endif

#if defined(__has_builtin)
#define FOLLY_HAS_BUILTIN(...) __has_builtin(__VA_ARGS__)
#else
#define FOLLY_HAS_BUILTIN(...) 0
#endif

#if defined(__has_feature)
#define FOLLY_HAS_FEATURE(...) __has_feature(__VA_ARGS__)
#else
#define FOLLY_HAS_FEATURE(...) 0
#endif

/* FOLLY_SANITIZE_ADDRESS is defined to 1 if the current compilation unit
 * is being compiled with ASAN enabled.
 *
 * Beware when using this macro in a header file: this macro may change values
 * across compilation units if some libraries are built with ASAN enabled
 * and some built with ASAN disabled.  For instance, this may occur, if folly
 * itself was compiled without ASAN but a downstream project that uses folly is
 * compiling with ASAN enabled.
 *
 * Use FOLLY_LIBRARY_SANITIZE_ADDRESS (defined in folly-config.h) to check if
 * folly itself was compiled with ASAN enabled.
 */
#ifndef FOLLY_SANITIZE_ADDRESS
#if FOLLY_HAS_FEATURE(address_sanitizer) || __SANITIZE_ADDRESS__
#define FOLLY_SANITIZE_ADDRESS 1
#endif
#endif

/* Define attribute wrapper for function attribute used to disable
 * address sanitizer instrumentation. Unfortunately, this attribute
 * has issues when inlining is used, so disable that as well. */
#ifdef FOLLY_SANITIZE_ADDRESS
#if defined(__clang__)
#if __has_attribute(__no_sanitize__)
#define FOLLY_DISABLE_ADDRESS_SANITIZER \
  __attribute__((__no_sanitize__("address"), __noinline__))
#elif __has_attribute(__no_address_safety_analysis__)
#define FOLLY_DISABLE_ADDRESS_SANITIZER \
  __attribute__((__no_address_safety_analysis__, __noinline__))
#elif __has_attribute(__no_sanitize_address__)
#define FOLLY_DISABLE_ADDRESS_SANITIZER \
  __attribute__((__no_sanitize_address__, __noinline__))
#endif
#elif defined(__GNUC__)
#define FOLLY_DISABLE_ADDRESS_SANITIZER \
  __attribute__((__no_address_safety_analysis__, __noinline__))
#endif
#endif
#ifndef FOLLY_DISABLE_ADDRESS_SANITIZER
#define FOLLY_DISABLE_ADDRESS_SANITIZER
#endif

/* Define a convenience macro to test when thread sanitizer is being used
 * across the different compilers (e.g. clang, gcc) */
#ifndef FOLLY_SANITIZE_THREAD
#if FOLLY_HAS_FEATURE(thread_sanitizer) || __SANITIZE_THREAD__
#define FOLLY_SANITIZE_THREAD 1
#endif
#endif

#if FOLLY_SANITIZE_THREAD
#define FOLLY_DISABLE_THREAD_SANITIZER \
  __attribute__((no_sanitize_thread, noinline))
#else
#define FOLLY_DISABLE_THREAD_SANITIZER
#endif

/**
 * Define a convenience macro to test when memory sanitizer is being used
 * across the different compilers (e.g. clang, gcc)
 */
#ifndef FOLLY_SANITIZE_MEMORY
#if FOLLY_HAS_FEATURE(memory_sanitizer) || __SANITIZE_MEMORY__
#define FOLLY_SANITIZE_MEMORY 1
#endif
#endif

#if FOLLY_SANITIZE_MEMORY
#define FOLLY_DISABLE_MEMORY_SANITIZER \
  __attribute__((no_sanitize_memory, noinline))
#else
#define FOLLY_DISABLE_MEMORY_SANITIZER
#endif

/**
 * Define a convenience macro to test when ASAN, UBSAN, TSAN or MSAN sanitizer
 * are being used
 */
#ifndef FOLLY_SANITIZE
#if defined(FOLLY_SANITIZE_ADDRESS) || defined(FOLLY_SANITIZE_THREAD) || \
    defined(FOLLY_SANITIZE_MEMORY)
#define FOLLY_SANITIZE 1
#endif
#endif

#if FOLLY_SANITIZE
#define FOLLY_DISABLE_UNDEFINED_BEHAVIOR_SANITIZER(...) \
  __attribute__((no_sanitize(__VA_ARGS__)))
#else
#define FOLLY_DISABLE_UNDEFINED_BEHAVIOR_SANITIZER(...)
#endif // FOLLY_SANITIZE

#define FOLLY_DISABLE_SANITIZERS                                 \
  FOLLY_DISABLE_ADDRESS_SANITIZER FOLLY_DISABLE_THREAD_SANITIZER \
      FOLLY_DISABLE_UNDEFINED_BEHAVIOR_SANITIZER("undefined")

/**
 * Macro for marking functions as having public visibility.
 */
#if defined(__GNUC__)
#define FOLLY_EXPORT __attribute__((__visibility__("default")))
#else
#define FOLLY_EXPORT
#endif

// noinline
#ifdef _MSC_VER
#define FOLLY_NOINLINE __declspec(noinline)
#elif defined(__GNUC__)
#define FOLLY_NOINLINE __attribute__((__noinline__))
#else
#define FOLLY_NOINLINE
#endif

// always inline
#ifdef _MSC_VER
#define FOLLY_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__)
#define FOLLY_ALWAYS_INLINE inline __attribute__((__always_inline__))
#else
#define FOLLY_ALWAYS_INLINE inline
#endif

// attribute hidden
#if defined(_MSC_VER)
#define FOLLY_ATTR_VISIBILITY_HIDDEN
#elif defined(__GNUC__)
#define FOLLY_ATTR_VISIBILITY_HIDDEN __attribute__((__visibility__("hidden")))
#else
#define FOLLY_ATTR_VISIBILITY_HIDDEN
#endif

// An attribute for marking symbols as weak, if supported
#if FOLLY_HAVE_WEAK_SYMBOLS
#define FOLLY_ATTR_WEAK __attribute__((__weak__))
#else
#define FOLLY_ATTR_WEAK
#endif

// Microsoft ABI version (can be overridden manually if necessary)
#ifndef FOLLY_MICROSOFT_ABI_VER
#ifdef _MSC_VER
#define FOLLY_MICROSOFT_ABI_VER _MSC_VER
#endif
#endif

//  FOLLY_ERASE
//
//  A conceptual attribute/syntax combo for erasing a function from the build
//  artifacts and forcing all call-sites to inline the callee, at least as far
//  as each compiler supports.
//
//  Semantically includes the inline specifier.
#define FOLLY_ERASE FOLLY_ALWAYS_INLINE FOLLY_ATTR_VISIBILITY_HIDDEN

//  FOLLY_ERASE_HACK_GCC
//
//  Equivalent to FOLLY_ERASE, but without hiding under gcc. Useful when applied
//  to a function which may sometimes be hidden separately, for example by being
//  declared in an anonymous namespace, since in such cases with -Wattributes
//  enabled, gcc would emit: 'visibility' attribute ignored.
//
//  Semantically includes the inline specifier.
#if defined(__GNUC__) && !defined(__clang__)
#define FOLLY_ERASE_HACK_GCC FOLLY_ALWAYS_INLINE
#else
#define FOLLY_ERASE_HACK_GCC FOLLY_ERASE
#endif

//  FOLLY_ERASE_TRYCATCH
//
//  Equivalent to FOLLY_ERASE, but for code which might contain explicit
//  exception handling. Has the effect of FOLLY_ERASE, except under MSVC which
//  warns about __forceinline when functions contain exception handling.
//
//  Semantically includes the inline specifier.
#ifdef _MSC_VER
#define FOLLY_ERASE_TRYCATCH inline
#else
#define FOLLY_ERASE_TRYCATCH FOLLY_ERASE
#endif

// Unaligned loads and stores
namespace folly {
#if FOLLY_HAVE_UNALIGNED_ACCESS
constexpr bool kHasUnalignedAccess = true;
#else
constexpr bool kHasUnalignedAccess = false;
#endif
} // namespace folly

// compiler specific attribute translation
// msvc should come first, so if clang is in msvc mode it gets the right defines

// NOTE: this will only do checking in msvc with versions that support /analyze
#if _MSC_VER
#ifdef _USE_ATTRIBUTES_FOR_SAL
#undef _USE_ATTRIBUTES_FOR_SAL
#endif
/* nolint */
#define _USE_ATTRIBUTES_FOR_SAL 1
#include <sal.h> // @manual
#define FOLLY_PRINTF_FORMAT _Printf_format_string_
#define FOLLY_PRINTF_FORMAT_ATTR(format_param, dots_param) /**/
#else
#define FOLLY_PRINTF_FORMAT /**/
#define FOLLY_PRINTF_FORMAT_ATTR(format_param, dots_param) \
  __attribute__((__format__(__printf__, format_param, dots_param)))
#endif

// warn unused result
#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(nodiscard)
#define FOLLY_NODISCARD [[nodiscard]]
#endif
#endif
#if !defined FOLLY_NODISCARD
#if defined(_MSC_VER) && (_MSC_VER >= 1700)
#define FOLLY_NODISCARD _Check_return_
#elif defined(__GNUC__)
#define FOLLY_NODISCARD __attribute__((__warn_unused_result__))
#else
#define FOLLY_NODISCARD
#endif
#endif

// target
#ifdef _MSC_VER
#define FOLLY_TARGET_ATTRIBUTE(target)
#else
#define FOLLY_TARGET_ATTRIBUTE(target) __attribute__((__target__(target)))
#endif

// detection for 64 bit
#if defined(__x86_64__) || defined(_M_X64)
#define FOLLY_X64 1
#else
#define FOLLY_X64 0
#endif

#if defined(__arm__)
#define FOLLY_ARM 1
#else
#define FOLLY_ARM 0
#endif

#if defined(__aarch64__)
#define FOLLY_AARCH64 1
#else
#define FOLLY_AARCH64 0
#endif

#if defined(__powerpc64__)
#define FOLLY_PPC64 1
#else
#define FOLLY_PPC64 0
#endif

#if defined(__s390x__)
#define FOLLY_S390X 1
#else
#define FOLLY_S390X 0
#endif

namespace folly {
constexpr bool kIsArchArm = FOLLY_ARM == 1;
constexpr bool kIsArchAmd64 = FOLLY_X64 == 1;
constexpr bool kIsArchAArch64 = FOLLY_AARCH64 == 1;
constexpr bool kIsArchPPC64 = FOLLY_PPC64 == 1;
constexpr bool kIsArchS390X = FOLLY_S390X == 1;
} // namespace folly

namespace folly {

/**
 * folly::kIsLibrarySanitizeAddress reports if folly was compiled with ASAN
 * enabled.  Note that for compilation units outside of folly that include
 * folly/Portability.h, the value of kIsLibrarySanitizeAddress may be different
 * from whether or not the current compilation unit is being compiled with ASAN.
 */
#if FOLLY_LIBRARY_SANITIZE_ADDRESS
constexpr bool kIsLibrarySanitizeAddress = true;
#else
constexpr bool kIsLibrarySanitizeAddress = false;
#endif

#if FOLLY_SANITIZE_ADDRESS
constexpr bool kIsSanitizeAddress = true;
#else
constexpr bool kIsSanitizeAddress = false;
#endif

#if FOLLY_SANITIZE_THREAD
constexpr bool kIsSanitizeThread = true;
#else
constexpr bool kIsSanitizeThread = false;
#endif

#if FOLLY_SANITIZE
constexpr bool kIsSanitize = true;
#else
constexpr bool kIsSanitize = false;
#endif
} // namespace folly

// packing is very ugly in msvc
#ifdef _MSC_VER
#define FOLLY_PACK_ATTR /**/
#define FOLLY_PACK_PUSH __pragma(pack(push, 1))
#define FOLLY_PACK_POP __pragma(pack(pop))
#elif defined(__GNUC__)
#define FOLLY_PACK_ATTR __attribute__((__packed__))
#define FOLLY_PACK_PUSH /**/
#define FOLLY_PACK_POP /**/
#else
#define FOLLY_PACK_ATTR /**/
#define FOLLY_PACK_PUSH /**/
#define FOLLY_PACK_POP /**/
#endif

// Generalize warning push/pop.
#if defined(__GNUC__) || defined(__clang__)
// Clang & GCC
#define FOLLY_PUSH_WARNING _Pragma("GCC diagnostic push")
#define FOLLY_POP_WARNING _Pragma("GCC diagnostic pop")
#define FOLLY_GNU_DISABLE_WARNING_INTERNAL2(warningName) #warningName
#define FOLLY_GNU_DISABLE_WARNING(warningName) \
  _Pragma(                                     \
      FOLLY_GNU_DISABLE_WARNING_INTERNAL2(GCC diagnostic ignored warningName))
#ifdef __clang__
#define FOLLY_CLANG_DISABLE_WARNING(warningName) \
  FOLLY_GNU_DISABLE_WARNING(warningName)
#define FOLLY_GCC_DISABLE_WARNING(warningName)
#else
#define FOLLY_CLANG_DISABLE_WARNING(warningName)
#define FOLLY_GCC_DISABLE_WARNING(warningName) \
  FOLLY_GNU_DISABLE_WARNING(warningName)
#endif
#define FOLLY_MSVC_DISABLE_WARNING(warningNumber)
#elif defined(_MSC_VER)
#define FOLLY_PUSH_WARNING __pragma(warning(push))
#define FOLLY_POP_WARNING __pragma(warning(pop))
// Disable the GCC warnings.
#define FOLLY_GNU_DISABLE_WARNING(warningName)
#define FOLLY_GCC_DISABLE_WARNING(warningName)
#define FOLLY_CLANG_DISABLE_WARNING(warningName)
#define FOLLY_MSVC_DISABLE_WARNING(warningNumber) \
  __pragma(warning(disable : warningNumber))
#else
#define FOLLY_PUSH_WARNING
#define FOLLY_POP_WARNING
#define FOLLY_GNU_DISABLE_WARNING(warningName)
#define FOLLY_GCC_DISABLE_WARNING(warningName)
#define FOLLY_CLANG_DISABLE_WARNING(warningName)
#define FOLLY_MSVC_DISABLE_WARNING(warningNumber)
#endif

#ifdef FOLLY_HAVE_SHADOW_LOCAL_WARNINGS
#define FOLLY_GCC_DISABLE_NEW_SHADOW_WARNINGS            \
  FOLLY_GNU_DISABLE_WARNING("-Wshadow-compatible-local") \
  FOLLY_GNU_DISABLE_WARNING("-Wshadow-local")            \
  FOLLY_GNU_DISABLE_WARNING("-Wshadow")
#else
#define FOLLY_GCC_DISABLE_NEW_SHADOW_WARNINGS /* empty */
#endif

/* Platform specific TLS support
 * gcc implements __thread
 * msvc implements __declspec(thread)
 * the semantics are the same
 * (but remember __thread has different semantics when using emutls (ex. apple))
 */
#if defined(_MSC_VER)
#define FOLLY_TLS __declspec(thread)
#elif defined(__GNUC__)
#define FOLLY_TLS __thread
#else
#error cannot define platform specific thread local storage
#endif

// disable FOLLY_TLS on 32 bit Apple/iOS
#if defined(__APPLE__) && FOLLY_MOBILE
#if (__SIZEOF_POINTER__ == 4)
#undef FOLLY_TLS
#endif
#endif

// It turns out that GNU libstdc++ and LLVM libc++ differ on how they implement
// the 'std' namespace; the latter uses inline namespaces. Wrap this decision
// up in a macro to make forward-declarations easier.
#if FOLLY_USE_LIBCPP
#include <__config> // @manual
#define FOLLY_NAMESPACE_STD_BEGIN _LIBCPP_BEGIN_NAMESPACE_STD
#define FOLLY_NAMESPACE_STD_END _LIBCPP_END_NAMESPACE_STD
#else
#define FOLLY_NAMESPACE_STD_BEGIN namespace std {
#define FOLLY_NAMESPACE_STD_END }
#endif

// If the new c++ ABI is used, __cxx11 inline namespace needs to be added to
// some types, e.g. std::list.
#if _GLIBCXX_USE_CXX11_ABI
#define FOLLY_GLIBCXX_NAMESPACE_CXX11_BEGIN \
  inline _GLIBCXX_BEGIN_NAMESPACE_CXX11
#define FOLLY_GLIBCXX_NAMESPACE_CXX11_END _GLIBCXX_END_NAMESPACE_CXX11
#else
#define FOLLY_GLIBCXX_NAMESPACE_CXX11_BEGIN
#define FOLLY_GLIBCXX_NAMESPACE_CXX11_END
#endif

// MSVC specific defines
// mainly for posix compat
#ifdef _MSC_VER
#include <folly/portability/SysTypes.h>

// Hide a GCC specific thing that breaks MSVC if left alone.
#define __extension__

// We have compiler support for the newest of the new, but
// MSVC doesn't tell us that.
//
// Clang pretends to be MSVC on Windows, but it refuses to compile
// SSE4.2 intrinsics unless -march argument is specified.
// So cannot unconditionally define __SSE4_2__ in clang.
#ifndef __clang__
#define __SSE4_2__ 1
// compiler specific to compiler specific
// nolint
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

#endif

// Define FOLLY_HAS_EXCEPTIONS
#if __cpp_exceptions >= 199711 || FOLLY_HAS_FEATURE(cxx_exceptions)
#define FOLLY_HAS_EXCEPTIONS 1
#elif __GNUC__
#if __EXCEPTIONS
#define FOLLY_HAS_EXCEPTIONS 1
#else // __EXCEPTIONS
#define FOLLY_HAS_EXCEPTIONS 0
#endif // __EXCEPTIONS
#elif FOLLY_MICROSOFT_ABI_VER
#if _CPPUNWIND
#define FOLLY_HAS_EXCEPTIONS 1
#else // _CPPUNWIND
#define FOLLY_HAS_EXCEPTIONS 0
#endif // _CPPUNWIND
#else
#define FOLLY_HAS_EXCEPTIONS 1 // default assumption for unknown platforms
#endif

// Debug
namespace folly {
#ifdef NDEBUG
constexpr auto kIsDebug = false;
#else
constexpr auto kIsDebug = true;
#endif
} // namespace folly

// Exceptions
namespace folly {
#if FOLLY_HAS_EXCEPTIONS
constexpr auto kHasExceptions = true;
#else
constexpr auto kHasExceptions = false;
#endif
} // namespace folly

// Endianness
namespace folly {
#ifdef _MSC_VER
// It's MSVC, so we just have to guess ... and allow an override
#ifdef FOLLY_ENDIAN_BE
constexpr auto kIsLittleEndian = false;
#else
constexpr auto kIsLittleEndian = true;
#endif
#else
constexpr auto kIsLittleEndian = __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;
#endif
constexpr auto kIsBigEndian = !kIsLittleEndian;
} // namespace folly

#ifndef FOLLY_SSE
#if defined(__SSE4_2__)
#define FOLLY_SSE 4
#define FOLLY_SSE_MINOR 2
#elif defined(__SSE4_1__)
#define FOLLY_SSE 4
#define FOLLY_SSE_MINOR 1
#elif defined(__SSE4__)
#define FOLLY_SSE 4
#define FOLLY_SSE_MINOR 0
#elif defined(__SSE3__)
#define FOLLY_SSE 3
#define FOLLY_SSE_MINOR 0
#elif defined(__SSE2__)
#define FOLLY_SSE 2
#define FOLLY_SSE_MINOR 0
#elif defined(__SSE__)
#define FOLLY_SSE 1
#define FOLLY_SSE_MINOR 0
#else
#define FOLLY_SSE 0
#define FOLLY_SSE_MINOR 0
#endif
#endif

#ifndef FOLLY_SSSE
#if defined(__SSSE3__)
#define FOLLY_SSSE 3
#else
#define FOLLY_SSSE 0
#endif
#endif

#define FOLLY_SSE_PREREQ(major, minor) \
  (FOLLY_SSE > major || FOLLY_SSE == major && FOLLY_SSE_MINOR >= minor)

#ifndef FOLLY_NEON
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define FOLLY_NEON 1
#endif
#endif

#if FOLLY_UNUSUAL_GFLAGS_NAMESPACE
namespace FOLLY_GFLAGS_NAMESPACE {}
namespace gflags {
using namespace FOLLY_GFLAGS_NAMESPACE;
} // namespace gflags
#endif

// for TARGET_OS_IPHONE
#ifdef __APPLE__
#include <TargetConditionals.h> // @manual
#endif

// RTTI may not be enabled for this compilation unit.
#if defined(__GXX_RTTI) || defined(__cpp_rtti) || \
    (defined(_MSC_VER) && defined(_CPPRTTI))
#define FOLLY_HAS_RTTI 1
#else
#define FOLLY_HAS_RTTI 0
#endif

namespace folly {
constexpr bool const kHasRtti = FOLLY_HAS_RTTI;
} // namespace folly

#if defined(__APPLE__) || defined(_MSC_VER)
#define FOLLY_STATIC_CTOR_PRIORITY_MAX
#else
// 101 is the highest priority allowed by the init_priority attribute.
// This priority is already used by JEMalloc and other memory allocators so
// we will take the next one.
#define FOLLY_STATIC_CTOR_PRIORITY_MAX __attribute__((__init_priority__(102)))
#endif

namespace folly {

#if __OBJC__
constexpr auto kIsObjC = true;
#else
constexpr auto kIsObjC = false;
#endif

#if FOLLY_MOBILE
constexpr auto kIsMobile = true;
#else
constexpr auto kIsMobile = false;
#endif

#if defined(__linux__) && !FOLLY_MOBILE
constexpr auto kIsLinux = true;
#else
constexpr auto kIsLinux = false;
#endif

#if defined(_WIN32)
constexpr auto kIsWindows = true;
#else
constexpr auto kIsWindows = false;
#endif

#if __GLIBCXX__
constexpr auto kIsGlibcxx = true;
#else
constexpr auto kIsGlibcxx = false;
#endif

#if __GLIBCXX__ && _GLIBCXX_RELEASE // major version, 7+
constexpr auto kGlibcxxVer = _GLIBCXX_RELEASE;
#else
constexpr auto kGlibcxxVer = 0;
#endif

#if _LIBCPP_VERSION
constexpr auto kIsLibcpp = true;
#else
constexpr auto kIsLibcpp = false;
#endif

#if FOLLY_USE_LIBSTDCPP
constexpr auto kIsLibstdcpp = true;
#else
constexpr auto kIsLibstdcpp = false;
#endif

#if _MSC_VER
constexpr auto kMscVer = _MSC_VER;
#else
constexpr auto kMscVer = 0;
#endif

#if __GNUC__
constexpr auto kGnuc = __GNUC__;
#else
constexpr auto kGnuc = 0;
#endif

#if __clang__
constexpr auto kIsClang = true;
#else
constexpr auto kIsClang = false;
#endif

#if FOLLY_MICROSOFT_ABI_VER
constexpr auto kMicrosoftAbiVer = FOLLY_MICROSOFT_ABI_VER;
#else
constexpr auto kMicrosoftAbiVer = 0;
#endif

// cpplib is an implementation of the standard library, and is the one typically
// used with the msvc compiler
#if _CPPLIB_VER
constexpr auto kCpplibVer = _CPPLIB_VER;
#else
constexpr auto kCpplibVer = 0;
#endif
} // namespace folly

//  MSVC does not permit:
//
//    extern int const num;
//    constexpr int const num = 3;
//
//  Instead:
//
//    extern int const num;
//    FOLLY_STORAGE_CONSTEXPR int const num = 3;
//
//  True as of MSVC 2017.
#if _MSC_VER
#define FOLLY_STORAGE_CONSTEXPR
#else
#define FOLLY_STORAGE_CONSTEXPR constexpr
#endif

#if __cplusplus >= 201703L
// folly::coro requires C++17 support
#if __cpp_coroutines >= 201703L && __has_include(<experimental/coroutine>)
#define FOLLY_HAS_COROUTINES 1
// This is mainly to workaround bugs triggered by LTO, when stack allocated
// variables in await_suspend end up on a coroutine frame.
#define FOLLY_CORO_AWAIT_SUSPEND_NONTRIVIAL_ATTRIBUTES FOLLY_NOINLINE
#elif _MSC_VER && _RESUMABLE_FUNCTIONS_SUPPORTED
// NOTE: MSVC 2017 does not currently support the full Coroutines TS since it
// does not yet support symmetric-transfer.
#define FOLLY_HAS_COROUTINES 0
#else
#define FOLLY_HAS_COROUTINES 0
#endif
#else
#define FOLLY_HAS_COROUTINES 0
#endif // __cplusplus >= 201703L

// MSVC 2017.5 && C++17
#if __cpp_noexcept_function_type >= 201510 || \
    (_MSC_FULL_VER >= 191225816 && _MSVC_LANG > 201402)
#define FOLLY_HAVE_NOEXCEPT_FUNCTION_TYPE 1
#endif

#if __cpp_inline_variables >= 201606L
#define FOLLY_HAS_INLINE_VARIABLES 1
#define FOLLY_INLINE_VARIABLE inline
#else
#define FOLLY_HAS_INLINE_VARIABLES 0
#define FOLLY_INLINE_VARIABLE
#endif

// feature test __cpp_lib_string_view is defined in <string>, which is
// too heavy to include here.  MSVC __has_include support arrived later
// than string_view, so we need an alternate case for it.
#ifdef __has_include
#if __has_include(<string_view>) && __cplusplus >= 201703L
#define FOLLY_HAS_STRING_VIEW 1
#else
#define FOLLY_HAS_STRING_VIEW 0
#endif
#else // __has_include
#if _MSC_VER >= 1910 && (_MSVC_LANG > 201402 || __cplusplus > 201402)
#define FOLLY_HAS_STRING_VIEW 1
#else
#define FOLLY_HAS_STRING_VIEW 0
#endif
#endif // __has_include

#if defined(__linux__)
#define FOLLY_ELF_NATIVE_CLASS __ELF_NATIVE_CLASS
#elif defined(__FreeBSD__)
#if defined(__LP64__)
#define FOLLY_ELF_NATIVE_CLASS 64
#else // __linux__
#define FOLLY_ELF_NATIVE_CLASS 32
#endif
#endif // __linux__