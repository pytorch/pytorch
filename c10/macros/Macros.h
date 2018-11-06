#ifndef C10_MACROS_MACROS_H_
#define C10_MACROS_MACROS_H_

/* Main entry for c10/macros.
 *
 * In your code, include c10/macros/Macros.h directly, instead of individual
 * files in this folder.
 */

// For build systems that do not directly depend on CMake and directly build
// from the source directory (such as Buck), one may not have a cmake_macros.h
// file at all. In this case, the build system is responsible for providing
// correct macro definitions corresponding to the cmake_macros.h.in file.
//
// In such scenarios, one should define the macro
//     C10_USING_CUSTOM_GENERATED_MACROS
// to inform this header that it does not need to include the cmake_macros.h
// file.

#ifndef C10_USING_CUSTOM_GENERATED_MACROS
#include "c10/macros/cmake_macros.h"
#endif // C10_USING_CUSTOM_GENERATED_MACROS

#include "c10/macros/Export.h"

// Disable the copy and assignment operator for a class. Note that this will
// disable the usage of the class in std containers.
#define C10_DISABLE_COPY_AND_ASSIGN(classname) \
  classname(const classname&) = delete;        \
  classname& operator=(const classname&) = delete

#define CONCAT_IMPL(x, y) x##y
#define MACRO_CONCAT(x, y) CONCAT_IMPL(x, y)

#define MACRO_EXPAND(args) args

/// C10_NODISCARD - Warn if a type or return value is discarded.
#define C10_NODISCARD
#if __cplusplus > 201402L && defined(__has_cpp_attribute)
#if __has_cpp_attribute(nodiscard)
#undef C10_NODISCARD
#define C10_NODISCARD [[nodiscard]]
#endif
// Workaround for llvm.org/PR23435, since clang 3.6 and below emit a spurious
// error when __has_cpp_attribute is given a scoped attribute in C mode.
#elif __cplusplus && defined(__has_cpp_attribute)
#if __has_cpp_attribute(clang::warn_unused_result)
#undef C10_NODISCARD
#define C10_NODISCARD [[clang::warn_unused_result]]
#endif
#endif

// Simply define the namespace, in case a dependent library want to refer to
// the c10 namespace but not any nontrivial files.
namespace c10 {} // namespace c10
namespace c10 { namespace detail {} }

// Since C10 is the core library for caffe2 (and aten), we will simply reroute
// all abstractions defined in c10 to be available in caffe2 as well.
// This is only for backwards compatibility. Please use the symbols from the
// c10 namespace where possible.
namespace caffe2 {using namespace c10;}
namespace at {using namespace c10;}
namespace at { namespace detail { using namespace c10::detail; }}

// C10_NORETURN
#if defined(_MSC_VER)
#define C10_NORETURN __declspec(noreturn)
#else
#define C10_NORETURN __attribute__((noreturn))
#endif

// C10_LIKELY/C10_UNLIKELY
//
// These macros provide parentheses, so you can use these macros as:
//
//    if C10_LIKELY(some_expr) {
//      ...
//    }
//
// NB: static_cast to boolean is mandatory in C++, because __builtin_expect
// takes a long argument, which means you may trigger the wrong conversion
// without it.
//
#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define C10_LIKELY(expr)    (__builtin_expect(static_cast<bool>(expr), 1))
#define C10_UNLIKELY(expr)  (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define C10_LIKELY(expr)    (expr)
#define C10_UNLIKELY(expr)  (expr)
#endif

#include <sstream>
#include <string>

#if defined(__CUDACC__) || defined(__HIPCC__)
// Designates functions callable from the host (CPU) and the device (GPU)
#define C10_HOST_DEVICE __host__ __device__
#define C10_DEVICE __device__
#define C10_HOST __host__
#else
#define C10_HOST_DEVICE
#define C10_HOST
#define C10_DEVICE
#endif

#ifdef __HIP_PLATFORM_HCC__
#define C10_HIP_HOST_DEVICE __host__ __device__
#else
#define C10_HIP_HOST_DEVICE
#endif

#if defined(__ANDROID__)
#define C10_ANDROID 1
#define C10_MOBILE 1
#elif (                   \
    defined(__APPLE__) && \
    (TARGET_IPHONE_SIMULATOR || TARGET_OS_SIMULATOR || TARGET_OS_IPHONE))
#define C10_IOS 1
#define C10_MOBILE 1
#elif (defined(__APPLE__) && TARGET_OS_MAC)
#define C10_IOS 1
#define C10_MOBILE 0
#else
#define C10_MOBILE 0
#endif // ANDROID / IOS / MACOS

#endif // C10_MACROS_MACROS_H_
