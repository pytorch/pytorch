#ifndef C10_MACROS_MACROS_H_
#define C10_MACROS_MACROS_H_
#include <cassert>

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
#include <c10/macros/cmake_macros.h>
#endif // C10_USING_CUSTOM_GENERATED_MACROS

#include <c10/macros/Export.h>

#if defined(__clang__)
#define __ubsan_ignore_float_divide_by_zero__ \
  __attribute__((no_sanitize("float-divide-by-zero")))
#define __ubsan_ignore_undefined__ __attribute__((no_sanitize("undefined")))
#define __ubsan_ignore_signed_int_overflow__ \
  __attribute__((no_sanitize("signed-integer-overflow")))
#define __ubsan_ignore_pointer_overflow__ \
  __attribute__((no_sanitize("pointer-overflow")))
#define __ubsan_ignore_function__ __attribute__((no_sanitize("function")))
#else
#define __ubsan_ignore_float_divide_by_zero__
#define __ubsan_ignore_undefined__
#define __ubsan_ignore_signed_int_overflow__
#define __ubsan_ignore_pointer_overflow__
#define __ubsan_ignore_function__
#endif

// Detect address sanitizer as some stuff doesn't work with it
#undef C10_ASAN_ENABLED

// for clang
#if defined(__has_feature)
#if ((__has_feature(address_sanitizer)))
#define C10_ASAN_ENABLED 1
#endif
#endif

// for gcc
#if defined(__SANITIZE_ADDRESS__)
#if __SANITIZE_ADDRESS__
#if !defined(C10_ASAN_ENABLED)
#define C10_ASAN_ENABLED 1
#endif
#endif
#endif

#if !defined(C10_ASAN_ENABLED)
#define C10_ASAN_ENABLED 0
#endif

// Detect undefined-behavior sanitizer (UBSAN)
#undef C10_UBSAN_ENABLED

// for clang or gcc >= 14
// NB: gcc 14 adds support for Clang's __has_feature
//   https://gcc.gnu.org/gcc-14/changes.html
//   gcc < 14 doesn't have a macro for UBSAN
//   (e.g. __SANITIZE_UNDEFINED__ does not exist in gcc)
//   https://github.com/google/sanitizers/issues/765
#if defined(__has_feature)
#if ((__has_feature(undefined_behavior_sanitizer)))
#define C10_UBSAN_ENABLED 1
#endif
#endif

#if !defined(C10_UBSAN_ENABLED)
#define C10_UBSAN_ENABLED 0
#endif

// Disable the copy and assignment operator for a class. Note that this will
// disable the usage of the class in std containers.
#define C10_DISABLE_COPY_AND_ASSIGN(classname) \
  classname(const classname&) = delete;        \
  classname& operator=(const classname&) = delete

#define C10_CONCATENATE_IMPL(s1, s2) s1##s2
#define C10_CONCATENATE(s1, s2) C10_CONCATENATE_IMPL(s1, s2)

#define C10_MACRO_EXPAND(args) args

#define C10_STRINGIZE_IMPL(x) #x
#define C10_STRINGIZE(x) C10_STRINGIZE_IMPL(x)

/**
 * C10_ANONYMOUS_VARIABLE(str) introduces a new identifier which starts with
 * str and ends with a unique number.
 */
#ifdef __COUNTER__
#define C10_UID __COUNTER__
#define C10_ANONYMOUS_VARIABLE(str) C10_CONCATENATE(str, __COUNTER__)
#else
#define C10_UID __LINE__
#define C10_ANONYMOUS_VARIABLE(str) C10_CONCATENATE(str, __LINE__)
#endif

#ifdef __has_cpp_attribute
#define C10_HAS_CPP_ATTRIBUTE(x) __has_cpp_attribute(x)
#else
#define C10_HAS_CPP_ATTRIBUTE(x) (0)
#endif

/// C10_NODISCARD - Warn if a type or return value is discarded.

// Technically, we should check if __cplusplus > 201402L here, because
// [[nodiscard]] is only defined in C++17.  However, some compilers
// we care about don't advertise being C++17 (e.g., clang), but
// support the attribute anyway.  In fact, this is not just a good idea,
// it's the law: clang::warn_unused_result doesn't work on nvcc + clang
// and the best workaround for this case is to use [[nodiscard]]
// instead; see https://github.com/pytorch/pytorch/issues/13118
//
// Note to future editors: if you have noticed that a compiler is
// misbehaving (e.g., it advertises support, but the support doesn't
// actually work, or it is emitting warnings).  Some compilers which
// are strict about the matter include MSVC, which will complain:
//
//  error C2429: attribute 'nodiscard' requires compiler flag '/std:c++latest'
//
// Exhibits:
//  - MSVC 19.14: https://godbolt.org/z/Dzd7gn (requires /std:c++latest)
//  - Clang 8.0.0: https://godbolt.org/z/3PYL4Z (always advertises support)
//  - gcc 8.3: https://godbolt.org/z/4tLMQS (always advertises support)
#if C10_HAS_CPP_ATTRIBUTE(nodiscard)
#define C10_NODISCARD [[nodiscard]]
// Workaround for llvm.org/PR23435, since clang 3.6 and below emit a spurious
// error when __has_cpp_attribute is given a scoped attribute in C mode.
#elif __cplusplus && C10_HAS_CPP_ATTRIBUTE(clang::warn_unused_result)
// TODO: It's possible this is still triggering
// https://github.com/pytorch/pytorch/issues/13118 on Windows; if it is, better
// fix it.
#define C10_NODISCARD [[clang::warn_unused_result]]
#else
#define C10_NODISCARD
#endif

// suppress an unused variable.
#if defined(_MSC_VER) && !defined(__clang__)
#define C10_UNUSED __pragma(warning(suppress : 4100 4101))
#else
#define C10_UNUSED __attribute__((__unused__))
#endif //_MSC_VER

#if !defined(__has_attribute)
#define __has_attribute(x) 0
#endif

// Direct port of LLVM_ATTRIBUTE_USED.
#if __has_attribute(used)
#define C10_USED __attribute__((__used__))
#else
#define C10_USED
#endif

#define C10_RESTRICT __restrict

// Simply define the namespace, in case a dependent library want to refer to
// the c10 namespace but not any nontrivial files.
namespace c10 {}
namespace c10::cuda {}
namespace c10::hip {}
namespace c10::xpu {}

// Since C10 is the core library for caffe2 (and aten), we will simply reroute
// all abstractions defined in c10 to be available in caffe2 as well.
// This is only for backwards compatibility. Please use the symbols from the
// c10 namespace where possible.
namespace caffe2 {
using namespace c10;
}
namespace at {
using namespace c10;
}
namespace at::cuda {
using namespace c10::cuda;
} // namespace at::cuda

// WARNING!!! THIS IS A GIANT HACK!!!
// This line means you cannot simultaneously include c10/hip
// and c10/cuda and then use them from the at::cuda namespace.
// This is true in practice, because HIPIFY works inplace on
// files in ATen/cuda, so it assumes that c10::hip is available
// from at::cuda.  This namespace makes that happen.  When
// HIPIFY is no longer out-of-place, we can switch the cuda
// here to hip and everyone is happy.
namespace at::cuda {
using namespace c10::hip;
} // namespace at::cuda

namespace at::xpu {
using namespace c10::xpu;
} // namespace at::xpu

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
#define C10_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define C10_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define C10_LIKELY(expr) (expr)
#define C10_UNLIKELY(expr) (expr)
#endif

/// C10_NOINLINE - Functions whose declaration is annotated with this will not
/// be inlined.
#ifdef __GNUC__
#define C10_NOINLINE __attribute__((noinline))
#elif _MSC_VER
#define C10_NOINLINE __declspec(noinline)
#else
#define C10_NOINLINE
#endif

#if defined(_MSC_VER)
#define C10_ALWAYS_INLINE __forceinline
#elif __has_attribute(always_inline) || defined(__GNUC__)
#define C10_ALWAYS_INLINE __attribute__((__always_inline__)) inline
#else
#define C10_ALWAYS_INLINE inline
#endif

#if defined(_MSC_VER)
#define C10_ATTR_VISIBILITY_HIDDEN
#elif defined(__GNUC__)
#define C10_ATTR_VISIBILITY_HIDDEN __attribute__((__visibility__("hidden")))
#else
#define C10_ATTR_VISIBILITY_HIDDEN
#endif

#define C10_ERASE C10_ALWAYS_INLINE C10_ATTR_VISIBILITY_HIDDEN

#include <cstdint>

#ifdef __HIPCC__
// Unlike CUDA, HIP requires a HIP header to be included for __host__ to work.
// We do this #include here so that C10_HOST_DEVICE and friends will Just Work.
// See https://github.com/ROCm-Developer-Tools/HIP/issues/441
#include <hip/hip_runtime.h>
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
// Designates functions callable from the host (CPU) and the device (GPU)
#define C10_HOST_DEVICE __host__ __device__
#define C10_DEVICE __device__
#define C10_HOST __host__
// constants from
// (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications)
// The maximum number of threads per multiprocessor is 1024 for Turing
// architecture (7.5), 1536 for Geforce Ampere (8.6)/Jetson Orin (8.7), and
// 2048 for all other architectures. You'll get warnings if you exceed these
// constants. Hence, the following macros adjust the input values from the user
// to resolve potential warnings.
#if __CUDA_ARCH__ == 750
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 1024;
#elif __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 870 || __CUDA_ARCH__ == 890
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 1536;
#else
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 2048;
#endif
// CUDA_MAX_THREADS_PER_BLOCK is same for all architectures currently
constexpr uint32_t CUDA_MAX_THREADS_PER_BLOCK = 1024;
// CUDA_THREADS_PER_BLOCK_FALLBACK is the "canonical fallback" choice of block
// size. 256 is a good number for this fallback and should give good occupancy
// and versatility across all architectures.
constexpr uint32_t CUDA_THREADS_PER_BLOCK_FALLBACK = 256;
// NOTE: if you are thinking of constexpr-ify the inputs to launch bounds, it
//       turns out that although __launch_bounds__ can take constexpr, it
//       can't take a constexpr that has anything to do with templates.
//       Currently we use launch_bounds that depend on template arguments in
//       Loops.cuh, Reduce.cuh and LossCTC.cuh. Hence, C10_MAX_THREADS_PER_BLOCK
//       and C10_MIN_BLOCKS_PER_SM are kept as macros.
// Suppose you were planning to write __launch_bounds__(a, b), based on your
// performance tuning on a modern GPU. Instead, you should write
// __launch_bounds__(C10_MAX_THREADS_PER_BLOCK(a), C10_MIN_BLOCKS_PER_SM(a, b)),
// which will also properly respect limits on old architectures.
#define C10_MAX_THREADS_PER_BLOCK(val)           \
  (((val) <= CUDA_MAX_THREADS_PER_BLOCK) ? (val) \
                                         : CUDA_THREADS_PER_BLOCK_FALLBACK)
#define C10_MIN_BLOCKS_PER_SM(threads_per_block, blocks_per_sm)        \
  ((((threads_per_block) * (blocks_per_sm) <= CUDA_MAX_THREADS_PER_SM) \
        ? (blocks_per_sm)                                              \
        : ((CUDA_MAX_THREADS_PER_SM + (threads_per_block)-1) /         \
           (threads_per_block))))
// C10_LAUNCH_BOUNDS is analogous to __launch_bounds__
#define C10_LAUNCH_BOUNDS_0 \
  __launch_bounds__(        \
      256, 4) // default launch bounds that should give good occupancy and
              // versatility across all architectures.
#define C10_LAUNCH_BOUNDS_1(max_threads_per_block) \
  __launch_bounds__((C10_MAX_THREADS_PER_BLOCK((max_threads_per_block))))
#define C10_LAUNCH_BOUNDS_2(max_threads_per_block, min_blocks_per_sm) \
  __launch_bounds__(                                                  \
      (C10_MAX_THREADS_PER_BLOCK((max_threads_per_block))),           \
      (C10_MIN_BLOCKS_PER_SM((max_threads_per_block), (min_blocks_per_sm))))
#else
#define C10_HOST_DEVICE
#define C10_HOST
#define C10_DEVICE
#endif

#if defined(USE_ROCM)
#define C10_HIP_HOST_DEVICE __host__ __device__
#else
#define C10_HIP_HOST_DEVICE
#endif

#if defined(USE_ROCM)
#define C10_WARP_SIZE warpSize // = 64 or 32 (Defined in hip_runtime.h)
#else
#define C10_WARP_SIZE 32
#endif

#if defined(_MSC_VER) && _MSC_VER <= 1900
#define __func__ __FUNCTION__
#endif

// CUDA_KERNEL_ASSERT checks the assertion
// even when NDEBUG is defined. This is useful for important assertions in CUDA
// code that would otherwise be suppressed when building Release.
#if defined(__ANDROID__) || defined(__APPLE__) || defined(__FreeBSD__)
// Those platforms do not support assert()
#define CUDA_KERNEL_ASSERT(cond)
#define CUDA_KERNEL_ASSERT_MSG(cond, msg)
#define SYCL_KERNEL_ASSERT(cond)
#elif defined(_MSC_VER)
#if defined(NDEBUG)
extern "C" {
C10_IMPORT
#if defined(__SYCL_DEVICE_ONLY__)
extern SYCL_EXTERNAL void _wassert(
    const wchar_t* wexpr,
    const wchar_t* wfile,
    unsigned line);
#else
#if defined(__CUDA_ARCH__)
__host__ __device__
#endif // __CUDA_ARCH__
    void
    _wassert(wchar_t const* _Message, wchar_t const* _File, unsigned _Line);
#endif // __SYCL_DEVICE_ONLY__
}
#endif // NDEBUG
#define CUDA_KERNEL_ASSERT(cond)                 \
  if (C10_UNLIKELY(!(cond))) {                   \
    (void)(_wassert(                             \
               _CRT_WIDE(#cond),                 \
               _CRT_WIDE(__FILE__),              \
               static_cast<unsigned>(__LINE__)), \
           0);                                   \
  }
// TODO: This doesn't assert the message because I (chilli) couldn't figure out
// a nice way to convert a char* to a wchar_t*
#define CUDA_KERNEL_ASSERT_MSG(cond, msg)        \
  if (C10_UNLIKELY(!(cond))) {                   \
    (void)(_wassert(                             \
               _CRT_WIDE(#cond),                 \
               _CRT_WIDE(__FILE__),              \
               static_cast<unsigned>(__LINE__)), \
           0);                                   \
  }
#define SYCL_KERNEL_ASSERT(cond)                 \
  if (C10_UNLIKELY(!(cond))) {                   \
    (void)(_wassert(                             \
               _CRT_WIDE(#cond),                 \
               _CRT_WIDE(__FILE__),              \
               static_cast<unsigned>(__LINE__)), \
           0);                                   \
  }
#else // __APPLE__, _MSC_VER
#if defined(NDEBUG)
extern "C" {
#if defined(__SYCL_DEVICE_ONLY__)
extern SYCL_EXTERNAL void __assert_fail(
    const char* expr,
    const char* file,
    unsigned int line,
    const char* func);
#else // __SYCL_DEVICE_ONLY__
#if (defined(__CUDA_ARCH__) && !(defined(__clang__) && defined(__CUDA__)))
// CUDA supports __assert_fail function which are common for both device
// and host side code.
__host__ __device__
#endif

    // This forward declaration matching the declaration of __assert_fail
    // exactly how it is in glibc in case parts of the program are compiled with
    // different NDEBUG settings. Otherwise we might get 'ambiguous declaration'
    // error. Note: On ROCm - this declaration serves for host side compilation.
    void
    __assert_fail(
        const char* assertion,
        const char* file,
        unsigned int line,
        const char* function) noexcept __attribute__((__noreturn__));

#endif // __SYCL_DEVICE_ONLY__
}
#endif // NDEBUG
// ROCm disable kernel assert by default
#if !defined(C10_USE_ROCM_KERNEL_ASSERT) and defined(USE_ROCM)
#define CUDA_KERNEL_ASSERT(cond)
#define CUDA_KERNEL_ASSERT_MSG(cond, msg)
#define SYCL_KERNEL_ASSERT(cond)
#else
#define CUDA_KERNEL_ASSERT(cond)                                         \
  if (C10_UNLIKELY(!(cond))) {                                           \
    __assert_fail(                                                       \
        #cond, __FILE__, static_cast<unsigned int>(__LINE__), __func__); \
  }
#define CUDA_KERNEL_ASSERT_MSG(cond, msg)                              \
  if (C10_UNLIKELY(!(cond))) {                                         \
    __assert_fail(                                                     \
        msg, __FILE__, static_cast<unsigned int>(__LINE__), __func__); \
  }
#define SYCL_KERNEL_ASSERT(cond)                                         \
  if (C10_UNLIKELY(!(cond))) {                                           \
    __assert_fail(                                                       \
        #cond, __FILE__, static_cast<unsigned int>(__LINE__), __func__); \
  }
#endif //  C10_USE_ROCM_KERNEL_ASSERT and USE_ROCM
#endif // __APPLE__

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#if defined(__ANDROID__)
#define C10_ANDROID 1
#define C10_MOBILE 1
#elif (                   \
    defined(__APPLE__) && \
    (TARGET_IPHONE_SIMULATOR || TARGET_OS_SIMULATOR || TARGET_OS_IPHONE))
#define C10_IOS 1
#define C10_MOBILE 1
#endif // ANDROID / IOS

#if defined(C10_MOBILE) && C10_MOBILE
#define C10_ALWAYS_INLINE_UNLESS_MOBILE inline
#else
#define C10_ALWAYS_INLINE_UNLESS_MOBILE C10_ALWAYS_INLINE
#endif

#if defined(__CUDA_ARCH__)
#if defined(_MSC_VER) && defined(__CUDACC__)
#define CONSTEXPR_EXCEPT_WIN_CUDA const
#define C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA __host__

// Note [static constexpr char* members for windows NVCC]
// The Windows NVCC compiler doesn't handle static constexpr class members,
// although it's fixed in a later version.
// (see
// https://developercommunity.visualstudio.com/t/intellisense-error-c11-static-constexpr-member-ini/245425)
//
// If we want to ensure that our field is static under all builds, then we need
// to work around it specifically for windows NVCC by making it (a) const, (b)
// defined outside of the class definition We need to define it outside of the
// class definition because of the C++ standard; char* is not an integral type
// (see
// https://stackoverflow.com/questions/24278473/intellisense-a-member-of-type-const-char-const-cannot-have-an-in-class-in)
//
// So instead of this:
// struct Foo {
//     static constexpr const char* name = "foo";
// }
// In Windows NVCC, we end up with this:
// struct Foo {
//     static const char* name;
// }
// const char* Foo::name = "foo";
//
// This gives us a small perf hit for any code that wants to access these field
// members, but right now it isn't used in any perf-critical code paths.
#define STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(field, val) \
  static const char* field;
#define STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cls, field, val) \
  const char* cls::field = val;
#else
#define CONSTEXPR_EXCEPT_WIN_CUDA constexpr
#define C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA __host__

#define STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(field, val) \
  static constexpr const char* field = val;
#define STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cls, field, val)
#endif
#else
#if defined(_MSC_VER) && defined(__CUDACC__)
#define CONSTEXPR_EXCEPT_WIN_CUDA const
#define C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA

#define STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(field, val) \
  static const char* field;
#define STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cls, field, val) \
  const char* cls::field = val;
#else
#define CONSTEXPR_EXCEPT_WIN_CUDA constexpr
#define C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA constexpr

#define STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(field, val) \
  static constexpr const char* field = val;
#define STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cls, field, val)
#endif
#endif

#ifndef HAS_DEMANGLE
#if defined(__ANDROID__) || defined(_WIN32) || defined(__EMSCRIPTEN__)
#define HAS_DEMANGLE 0
#elif defined(__APPLE__) && \
    (TARGET_IPHONE_SIMULATOR || TARGET_OS_SIMULATOR || TARGET_OS_IPHONE)
#define HAS_DEMANGLE 0
#else
#define HAS_DEMANGLE 1
#endif
#endif // HAS_DEMANGLE

#define _C10_PRAGMA__(string) _Pragma(#string)
#define _C10_PRAGMA_(string) _C10_PRAGMA__(string)

#ifdef __clang__
#define C10_CLANG_DIAGNOSTIC_PUSH() _Pragma("clang diagnostic push")
#define C10_CLANG_DIAGNOSTIC_POP() _Pragma("clang diagnostic pop")
#define C10_CLANG_DIAGNOSTIC_IGNORE(flag) \
  _C10_PRAGMA_(clang diagnostic ignored flag)
#define C10_CLANG_HAS_WARNING(flag) __has_warning(flag)
#else
#define C10_CLANG_DIAGNOSTIC_PUSH()
#define C10_CLANG_DIAGNOSTIC_POP()
#define C10_CLANG_DIAGNOSTIC_IGNORE(flag)
#define C10_CLANG_HAS_WARNING(flag) 0
#endif

#ifdef __clang__

#define C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED(warning)         \
  _C10_PRAGMA_(clang diagnostic push)                               \
  _C10_PRAGMA_(clang diagnostic ignored "-Wunknown-warning-option") \
  _C10_PRAGMA_(clang diagnostic ignored warning)

#define C10_DIAGNOSTIC_POP() _C10_PRAGMA_(clang diagnostic pop)

#elif __GNUC__

#define C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED(warning) \
  _C10_PRAGMA_(GCC diagnostic push)                         \
  _C10_PRAGMA_(GCC diagnostic ignored "-Wpragmas")          \
  _C10_PRAGMA_(GCC diagnostic ignored warning)

#define C10_DIAGNOSTIC_POP() _C10_PRAGMA_(GCC diagnostic pop)

#else

#define C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED(warning)
#define C10_DIAGNOSTIC_POP()

#endif

#endif // C10_MACROS_MACROS_H_
