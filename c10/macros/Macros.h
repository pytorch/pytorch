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
#include <c10/macros/cmake_macros.h>
#endif // C10_USING_CUSTOM_GENERATED_MACROS

#include <c10/macros/Export.h>

#if defined(__clang__)
  #define __ubsan_ignore_float_divide_by_zero__ __attribute__((no_sanitize("float-divide-by-zero")))
  #define __ubsan_ignore_undefined__ __attribute__((no_sanitize("undefined")))
  #define __ubsan_ignore_signed_int_overflow__ __attribute__((no_sanitize("signed-integer-overflow")))
#else
  #define __ubsan_ignore_float_divide_by_zero__
  #define __ubsan_ignore_undefined__
  #define __ubsan_ignore_signed_int_overflow__
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
 * C10_ANONYMOUS_VARIABLE(str) introduces an identifier starting with
 * str and ending with a number that varies with the line.
 */
#ifdef __COUNTER__
#define C10_UID __COUNTER__
#define C10_ANONYMOUS_VARIABLE(str) C10_CONCATENATE(str, __COUNTER__)
#else
#define C10_UID __LINE__
#define C10_ANONYMOUS_VARIABLE(str) C10_CONCATENATE(str, __LINE__)
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
#define C10_NODISCARD
#if defined(__has_cpp_attribute)
# if __has_cpp_attribute(nodiscard)
#  undef C10_NODISCARD
#  define C10_NODISCARD [[nodiscard]]
# endif
// Workaround for llvm.org/PR23435, since clang 3.6 and below emit a spurious
// error when __has_cpp_attribute is given a scoped attribute in C mode.
#elif __cplusplus && defined(__has_cpp_attribute)
# if __has_cpp_attribute(clang::warn_unused_result)
// TODO: It's possible this is still triggering https://github.com/pytorch/pytorch/issues/13118
// on Windows; if it is, better fix it.
#  undef C10_NODISCARD
#  define C10_NODISCARD [[clang::warn_unused_result]]
# endif
#endif

// suppress an unused variable.
#if defined(_MSC_VER) && !defined(__clang__)
#define C10_UNUSED __pragma(warning(suppress: 4100 4101))
#else
#define C10_UNUSED __attribute__((__unused__))
#endif //_MSC_VER

#define C10_RESTRICT __restrict

// Simply define the namespace, in case a dependent library want to refer to
// the c10 namespace but not any nontrivial files.
namespace c10 {} // namespace c10
namespace c10 { namespace cuda {} }
namespace c10 { namespace hip {} }

// Since C10 is the core library for caffe2 (and aten), we will simply reroute
// all abstractions defined in c10 to be available in caffe2 as well.
// This is only for backwards compatibility. Please use the symbols from the
// c10 namespace where possible.
namespace caffe2 { using namespace c10; }
namespace at { using namespace c10; }
namespace at { namespace cuda { using namespace c10::cuda; }}

// WARNING!!! THIS IS A GIANT HACK!!!
// This line means you cannot simultaneously include c10/hip
// and c10/cuda and then use them from the at::cuda namespace.
// This is true in practice, because HIPIFY works inplace on
// files in ATen/cuda, so it assumes that c10::hip is available
// from at::cuda.  This namespace makes that happen.  When
// HIPIFY is no longer out-of-place, we can switch the cuda
// here to hip and everyone is happy.
namespace at { namespace cuda { using namespace c10::hip; }}

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

#include <sstream>
#include <string>

#if defined(__CUDACC__) || defined(__HIPCC__)
// Designates functions callable from the host (CPU) and the device (GPU)
#define C10_HOST_DEVICE __host__ __device__
#define C10_DEVICE __device__
#define C10_HOST __host__
// constants from (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications)
// The maximum number of threads per multiprocessor is 1024 for Turing architecture (7.5),
// 1536 for Geforce Ampere (8.6),
// and 2048 for all other architectures. You'll get warnings if you exceed these constants.
// Hence, the following macros adjust the input values from the user to resolve potential warnings.
#if __CUDA_ARCH__ == 750
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 1024;
#elif __CUDA_ARCH__ == 860
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 1536;
#else
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 2048;
#endif
// CUDA_MAX_THREADS_PER_BLOCK is same for all architectures currently
constexpr uint32_t CUDA_MAX_THREADS_PER_BLOCK = 1024;
// CUDA_THREADS_PER_BLOCK_FALLBACK is the "canonical fallback" choice of block size.
// 256 is a good number for this fallback and should give good occupancy and
// versatility across all architectures.
constexpr uint32_t CUDA_THREADS_PER_BLOCK_FALLBACK = 256;
// NOTE: if you are thinking of constexpr-ify the inputs to launch bounds, it
//       turns out that although __launch_bounds__ can take constexpr, it
//       can't take a constexpr that has anything to do with templates.
//       Currently we use launch_bounds that depend on template arguments in
//       Loops.cuh, Reduce.cuh and LossCTC.cuh. Hence, C10_MAX_THREADS_PER_BLOCK and
//       C10_MIN_BLOCKS_PER_SM are kept as macros.
// Suppose you were planning to write __launch_bounds__(a, b), based on your performance tuning on a modern GPU.
// Instead, you should write __launch_bounds__(C10_MAX_THREADS_PER_BLOCK(a), C10_MIN_BLOCKS_PER_SM(a, b)),
// which will also properly respect limits on old architectures.
#define C10_MAX_THREADS_PER_BLOCK(val) (((val) <= CUDA_MAX_THREADS_PER_BLOCK) ? (val) : CUDA_THREADS_PER_BLOCK_FALLBACK)
#define C10_MIN_BLOCKS_PER_SM(threads_per_block, blocks_per_sm) ((((threads_per_block)*(blocks_per_sm) <= CUDA_MAX_THREADS_PER_SM) ? (blocks_per_sm) : ((CUDA_MAX_THREADS_PER_SM + (threads_per_block) - 1) / (threads_per_block))))
// C10_LAUNCH_BOUNDS is analogous to __launch_bounds__
#define C10_LAUNCH_BOUNDS_0 __launch_bounds__(256, 4) // default launch bounds that should give good occupancy and versatility across all architectures.
#define C10_LAUNCH_BOUNDS_1(max_threads_per_block) __launch_bounds__((C10_MAX_THREADS_PER_BLOCK((max_threads_per_block))))
#define C10_LAUNCH_BOUNDS_2(max_threads_per_block, min_blocks_per_sm) __launch_bounds__((C10_MAX_THREADS_PER_BLOCK((max_threads_per_block))), (C10_MIN_BLOCKS_PER_SM((max_threads_per_block), (min_blocks_per_sm))))
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

#ifdef __HIP_PLATFORM_HCC__
#define C10_WARP_SIZE 64
#else
#define C10_WARP_SIZE 32
#endif

#if defined(_MSC_VER) && _MSC_VER <= 1900
#define __func__ __FUNCTION__
#endif

// CUDA_KERNEL_ASSERT checks the assertion
// even when NDEBUG is defined. This is useful for important assertions in CUDA
// code that would otherwise be suppressed when building Release.
#if defined(__ANDROID__) || defined(__APPLE__) || defined(__HIP_PLATFORM_HCC__)
// Those platforms do not support assert()
#define CUDA_KERNEL_ASSERT(cond)
#elif defined(_MSC_VER)
#if defined(NDEBUG)
extern "C" {
  C10_IMPORT
#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__) || defined(__HIP__)
    __host__ __device__
#endif // __CUDA_ARCH__
 void _wassert(
    wchar_t const* _Message,
    wchar_t const* _File,
    unsigned _Line);
}
#endif
#define CUDA_KERNEL_ASSERT(cond)                                                                 \
  if (C10_UNLIKELY(!(cond))) {                                                                   \
    (void)(_wassert(_CRT_WIDE(#cond), _CRT_WIDE(__FILE__), static_cast<unsigned>(__LINE__)), 0); \
  }
#else // __APPLE__, _MSC_VER
#if defined(NDEBUG)
extern "C" {
#if (defined(__CUDA_ARCH__) && !(defined(__clang__) && defined(__CUDA__))) || \
    defined(__HIP_ARCH__) || defined(__HIP__)
__host__ __device__
#endif // __CUDA_ARCH__
    void
    __assert_fail(
        const char* assertion,
        const char* file,
        unsigned int line,
        const char* function) throw();
}
#endif // NDEBUG
#define CUDA_KERNEL_ASSERT(cond)                                         \
  if (C10_UNLIKELY(!(cond))) {                                           \
    __assert_fail(#cond, __FILE__, static_cast<unsigned int>(__LINE__),  \
                  __func__);                                             \
  }
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

// Portable determination of whether type T is trivially copyable.
// Warning: __has_trivial_copy for GCC may not always detect the non-POD
// correctly. For example, T = std::unique_ptr may evaluate to true and be
// treated as POD. This can cause unexpected behavior.
#if defined(__GNUG__) && __GNUC__ < 5
#define C10_IS_TRIVIALLY_COPYABLE(T) __has_trivial_copy(T)
#else
#define C10_IS_TRIVIALLY_COPYABLE(T) std::is_trivially_copyable<T>::value
#endif

// We need --expt-relaxed-constexpr in CUDA because of Eigen. This flag allows
// device code in CUDA to call host constexpr functions. Unfortunately,
// the CUDA compiler (at least for CUDA 9.0, 9.1 and 9.2) isn't compatible
// with many of the constexpr things we'd like to do and the device code
// compiler crashes when it sees one of these host-only functions.
// It works when nvcc builds host code, but not when it builds device code
// and notices it can call these constexpr functions from device code.
// As a workaround, we use C10_HOST_CONSTEXPR instead of constexpr for these
// functions. This enables constexpr when compiled on the host and applies
// __host__ when it is compiled on the device in an attempt to stop it from
// being called from device functions. Not sure if the latter works, but
// even if not, it not being constexpr anymore should be enough to stop
// it from being called from device code.
// TODO This occurred in CUDA 9 (9.0 to 9.2). Test if this is fixed in CUDA 10.
#if defined(__CUDA_ARCH__)
#define C10_HOST_CONSTEXPR __host__
#define C10_HOST_CONSTEXPR_VAR
#else
#define C10_HOST_CONSTEXPR constexpr
#define C10_HOST_CONSTEXPR_VAR constexpr
#endif

#if !defined(__clang__) && !defined(_MSC_VER) && defined(__GNUC__) && \
    __GNUC__ < 6
#define CONSTEXPR_EXCEPT_GCC5
#define IS_NOT_GCC5_CONSTEXPR 0
#else
#define CONSTEXPR_EXCEPT_GCC5 constexpr
#define IS_NOT_GCC5_CONSTEXPR 1
#endif

#if defined(__CUDA_ARCH__)
#if defined(_MSC_VER) && defined(__CUDACC__)
#define CONSTEXPR_EXCEPT_WIN_CUDA const
#define C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA __host__
#else
#define CONSTEXPR_EXCEPT_WIN_CUDA constexpr
#define C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA __host__
#endif
#else
#if defined(_MSC_VER) && defined(__CUDACC__)
#define CONSTEXPR_EXCEPT_WIN_CUDA const
#define C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA
#else
#define CONSTEXPR_EXCEPT_WIN_CUDA constexpr
#define C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA constexpr
#endif
#endif

#endif // C10_MACROS_MACROS_H_
