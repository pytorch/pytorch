#ifndef C10_MACROS_MACROS_H_
#define C10_MACROS_MACROS_H_

#ifdef __cplusplus
#include <cassert>
#else
#include <assert.h>
#endif

/* Main entry for torch/headeronly/macros (used to be c10/macros).
 *
 * In your code, include torch/headeronly/macros/Macros.h directly, instead of
 * individual files in this folder.
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
#include <torch/headeronly/macros/cmake_macros.h>
#endif // C10_USING_CUSTOM_GENERATED_MACROS

#include <torch/headeronly/macros/Export.h>

#if defined(__clang__)
#define __ubsan_ignore_float_divide_by_zero__ \
  __attribute__((no_sanitize("float-divide-by-zero")))
#define __ubsan_ignore_undefined__ __attribute__((no_sanitize("undefined")))
#define __ubsan_ignore_signed_int_overflow__ \
  __attribute__((no_sanitize("signed-integer-overflow")))
#define __ubsan_ignore_pointer_overflow__ \
  __attribute__((no_sanitize("pointer-overflow")))
#define __ubsan_ignore_function__ __attribute__((no_sanitize("function")))
#define __ubsan_ignore_float_cast_overflow__ \
  __attribute__((no_sanitize("float-cast-overflow")))
#else
#define __ubsan_ignore_float_divide_by_zero__
#define __ubsan_ignore_undefined__
#define __ubsan_ignore_signed_int_overflow__
#define __ubsan_ignore_pointer_overflow__
#define __ubsan_ignore_function__
#define __ubsan_ignore_float_cast_overflow__
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

#ifndef FBCODE_CAFFE2
/// DEPRECATED: Warn if a type or return value is discarded.
#define C10_NODISCARD [[nodiscard]]

/// DEPRECATED: Suppress an unused variable.
#define C10_UNUSED [[maybe_unused]]
#endif

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

#ifdef __cplusplus

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

#endif // __cplusplus

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

// Unlike C10_ALWAYS_INLINE, C10_ALWAYS_INLINE_ATTRIBUTE can be used
// on a lambda.
#if defined(_MSC_VER)
// MSVC 14.39 is reasonably recent and doesn't like
// [[msvc::forceinline]] on a lambda, so don't try to use it.
#define C10_ALWAYS_INLINE_ATTRIBUTE
#elif __has_attribute(always_inline) || defined(__GNUC__)
#define C10_ALWAYS_INLINE_ATTRIBUTE __attribute__((__always_inline__))
#else
#define C10_ALWAYS_INLINE_ATTRIBUTE
#endif

#if defined(_MSC_VER)
#define C10_ATTR_VISIBILITY_HIDDEN
#elif defined(__GNUC__)
#define C10_ATTR_VISIBILITY_HIDDEN __attribute__((__visibility__("hidden")))
#else
#define C10_ATTR_VISIBILITY_HIDDEN
#endif

#define C10_ERASE C10_ALWAYS_INLINE C10_ATTR_VISIBILITY_HIDDEN

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

#ifdef __HIPCC__
// Unlike CUDA, HIP requires a HIP header to be included for __host__ to work.
// We do this #include here so that C10_HOST_DEVICE and friends will Just Work.
// See https://github.com/ROCm/hip/issues/441
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
#elif __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 870 || __CUDA_ARCH__ == 890 || \
    __CUDA_ARCH__ == 1200
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
        : ((CUDA_MAX_THREADS_PER_SM + (threads_per_block) - 1) /       \
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
// C10_WARP_SIZE is only allowed for device code.
// Host code _must_ use at::cuda::warp_size()
// HIP header used to define warpSize as a constexpr that was either 32 or 64
// depending on the target device, and then always set it to 64 for host code.
// Host pass of HIP compiler needs C10_WARP_SIZE defined to _something_ so we
// set it to something unreasonable to trigger obvious host code errors.

namespace at::cuda {
TORCH_CUDA_CPP_API int warp_size();
}
#ifdef __HIPCC__
static inline int __host__ C10_WARP_SIZE_INTERNAL() {
  return at::cuda::warp_size();
}

static inline constexpr int __device__ C10_WARP_SIZE_INTERNAL() {
#if defined(__GFX9__)
  return 64;
#else // __GFX9__
  return 32;
#endif // __GFX9__
}
#else // __HIPCC__
static inline int C10_WARP_SIZE_INTERNAL() {
  return at::cuda::warp_size();
}
#endif // __HIPCC__

#define C10_WARP_SIZE (C10_WARP_SIZE_INTERNAL())
#define C10_WARP_SIZE_STATIC 64

#else // defined(USE_ROCM)
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
#define CUDA_KERNEL_ASSERT_PRINTF(cond, msg, ...)
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
#define CUDA_KERNEL_ASSERT_PRINTF(cond, msg, ...)                     \
  if (C10_UNLIKELY(!(cond))) {                                        \
    (void)(printf(                                                    \
        "[CUDA_KERNEL_ASSERT] " __FILE__ ":" C10_STRINGIZE(           \
            __LINE__) ": %s: block: [%d,%d,%d], thread: [%d,%d,%d]: " \
                      "Assertion failed: `" #cond "`: " msg "\n",     \
        __func__,                                                     \
        blockIdx.x,                                                   \
        blockIdx.y,                                                   \
        blockIdx.z,                                                   \
        threadIdx.x,                                                  \
        threadIdx.y,                                                  \
        threadIdx.z,                                                  \
        ##__VA_ARGS__));                                              \
    (void)(_wassert(                                                  \
               _CRT_WIDE(#cond),                                      \
               _CRT_WIDE(__FILE__),                                   \
               static_cast<unsigned>(__LINE__)),                      \
           0);                                                        \
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
#elif (defined(__EMSCRIPTEN__))
// As defined in assert.h in the Emscripten stdlib
_Noreturn void __assert_fail(
    const char* expr,
    const char* file,
    int line,
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
// ROCm disables kernel assert by default for performance considerations.
// Though ROCm supports __assert_fail, it uses kernel printf which has
// a non-negligible performance impact even if the assert condition is
// never triggered. We choose to use abort() instead which will still
// terminate the application but without a more useful error message.
#if !defined(C10_USE_ROCM_KERNEL_ASSERT) && defined(USE_ROCM)
#define CUDA_KERNEL_ASSERT(cond) \
  if C10_UNLIKELY (!(cond)) {    \
    abort();                     \
  }
#define CUDA_KERNEL_ASSERT_MSG(cond, msg) \
  if C10_UNLIKELY (!(cond)) {             \
    abort();                              \
  }
#define CUDA_KERNEL_ASSERT_PRINTF(cond, msg, ...) \
  if C10_UNLIKELY (!(cond)) {                     \
    abort();                                      \
  }
#define SYCL_KERNEL_ASSERT(cond) \
  if C10_UNLIKELY (!(cond)) {    \
    abort();                     \
  }
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
#define CUDA_KERNEL_ASSERT_PRINTF(cond, msg, ...)                        \
  if (C10_UNLIKELY(!(cond))) {                                           \
    printf(                                                            \
        "[CUDA_KERNEL_ASSERT] " __FILE__ ":" C10_STRINGIZE(            \
            __LINE__) ": %s: block: [%d,%d,%d], thread: [%d,%d,%d]: "  \
            "Assertion failed: `" #cond "`: " msg "\n",                \
        __func__,                                                      \
        blockIdx.x,                                                    \
        blockIdx.y,                                                    \
        blockIdx.z,                                                    \
        threadIdx.x,                                                   \
        threadIdx.y,                                                   \
        threadIdx.z,                                                   \
        ##__VA_ARGS__); \
    __assert_fail(                                                       \
        #cond, __FILE__, static_cast<unsigned int>(__LINE__), __func__); \
  }
#define SYCL_KERNEL_ASSERT(cond)                                         \
  if (C10_UNLIKELY(!(cond))) {                                           \
    __assert_fail(                                                       \
        #cond, __FILE__, static_cast<unsigned int>(__LINE__), __func__); \
  }
#endif //  C10_USE_ROCM_KERNEL_ASSERT && USE_ROCM
#endif // __APPLE__

// Compile-time switch to control how assertions are logged inside CUDA kernels.
// If C10_CUDA_VERBOSE_ASSERT is defined,  CUDA_KERNEL_ASSERT_VERBOSE will
// take addition information passed to the macro and forward them to
// CUDA_KERNEL_ASSERT_PRINTF If C10_CUDA_VERBOSE_ASSERT is not defined,
// CUDA_KERNEL_ASSERT_VERBOSE will behave the same as CUDA_KERNEL_ASSERT.
#ifdef C10_ENABLE_VERBOSE_ASSERT
#define CUDA_KERNEL_ASSERT_VERBOSE(cond, ...) \
  CUDA_KERNEL_ASSERT_PRINTF(cond, __VA_ARGS__)
#else
#define CUDA_KERNEL_ASSERT_VERBOSE(cond, ...) CUDA_KERNEL_ASSERT(cond)
#endif

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

#if !defined(FBCODE_CAFFE2) && !defined(C10_NODEPRECATED)
#define CONSTEXPR_EXCEPT_WIN_CUDA constexpr
#define C10_HOST_CONSTEXPR_EXCEPT_WIN_CUDA constexpr

#define STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(field, val) \
  static constexpr const char field[] = val;
#define STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cls, field, val)
#endif // !defined(FBCODE_CAFFE2) && !defined(C10_NODEPRECATED)

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

// This macro is used to find older C++ compilers
// that don't support move optimization for return values.

#if (defined(__GNUC__) && __GNUC__ < 13) || \
    (defined(__clang_major__) && __clang_major__ < 13)
#define C10_RETURN_MOVE_IF_OLD_COMPILER 1
#else
#define C10_RETURN_MOVE_IF_OLD_COMPILER 0
#endif

// The HIDDEN_NAMESPACE_BEGIN and HIDDEN_NAMESPACE_END below
// are needed for maintaining robustness in our header APIs in
// torch/headeronly and torch/csrc/stable under the namespaces
// torch::headeronly and torch::stable respectively. We enforce
// hidden visibility for these APIs because we want to enable
// loading custom extensions compiled against different libtorch
// versions where these APIs may have changed.

// Helper macros to handle 1-3 hidden namespace levels when not windows
#define _HIDDEN_NS_GET_MACRO(_1, _2, _3, NAME, ...) NAME
#define _HIDDEN_NS_1(n1) namespace n1 __attribute__((visibility("hidden"))) {
#define _HIDDEN_NS_2(n1, n2) \
  namespace n1 {             \
  namespace n2 __attribute__((visibility("hidden"))) {
#define _HIDDEN_NS_3(n1, n2, n3) \
  namespace n1::n2 {             \
  namespace n3 __attribute__((visibility("hidden"))) {

// Helper macros to close namespaces when not windows
#define _HIDDEN_NS_END_1(n1) }
#define _HIDDEN_NS_END_N(n1, ...) \
  }                               \
  }

// Helper macros to join strs with :: (for win, where symbols are hidden by
// default)
#define _EXPAND(...) __VA_ARGS__
#define _JOIN_GET_MACRO(_1, _2, _3, NAME, ...) NAME
#define _JOIN_NS1(a) a
#define _JOIN_NS2(a, b) a::b
#define _JOIN_NS3(a, b, c) a::b::c

#if !defined(HIDDEN_NAMESPACE_BEGIN)
#if defined(__GNUG__) && !defined(_WIN32)
#define HIDDEN_NAMESPACE_BEGIN(...) \
  _HIDDEN_NS_GET_MACRO(             \
      __VA_ARGS__, _HIDDEN_NS_3, _HIDDEN_NS_2, _HIDDEN_NS_1)(__VA_ARGS__)
#else
#define HIDDEN_NAMESPACE_BEGIN(...)  \
  namespace _EXPAND(_JOIN_GET_MACRO( \
      __VA_ARGS__, _JOIN_NS3, _JOIN_NS2, _JOIN_NS1)(__VA_ARGS__)) {
#endif
#endif

#if !defined(HIDDEN_NAMESPACE_END)
#if defined(__GNUG__) && !defined(_WIN32)
#define HIDDEN_NAMESPACE_END(...)                                         \
  _HIDDEN_NS_GET_MACRO(                                                   \
      __VA_ARGS__, _HIDDEN_NS_END_N, _HIDDEN_NS_END_N, _HIDDEN_NS_END_1)( \
      __VA_ARGS__)
#else
#define HIDDEN_NAMESPACE_END(...) }
#endif
#endif

#endif // C10_MACROS_MACROS_H_
