#pragma once

#define TORCH_STANDALONE_CONCATENATE_IMPL(s1, s2) s1##s2
#define TORCH_STANDALONE_CONCATENATE(s1, s2) \
  TORCH_STANDALONE_CONCATENATE_IMPL(s1, s2)

#define TORCH_STANDALONE_MACRO_EXPAND(args) args

#define TORCH_STANDALONE_STRINGIZE_IMPL(x) #x
#define TORCH_STANDALONE_STRINGIZE(x) TORCH_STANDALONE_STRINGIZE_IMPL(x)

// TORCH_STANDALONE_LIKELY/TORCH_STANDALONE_UNLIKELY
//
// These macros provide parentheses, so you can use these macros as:
//
//    if TORCH_STANDALONE_LIKELY(some_expr) {
//      ...
//    }
//
// NB: static_cast to boolean is mandatory in C++, because __builtin_expect
// takes a long argument, which means you may trigger the wrong conversion
// without it.
//
#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define TORCH_STANDALONE_LIKELY(expr) \
  (__builtin_expect(static_cast<bool>(expr), 1))
#define TORCH_STANDALONE_UNLIKELY(expr) \
  (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define TORCH_STANDALONE_LIKELY(expr) (expr)
#define TORCH_STANDALONE_UNLIKELY(expr) (expr)
#endif

// On nvcc, TORCH_STANDALONE_UNLIKELY thwarts missing return statement analysis.
// In cases where the unlikely expression may be a constant, use this macro to
// ensure return statement analysis keeps working (at the cost of not getting
// the likely/unlikely annotation on nvcc).
// https://github.com/pytorch/pytorch/issues/21418
//
// Currently, this is only used in the error reporting macros below.  If you
// want to use it more generally, move me to Macros.h
//
// TODO: Brian Vaughan observed that we might be able to get this to work on
// nvcc by writing some sort of C++ overload that distinguishes constexpr inputs
// from non-constexpr.  Since there isn't any evidence that losing
// TORCH_STANDALONE_UNLIKELY in nvcc is causing us perf problems, this is not
// yet implemented, but this might be an interesting piece of C++ code for an
// intrepid bootcamper to write.
#if defined(__CUDACC__)
#define TORCH_STANDALONE_UNLIKELY_OR_CONST(e) e
#else
#define TORCH_STANDALONE_UNLIKELY_OR_CONST(e) TORCH_STANDALONE_UNLIKELY(e)
#endif
