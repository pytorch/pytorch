#pragma once

/**
 * This file provides portable macros for marking declarations
 * as deprecated.  You should generally use C10_DEPRECATED,
 * except when marking 'using' declarations as deprecated,
 * in which case you should use C10_DEPRECATED_USING (due to
 * portability concerns).
 */

// Sample usage:
//
//    C10_DEPRECATED void bad_func();
//    struct C10_DEPRECATED BadStruct {
//      ...
//    };

// NB: In PyTorch, this block is not actually used at the moment
// because we are C++11.  However, aspirationally, we would like
// to use this version, because as of C++14 it is the correct and
// portable way to declare something deprecated.
#if defined(__cplusplus) && __cplusplus >= 201402L
# define C10_DEPRECATED [[deprecated]]
# define C10_DEPRECATED_MESSAGE(message) [[deprecated(message)]]
#elif defined(__GNUC__)
# define C10_DEPRECATED __attribute__((deprecated))
// TODO: is there some way to implement this?
# define C10_DEPRECATED_MESSAGE(message) __attribute__((deprecated))
#elif defined(_MSC_VER)
# define C10_DEPRECATED __declspec(deprecated)
// TODO: is there some way to implement this?
# define C10_DEPRECATED_MESSAGE(message) __declspec(deprecated)
#else
# warning "You need to implement C10_DEPRECATED for this compiler"
# define C10_DEPRECATED
#endif


// Sample usage:
//
//    using BadType C10_DEPRECATED_USING = int;

// technically [[deprecated]] syntax is from c++14 standard, but it works in
// many compilers.
#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(deprecated)
# define C10_DEPRECATED_USING [[deprecated]]
#endif
#endif

#if !defined(C10_DEPRECATED_USING) && defined(_MSC_VER)
#if defined(__CUDACC__)
// [[deprecated]] doesn't work on nvcc on Windows;
// you get the error:
//
//    error: attribute does not apply to any entity
//
// So we just turn the macro off in this case.
# define C10_DEPRECATED_USING
#else
// [[deprecated]] does work in windows without nvcc, though msc doesn't support
// `__has_cpp_attribute`.
# define C10_DEPRECATED_USING [[deprecated]]
#endif
#endif

#if !defined(C10_DEPRECATED_USING) && defined(__GNUC__)
// nvcc has a bug where it doesn't understand __attribute__((deprecated))
// declarations even when the host compiler supports it. We'll only use this gcc
// attribute when not cuda, and when using a GCC compiler that doesn't support
// the c++14 syntax we checked for above (availble in __GNUC__ >= 5)
#if !defined(__CUDACC__)
# define C10_DEPRECATED_USING __attribute__((deprecated))
#else
// using cuda + gcc < 5, neither deprecated syntax is available so turning off.
# define C10_DEPRECATED_USING
#endif
#endif

#if ! defined(C10_DEPRECATED_USING)
# warning "You need to implement C10_DEPRECATED_USING for this compiler"
# define C10_DEPRECATED_USING
#endif
