#pragma once

/**
 * This file provides portable macros for marking declarations
 * as deprecated.  You should generally use C10_DEPRECATED,
 * except when marking 'using' declarations as deprecated,
 * in which case you should use C10_DEFINE_DEPRECATED_USING
 * (due to portability concerns).
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
// NB: __cplusplus doesn't work for MSVC, so for now MSVC always uses
// the "__declspec(deprecated)" implementation and not the C++14 "[[deprecated]]"
// attribute. We tried enabling "[[deprecated]]" for C++14 on MSVC, but
// ran into issues with some older MSVC versions.
#if (defined(__cplusplus) && __cplusplus >= 201402L)
# define C10_DEPRECATED [[deprecated]]
# define C10_DEPRECATED_MESSAGE(message) [[deprecated(message)]]
#elif defined(__GNUC__)
# define C10_DEPRECATED __attribute__((deprecated))
// TODO Is there some way to implement this?
# define C10_DEPRECATED_MESSAGE(message) __attribute__((deprecated))

#elif defined(_MSC_VER)
# define C10_DEPRECATED __declspec(deprecated)
# define C10_DEPRECATED_MESSAGE(message) __declspec(deprecated(message))
#else
# warning "You need to implement C10_DEPRECATED for this compiler"
# define C10_DEPRECATED
#endif


// Sample usage:
//
//    C10_DEFINE_DEPRECATED_USING(BadType, int)
//
//   which is the portable version of
//
//    using BadType [[deprecated]] = int;

// technically [[deprecated]] syntax is from c++14 standard, but it works in
// many compilers.
#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(deprecated)
# define C10_DEFINE_DEPRECATED_USING(TypeName, TypeThingy) using TypeName [[deprecated]] = TypeThingy;
#endif
#endif

#if defined(_MSC_VER)
#if defined(__CUDACC__)
// neither [[deprecated]] nor __declspec(deprecated) work on nvcc on Windows;
// you get the error:
//
//    error: attribute does not apply to any entity
//
// So we just turn the macro off in this case.
#if defined(C10_DEFINE_DEPRECATED_USING)
# undef C10_DEFINE_DEPRECATED_USING
#endif
# define C10_DEFINE_DEPRECATED_USING(TypeName, TypeThingy) using TypeName = TypeThingy;
#else
// [[deprecated]] does work in windows without nvcc, though msc doesn't support
// `__has_cpp_attribute` when c++14 is supported, otherwise __declspec(deprecated)
// is used as the alternative.
#ifndef C10_DEFINE_DEPRECATED_USING
#if defined(_MSVC_LANG) && _MSVC_LANG >= 201402L
# define C10_DEFINE_DEPRECATED_USING(TypeName, TypeThingy) using TypeName [[deprecated]] = TypeThingy;
#else
# define C10_DEFINE_DEPRECATED_USING(TypeName, TypeThingy) using TypeName = __declspec(deprecated) TypeThingy;
#endif
#endif
#endif
#endif

#if !defined(C10_DEFINE_DEPRECATED_USING) && defined(__GNUC__)
// nvcc has a bug where it doesn't understand __attribute__((deprecated))
// declarations even when the host compiler supports it. We'll only use this gcc
// attribute when not cuda, and when using a GCC compiler that doesn't support
// the c++14 syntax we checked for above (availble in __GNUC__ >= 5)
#if !defined(__CUDACC__)
# define C10_DEFINE_DEPRECATED_USING(TypeName, TypeThingy) using TypeName __attribute__((deprecated)) = TypeThingy;
#else
// using cuda + gcc < 5, neither deprecated syntax is available so turning off.
# define C10_DEFINE_DEPRECATED_USING(TypeName, TypeThingy) using TypeName = TypeThingy;
#endif
#endif

#if ! defined(C10_DEFINE_DEPRECATED_USING)
# warning "You need to implement C10_DEFINE_DEPRECATED_USING for this compiler"
# define C10_DEFINE_DEPRECATED_USING
#endif
