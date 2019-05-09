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
//

#if defined(__cplusplus) && __cplusplus >= 201402L
# define C10_DEPRECATED_USING [[deprecated]]
#elif defined(_MSC_VER) && defined(__CUDACC__)
// Apparently, [[deprecated]] doesn't work on nvcc on Windows;
// you get the error:
//
//    error: attribute does not apply to any entity
//
// So we just turn the macro off in this case.
# define C10_DEPRECATED_USING
#elif defined(_MSC_VER)
// __declspec(deprecated) does not work in using declarations:
//  https://godbolt.org/z/lOwe1h
// but it seems that most of C++14 is available in MSVC even if you don't ask for
// it. (It's also harmless to specify an attribute because it is C++11 supported
// syntax; you mostly risk it not being understood).  Some more notes at
// https://blogs.msdn.microsoft.com/vcblog/2016/06/07/standards-version-switches-in-the-compiler/
# define C10_DEPRECATED_USING [[deprecated]]
#elif defined(__CUDACC__)
// nvcc has a bug where it doesn't understand __attribute__((deprecated))
// declarations even when the host compiler supports it.  It's OK
// with [[deprecated]] though (although, if you are on an old version
// of gcc which doesn't understand attributes, you'll get a -Wattributes
// error that it is ignored
# define C10_DEPRECATED_USING [[deprecated]]
#elif defined(__GNUC__)
# define C10_DEPRECATED_USING __attribute__((deprecated))
#else
# warning "You need to implement C10_DEPRECATED_USING for this compiler"
# define C10_DEPRECATED_USING
#endif
