// Copyright 2005, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Author: wan@google.com (Zhanyong Wan)
//
// The Google C++ Testing Framework (Google Test)
//
// This header file defines the public API for Google Test.  It should be
// included by any test program that uses Google Test.
//
// IMPORTANT NOTE: Due to limitation of the C++ language, we have to
// leave some internal implementation details in this header file.
// They are clearly marked by comments like this:
//
//   // INTERNAL IMPLEMENTATION - DO NOT USE IN A USER PROGRAM.
//
// Such code is NOT meant to be used by a user directly, and is subject
// to CHANGE WITHOUT NOTICE.  Therefore DO NOT DEPEND ON IT in a user
// program!
//
// Acknowledgment: Google Test borrowed the idea of automatic test
// registration from Barthelemy Dagenais' (barthelemy@prologique.com)
// easyUnit framework.

#ifndef GTEST_INCLUDE_GTEST_GTEST_H_
#define GTEST_INCLUDE_GTEST_GTEST_H_

#include <limits>
#include <vector>

// Copyright 2005, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Authors: wan@google.com (Zhanyong Wan), eefacm@gmail.com (Sean Mcafee)
//
// The Google C++ Testing Framework (Google Test)
//
// This header file declares functions and macros used internally by
// Google Test.  They are subject to change without notice.

#ifndef GTEST_INCLUDE_GTEST_INTERNAL_GTEST_INTERNAL_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_GTEST_INTERNAL_H_

// Copyright 2005, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Authors: wan@google.com (Zhanyong Wan)
//
// Low-level types and utilities for porting Google Test to various
// platforms.  They are subject to change without notice.  DO NOT USE
// THEM IN USER CODE.

#ifndef GTEST_INCLUDE_GTEST_INTERNAL_GTEST_PORT_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_GTEST_PORT_H_

// The user can define the following macros in the build script to
// control Google Test's behavior.  If the user doesn't define a macro
// in this list, Google Test will define it.
//
//   GTEST_HAS_CLONE          - Define it to 1/0 to indicate that clone(2)
//                              is/isn't available.
//   GTEST_HAS_EXCEPTIONS     - Define it to 1/0 to indicate that exceptions
//                              are enabled.
//   GTEST_HAS_GLOBAL_STRING  - Define it to 1/0 to indicate that ::string
//                              is/isn't available (some systems define
//                              ::string, which is different to std::string).
//   GTEST_HAS_GLOBAL_WSTRING - Define it to 1/0 to indicate that ::string
//                              is/isn't available (some systems define
//                              ::wstring, which is different to std::wstring).
//   GTEST_HAS_POSIX_RE       - Define it to 1/0 to indicate that POSIX regular
//                              expressions are/aren't available.
//   GTEST_HAS_PTHREAD        - Define it to 1/0 to indicate that <pthread.h>
//                              is/isn't available.
//   GTEST_HAS_RTTI           - Define it to 1/0 to indicate that RTTI is/isn't
//                              enabled.
//   GTEST_HAS_STD_WSTRING    - Define it to 1/0 to indicate that
//                              std::wstring does/doesn't work (Google Test can
//                              be used where std::wstring is unavailable).
//   GTEST_HAS_TR1_TUPLE      - Define it to 1/0 to indicate tr1::tuple
//                              is/isn't available.
//   GTEST_HAS_SEH            - Define it to 1/0 to indicate whether the
//                              compiler supports Microsoft's "Structured
//                              Exception Handling".
//   GTEST_HAS_STREAM_REDIRECTION
//                            - Define it to 1/0 to indicate whether the
//                              platform supports I/O stream redirection using
//                              dup() and dup2().
//   GTEST_USE_OWN_TR1_TUPLE  - Define it to 1/0 to indicate whether Google
//                              Test's own tr1 tuple implementation should be
//                              used.  Unused when the user sets
//                              GTEST_HAS_TR1_TUPLE to 0.
//   GTEST_LINKED_AS_SHARED_LIBRARY
//                            - Define to 1 when compiling tests that use
//                              Google Test as a shared library (known as
//                              DLL on Windows).
//   GTEST_CREATE_SHARED_LIBRARY
//                            - Define to 1 when compiling Google Test itself
//                              as a shared library.

// This header defines the following utilities:
//
// Macros indicating the current platform (defined to 1 if compiled on
// the given platform; otherwise undefined):
//   GTEST_OS_AIX      - IBM AIX
//   GTEST_OS_CYGWIN   - Cygwin
//   GTEST_OS_HPUX     - HP-UX
//   GTEST_OS_LINUX    - Linux
//     GTEST_OS_LINUX_ANDROID - Google Android
//   GTEST_OS_MAC      - Mac OS X
//   GTEST_OS_NACL     - Google Native Client (NaCl)
//   GTEST_OS_SOLARIS  - Sun Solaris
//   GTEST_OS_SYMBIAN  - Symbian
//   GTEST_OS_WINDOWS  - Windows (Desktop, MinGW, or Mobile)
//     GTEST_OS_WINDOWS_DESKTOP  - Windows Desktop
//     GTEST_OS_WINDOWS_MINGW    - MinGW
//     GTEST_OS_WINDOWS_MOBILE   - Windows Mobile
//   GTEST_OS_ZOS      - z/OS
//
// Among the platforms, Cygwin, Linux, Max OS X, and Windows have the
// most stable support.  Since core members of the Google Test project
// don't have access to other platforms, support for them may be less
// stable.  If you notice any problems on your platform, please notify
// googletestframework@googlegroups.com (patches for fixing them are
// even more welcome!).
//
// Note that it is possible that none of the GTEST_OS_* macros are defined.
//
// Macros indicating available Google Test features (defined to 1 if
// the corresponding feature is supported; otherwise undefined):
//   GTEST_HAS_COMBINE      - the Combine() function (for value-parameterized
//                            tests)
//   GTEST_HAS_DEATH_TEST   - death tests
//   GTEST_HAS_PARAM_TEST   - value-parameterized tests
//   GTEST_HAS_TYPED_TEST   - typed tests
//   GTEST_HAS_TYPED_TEST_P - type-parameterized tests
//   GTEST_USES_POSIX_RE    - enhanced POSIX regex is used. Do not confuse with
//                            GTEST_HAS_POSIX_RE (see above) which users can
//                            define themselves.
//   GTEST_USES_SIMPLE_RE   - our own simple regex is used;
//                            the above two are mutually exclusive.
//   GTEST_CAN_COMPARE_NULL - accepts untyped NULL in EXPECT_EQ().
//
// Macros for basic C++ coding:
//   GTEST_AMBIGUOUS_ELSE_BLOCKER_ - for disabling a gcc warning.
//   GTEST_ATTRIBUTE_UNUSED_  - declares that a class' instances or a
//                              variable don't have to be used.
//   GTEST_DISALLOW_ASSIGN_   - disables operator=.
//   GTEST_DISALLOW_COPY_AND_ASSIGN_ - disables copy ctor and operator=.
//   GTEST_MUST_USE_RESULT_   - declares that a function's result must be used.
//
// Synchronization:
//   Mutex, MutexLock, ThreadLocal, GetThreadCount()
//                  - synchronization primitives.
//   GTEST_IS_THREADSAFE - defined to 1 to indicate that the above
//                         synchronization primitives have real implementations
//                         and Google Test is thread-safe; or 0 otherwise.
//
// Template meta programming:
//   is_pointer     - as in TR1; needed on Symbian and IBM XL C/C++ only.
//   IteratorTraits - partial implementation of std::iterator_traits, which
//                    is not available in libCstd when compiled with Sun C++.
//
// Smart pointers:
//   scoped_ptr     - as in TR2.
//
// Regular expressions:
//   RE             - a simple regular expression class using the POSIX
//                    Extended Regular Expression syntax on UNIX-like
//                    platforms, or a reduced regular exception syntax on
//                    other platforms, including Windows.
//
// Logging:
//   GTEST_LOG_()   - logs messages at the specified severity level.
//   LogToStderr()  - directs all log messages to stderr.
//   FlushInfoLog() - flushes informational log messages.
//
// Stdout and stderr capturing:
//   CaptureStdout()     - starts capturing stdout.
//   GetCapturedStdout() - stops capturing stdout and returns the captured
//                         string.
//   CaptureStderr()     - starts capturing stderr.
//   GetCapturedStderr() - stops capturing stderr and returns the captured
//                         string.
//
// Integer types:
//   TypeWithSize   - maps an integer to a int type.
//   Int32, UInt32, Int64, UInt64, TimeInMillis
//                  - integers of known sizes.
//   BiggestInt     - the biggest signed integer type.
//
// Command-line utilities:
//   GTEST_FLAG()       - references a flag.
//   GTEST_DECLARE_*()  - declares a flag.
//   GTEST_DEFINE_*()   - defines a flag.
//   GetArgvs()         - returns the command line as a vector of strings.
//
// Environment variable utilities:
//   GetEnv()             - gets the value of an environment variable.
//   BoolFromGTestEnv()   - parses a bool environment variable.
//   Int32FromGTestEnv()  - parses an Int32 environment variable.
//   StringFromGTestEnv() - parses a string environment variable.

#include <ctype.h>   // for isspace, etc
#include <stddef.h>  // for ptrdiff_t
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifndef _WIN32_WCE
# include <sys/types.h>
# include <sys/stat.h>
#endif  // !_WIN32_WCE

#include <iostream>  // NOLINT
#include <sstream>  // NOLINT
#include <string>  // NOLINT

#define GTEST_DEV_EMAIL_ "googletestframework@@googlegroups.com"
#define GTEST_FLAG_PREFIX_ "gtest_"
#define GTEST_FLAG_PREFIX_DASH_ "gtest-"
#define GTEST_FLAG_PREFIX_UPPER_ "GTEST_"
#define GTEST_NAME_ "Google Test"
#define GTEST_PROJECT_URL_ "http://code.google.com/p/googletest/"

// Determines the version of gcc that is used to compile this.
#ifdef __GNUC__
// 40302 means version 4.3.2.
# define GTEST_GCC_VER_ \
    (__GNUC__*10000 + __GNUC_MINOR__*100 + __GNUC_PATCHLEVEL__)
#endif  // __GNUC__

// Determines the platform on which Google Test is compiled.
#ifdef __CYGWIN__
# define GTEST_OS_CYGWIN 1
#elif defined __SYMBIAN32__
# define GTEST_OS_SYMBIAN 1
#elif defined _WIN32
# define GTEST_OS_WINDOWS 1
# ifdef _WIN32_WCE
#  define GTEST_OS_WINDOWS_MOBILE 1
# elif defined(__MINGW__) || defined(__MINGW32__)
#  define GTEST_OS_WINDOWS_MINGW 1
# else
#  define GTEST_OS_WINDOWS_DESKTOP 1
# endif  // _WIN32_WCE
#elif defined __APPLE__
# define GTEST_OS_MAC 1
#elif defined __linux__
# define GTEST_OS_LINUX 1
# ifdef ANDROID
#  define GTEST_OS_LINUX_ANDROID 1
# endif  // ANDROID
#elif defined __MVS__
# define GTEST_OS_ZOS 1
#elif defined(__sun) && defined(__SVR4)
# define GTEST_OS_SOLARIS 1
#elif defined(_AIX)
# define GTEST_OS_AIX 1
#elif defined(__hpux)
# define GTEST_OS_HPUX 1
#elif defined __native_client__
# define GTEST_OS_NACL 1
#endif  // __CYGWIN__

// Brings in definitions for functions used in the testing::internal::posix
// namespace (read, write, close, chdir, isatty, stat). We do not currently
// use them on Windows Mobile.
#if !GTEST_OS_WINDOWS
// This assumes that non-Windows OSes provide unistd.h. For OSes where this
// is not the case, we need to include headers that provide the functions
// mentioned above.
# include <unistd.h>
# if !GTEST_OS_NACL
// TODO(vladl@google.com): Remove this condition when Native Client SDK adds
// strings.h (tracked in
// http://code.google.com/p/nativeclient/issues/detail?id=1175).
#  include <strings.h>  // Native Client doesn't provide strings.h.
# endif
#elif !GTEST_OS_WINDOWS_MOBILE
# include <direct.h>
# include <io.h>
#endif

// Defines this to true iff Google Test can use POSIX regular expressions.
#ifndef GTEST_HAS_POSIX_RE
# define GTEST_HAS_POSIX_RE (!GTEST_OS_WINDOWS)
#endif

#if GTEST_HAS_POSIX_RE

// On some platforms, <regex.h> needs someone to define size_t, and
// won't compile otherwise.  We can #include it here as we already
// included <stdlib.h>, which is guaranteed to define size_t through
// <stddef.h>.
# include <regex.h>  // NOLINT

# define GTEST_USES_POSIX_RE 1

#elif GTEST_OS_WINDOWS

// <regex.h> is not available on Windows.  Use our own simple regex
// implementation instead.
# define GTEST_USES_SIMPLE_RE 1

#else

// <regex.h> may not be available on this platform.  Use our own
// simple regex implementation instead.
# define GTEST_USES_SIMPLE_RE 1

#endif  // GTEST_HAS_POSIX_RE

#ifndef GTEST_HAS_EXCEPTIONS
// The user didn't tell us whether exceptions are enabled, so we need
// to figure it out.
# if defined(_MSC_VER) || defined(__BORLANDC__)
// MSVC's and C++Builder's implementations of the STL use the _HAS_EXCEPTIONS
// macro to enable exceptions, so we'll do the same.
// Assumes that exceptions are enabled by default.
#  ifndef _HAS_EXCEPTIONS
#   define _HAS_EXCEPTIONS 1
#  endif  // _HAS_EXCEPTIONS
#  define GTEST_HAS_EXCEPTIONS _HAS_EXCEPTIONS
# elif defined(__GNUC__) && __EXCEPTIONS
// gcc defines __EXCEPTIONS to 1 iff exceptions are enabled.
#  define GTEST_HAS_EXCEPTIONS 1
# elif defined(__SUNPRO_CC)
// Sun Pro CC supports exceptions.  However, there is no compile-time way of
// detecting whether they are enabled or not.  Therefore, we assume that
// they are enabled unless the user tells us otherwise.
#  define GTEST_HAS_EXCEPTIONS 1
# elif defined(__IBMCPP__) && __EXCEPTIONS
// xlC defines __EXCEPTIONS to 1 iff exceptions are enabled.
#  define GTEST_HAS_EXCEPTIONS 1
# elif defined(__HP_aCC)
// Exception handling is in effect by default in HP aCC compiler. It has to
// be turned of by +noeh compiler option if desired.
#  define GTEST_HAS_EXCEPTIONS 1
# else
// For other compilers, we assume exceptions are disabled to be
// conservative.
#  define GTEST_HAS_EXCEPTIONS 0
# endif  // defined(_MSC_VER) || defined(__BORLANDC__)
#endif  // GTEST_HAS_EXCEPTIONS

#if !defined(GTEST_HAS_STD_STRING)
// Even though we don't use this macro any longer, we keep it in case
// some clients still depend on it.
# define GTEST_HAS_STD_STRING 1
#elif !GTEST_HAS_STD_STRING
// The user told us that ::std::string isn't available.
# error "Google Test cannot be used where ::std::string isn't available."
#endif  // !defined(GTEST_HAS_STD_STRING)

#ifndef GTEST_HAS_GLOBAL_STRING
// The user didn't tell us whether ::string is available, so we need
// to figure it out.

# define GTEST_HAS_GLOBAL_STRING 0

#endif  // GTEST_HAS_GLOBAL_STRING

#ifndef GTEST_HAS_STD_WSTRING
// The user didn't tell us whether ::std::wstring is available, so we need
// to figure it out.
// TODO(wan@google.com): uses autoconf to detect whether ::std::wstring
//   is available.

// Cygwin 1.7 and below doesn't support ::std::wstring.
// Solaris' libc++ doesn't support it either.  Android has
// no support for it at least as recent as Froyo (2.2).
# define GTEST_HAS_STD_WSTRING \
    (!(GTEST_OS_LINUX_ANDROID || GTEST_OS_CYGWIN || GTEST_OS_SOLARIS))

#endif  // GTEST_HAS_STD_WSTRING

#ifndef GTEST_HAS_GLOBAL_WSTRING
// The user didn't tell us whether ::wstring is available, so we need
// to figure it out.
# define GTEST_HAS_GLOBAL_WSTRING \
    (GTEST_HAS_STD_WSTRING && GTEST_HAS_GLOBAL_STRING)
#endif  // GTEST_HAS_GLOBAL_WSTRING

// Determines whether RTTI is available.
#ifndef GTEST_HAS_RTTI
// The user didn't tell us whether RTTI is enabled, so we need to
// figure it out.

# ifdef _MSC_VER

#  ifdef _CPPRTTI  // MSVC defines this macro iff RTTI is enabled.
#   define GTEST_HAS_RTTI 1
#  else
#   define GTEST_HAS_RTTI 0
#  endif

// Starting with version 4.3.2, gcc defines __GXX_RTTI iff RTTI is enabled.
# elif defined(__GNUC__) && (GTEST_GCC_VER_ >= 40302)

#  ifdef __GXX_RTTI
#   define GTEST_HAS_RTTI 1
#  else
#   define GTEST_HAS_RTTI 0
#  endif  // __GXX_RTTI

// Starting with version 9.0 IBM Visual Age defines __RTTI_ALL__ to 1 if
// both the typeid and dynamic_cast features are present.
# elif defined(__IBMCPP__) && (__IBMCPP__ >= 900)

#  ifdef __RTTI_ALL__
#   define GTEST_HAS_RTTI 1
#  else
#   define GTEST_HAS_RTTI 0
#  endif

# else

// For all other compilers, we assume RTTI is enabled.
#  define GTEST_HAS_RTTI 1

# endif  // _MSC_VER

#endif  // GTEST_HAS_RTTI

// It's this header's responsibility to #include <typeinfo> when RTTI
// is enabled.
#if GTEST_HAS_RTTI
# include <typeinfo>
#endif

// Determines whether Google Test can use the pthreads library.
#ifndef GTEST_HAS_PTHREAD
// The user didn't tell us explicitly, so we assume pthreads support is
// available on Linux and Mac.
//
// To disable threading support in Google Test, add -DGTEST_HAS_PTHREAD=0
// to your compiler flags.
# define GTEST_HAS_PTHREAD (GTEST_OS_LINUX || GTEST_OS_MAC || GTEST_OS_HPUX)
#endif  // GTEST_HAS_PTHREAD

#if GTEST_HAS_PTHREAD
// gtest-port.h guarantees to #include <pthread.h> when GTEST_HAS_PTHREAD is
// true.
# include <pthread.h>  // NOLINT

// For timespec and nanosleep, used below.
# include <time.h>  // NOLINT
#endif

// Determines whether Google Test can use tr1/tuple.  You can define
// this macro to 0 to prevent Google Test from using tuple (any
// feature depending on tuple with be disabled in this mode).
#ifndef GTEST_HAS_TR1_TUPLE
// The user didn't tell us not to do it, so we assume it's OK.
# define GTEST_HAS_TR1_TUPLE 1
#endif  // GTEST_HAS_TR1_TUPLE

// Determines whether Google Test's own tr1 tuple implementation
// should be used.
#ifndef GTEST_USE_OWN_TR1_TUPLE
// The user didn't tell us, so we need to figure it out.

// We use our own TR1 tuple if we aren't sure the user has an
// implementation of it already.  At this time, GCC 4.0.0+ and MSVC
// 2010 are the only mainstream compilers that come with a TR1 tuple
// implementation.  NVIDIA's CUDA NVCC compiler pretends to be GCC by
// defining __GNUC__ and friends, but cannot compile GCC's tuple
// implementation.  MSVC 2008 (9.0) provides TR1 tuple in a 323 MB
// Feature Pack download, which we cannot assume the user has.
# if (defined(__GNUC__) && !defined(__CUDACC__) && (GTEST_GCC_VER_ >= 40000)) \
    || _MSC_VER >= 1600
#  define GTEST_USE_OWN_TR1_TUPLE 0
# else
#  define GTEST_USE_OWN_TR1_TUPLE 1
# endif

#endif  // GTEST_USE_OWN_TR1_TUPLE

// To avoid conditional compilation everywhere, we make it
// gtest-port.h's responsibility to #include the header implementing
// tr1/tuple.
#if GTEST_HAS_TR1_TUPLE

# if GTEST_USE_OWN_TR1_TUPLE
// This file was GENERATED by a script.  DO NOT EDIT BY HAND!!!

// Copyright 2009 Google Inc.
// All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Author: wan@google.com (Zhanyong Wan)

// Implements a subset of TR1 tuple needed by Google Test and Google Mock.

#ifndef GTEST_INCLUDE_GTEST_INTERNAL_GTEST_TUPLE_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_GTEST_TUPLE_H_

#include <utility>  // For ::std::pair.

// The compiler used in Symbian has a bug that prevents us from declaring the
// tuple template as a friend (it complains that tuple is redefined).  This
// hack bypasses the bug by declaring the members that should otherwise be
// private as public.
// Sun Studio versions < 12 also have the above bug.
#if defined(__SYMBIAN32__) || (defined(__SUNPRO_CC) && __SUNPRO_CC < 0x590)
# define GTEST_DECLARE_TUPLE_AS_FRIEND_ public:
#else
# define GTEST_DECLARE_TUPLE_AS_FRIEND_ \
    template <GTEST_10_TYPENAMES_(U)> friend class tuple; \
   private:
#endif

// GTEST_n_TUPLE_(T) is the type of an n-tuple.
#define GTEST_0_TUPLE_(T) tuple<>
#define GTEST_1_TUPLE_(T) tuple<T##0, void, void, void, void, void, void, \
    void, void, void>
#define GTEST_2_TUPLE_(T) tuple<T##0, T##1, void, void, void, void, void, \
    void, void, void>
#define GTEST_3_TUPLE_(T) tuple<T##0, T##1, T##2, void, void, void, void, \
    void, void, void>
#define GTEST_4_TUPLE_(T) tuple<T##0, T##1, T##2, T##3, void, void, void, \
    void, void, void>
#define GTEST_5_TUPLE_(T) tuple<T##0, T##1, T##2, T##3, T##4, void, void, \
    void, void, void>
#define GTEST_6_TUPLE_(T) tuple<T##0, T##1, T##2, T##3, T##4, T##5, void, \
    void, void, void>
#define GTEST_7_TUPLE_(T) tuple<T##0, T##1, T##2, T##3, T##4, T##5, T##6, \
    void, void, void>
#define GTEST_8_TUPLE_(T) tuple<T##0, T##1, T##2, T##3, T##4, T##5, T##6, \
    T##7, void, void>
#define GTEST_9_TUPLE_(T) tuple<T##0, T##1, T##2, T##3, T##4, T##5, T##6, \
    T##7, T##8, void>
#define GTEST_10_TUPLE_(T) tuple<T##0, T##1, T##2, T##3, T##4, T##5, T##6, \
    T##7, T##8, T##9>

// GTEST_n_TYPENAMES_(T) declares a list of n typenames.
#define GTEST_0_TYPENAMES_(T)
#define GTEST_1_TYPENAMES_(T) typename T##0
#define GTEST_2_TYPENAMES_(T) typename T##0, typename T##1
#define GTEST_3_TYPENAMES_(T) typename T##0, typename T##1, typename T##2
#define GTEST_4_TYPENAMES_(T) typename T##0, typename T##1, typename T##2, \
    typename T##3
#define GTEST_5_TYPENAMES_(T) typename T##0, typename T##1, typename T##2, \
    typename T##3, typename T##4
#define GTEST_6_TYPENAMES_(T) typename T##0, typename T##1, typename T##2, \
    typename T##3, typename T##4, typename T##5
#define GTEST_7_TYPENAMES_(T) typename T##0, typename T##1, typename T##2, \
    typename T##3, typename T##4, typename T##5, typename T##6
#define GTEST_8_TYPENAMES_(T) typename T##0, typename T##1, typename T##2, \
    typename T##3, typename T##4, typename T##5, typename T##6, typename T##7
#define GTEST_9_TYPENAMES_(T) typename T##0, typename T##1, typename T##2, \
    typename T##3, typename T##4, typename T##5, typename T##6, \
    typename T##7, typename T##8
#define GTEST_10_TYPENAMES_(T) typename T##0, typename T##1, typename T##2, \
    typename T##3, typename T##4, typename T##5, typename T##6, \
    typename T##7, typename T##8, typename T##9

// In theory, defining stuff in the ::std namespace is undefined
// behavior.  We can do this as we are playing the role of a standard
// library vendor.
namespace std {
namespace tr1 {

template <typename T0 = void, typename T1 = void, typename T2 = void,
    typename T3 = void, typename T4 = void, typename T5 = void,
    typename T6 = void, typename T7 = void, typename T8 = void,
    typename T9 = void>
class tuple;

// Anything in namespace gtest_internal is Google Test's INTERNAL
// IMPLEMENTATION DETAIL and MUST NOT BE USED DIRECTLY in user code.
namespace gtest_internal {

// ByRef<T>::type is T if T is a reference; otherwise it's const T&.
template <typename T>
struct ByRef { typedef const T& type; };  // NOLINT
template <typename T>
struct ByRef<T&> { typedef T& type; };  // NOLINT

// A handy wrapper for ByRef.
#define GTEST_BY_REF_(T) typename ::std::tr1::gtest_internal::ByRef<T>::type

// AddRef<T>::type is T if T is a reference; otherwise it's T&.  This
// is the same as tr1::add_reference<T>::type.
template <typename T>
struct AddRef { typedef T& type; };  // NOLINT
template <typename T>
struct AddRef<T&> { typedef T& type; };  // NOLINT

// A handy wrapper for AddRef.
#define GTEST_ADD_REF_(T) typename ::std::tr1::gtest_internal::AddRef<T>::type

// A helper for implementing get<k>().
template <int k> class Get;

// A helper for implementing tuple_element<k, T>.  kIndexValid is true
// iff k < the number of fields in tuple type T.
template <bool kIndexValid, int kIndex, class Tuple>
struct TupleElement;

template <GTEST_10_TYPENAMES_(T)>
struct TupleElement<true, 0, GTEST_10_TUPLE_(T)> { typedef T0 type; };

template <GTEST_10_TYPENAMES_(T)>
struct TupleElement<true, 1, GTEST_10_TUPLE_(T)> { typedef T1 type; };

template <GTEST_10_TYPENAMES_(T)>
struct TupleElement<true, 2, GTEST_10_TUPLE_(T)> { typedef T2 type; };

template <GTEST_10_TYPENAMES_(T)>
struct TupleElement<true, 3, GTEST_10_TUPLE_(T)> { typedef T3 type; };

template <GTEST_10_TYPENAMES_(T)>
struct TupleElement<true, 4, GTEST_10_TUPLE_(T)> { typedef T4 type; };

template <GTEST_10_TYPENAMES_(T)>
struct TupleElement<true, 5, GTEST_10_TUPLE_(T)> { typedef T5 type; };

template <GTEST_10_TYPENAMES_(T)>
struct TupleElement<true, 6, GTEST_10_TUPLE_(T)> { typedef T6 type; };

template <GTEST_10_TYPENAMES_(T)>
struct TupleElement<true, 7, GTEST_10_TUPLE_(T)> { typedef T7 type; };

template <GTEST_10_TYPENAMES_(T)>
struct TupleElement<true, 8, GTEST_10_TUPLE_(T)> { typedef T8 type; };

template <GTEST_10_TYPENAMES_(T)>
struct TupleElement<true, 9, GTEST_10_TUPLE_(T)> { typedef T9 type; };

}  // namespace gtest_internal

template <>
class tuple<> {
 public:
  tuple() {}
  tuple(const tuple& /* t */)  {}
  tuple& operator=(const tuple& /* t */) { return *this; }
};

template <GTEST_1_TYPENAMES_(T)>
class GTEST_1_TUPLE_(T) {
 public:
  template <int k> friend class gtest_internal::Get;

  tuple() : f0_() {}

  explicit tuple(GTEST_BY_REF_(T0) f0) : f0_(f0) {}

  tuple(const tuple& t) : f0_(t.f0_) {}

  template <GTEST_1_TYPENAMES_(U)>
  tuple(const GTEST_1_TUPLE_(U)& t) : f0_(t.f0_) {}

  tuple& operator=(const tuple& t) { return CopyFrom(t); }

  template <GTEST_1_TYPENAMES_(U)>
  tuple& operator=(const GTEST_1_TUPLE_(U)& t) {
    return CopyFrom(t);
  }

  GTEST_DECLARE_TUPLE_AS_FRIEND_

  template <GTEST_1_TYPENAMES_(U)>
  tuple& CopyFrom(const GTEST_1_TUPLE_(U)& t) {
    f0_ = t.f0_;
    return *this;
  }

  T0 f0_;
};

template <GTEST_2_TYPENAMES_(T)>
class GTEST_2_TUPLE_(T) {
 public:
  template <int k> friend class gtest_internal::Get;

  tuple() : f0_(), f1_() {}

  explicit tuple(GTEST_BY_REF_(T0) f0, GTEST_BY_REF_(T1) f1) : f0_(f0),
      f1_(f1) {}

  tuple(const tuple& t) : f0_(t.f0_), f1_(t.f1_) {}

  template <GTEST_2_TYPENAMES_(U)>
  tuple(const GTEST_2_TUPLE_(U)& t) : f0_(t.f0_), f1_(t.f1_) {}
  template <typename U0, typename U1>
  tuple(const ::std::pair<U0, U1>& p) : f0_(p.first), f1_(p.second) {}

  tuple& operator=(const tuple& t) { return CopyFrom(t); }

  template <GTEST_2_TYPENAMES_(U)>
  tuple& operator=(const GTEST_2_TUPLE_(U)& t) {
    return CopyFrom(t);
  }
  template <typename U0, typename U1>
  tuple& operator=(const ::std::pair<U0, U1>& p) {
    f0_ = p.first;
    f1_ = p.second;
    return *this;
  }

  GTEST_DECLARE_TUPLE_AS_FRIEND_

  template <GTEST_2_TYPENAMES_(U)>
  tuple& CopyFrom(const GTEST_2_TUPLE_(U)& t) {
    f0_ = t.f0_;
    f1_ = t.f1_;
    return *this;
  }

  T0 f0_;
  T1 f1_;
};

template <GTEST_3_TYPENAMES_(T)>
class GTEST_3_TUPLE_(T) {
 public:
  template <int k> friend class gtest_internal::Get;

  tuple() : f0_(), f1_(), f2_() {}

  explicit tuple(GTEST_BY_REF_(T0) f0, GTEST_BY_REF_(T1) f1,
      GTEST_BY_REF_(T2) f2) : f0_(f0), f1_(f1), f2_(f2) {}

  tuple(const tuple& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_) {}

  template <GTEST_3_TYPENAMES_(U)>
  tuple(const GTEST_3_TUPLE_(U)& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_) {}

  tuple& operator=(const tuple& t) { return CopyFrom(t); }

  template <GTEST_3_TYPENAMES_(U)>
  tuple& operator=(const GTEST_3_TUPLE_(U)& t) {
    return CopyFrom(t);
  }

  GTEST_DECLARE_TUPLE_AS_FRIEND_

  template <GTEST_3_TYPENAMES_(U)>
  tuple& CopyFrom(const GTEST_3_TUPLE_(U)& t) {
    f0_ = t.f0_;
    f1_ = t.f1_;
    f2_ = t.f2_;
    return *this;
  }

  T0 f0_;
  T1 f1_;
  T2 f2_;
};

template <GTEST_4_TYPENAMES_(T)>
class GTEST_4_TUPLE_(T) {
 public:
  template <int k> friend class gtest_internal::Get;

  tuple() : f0_(), f1_(), f2_(), f3_() {}

  explicit tuple(GTEST_BY_REF_(T0) f0, GTEST_BY_REF_(T1) f1,
      GTEST_BY_REF_(T2) f2, GTEST_BY_REF_(T3) f3) : f0_(f0), f1_(f1), f2_(f2),
      f3_(f3) {}

  tuple(const tuple& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_), f3_(t.f3_) {}

  template <GTEST_4_TYPENAMES_(U)>
  tuple(const GTEST_4_TUPLE_(U)& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_),
      f3_(t.f3_) {}

  tuple& operator=(const tuple& t) { return CopyFrom(t); }

  template <GTEST_4_TYPENAMES_(U)>
  tuple& operator=(const GTEST_4_TUPLE_(U)& t) {
    return CopyFrom(t);
  }

  GTEST_DECLARE_TUPLE_AS_FRIEND_

  template <GTEST_4_TYPENAMES_(U)>
  tuple& CopyFrom(const GTEST_4_TUPLE_(U)& t) {
    f0_ = t.f0_;
    f1_ = t.f1_;
    f2_ = t.f2_;
    f3_ = t.f3_;
    return *this;
  }

  T0 f0_;
  T1 f1_;
  T2 f2_;
  T3 f3_;
};

template <GTEST_5_TYPENAMES_(T)>
class GTEST_5_TUPLE_(T) {
 public:
  template <int k> friend class gtest_internal::Get;

  tuple() : f0_(), f1_(), f2_(), f3_(), f4_() {}

  explicit tuple(GTEST_BY_REF_(T0) f0, GTEST_BY_REF_(T1) f1,
      GTEST_BY_REF_(T2) f2, GTEST_BY_REF_(T3) f3,
      GTEST_BY_REF_(T4) f4) : f0_(f0), f1_(f1), f2_(f2), f3_(f3), f4_(f4) {}

  tuple(const tuple& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_), f3_(t.f3_),
      f4_(t.f4_) {}

  template <GTEST_5_TYPENAMES_(U)>
  tuple(const GTEST_5_TUPLE_(U)& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_),
      f3_(t.f3_), f4_(t.f4_) {}

  tuple& operator=(const tuple& t) { return CopyFrom(t); }

  template <GTEST_5_TYPENAMES_(U)>
  tuple& operator=(const GTEST_5_TUPLE_(U)& t) {
    return CopyFrom(t);
  }

  GTEST_DECLARE_TUPLE_AS_FRIEND_

  template <GTEST_5_TYPENAMES_(U)>
  tuple& CopyFrom(const GTEST_5_TUPLE_(U)& t) {
    f0_ = t.f0_;
    f1_ = t.f1_;
    f2_ = t.f2_;
    f3_ = t.f3_;
    f4_ = t.f4_;
    return *this;
  }

  T0 f0_;
  T1 f1_;
  T2 f2_;
  T3 f3_;
  T4 f4_;
};

template <GTEST_6_TYPENAMES_(T)>
class GTEST_6_TUPLE_(T) {
 public:
  template <int k> friend class gtest_internal::Get;

  tuple() : f0_(), f1_(), f2_(), f3_(), f4_(), f5_() {}

  explicit tuple(GTEST_BY_REF_(T0) f0, GTEST_BY_REF_(T1) f1,
      GTEST_BY_REF_(T2) f2, GTEST_BY_REF_(T3) f3, GTEST_BY_REF_(T4) f4,
      GTEST_BY_REF_(T5) f5) : f0_(f0), f1_(f1), f2_(f2), f3_(f3), f4_(f4),
      f5_(f5) {}

  tuple(const tuple& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_), f3_(t.f3_),
      f4_(t.f4_), f5_(t.f5_) {}

  template <GTEST_6_TYPENAMES_(U)>
  tuple(const GTEST_6_TUPLE_(U)& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_),
      f3_(t.f3_), f4_(t.f4_), f5_(t.f5_) {}

  tuple& operator=(const tuple& t) { return CopyFrom(t); }

  template <GTEST_6_TYPENAMES_(U)>
  tuple& operator=(const GTEST_6_TUPLE_(U)& t) {
    return CopyFrom(t);
  }

  GTEST_DECLARE_TUPLE_AS_FRIEND_

  template <GTEST_6_TYPENAMES_(U)>
  tuple& CopyFrom(const GTEST_6_TUPLE_(U)& t) {
    f0_ = t.f0_;
    f1_ = t.f1_;
    f2_ = t.f2_;
    f3_ = t.f3_;
    f4_ = t.f4_;
    f5_ = t.f5_;
    return *this;
  }

  T0 f0_;
  T1 f1_;
  T2 f2_;
  T3 f3_;
  T4 f4_;
  T5 f5_;
};

template <GTEST_7_TYPENAMES_(T)>
class GTEST_7_TUPLE_(T) {
 public:
  template <int k> friend class gtest_internal::Get;

  tuple() : f0_(), f1_(), f2_(), f3_(), f4_(), f5_(), f6_() {}

  explicit tuple(GTEST_BY_REF_(T0) f0, GTEST_BY_REF_(T1) f1,
      GTEST_BY_REF_(T2) f2, GTEST_BY_REF_(T3) f3, GTEST_BY_REF_(T4) f4,
      GTEST_BY_REF_(T5) f5, GTEST_BY_REF_(T6) f6) : f0_(f0), f1_(f1), f2_(f2),
      f3_(f3), f4_(f4), f5_(f5), f6_(f6) {}

  tuple(const tuple& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_), f3_(t.f3_),
      f4_(t.f4_), f5_(t.f5_), f6_(t.f6_) {}

  template <GTEST_7_TYPENAMES_(U)>
  tuple(const GTEST_7_TUPLE_(U)& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_),
      f3_(t.f3_), f4_(t.f4_), f5_(t.f5_), f6_(t.f6_) {}

  tuple& operator=(const tuple& t) { return CopyFrom(t); }

  template <GTEST_7_TYPENAMES_(U)>
  tuple& operator=(const GTEST_7_TUPLE_(U)& t) {
    return CopyFrom(t);
  }

  GTEST_DECLARE_TUPLE_AS_FRIEND_

  template <GTEST_7_TYPENAMES_(U)>
  tuple& CopyFrom(const GTEST_7_TUPLE_(U)& t) {
    f0_ = t.f0_;
    f1_ = t.f1_;
    f2_ = t.f2_;
    f3_ = t.f3_;
    f4_ = t.f4_;
    f5_ = t.f5_;
    f6_ = t.f6_;
    return *this;
  }

  T0 f0_;
  T1 f1_;
  T2 f2_;
  T3 f3_;
  T4 f4_;
  T5 f5_;
  T6 f6_;
};

template <GTEST_8_TYPENAMES_(T)>
class GTEST_8_TUPLE_(T) {
 public:
  template <int k> friend class gtest_internal::Get;

  tuple() : f0_(), f1_(), f2_(), f3_(), f4_(), f5_(), f6_(), f7_() {}

  explicit tuple(GTEST_BY_REF_(T0) f0, GTEST_BY_REF_(T1) f1,
      GTEST_BY_REF_(T2) f2, GTEST_BY_REF_(T3) f3, GTEST_BY_REF_(T4) f4,
      GTEST_BY_REF_(T5) f5, GTEST_BY_REF_(T6) f6,
      GTEST_BY_REF_(T7) f7) : f0_(f0), f1_(f1), f2_(f2), f3_(f3), f4_(f4),
      f5_(f5), f6_(f6), f7_(f7) {}

  tuple(const tuple& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_), f3_(t.f3_),
      f4_(t.f4_), f5_(t.f5_), f6_(t.f6_), f7_(t.f7_) {}

  template <GTEST_8_TYPENAMES_(U)>
  tuple(const GTEST_8_TUPLE_(U)& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_),
      f3_(t.f3_), f4_(t.f4_), f5_(t.f5_), f6_(t.f6_), f7_(t.f7_) {}

  tuple& operator=(const tuple& t) { return CopyFrom(t); }

  template <GTEST_8_TYPENAMES_(U)>
  tuple& operator=(const GTEST_8_TUPLE_(U)& t) {
    return CopyFrom(t);
  }

  GTEST_DECLARE_TUPLE_AS_FRIEND_

  template <GTEST_8_TYPENAMES_(U)>
  tuple& CopyFrom(const GTEST_8_TUPLE_(U)& t) {
    f0_ = t.f0_;
    f1_ = t.f1_;
    f2_ = t.f2_;
    f3_ = t.f3_;
    f4_ = t.f4_;
    f5_ = t.f5_;
    f6_ = t.f6_;
    f7_ = t.f7_;
    return *this;
  }

  T0 f0_;
  T1 f1_;
  T2 f2_;
  T3 f3_;
  T4 f4_;
  T5 f5_;
  T6 f6_;
  T7 f7_;
};

template <GTEST_9_TYPENAMES_(T)>
class GTEST_9_TUPLE_(T) {
 public:
  template <int k> friend class gtest_internal::Get;

  tuple() : f0_(), f1_(), f2_(), f3_(), f4_(), f5_(), f6_(), f7_(), f8_() {}

  explicit tuple(GTEST_BY_REF_(T0) f0, GTEST_BY_REF_(T1) f1,
      GTEST_BY_REF_(T2) f2, GTEST_BY_REF_(T3) f3, GTEST_BY_REF_(T4) f4,
      GTEST_BY_REF_(T5) f5, GTEST_BY_REF_(T6) f6, GTEST_BY_REF_(T7) f7,
      GTEST_BY_REF_(T8) f8) : f0_(f0), f1_(f1), f2_(f2), f3_(f3), f4_(f4),
      f5_(f5), f6_(f6), f7_(f7), f8_(f8) {}

  tuple(const tuple& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_), f3_(t.f3_),
      f4_(t.f4_), f5_(t.f5_), f6_(t.f6_), f7_(t.f7_), f8_(t.f8_) {}

  template <GTEST_9_TYPENAMES_(U)>
  tuple(const GTEST_9_TUPLE_(U)& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_),
      f3_(t.f3_), f4_(t.f4_), f5_(t.f5_), f6_(t.f6_), f7_(t.f7_), f8_(t.f8_) {}

  tuple& operator=(const tuple& t) { return CopyFrom(t); }

  template <GTEST_9_TYPENAMES_(U)>
  tuple& operator=(const GTEST_9_TUPLE_(U)& t) {
    return CopyFrom(t);
  }

  GTEST_DECLARE_TUPLE_AS_FRIEND_

  template <GTEST_9_TYPENAMES_(U)>
  tuple& CopyFrom(const GTEST_9_TUPLE_(U)& t) {
    f0_ = t.f0_;
    f1_ = t.f1_;
    f2_ = t.f2_;
    f3_ = t.f3_;
    f4_ = t.f4_;
    f5_ = t.f5_;
    f6_ = t.f6_;
    f7_ = t.f7_;
    f8_ = t.f8_;
    return *this;
  }

  T0 f0_;
  T1 f1_;
  T2 f2_;
  T3 f3_;
  T4 f4_;
  T5 f5_;
  T6 f6_;
  T7 f7_;
  T8 f8_;
};

template <GTEST_10_TYPENAMES_(T)>
class tuple {
 public:
  template <int k> friend class gtest_internal::Get;

  tuple() : f0_(), f1_(), f2_(), f3_(), f4_(), f5_(), f6_(), f7_(), f8_(),
      f9_() {}

  explicit tuple(GTEST_BY_REF_(T0) f0, GTEST_BY_REF_(T1) f1,
      GTEST_BY_REF_(T2) f2, GTEST_BY_REF_(T3) f3, GTEST_BY_REF_(T4) f4,
      GTEST_BY_REF_(T5) f5, GTEST_BY_REF_(T6) f6, GTEST_BY_REF_(T7) f7,
      GTEST_BY_REF_(T8) f8, GTEST_BY_REF_(T9) f9) : f0_(f0), f1_(f1), f2_(f2),
      f3_(f3), f4_(f4), f5_(f5), f6_(f6), f7_(f7), f8_(f8), f9_(f9) {}

  tuple(const tuple& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_), f3_(t.f3_),
      f4_(t.f4_), f5_(t.f5_), f6_(t.f6_), f7_(t.f7_), f8_(t.f8_), f9_(t.f9_) {}

  template <GTEST_10_TYPENAMES_(U)>
  tuple(const GTEST_10_TUPLE_(U)& t) : f0_(t.f0_), f1_(t.f1_), f2_(t.f2_),
      f3_(t.f3_), f4_(t.f4_), f5_(t.f5_), f6_(t.f6_), f7_(t.f7_), f8_(t.f8_),
      f9_(t.f9_) {}

  tuple& operator=(const tuple& t) { return CopyFrom(t); }

  template <GTEST_10_TYPENAMES_(U)>
  tuple& operator=(const GTEST_10_TUPLE_(U)& t) {
    return CopyFrom(t);
  }

  GTEST_DECLARE_TUPLE_AS_FRIEND_

  template <GTEST_10_TYPENAMES_(U)>
  tuple& CopyFrom(const GTEST_10_TUPLE_(U)& t) {
    f0_ = t.f0_;
    f1_ = t.f1_;
    f2_ = t.f2_;
    f3_ = t.f3_;
    f4_ = t.f4_;
    f5_ = t.f5_;
    f6_ = t.f6_;
    f7_ = t.f7_;
    f8_ = t.f8_;
    f9_ = t.f9_;
    return *this;
  }

  T0 f0_;
  T1 f1_;
  T2 f2_;
  T3 f3_;
  T4 f4_;
  T5 f5_;
  T6 f6_;
  T7 f7_;
  T8 f8_;
  T9 f9_;
};

// 6.1.3.2 Tuple creation functions.

// Known limitations: we don't support passing an
// std::tr1::reference_wrapper<T> to make_tuple().  And we don't
// implement tie().

inline tuple<> make_tuple() { return tuple<>(); }

template <GTEST_1_TYPENAMES_(T)>
inline GTEST_1_TUPLE_(T) make_tuple(const T0& f0) {
  return GTEST_1_TUPLE_(T)(f0);
}

template <GTEST_2_TYPENAMES_(T)>
inline GTEST_2_TUPLE_(T) make_tuple(const T0& f0, const T1& f1) {
  return GTEST_2_TUPLE_(T)(f0, f1);
}

template <GTEST_3_TYPENAMES_(T)>
inline GTEST_3_TUPLE_(T) make_tuple(const T0& f0, const T1& f1, const T2& f2) {
  return GTEST_3_TUPLE_(T)(f0, f1, f2);
}

template <GTEST_4_TYPENAMES_(T)>
inline GTEST_4_TUPLE_(T) make_tuple(const T0& f0, const T1& f1, const T2& f2,
    const T3& f3) {
  return GTEST_4_TUPLE_(T)(f0, f1, f2, f3);
}

template <GTEST_5_TYPENAMES_(T)>
inline GTEST_5_TUPLE_(T) make_tuple(const T0& f0, const T1& f1, const T2& f2,
    const T3& f3, const T4& f4) {
  return GTEST_5_TUPLE_(T)(f0, f1, f2, f3, f4);
}

template <GTEST_6_TYPENAMES_(T)>
inline GTEST_6_TUPLE_(T) make_tuple(const T0& f0, const T1& f1, const T2& f2,
    const T3& f3, const T4& f4, const T5& f5) {
  return GTEST_6_TUPLE_(T)(f0, f1, f2, f3, f4, f5);
}

template <GTEST_7_TYPENAMES_(T)>
inline GTEST_7_TUPLE_(T) make_tuple(const T0& f0, const T1& f1, const T2& f2,
    const T3& f3, const T4& f4, const T5& f5, const T6& f6) {
  return GTEST_7_TUPLE_(T)(f0, f1, f2, f3, f4, f5, f6);
}

template <GTEST_8_TYPENAMES_(T)>
inline GTEST_8_TUPLE_(T) make_tuple(const T0& f0, const T1& f1, const T2& f2,
    const T3& f3, const T4& f4, const T5& f5, const T6& f6, const T7& f7) {
  return GTEST_8_TUPLE_(T)(f0, f1, f2, f3, f4, f5, f6, f7);
}

template <GTEST_9_TYPENAMES_(T)>
inline GTEST_9_TUPLE_(T) make_tuple(const T0& f0, const T1& f1, const T2& f2,
    const T3& f3, const T4& f4, const T5& f5, const T6& f6, const T7& f7,
    const T8& f8) {
  return GTEST_9_TUPLE_(T)(f0, f1, f2, f3, f4, f5, f6, f7, f8);
}

template <GTEST_10_TYPENAMES_(T)>
inline GTEST_10_TUPLE_(T) make_tuple(const T0& f0, const T1& f1, const T2& f2,
    const T3& f3, const T4& f4, const T5& f5, const T6& f6, const T7& f7,
    const T8& f8, const T9& f9) {
  return GTEST_10_TUPLE_(T)(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9);
}

// 6.1.3.3 Tuple helper classes.

template <typename Tuple> struct tuple_size;

template <GTEST_0_TYPENAMES_(T)>
struct tuple_size<GTEST_0_TUPLE_(T)> { static const int value = 0; };

template <GTEST_1_TYPENAMES_(T)>
struct tuple_size<GTEST_1_TUPLE_(T)> { static const int value = 1; };

template <GTEST_2_TYPENAMES_(T)>
struct tuple_size<GTEST_2_TUPLE_(T)> { static const int value = 2; };

template <GTEST_3_TYPENAMES_(T)>
struct tuple_size<GTEST_3_TUPLE_(T)> { static const int value = 3; };

template <GTEST_4_TYPENAMES_(T)>
struct tuple_size<GTEST_4_TUPLE_(T)> { static const int value = 4; };

template <GTEST_5_TYPENAMES_(T)>
struct tuple_size<GTEST_5_TUPLE_(T)> { static const int value = 5; };

template <GTEST_6_TYPENAMES_(T)>
struct tuple_size<GTEST_6_TUPLE_(T)> { static const int value = 6; };

template <GTEST_7_TYPENAMES_(T)>
struct tuple_size<GTEST_7_TUPLE_(T)> { static const int value = 7; };

template <GTEST_8_TYPENAMES_(T)>
struct tuple_size<GTEST_8_TUPLE_(T)> { static const int value = 8; };

template <GTEST_9_TYPENAMES_(T)>
struct tuple_size<GTEST_9_TUPLE_(T)> { static const int value = 9; };

template <GTEST_10_TYPENAMES_(T)>
struct tuple_size<GTEST_10_TUPLE_(T)> { static const int value = 10; };

template <int k, class Tuple>
struct tuple_element {
  typedef typename gtest_internal::TupleElement<
      k < (tuple_size<Tuple>::value), k, Tuple>::type type;
};

#define GTEST_TUPLE_ELEMENT_(k, Tuple) typename tuple_element<k, Tuple >::type

// 6.1.3.4 Element access.

namespace gtest_internal {

template <>
class Get<0> {
 public:
  template <class Tuple>
  static GTEST_ADD_REF_(GTEST_TUPLE_ELEMENT_(0, Tuple))
  Field(Tuple& t) { return t.f0_; }  // NOLINT

  template <class Tuple>
  static GTEST_BY_REF_(GTEST_TUPLE_ELEMENT_(0, Tuple))
  ConstField(const Tuple& t) { return t.f0_; }
};

template <>
class Get<1> {
 public:
  template <class Tuple>
  static GTEST_ADD_REF_(GTEST_TUPLE_ELEMENT_(1, Tuple))
  Field(Tuple& t) { return t.f1_; }  // NOLINT

  template <class Tuple>
  static GTEST_BY_REF_(GTEST_TUPLE_ELEMENT_(1, Tuple))
  ConstField(const Tuple& t) { return t.f1_; }
};

template <>
class Get<2> {
 public:
  template <class Tuple>
  static GTEST_ADD_REF_(GTEST_TUPLE_ELEMENT_(2, Tuple))
  Field(Tuple& t) { return t.f2_; }  // NOLINT

  template <class Tuple>
  static GTEST_BY_REF_(GTEST_TUPLE_ELEMENT_(2, Tuple))
  ConstField(const Tuple& t) { return t.f2_; }
};

template <>
class Get<3> {
 public:
  template <class Tuple>
  static GTEST_ADD_REF_(GTEST_TUPLE_ELEMENT_(3, Tuple))
  Field(Tuple& t) { return t.f3_; }  // NOLINT

  template <class Tuple>
  static GTEST_BY_REF_(GTEST_TUPLE_ELEMENT_(3, Tuple))
  ConstField(const Tuple& t) { return t.f3_; }
};

template <>
class Get<4> {
 public:
  template <class Tuple>
  static GTEST_ADD_REF_(GTEST_TUPLE_ELEMENT_(4, Tuple))
  Field(Tuple& t) { return t.f4_; }  // NOLINT

  template <class Tuple>
  static GTEST_BY_REF_(GTEST_TUPLE_ELEMENT_(4, Tuple))
  ConstField(const Tuple& t) { return t.f4_; }
};

template <>
class Get<5> {
 public:
  template <class Tuple>
  static GTEST_ADD_REF_(GTEST_TUPLE_ELEMENT_(5, Tuple))
  Field(Tuple& t) { return t.f5_; }  // NOLINT

  template <class Tuple>
  static GTEST_BY_REF_(GTEST_TUPLE_ELEMENT_(5, Tuple))
  ConstField(const Tuple& t) { return t.f5_; }
};

template <>
class Get<6> {
 public:
  template <class Tuple>
  static GTEST_ADD_REF_(GTEST_TUPLE_ELEMENT_(6, Tuple))
  Field(Tuple& t) { return t.f6_; }  // NOLINT

  template <class Tuple>
  static GTEST_BY_REF_(GTEST_TUPLE_ELEMENT_(6, Tuple))
  ConstField(const Tuple& t) { return t.f6_; }
};

template <>
class Get<7> {
 public:
  template <class Tuple>
  static GTEST_ADD_REF_(GTEST_TUPLE_ELEMENT_(7, Tuple))
  Field(Tuple& t) { return t.f7_; }  // NOLINT

  template <class Tuple>
  static GTEST_BY_REF_(GTEST_TUPLE_ELEMENT_(7, Tuple))
  ConstField(const Tuple& t) { return t.f7_; }
};

template <>
class Get<8> {
 public:
  template <class Tuple>
  static GTEST_ADD_REF_(GTEST_TUPLE_ELEMENT_(8, Tuple))
  Field(Tuple& t) { return t.f8_; }  // NOLINT

  template <class Tuple>
  static GTEST_BY_REF_(GTEST_TUPLE_ELEMENT_(8, Tuple))
  ConstField(const Tuple& t) { return t.f8_; }
};

template <>
class Get<9> {
 public:
  template <class Tuple>
  static GTEST_ADD_REF_(GTEST_TUPLE_ELEMENT_(9, Tuple))
  Field(Tuple& t) { return t.f9_; }  // NOLINT

  template <class Tuple>
  static GTEST_BY_REF_(GTEST_TUPLE_ELEMENT_(9, Tuple))
  ConstField(const Tuple& t) { return t.f9_; }
};

}  // namespace gtest_internal

template <int k, GTEST_10_TYPENAMES_(T)>
GTEST_ADD_REF_(GTEST_TUPLE_ELEMENT_(k, GTEST_10_TUPLE_(T)))
get(GTEST_10_TUPLE_(T)& t) {
  return gtest_internal::Get<k>::Field(t);
}

template <int k, GTEST_10_TYPENAMES_(T)>
GTEST_BY_REF_(GTEST_TUPLE_ELEMENT_(k,  GTEST_10_TUPLE_(T)))
get(const GTEST_10_TUPLE_(T)& t) {
  return gtest_internal::Get<k>::ConstField(t);
}

// 6.1.3.5 Relational operators

// We only implement == and !=, as we don't have a need for the rest yet.

namespace gtest_internal {

// SameSizeTuplePrefixComparator<k, k>::Eq(t1, t2) returns true if the
// first k fields of t1 equals the first k fields of t2.
// SameSizeTuplePrefixComparator(k1, k2) would be a compiler error if
// k1 != k2.
template <int kSize1, int kSize2>
struct SameSizeTuplePrefixComparator;

template <>
struct SameSizeTuplePrefixComparator<0, 0> {
  template <class Tuple1, class Tuple2>
  static bool Eq(const Tuple1& /* t1 */, const Tuple2& /* t2 */) {
    return true;
  }
};

template <int k>
struct SameSizeTuplePrefixComparator<k, k> {
  template <class Tuple1, class Tuple2>
  static bool Eq(const Tuple1& t1, const Tuple2& t2) {
    return SameSizeTuplePrefixComparator<k - 1, k - 1>::Eq(t1, t2) &&
        ::std::tr1::get<k - 1>(t1) == ::std::tr1::get<k - 1>(t2);
  }
};

}  // namespace gtest_internal

template <GTEST_10_TYPENAMES_(T), GTEST_10_TYPENAMES_(U)>
inline bool operator==(const GTEST_10_TUPLE_(T)& t,
                       const GTEST_10_TUPLE_(U)& u) {
  return gtest_internal::SameSizeTuplePrefixComparator<
      tuple_size<GTEST_10_TUPLE_(T)>::value,
      tuple_size<GTEST_10_TUPLE_(U)>::value>::Eq(t, u);
}

template <GTEST_10_TYPENAMES_(T), GTEST_10_TYPENAMES_(U)>
inline bool operator!=(const GTEST_10_TUPLE_(T)& t,
                       const GTEST_10_TUPLE_(U)& u) { return !(t == u); }

// 6.1.4 Pairs.
// Unimplemented.

}  // namespace tr1
}  // namespace std

#undef GTEST_0_TUPLE_
#undef GTEST_1_TUPLE_
#undef GTEST_2_TUPLE_
#undef GTEST_3_TUPLE_
#undef GTEST_4_TUPLE_
#undef GTEST_5_TUPLE_
#undef GTEST_6_TUPLE_
#undef GTEST_7_TUPLE_
#undef GTEST_8_TUPLE_
#undef GTEST_9_TUPLE_
#undef GTEST_10_TUPLE_

#undef GTEST_0_TYPENAMES_
#undef GTEST_1_TYPENAMES_
#undef GTEST_2_TYPENAMES_
#undef GTEST_3_TYPENAMES_
#undef GTEST_4_TYPENAMES_
#undef GTEST_5_TYPENAMES_
#undef GTEST_6_TYPENAMES_
#undef GTEST_7_TYPENAMES_
#undef GTEST_8_TYPENAMES_
#undef GTEST_9_TYPENAMES_
#undef GTEST_10_TYPENAMES_

#undef GTEST_DECLARE_TUPLE_AS_FRIEND_
#undef GTEST_BY_REF_
#undef GTEST_ADD_REF_
#undef GTEST_TUPLE_ELEMENT_

#endif  // GTEST_INCLUDE_GTEST_INTERNAL_GTEST_TUPLE_H_
# elif GTEST_OS_SYMBIAN

// On Symbian, BOOST_HAS_TR1_TUPLE causes Boost's TR1 tuple library to
// use STLport's tuple implementation, which unfortunately doesn't
// work as the copy of STLport distributed with Symbian is incomplete.
// By making sure BOOST_HAS_TR1_TUPLE is undefined, we force Boost to
// use its own tuple implementation.
#  ifdef BOOST_HAS_TR1_TUPLE
#   undef BOOST_HAS_TR1_TUPLE
#  endif  // BOOST_HAS_TR1_TUPLE

// This prevents <boost/tr1/detail/config.hpp>, which defines
// BOOST_HAS_TR1_TUPLE, from being #included by Boost's <tuple>.
#  define BOOST_TR1_DETAIL_CONFIG_HPP_INCLUDED
#  include <tuple>

# elif defined(__GNUC__) && (GTEST_GCC_VER_ >= 40000)
// GCC 4.0+ implements tr1/tuple in the <tr1/tuple> header.  This does
// not conform to the TR1 spec, which requires the header to be <tuple>.

#  if !GTEST_HAS_RTTI && GTEST_GCC_VER_ < 40302
// Until version 4.3.2, gcc has a bug that causes <tr1/functional>,
// which is #included by <tr1/tuple>, to not compile when RTTI is
// disabled.  _TR1_FUNCTIONAL is the header guard for
// <tr1/functional>.  Hence the following #define is a hack to prevent
// <tr1/functional> from being included.
#   define _TR1_FUNCTIONAL 1
#   include <tr1/tuple>
#   undef _TR1_FUNCTIONAL  // Allows the user to #include
                        // <tr1/functional> if he chooses to.
#  else
#   include <tr1/tuple>  // NOLINT
#  endif  // !GTEST_HAS_RTTI && GTEST_GCC_VER_ < 40302

# else
// If the compiler is not GCC 4.0+, we assume the user is using a
// spec-conforming TR1 implementation.
#  include <tuple>  // NOLINT
# endif  // GTEST_USE_OWN_TR1_TUPLE

#endif  // GTEST_HAS_TR1_TUPLE

// Determines whether clone(2) is supported.
// Usually it will only be available on Linux, excluding
// Linux on the Itanium architecture.
// Also see http://linux.die.net/man/2/clone.
#ifndef GTEST_HAS_CLONE
// The user didn't tell us, so we need to figure it out.

# if GTEST_OS_LINUX && !defined(__ia64__)
#  define GTEST_HAS_CLONE 1
# else
#  define GTEST_HAS_CLONE 0
# endif  // GTEST_OS_LINUX && !defined(__ia64__)

#endif  // GTEST_HAS_CLONE

// Determines whether to support stream redirection. This is used to test
// output correctness and to implement death tests.
#ifndef GTEST_HAS_STREAM_REDIRECTION
// By default, we assume that stream redirection is supported on all
// platforms except known mobile ones.
# if GTEST_OS_WINDOWS_MOBILE || GTEST_OS_SYMBIAN
#  define GTEST_HAS_STREAM_REDIRECTION 0
# else
#  define GTEST_HAS_STREAM_REDIRECTION 1
# endif  // !GTEST_OS_WINDOWS_MOBILE && !GTEST_OS_SYMBIAN
#endif  // GTEST_HAS_STREAM_REDIRECTION

// Determines whether to support death tests.
// Google Test does not support death tests for VC 7.1 and earlier as
// abort() in a VC 7.1 application compiled as GUI in debug config
// pops up a dialog window that cannot be suppressed programmatically.
#if (GTEST_OS_LINUX || GTEST_OS_MAC || GTEST_OS_CYGWIN || GTEST_OS_SOLARIS || \
     (GTEST_OS_WINDOWS_DESKTOP && _MSC_VER >= 1400) || \
     GTEST_OS_WINDOWS_MINGW || GTEST_OS_AIX || GTEST_OS_HPUX)
# define GTEST_HAS_DEATH_TEST 1
# include <vector>  // NOLINT
#endif

// We don't support MSVC 7.1 with exceptions disabled now.  Therefore
// all the compilers we care about are adequate for supporting
// value-parameterized tests.
#define GTEST_HAS_PARAM_TEST 1

// Determines whether to support type-driven tests.

// Typed tests need <typeinfo> and variadic macros, which GCC, VC++ 8.0,
// Sun Pro CC, IBM Visual Age, and HP aCC support.
#if defined(__GNUC__) || (_MSC_VER >= 1400) || defined(__SUNPRO_CC) || \
    defined(__IBMCPP__) || defined(__HP_aCC)
# define GTEST_HAS_TYPED_TEST 1
# define GTEST_HAS_TYPED_TEST_P 1
#endif

// Determines whether to support Combine(). This only makes sense when
// value-parameterized tests are enabled.  The implementation doesn't
// work on Sun Studio since it doesn't understand templated conversion
// operators.
#if GTEST_HAS_PARAM_TEST && GTEST_HAS_TR1_TUPLE && !defined(__SUNPRO_CC)
# define GTEST_HAS_COMBINE 1
#endif

// Determines whether the system compiler uses UTF-16 for encoding wide strings.
#define GTEST_WIDE_STRING_USES_UTF16_ \
    (GTEST_OS_WINDOWS || GTEST_OS_CYGWIN || GTEST_OS_SYMBIAN || GTEST_OS_AIX)

// Determines whether test results can be streamed to a socket.
#if GTEST_OS_LINUX
# define GTEST_CAN_STREAM_RESULTS_ 1
#endif

// Defines some utility macros.

// The GNU compiler emits a warning if nested "if" statements are followed by
// an "else" statement and braces are not used to explicitly disambiguate the
// "else" binding.  This leads to problems with code like:
//
//   if (gate)
//     ASSERT_*(condition) << "Some message";
//
// The "switch (0) case 0:" idiom is used to suppress this.
#ifdef __INTEL_COMPILER
# define GTEST_AMBIGUOUS_ELSE_BLOCKER_
#else
# define GTEST_AMBIGUOUS_ELSE_BLOCKER_ switch (0) case 0: default:  // NOLINT
#endif

// Use this annotation at the end of a struct/class definition to
// prevent the compiler from optimizing away instances that are never
// used.  This is useful when all interesting logic happens inside the
// c'tor and / or d'tor.  Example:
//
//   struct Foo {
//     Foo() { ... }
//   } GTEST_ATTRIBUTE_UNUSED_;
//
// Also use it after a variable or parameter declaration to tell the
// compiler the variable/parameter does not have to be used.
#if defined(__GNUC__) && !defined(COMPILER_ICC)
# define GTEST_ATTRIBUTE_UNUSED_ __attribute__ ((unused))
#else
# define GTEST_ATTRIBUTE_UNUSED_
#endif

// A macro to disallow operator=
// This should be used in the private: declarations for a class.
#define GTEST_DISALLOW_ASSIGN_(type)\
  void operator=(type const &)

// A macro to disallow copy constructor and operator=
// This should be used in the private: declarations for a class.
#define GTEST_DISALLOW_COPY_AND_ASSIGN_(type)\
  type(type const &);\
  GTEST_DISALLOW_ASSIGN_(type)

// Tell the compiler to warn about unused return values for functions declared
// with this macro.  The macro should be used on function declarations
// following the argument list:
//
//   Sprocket* AllocateSprocket() GTEST_MUST_USE_RESULT_;
#if defined(__GNUC__) && (GTEST_GCC_VER_ >= 30400) && !defined(COMPILER_ICC)
# define GTEST_MUST_USE_RESULT_ __attribute__ ((warn_unused_result))
#else
# define GTEST_MUST_USE_RESULT_
#endif  // __GNUC__ && (GTEST_GCC_VER_ >= 30400) && !COMPILER_ICC

// Determine whether the compiler supports Microsoft's Structured Exception
// Handling.  This is supported by several Windows compilers but generally
// does not exist on any other system.
#ifndef GTEST_HAS_SEH
// The user didn't tell us, so we need to figure it out.

# if defined(_MSC_VER) || defined(__BORLANDC__)
// These two compilers are known to support SEH.
#  define GTEST_HAS_SEH 1
# else
// Assume no SEH.
#  define GTEST_HAS_SEH 0
# endif

#endif  // GTEST_HAS_SEH

#ifdef _MSC_VER

# if GTEST_LINKED_AS_SHARED_LIBRARY
#  define GTEST_API_ __declspec(dllimport)
# elif GTEST_CREATE_SHARED_LIBRARY
#  define GTEST_API_ __declspec(dllexport)
# endif

#endif  // _MSC_VER

#ifndef GTEST_API_
# define GTEST_API_
#endif

#ifdef __GNUC__
// Ask the compiler to never inline a given function.
# define GTEST_NO_INLINE_ __attribute__((noinline))
#else
# define GTEST_NO_INLINE_
#endif

namespace testing {

class Message;

namespace internal {

class String;

// The GTEST_COMPILE_ASSERT_ macro can be used to verify that a compile time
// expression is true. For example, you could use it to verify the
// size of a static array:
//
//   GTEST_COMPILE_ASSERT_(ARRAYSIZE(content_type_names) == CONTENT_NUM_TYPES,
//                         content_type_names_incorrect_size);
//
// or to make sure a struct is smaller than a certain size:
//
//   GTEST_COMPILE_ASSERT_(sizeof(foo) < 128, foo_too_large);
//
// The second argument to the macro is the name of the variable. If
// the expression is false, most compilers will issue a warning/error
// containing the name of the variable.

template <bool>
struct CompileAssert {
};

#define GTEST_COMPILE_ASSERT_(expr, msg) \
  typedef ::testing::internal::CompileAssert<(bool(expr))> \
      msg[bool(expr) ? 1 : -1]

// Implementation details of GTEST_COMPILE_ASSERT_:
//
// - GTEST_COMPILE_ASSERT_ works by defining an array type that has -1
//   elements (and thus is invalid) when the expression is false.
//
// - The simpler definition
//
//    #define GTEST_COMPILE_ASSERT_(expr, msg) typedef char msg[(expr) ? 1 : -1]
//
//   does not work, as gcc supports variable-length arrays whose sizes
//   are determined at run-time (this is gcc's extension and not part
//   of the C++ standard).  As a result, gcc fails to reject the
//   following code with the simple definition:
//
//     int foo;
//     GTEST_COMPILE_ASSERT_(foo, msg); // not supposed to compile as foo is
//                                      // not a compile-time constant.
//
// - By using the type CompileAssert<(bool(expr))>, we ensures that
//   expr is a compile-time constant.  (Template arguments must be
//   determined at compile-time.)
//
// - The outter parentheses in CompileAssert<(bool(expr))> are necessary
//   to work around a bug in gcc 3.4.4 and 4.0.1.  If we had written
//
//     CompileAssert<bool(expr)>
//
//   instead, these compilers will refuse to compile
//
//     GTEST_COMPILE_ASSERT_(5 > 0, some_message);
//
//   (They seem to think the ">" in "5 > 0" marks the end of the
//   template argument list.)
//
// - The array size is (bool(expr) ? 1 : -1), instead of simply
//
//     ((expr) ? 1 : -1).
//
//   This is to avoid running into a bug in MS VC 7.1, which
//   causes ((0.0) ? 1 : -1) to incorrectly evaluate to 1.

// StaticAssertTypeEqHelper is used by StaticAssertTypeEq defined in gtest.h.
//
// This template is declared, but intentionally undefined.
template <typename T1, typename T2>
struct StaticAssertTypeEqHelper;

template <typename T>
struct StaticAssertTypeEqHelper<T, T> {};

#if GTEST_HAS_GLOBAL_STRING
typedef ::string string;
#else
typedef ::std::string string;
#endif  // GTEST_HAS_GLOBAL_STRING

#if GTEST_HAS_GLOBAL_WSTRING
typedef ::wstring wstring;
#elif GTEST_HAS_STD_WSTRING
typedef ::std::wstring wstring;
#endif  // GTEST_HAS_GLOBAL_WSTRING

// A helper for suppressing warnings on constant condition.  It just
// returns 'condition'.
GTEST_API_ bool IsTrue(bool condition);

// Defines scoped_ptr.

// This implementation of scoped_ptr is PARTIAL - it only contains
// enough stuff to satisfy Google Test's need.
template <typename T>
class scoped_ptr {
 public:
  typedef T element_type;

  explicit scoped_ptr(T* p = NULL) : ptr_(p) {}
  ~scoped_ptr() { reset(); }

  T& operator*() const { return *ptr_; }
  T* operator->() const { return ptr_; }
  T* get() const { return ptr_; }

  T* release() {
    T* const ptr = ptr_;
    ptr_ = NULL;
    return ptr;
  }

  void reset(T* p = NULL) {
    if (p != ptr_) {
      if (IsTrue(sizeof(T) > 0)) {  // Makes sure T is a complete type.
        delete ptr_;
      }
      ptr_ = p;
    }
  }
 private:
  T* ptr_;

  GTEST_DISALLOW_COPY_AND_ASSIGN_(scoped_ptr);
};

// Defines RE.

// A simple C++ wrapper for <regex.h>.  It uses the POSIX Extended
// Regular Expression syntax.
class GTEST_API_ RE {
 public:
  // A copy constructor is required by the Standard to initialize object
  // references from r-values.
  RE(const RE& other) { Init(other.pattern()); }

  // Constructs an RE from a string.
  RE(const ::std::string& regex) { Init(regex.c_str()); }  // NOLINT

#if GTEST_HAS_GLOBAL_STRING

  RE(const ::string& regex) { Init(regex.c_str()); }  // NOLINT

#endif  // GTEST_HAS_GLOBAL_STRING

  RE(const char* regex) { Init(regex); }  // NOLINT
  ~RE();

  // Returns the string representation of the regex.
  const char* pattern() const { return pattern_; }

  // FullMatch(str, re) returns true iff regular expression re matches
  // the entire str.
  // PartialMatch(str, re) returns true iff regular expression re
  // matches a substring of str (including str itself).
  //
  // TODO(wan@google.com): make FullMatch() and PartialMatch() work
  // when str contains NUL characters.
  static bool FullMatch(const ::std::string& str, const RE& re) {
    return FullMatch(str.c_str(), re);
  }
  static bool PartialMatch(const ::std::string& str, const RE& re) {
    return PartialMatch(str.c_str(), re);
  }

#if GTEST_HAS_GLOBAL_STRING

  static bool FullMatch(const ::string& str, const RE& re) {
    return FullMatch(str.c_str(), re);
  }
  static bool PartialMatch(const ::string& str, const RE& re) {
    return PartialMatch(str.c_str(), re);
  }

#endif  // GTEST_HAS_GLOBAL_STRING

  static bool FullMatch(const char* str, const RE& re);
  static bool PartialMatch(const char* str, const RE& re);

 private:
  void Init(const char* regex);

  // We use a const char* instead of a string, as Google Test may be used
  // where string is not available.  We also do not use Google Test's own
  // String type here, in order to simplify dependencies between the
  // files.
  const char* pattern_;
  bool is_valid_;

#if GTEST_USES_POSIX_RE

  regex_t full_regex_;     // For FullMatch().
  regex_t partial_regex_;  // For PartialMatch().

#else  // GTEST_USES_SIMPLE_RE

  const char* full_pattern_;  // For FullMatch();

#endif

  GTEST_DISALLOW_ASSIGN_(RE);
};

// Formats a source file path and a line number as they would appear
// in an error message from the compiler used to compile this code.
GTEST_API_ ::std::string FormatFileLocation(const char* file, int line);

// Formats a file location for compiler-independent XML output.
// Although this function is not platform dependent, we put it next to
// FormatFileLocation in order to contrast the two functions.
GTEST_API_ ::std::string FormatCompilerIndependentFileLocation(const char* file,
                                                               int line);

// Defines logging utilities:
//   GTEST_LOG_(severity) - logs messages at the specified severity level. The
//                          message itself is streamed into the macro.
//   LogToStderr()  - directs all log messages to stderr.
//   FlushInfoLog() - flushes informational log messages.

enum GTestLogSeverity {
  GTEST_INFO,
  GTEST_WARNING,
  GTEST_ERROR,
  GTEST_FATAL
};

// Formats log entry severity, provides a stream object for streaming the
// log message, and terminates the message with a newline when going out of
// scope.
class GTEST_API_ GTestLog {
 public:
  GTestLog(GTestLogSeverity severity, const char* file, int line);

  // Flushes the buffers and, if severity is GTEST_FATAL, aborts the program.
  ~GTestLog();

  ::std::ostream& GetStream() { return ::std::cerr; }

 private:
  const GTestLogSeverity severity_;

  GTEST_DISALLOW_COPY_AND_ASSIGN_(GTestLog);
};

#define GTEST_LOG_(severity) \
    ::testing::internal::GTestLog(::testing::internal::GTEST_##severity, \
                                  __FILE__, __LINE__).GetStream()

inline void LogToStderr() {}
inline void FlushInfoLog() { fflush(NULL); }

// INTERNAL IMPLEMENTATION - DO NOT USE.
//
// GTEST_CHECK_ is an all-mode assert. It aborts the program if the condition
// is not satisfied.
//  Synopsys:
//    GTEST_CHECK_(boolean_condition);
//     or
//    GTEST_CHECK_(boolean_condition) << "Additional message";
//
//    This checks the condition and if the condition is not satisfied
//    it prints message about the condition violation, including the
//    condition itself, plus additional message streamed into it, if any,
//    and then it aborts the program. It aborts the program irrespective of
//    whether it is built in the debug mode or not.
#define GTEST_CHECK_(condition) \
    GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
    if (::testing::internal::IsTrue(condition)) \
      ; \
    else \
      GTEST_LOG_(FATAL) << "Condition " #condition " failed. "

// An all-mode assert to verify that the given POSIX-style function
// call returns 0 (indicating success).  Known limitation: this
// doesn't expand to a balanced 'if' statement, so enclose the macro
// in {} if you need to use it as the only statement in an 'if'
// branch.
#define GTEST_CHECK_POSIX_SUCCESS_(posix_call) \
  if (const int gtest_error = (posix_call)) \
    GTEST_LOG_(FATAL) << #posix_call << "failed with error " \
                      << gtest_error

// INTERNAL IMPLEMENTATION - DO NOT USE IN USER CODE.
//
// Use ImplicitCast_ as a safe version of static_cast for upcasting in
// the type hierarchy (e.g. casting a Foo* to a SuperclassOfFoo* or a
// const Foo*).  When you use ImplicitCast_, the compiler checks that
// the cast is safe.  Such explicit ImplicitCast_s are necessary in
// surprisingly many situations where C++ demands an exact type match
// instead of an argument type convertable to a target type.
//
// The syntax for using ImplicitCast_ is the same as for static_cast:
//
//   ImplicitCast_<ToType>(expr)
//
// ImplicitCast_ would have been part of the C++ standard library,
// but the proposal was submitted too late.  It will probably make
// its way into the language in the future.
//
// This relatively ugly name is intentional. It prevents clashes with
// similar functions users may have (e.g., implicit_cast). The internal
// namespace alone is not enough because the function can be found by ADL.
template<typename To>
inline To ImplicitCast_(To x) { return x; }

// When you upcast (that is, cast a pointer from type Foo to type
// SuperclassOfFoo), it's fine to use ImplicitCast_<>, since upcasts
// always succeed.  When you downcast (that is, cast a pointer from
// type Foo to type SubclassOfFoo), static_cast<> isn't safe, because
// how do you know the pointer is really of type SubclassOfFoo?  It
// could be a bare Foo, or of type DifferentSubclassOfFoo.  Thus,
// when you downcast, you should use this macro.  In debug mode, we
// use dynamic_cast<> to double-check the downcast is legal (we die
// if it's not).  In normal mode, we do the efficient static_cast<>
// instead.  Thus, it's important to test in debug mode to make sure
// the cast is legal!
//    This is the only place in the code we should use dynamic_cast<>.
// In particular, you SHOULDN'T be using dynamic_cast<> in order to
// do RTTI (eg code like this:
//    if (dynamic_cast<Subclass1>(foo)) HandleASubclass1Object(foo);
//    if (dynamic_cast<Subclass2>(foo)) HandleASubclass2Object(foo);
// You should design the code some other way not to need this.
//
// This relatively ugly name is intentional. It prevents clashes with
// similar functions users may have (e.g., down_cast). The internal
// namespace alone is not enough because the function can be found by ADL.
template<typename To, typename From>  // use like this: DownCast_<T*>(foo);
inline To DownCast_(From* f) {  // so we only accept pointers
  // Ensures that To is a sub-type of From *.  This test is here only
  // for compile-time type checking, and has no overhead in an
  // optimized build at run-time, as it will be optimized away
  // completely.
  if (false) {
    const To to = NULL;
    ::testing::internal::ImplicitCast_<From*>(to);
  }

#if GTEST_HAS_RTTI
  // RTTI: debug mode only!
  GTEST_CHECK_(f == NULL || dynamic_cast<To>(f) != NULL);
#endif
  return static_cast<To>(f);
}

// Downcasts the pointer of type Base to Derived.
// Derived must be a subclass of Base. The parameter MUST
// point to a class of type Derived, not any subclass of it.
// When RTTI is available, the function performs a runtime
// check to enforce this.
template <class Derived, class Base>
Derived* CheckedDowncastToActualType(Base* base) {
#if GTEST_HAS_RTTI
  GTEST_CHECK_(typeid(*base) == typeid(Derived));
  return dynamic_cast<Derived*>(base);  // NOLINT
#else
  return static_cast<Derived*>(base);  // Poor man's downcast.
#endif
}

#if GTEST_HAS_STREAM_REDIRECTION

// Defines the stderr capturer:
//   CaptureStdout     - starts capturing stdout.
//   GetCapturedStdout - stops capturing stdout and returns the captured string.
//   CaptureStderr     - starts capturing stderr.
//   GetCapturedStderr - stops capturing stderr and returns the captured string.
//
GTEST_API_ void CaptureStdout();
GTEST_API_ String GetCapturedStdout();
GTEST_API_ void CaptureStderr();
GTEST_API_ String GetCapturedStderr();

#endif  // GTEST_HAS_STREAM_REDIRECTION


#if GTEST_HAS_DEATH_TEST

// A copy of all command line arguments.  Set by InitGoogleTest().
extern ::std::vector<String> g_argvs;

// GTEST_HAS_DEATH_TEST implies we have ::std::string.
const ::std::vector<String>& GetArgvs();

#endif  // GTEST_HAS_DEATH_TEST

// Defines synchronization primitives.

#if GTEST_HAS_PTHREAD

// Sleeps for (roughly) n milli-seconds.  This function is only for
// testing Google Test's own constructs.  Don't use it in user tests,
// either directly or indirectly.
inline void SleepMilliseconds(int n) {
  const timespec time = {
    0,                  // 0 seconds.
    n * 1000L * 1000L,  // And n ms.
  };
  nanosleep(&time, NULL);
}

// Allows a controller thread to pause execution of newly created
// threads until notified.  Instances of this class must be created
// and destroyed in the controller thread.
//
// This class is only for testing Google Test's own constructs. Do not
// use it in user tests, either directly or indirectly.
class Notification {
 public:
  Notification() : notified_(false) {}

  // Notifies all threads created with this notification to start. Must
  // be called from the controller thread.
  void Notify() { notified_ = true; }

  // Blocks until the controller thread notifies. Must be called from a test
  // thread.
  void WaitForNotification() {
    while(!notified_) {
      SleepMilliseconds(10);
    }
  }

 private:
  volatile bool notified_;

  GTEST_DISALLOW_COPY_AND_ASSIGN_(Notification);
};

// As a C-function, ThreadFuncWithCLinkage cannot be templated itself.
// Consequently, it cannot select a correct instantiation of ThreadWithParam
// in order to call its Run(). Introducing ThreadWithParamBase as a
// non-templated base class for ThreadWithParam allows us to bypass this
// problem.
class ThreadWithParamBase {
 public:
  virtual ~ThreadWithParamBase() {}
  virtual void Run() = 0;
};

// pthread_create() accepts a pointer to a function type with the C linkage.
// According to the Standard (7.5/1), function types with different linkages
// are different even if they are otherwise identical.  Some compilers (for
// example, SunStudio) treat them as different types.  Since class methods
// cannot be defined with C-linkage we need to define a free C-function to
// pass into pthread_create().
extern "C" inline void* ThreadFuncWithCLinkage(void* thread) {
  static_cast<ThreadWithParamBase*>(thread)->Run();
  return NULL;
}

// Helper class for testing Google Test's multi-threading constructs.
// To use it, write:
//
//   void ThreadFunc(int param) { /* Do things with param */ }
//   Notification thread_can_start;
//   ...
//   // The thread_can_start parameter is optional; you can supply NULL.
//   ThreadWithParam<int> thread(&ThreadFunc, 5, &thread_can_start);
//   thread_can_start.Notify();
//
// These classes are only for testing Google Test's own constructs. Do
// not use them in user tests, either directly or indirectly.
template <typename T>
class ThreadWithParam : public ThreadWithParamBase {
 public:
  typedef void (*UserThreadFunc)(T);

  ThreadWithParam(
      UserThreadFunc func, T param, Notification* thread_can_start)
      : func_(func),
        param_(param),
        thread_can_start_(thread_can_start),
        finished_(false) {
    ThreadWithParamBase* const base = this;
    // The thread can be created only after all fields except thread_
    // have been initialized.
    GTEST_CHECK_POSIX_SUCCESS_(
        pthread_create(&thread_, 0, &ThreadFuncWithCLinkage, base));
  }
  ~ThreadWithParam() { Join(); }

  void Join() {
    if (!finished_) {
      GTEST_CHECK_POSIX_SUCCESS_(pthread_join(thread_, 0));
      finished_ = true;
    }
  }

  virtual void Run() {
    if (thread_can_start_ != NULL)
      thread_can_start_->WaitForNotification();
    func_(param_);
  }

 private:
  const UserThreadFunc func_;  // User-supplied thread function.
  const T param_;  // User-supplied parameter to the thread function.
  // When non-NULL, used to block execution until the controller thread
  // notifies.
  Notification* const thread_can_start_;
  bool finished_;  // true iff we know that the thread function has finished.
  pthread_t thread_;  // The native thread object.

  GTEST_DISALLOW_COPY_AND_ASSIGN_(ThreadWithParam);
};

// MutexBase and Mutex implement mutex on pthreads-based platforms. They
// are used in conjunction with class MutexLock:
//
//   Mutex mutex;
//   ...
//   MutexLock lock(&mutex);  // Acquires the mutex and releases it at the end
//                            // of the current scope.
//
// MutexBase implements behavior for both statically and dynamically
// allocated mutexes.  Do not use MutexBase directly.  Instead, write
// the following to define a static mutex:
//
//   GTEST_DEFINE_STATIC_MUTEX_(g_some_mutex);
//
// You can forward declare a static mutex like this:
//
//   GTEST_DECLARE_STATIC_MUTEX_(g_some_mutex);
//
// To create a dynamic mutex, just define an object of type Mutex.
class MutexBase {
 public:
  // Acquires this mutex.
  void Lock() {
    GTEST_CHECK_POSIX_SUCCESS_(pthread_mutex_lock(&mutex_));
    owner_ = pthread_self();
  }

  // Releases this mutex.
  void Unlock() {
    // We don't protect writing to owner_ here, as it's the caller's
    // responsibility to ensure that the current thread holds the
    // mutex when this is called.
    owner_ = 0;
    GTEST_CHECK_POSIX_SUCCESS_(pthread_mutex_unlock(&mutex_));
  }

  // Does nothing if the current thread holds the mutex. Otherwise, crashes
  // with high probability.
  void AssertHeld() const {
    GTEST_CHECK_(owner_ == pthread_self())
        << "The current thread is not holding the mutex @" << this;
  }

  // A static mutex may be used before main() is entered.  It may even
  // be used before the dynamic initialization stage.  Therefore we
  // must be able to initialize a static mutex object at link time.
  // This means MutexBase has to be a POD and its member variables
  // have to be public.
 public:
  pthread_mutex_t mutex_;  // The underlying pthread mutex.
  pthread_t owner_;  // The thread holding the mutex; 0 means no one holds it.
};

// Forward-declares a static mutex.
# define GTEST_DECLARE_STATIC_MUTEX_(mutex) \
    extern ::testing::internal::MutexBase mutex

// Defines and statically (i.e. at link time) initializes a static mutex.
# define GTEST_DEFINE_STATIC_MUTEX_(mutex) \
    ::testing::internal::MutexBase mutex = { PTHREAD_MUTEX_INITIALIZER, 0 }

// The Mutex class can only be used for mutexes created at runtime. It
// shares its API with MutexBase otherwise.
class Mutex : public MutexBase {
 public:
  Mutex() {
    GTEST_CHECK_POSIX_SUCCESS_(pthread_mutex_init(&mutex_, NULL));
    owner_ = 0;
  }
  ~Mutex() {
    GTEST_CHECK_POSIX_SUCCESS_(pthread_mutex_destroy(&mutex_));
  }

 private:
  GTEST_DISALLOW_COPY_AND_ASSIGN_(Mutex);
};

// We cannot name this class MutexLock as the ctor declaration would
// conflict with a macro named MutexLock, which is defined on some
// platforms.  Hence the typedef trick below.
class GTestMutexLock {
 public:
  explicit GTestMutexLock(MutexBase* mutex)
      : mutex_(mutex) { mutex_->Lock(); }

  ~GTestMutexLock() { mutex_->Unlock(); }

 private:
  MutexBase* const mutex_;

  GTEST_DISALLOW_COPY_AND_ASSIGN_(GTestMutexLock);
};

typedef GTestMutexLock MutexLock;

// Helpers for ThreadLocal.

// pthread_key_create() requires DeleteThreadLocalValue() to have
// C-linkage.  Therefore it cannot be templatized to access
// ThreadLocal<T>.  Hence the need for class
// ThreadLocalValueHolderBase.
class ThreadLocalValueHolderBase {
 public:
  virtual ~ThreadLocalValueHolderBase() {}
};

// Called by pthread to delete thread-local data stored by
// pthread_setspecific().
extern "C" inline void DeleteThreadLocalValue(void* value_holder) {
  delete static_cast<ThreadLocalValueHolderBase*>(value_holder);
}

// Implements thread-local storage on pthreads-based systems.
//
//   // Thread 1
//   ThreadLocal<int> tl(100);  // 100 is the default value for each thread.
//
//   // Thread 2
//   tl.set(150);  // Changes the value for thread 2 only.
//   EXPECT_EQ(150, tl.get());
//
//   // Thread 1
//   EXPECT_EQ(100, tl.get());  // In thread 1, tl has the original value.
//   tl.set(200);
//   EXPECT_EQ(200, tl.get());
//
// The template type argument T must have a public copy constructor.
// In addition, the default ThreadLocal constructor requires T to have
// a public default constructor.
//
// An object managed for a thread by a ThreadLocal instance is deleted
// when the thread exits.  Or, if the ThreadLocal instance dies in
// that thread, when the ThreadLocal dies.  It's the user's
// responsibility to ensure that all other threads using a ThreadLocal
// have exited when it dies, or the per-thread objects for those
// threads will not be deleted.
//
// Google Test only uses global ThreadLocal objects.  That means they
// will die after main() has returned.  Therefore, no per-thread
// object managed by Google Test will be leaked as long as all threads
// using Google Test have exited when main() returns.
template <typename T>
class ThreadLocal {
 public:
  ThreadLocal() : key_(CreateKey()),
                  default_() {}
  explicit ThreadLocal(const T& value) : key_(CreateKey()),
                                         default_(value) {}

  ~ThreadLocal() {
    // Destroys the managed object for the current thread, if any.
    DeleteThreadLocalValue(pthread_getspecific(key_));

    // Releases resources associated with the key.  This will *not*
    // delete managed objects for other threads.
    GTEST_CHECK_POSIX_SUCCESS_(pthread_key_delete(key_));
  }

  T* pointer() { return GetOrCreateValue(); }
  const T* pointer() const { return GetOrCreateValue(); }
  const T& get() const { return *pointer(); }
  void set(const T& value) { *pointer() = value; }

 private:
  // Holds a value of type T.
  class ValueHolder : public ThreadLocalValueHolderBase {
   public:
    explicit ValueHolder(const T& value) : value_(value) {}

    T* pointer() { return &value_; }

   private:
    T value_;
    GTEST_DISALLOW_COPY_AND_ASSIGN_(ValueHolder);
  };

  static pthread_key_t CreateKey() {
    pthread_key_t key;
    // When a thread exits, DeleteThreadLocalValue() will be called on
    // the object managed for that thread.
    GTEST_CHECK_POSIX_SUCCESS_(
        pthread_key_create(&key, &DeleteThreadLocalValue));
    return key;
  }

  T* GetOrCreateValue() const {
    ThreadLocalValueHolderBase* const holder =
        static_cast<ThreadLocalValueHolderBase*>(pthread_getspecific(key_));
    if (holder != NULL) {
      return CheckedDowncastToActualType<ValueHolder>(holder)->pointer();
    }

    ValueHolder* const new_holder = new ValueHolder(default_);
    ThreadLocalValueHolderBase* const holder_base = new_holder;
    GTEST_CHECK_POSIX_SUCCESS_(pthread_setspecific(key_, holder_base));
    return new_holder->pointer();
  }

  // A key pthreads uses for looking up per-thread values.
  const pthread_key_t key_;
  const T default_;  // The default value for each thread.

  GTEST_DISALLOW_COPY_AND_ASSIGN_(ThreadLocal);
};

# define GTEST_IS_THREADSAFE 1

#else  // GTEST_HAS_PTHREAD

// A dummy implementation of synchronization primitives (mutex, lock,
// and thread-local variable).  Necessary for compiling Google Test where
// mutex is not supported - using Google Test in multiple threads is not
// supported on such platforms.

class Mutex {
 public:
  Mutex() {}
  void AssertHeld() const {}
};

# define GTEST_DECLARE_STATIC_MUTEX_(mutex) \
  extern ::testing::internal::Mutex mutex

# define GTEST_DEFINE_STATIC_MUTEX_(mutex) ::testing::internal::Mutex mutex

class GTestMutexLock {
 public:
  explicit GTestMutexLock(Mutex*) {}  // NOLINT
};

typedef GTestMutexLock MutexLock;

template <typename T>
class ThreadLocal {
 public:
  ThreadLocal() : value_() {}
  explicit ThreadLocal(const T& value) : value_(value) {}
  T* pointer() { return &value_; }
  const T* pointer() const { return &value_; }
  const T& get() const { return value_; }
  void set(const T& value) { value_ = value; }
 private:
  T value_;
};

// The above synchronization primitives have dummy implementations.
// Therefore Google Test is not thread-safe.
# define GTEST_IS_THREADSAFE 0

#endif  // GTEST_HAS_PTHREAD

// Returns the number of threads running in the process, or 0 to indicate that
// we cannot detect it.
GTEST_API_ size_t GetThreadCount();

// Passing non-POD classes through ellipsis (...) crashes the ARM
// compiler and generates a warning in Sun Studio.  The Nokia Symbian
// and the IBM XL C/C++ compiler try to instantiate a copy constructor
// for objects passed through ellipsis (...), failing for uncopyable
// objects.  We define this to ensure that only POD is passed through
// ellipsis on these systems.
#if defined(__SYMBIAN32__) || defined(__IBMCPP__) || defined(__SUNPRO_CC)
// We lose support for NULL detection where the compiler doesn't like
// passing non-POD classes through ellipsis (...).
# define GTEST_ELLIPSIS_NEEDS_POD_ 1
#else
# define GTEST_CAN_COMPARE_NULL 1
#endif

// The Nokia Symbian and IBM XL C/C++ compilers cannot decide between
// const T& and const T* in a function template.  These compilers
// _can_ decide between class template specializations for T and T*,
// so a tr1::type_traits-like is_pointer works.
#if defined(__SYMBIAN32__) || defined(__IBMCPP__)
# define GTEST_NEEDS_IS_POINTER_ 1
#endif

template <bool bool_value>
struct bool_constant {
  typedef bool_constant<bool_value> type;
  static const bool value = bool_value;
};
template <bool bool_value> const bool bool_constant<bool_value>::value;

typedef bool_constant<false> false_type;
typedef bool_constant<true> true_type;

template <typename T>
struct is_pointer : public false_type {};

template <typename T>
struct is_pointer<T*> : public true_type {};

template <typename Iterator>
struct IteratorTraits {
  typedef typename Iterator::value_type value_type;
};

template <typename T>
struct IteratorTraits<T*> {
  typedef T value_type;
};

template <typename T>
struct IteratorTraits<const T*> {
  typedef T value_type;
};

#if GTEST_OS_WINDOWS
# define GTEST_PATH_SEP_ "\\"
# define GTEST_HAS_ALT_PATH_SEP_ 1
// The biggest signed integer type the compiler supports.
typedef __int64 BiggestInt;
#else
# define GTEST_PATH_SEP_ "/"
# define GTEST_HAS_ALT_PATH_SEP_ 0
typedef long long BiggestInt;  // NOLINT
#endif  // GTEST_OS_WINDOWS

// Utilities for char.

// isspace(int ch) and friends accept an unsigned char or EOF.  char
// may be signed, depending on the compiler (or compiler flags).
// Therefore we need to cast a char to unsigned char before calling
// isspace(), etc.

inline bool IsAlpha(char ch) {
  return isalpha(static_cast<unsigned char>(ch)) != 0;
}
inline bool IsAlNum(char ch) {
  return isalnum(static_cast<unsigned char>(ch)) != 0;
}
inline bool IsDigit(char ch) {
  return isdigit(static_cast<unsigned char>(ch)) != 0;
}
inline bool IsLower(char ch) {
  return islower(static_cast<unsigned char>(ch)) != 0;
}
inline bool IsSpace(char ch) {
  return isspace(static_cast<unsigned char>(ch)) != 0;
}
inline bool IsUpper(char ch) {
  return isupper(static_cast<unsigned char>(ch)) != 0;
}
inline bool IsXDigit(char ch) {
  return isxdigit(static_cast<unsigned char>(ch)) != 0;
}

inline char ToLower(char ch) {
  return static_cast<char>(tolower(static_cast<unsigned char>(ch)));
}
inline char ToUpper(char ch) {
  return static_cast<char>(toupper(static_cast<unsigned char>(ch)));
}

// The testing::internal::posix namespace holds wrappers for common
// POSIX functions.  These wrappers hide the differences between
// Windows/MSVC and POSIX systems.  Since some compilers define these
// standard functions as macros, the wrapper cannot have the same name
// as the wrapped function.

namespace posix {

// Functions with a different name on Windows.

#if GTEST_OS_WINDOWS

typedef struct _stat StatStruct;

# ifdef __BORLANDC__
inline int IsATTY(int fd) { return isatty(fd); }
inline int StrCaseCmp(const char* s1, const char* s2) {
  return stricmp(s1, s2);
}
inline char* StrDup(const char* src) { return strdup(src); }
# else  // !__BORLANDC__
#  if GTEST_OS_WINDOWS_MOBILE
inline int IsATTY(int /* fd */) { return 0; }
#  else
inline int IsATTY(int fd) { return _isatty(fd); }
#  endif  // GTEST_OS_WINDOWS_MOBILE
inline int StrCaseCmp(const char* s1, const char* s2) {
  return _stricmp(s1, s2);
}
inline char* StrDup(const char* src) { return _strdup(src); }
# endif  // __BORLANDC__

# if GTEST_OS_WINDOWS_MOBILE
inline int FileNo(FILE* file) { return reinterpret_cast<int>(_fileno(file)); }
// Stat(), RmDir(), and IsDir() are not needed on Windows CE at this
// time and thus not defined there.
# else
inline int FileNo(FILE* file) { return _fileno(file); }
inline int Stat(const char* path, StatStruct* buf) { return _stat(path, buf); }
inline int RmDir(const char* dir) { return _rmdir(dir); }
inline bool IsDir(const StatStruct& st) {
  return (_S_IFDIR & st.st_mode) != 0;
}
# endif  // GTEST_OS_WINDOWS_MOBILE

#else

typedef struct stat StatStruct;

inline int FileNo(FILE* file) { return fileno(file); }
inline int IsATTY(int fd) { return isatty(fd); }
inline int Stat(const char* path, StatStruct* buf) { return stat(path, buf); }
inline int StrCaseCmp(const char* s1, const char* s2) {
  return strcasecmp(s1, s2);
}
inline char* StrDup(const char* src) { return strdup(src); }
inline int RmDir(const char* dir) { return rmdir(dir); }
inline bool IsDir(const StatStruct& st) { return S_ISDIR(st.st_mode); }

#endif  // GTEST_OS_WINDOWS

// Functions deprecated by MSVC 8.0.

#ifdef _MSC_VER
// Temporarily disable warning 4996 (deprecated function).
# pragma warning(push)
# pragma warning(disable:4996)
#endif

inline const char* StrNCpy(char* dest, const char* src, size_t n) {
  return strncpy(dest, src, n);
}

// ChDir(), FReopen(), FDOpen(), Read(), Write(), Close(), and
// StrError() aren't needed on Windows CE at this time and thus not
// defined there.

#if !GTEST_OS_WINDOWS_MOBILE
inline int ChDir(const char* dir) { return chdir(dir); }
#endif
inline FILE* FOpen(const char* path, const char* mode) {
  return fopen(path, mode);
}
#if !GTEST_OS_WINDOWS_MOBILE
inline FILE *FReopen(const char* path, const char* mode, FILE* stream) {
  return freopen(path, mode, stream);
}
inline FILE* FDOpen(int fd, const char* mode) { return fdopen(fd, mode); }
#endif
inline int FClose(FILE* fp) { return fclose(fp); }
#if !GTEST_OS_WINDOWS_MOBILE
inline int Read(int fd, void* buf, unsigned int count) {
  return static_cast<int>(read(fd, buf, count));
}
inline int Write(int fd, const void* buf, unsigned int count) {
  return static_cast<int>(write(fd, buf, count));
}
inline int Close(int fd) { return close(fd); }
inline const char* StrError(int errnum) { return strerror(errnum); }
#endif
inline const char* GetEnv(const char* name) {
#if GTEST_OS_WINDOWS_MOBILE
  // We are on Windows CE, which has no environment variables.
  return NULL;
#elif defined(__BORLANDC__) || defined(__SunOS_5_8) || defined(__SunOS_5_9)
  // Environment variables which we programmatically clear will be set to the
  // empty string rather than unset (NULL).  Handle that case.
  const char* const env = getenv(name);
  return (env != NULL && env[0] != '\0') ? env : NULL;
#else
  return getenv(name);
#endif
}

#ifdef _MSC_VER
# pragma warning(pop)  // Restores the warning state.
#endif

#if GTEST_OS_WINDOWS_MOBILE
// Windows CE has no C library. The abort() function is used in
// several places in Google Test. This implementation provides a reasonable
// imitation of standard behaviour.
void Abort();
#else
inline void Abort() { abort(); }
#endif  // GTEST_OS_WINDOWS_MOBILE

}  // namespace posix

// The maximum number a BiggestInt can represent.  This definition
// works no matter BiggestInt is represented in one's complement or
// two's complement.
//
// We cannot rely on numeric_limits in STL, as __int64 and long long
// are not part of standard C++ and numeric_limits doesn't need to be
// defined for them.
const BiggestInt kMaxBiggestInt =
    ~(static_cast<BiggestInt>(1) << (8*sizeof(BiggestInt) - 1));

// This template class serves as a compile-time function from size to
// type.  It maps a size in bytes to a primitive type with that
// size. e.g.
//
//   TypeWithSize<4>::UInt
//
// is typedef-ed to be unsigned int (unsigned integer made up of 4
// bytes).
//
// Such functionality should belong to STL, but I cannot find it
// there.
//
// Google Test uses this class in the implementation of floating-point
// comparison.
//
// For now it only handles UInt (unsigned int) as that's all Google Test
// needs.  Other types can be easily added in the future if need
// arises.
template <size_t size>
class TypeWithSize {
 public:
  // This prevents the user from using TypeWithSize<N> with incorrect
  // values of N.
  typedef void UInt;
};

// The specialization for size 4.
template <>
class TypeWithSize<4> {
 public:
  // unsigned int has size 4 in both gcc and MSVC.
  //
  // As base/basictypes.h doesn't compile on Windows, we cannot use
  // uint32, uint64, and etc here.
  typedef int Int;
  typedef unsigned int UInt;
};

// The specialization for size 8.
template <>
class TypeWithSize<8> {
 public:

#if GTEST_OS_WINDOWS
  typedef __int64 Int;
  typedef unsigned __int64 UInt;
#else
  typedef long long Int;  // NOLINT
  typedef unsigned long long UInt;  // NOLINT
#endif  // GTEST_OS_WINDOWS
};

// Integer types of known sizes.
typedef TypeWithSize<4>::Int Int32;
typedef TypeWithSize<4>::UInt UInt32;
typedef TypeWithSize<8>::Int Int64;
typedef TypeWithSize<8>::UInt UInt64;
typedef TypeWithSize<8>::Int TimeInMillis;  // Represents time in milliseconds.

// Utilities for command line flags and environment variables.

// Macro for referencing flags.
#define GTEST_FLAG(name) FLAGS_gtest_##name

// Macros for declaring flags.
#define GTEST_DECLARE_bool_(name) GTEST_API_ extern bool GTEST_FLAG(name)
#define GTEST_DECLARE_int32_(name) \
    GTEST_API_ extern ::testing::internal::Int32 GTEST_FLAG(name)
#define GTEST_DECLARE_string_(name) \
    GTEST_API_ extern ::testing::internal::String GTEST_FLAG(name)

// Macros for defining flags.
#define GTEST_DEFINE_bool_(name, default_val, doc) \
    GTEST_API_ bool GTEST_FLAG(name) = (default_val)
#define GTEST_DEFINE_int32_(name, default_val, doc) \
    GTEST_API_ ::testing::internal::Int32 GTEST_FLAG(name) = (default_val)
#define GTEST_DEFINE_string_(name, default_val, doc) \
    GTEST_API_ ::testing::internal::String GTEST_FLAG(name) = (default_val)

// Parses 'str' for a 32-bit signed integer.  If successful, writes the result
// to *value and returns true; otherwise leaves *value unchanged and returns
// false.
// TODO(chandlerc): Find a better way to refactor flag and environment parsing
// out of both gtest-port.cc and gtest.cc to avoid exporting this utility
// function.
bool ParseInt32(const Message& src_text, const char* str, Int32* value);

// Parses a bool/Int32/string from the environment variable
// corresponding to the given Google Test flag.
bool BoolFromGTestEnv(const char* flag, bool default_val);
GTEST_API_ Int32 Int32FromGTestEnv(const char* flag, Int32 default_val);
const char* StringFromGTestEnv(const char* flag, const char* default_val);

}  // namespace internal
}  // namespace testing

#endif  // GTEST_INCLUDE_GTEST_INTERNAL_GTEST_PORT_H_

#if GTEST_OS_LINUX
# include <stdlib.h>
# include <sys/types.h>
# include <sys/wait.h>
# include <unistd.h>
#endif  // GTEST_OS_LINUX

#include <ctype.h>
#include <string.h>
#include <iomanip>
#include <limits>
#include <set>

// Copyright 2005, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Authors: wan@google.com (Zhanyong Wan), eefacm@gmail.com (Sean Mcafee)
//
// The Google C++ Testing Framework (Google Test)
//
// This header file declares the String class and functions used internally by
// Google Test.  They are subject to change without notice. They should not used
// by code external to Google Test.
//
// This header file is #included by <gtest/internal/gtest-internal.h>.
// It should not be #included by other files.

#ifndef GTEST_INCLUDE_GTEST_INTERNAL_GTEST_STRING_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_GTEST_STRING_H_

#ifdef __BORLANDC__
// string.h is not guaranteed to provide strcpy on C++ Builder.
# include <mem.h>
#endif

#include <string.h>

#include <string>

namespace testing {
namespace internal {

// String - a UTF-8 string class.
//
// For historic reasons, we don't use std::string.
//
// TODO(wan@google.com): replace this class with std::string or
// implement it in terms of the latter.
//
// Note that String can represent both NULL and the empty string,
// while std::string cannot represent NULL.
//
// NULL and the empty string are considered different.  NULL is less
// than anything (including the empty string) except itself.
//
// This class only provides minimum functionality necessary for
// implementing Google Test.  We do not intend to implement a full-fledged
// string class here.
//
// Since the purpose of this class is to provide a substitute for
// std::string on platforms where it cannot be used, we define a copy
// constructor and assignment operators such that we don't need
// conditional compilation in a lot of places.
//
// In order to make the representation efficient, the d'tor of String
// is not virtual.  Therefore DO NOT INHERIT FROM String.
class GTEST_API_ String {
 public:
  // Static utility methods

  // Returns the input enclosed in double quotes if it's not NULL;
  // otherwise returns "(null)".  For example, "\"Hello\"" is returned
  // for input "Hello".
  //
  // This is useful for printing a C string in the syntax of a literal.
  //
  // Known issue: escape sequences are not handled yet.
  static String ShowCStringQuoted(const char* c_str);

  // Clones a 0-terminated C string, allocating memory using new.  The
  // caller is responsible for deleting the return value using
  // delete[].  Returns the cloned string, or NULL if the input is
  // NULL.
  //
  // This is different from strdup() in string.h, which allocates
  // memory using malloc().
  static const char* CloneCString(const char* c_str);

#if GTEST_OS_WINDOWS_MOBILE
  // Windows CE does not have the 'ANSI' versions of Win32 APIs. To be
  // able to pass strings to Win32 APIs on CE we need to convert them
  // to 'Unicode', UTF-16.

  // Creates a UTF-16 wide string from the given ANSI string, allocating
  // memory using new. The caller is responsible for deleting the return
  // value using delete[]. Returns the wide string, or NULL if the
  // input is NULL.
  //
  // The wide string is created using the ANSI codepage (CP_ACP) to
  // match the behaviour of the ANSI versions of Win32 calls and the
  // C runtime.
  static LPCWSTR AnsiToUtf16(const char* c_str);

  // Creates an ANSI string from the given wide string, allocating
  // memory using new. The caller is responsible for deleting the return
  // value using delete[]. Returns the ANSI string, or NULL if the
  // input is NULL.
  //
  // The returned string is created using the ANSI codepage (CP_ACP) to
  // match the behaviour of the ANSI versions of Win32 calls and the
  // C runtime.
  static const char* Utf16ToAnsi(LPCWSTR utf16_str);
#endif

  // Compares two C strings.  Returns true iff they have the same content.
  //
  // Unlike strcmp(), this function can handle NULL argument(s).  A
  // NULL C string is considered different to any non-NULL C string,
  // including the empty string.
  static bool CStringEquals(const char* lhs, const char* rhs);

  // Converts a wide C string to a String using the UTF-8 encoding.
  // NULL will be converted to "(null)".  If an error occurred during
  // the conversion, "(failed to convert from wide string)" is
  // returned.
  static String ShowWideCString(const wchar_t* wide_c_str);

  // Similar to ShowWideCString(), except that this function encloses
  // the converted string in double quotes.
  static String ShowWideCStringQuoted(const wchar_t* wide_c_str);

  // Compares two wide C strings.  Returns true iff they have the same
  // content.
  //
  // Unlike wcscmp(), this function can handle NULL argument(s).  A
  // NULL C string is considered different to any non-NULL C string,
  // including the empty string.
  static bool WideCStringEquals(const wchar_t* lhs, const wchar_t* rhs);

  // Compares two C strings, ignoring case.  Returns true iff they
  // have the same content.
  //
  // Unlike strcasecmp(), this function can handle NULL argument(s).
  // A NULL C string is considered different to any non-NULL C string,
  // including the empty string.
  static bool CaseInsensitiveCStringEquals(const char* lhs,
                                           const char* rhs);

  // Compares two wide C strings, ignoring case.  Returns true iff they
  // have the same content.
  //
  // Unlike wcscasecmp(), this function can handle NULL argument(s).
  // A NULL C string is considered different to any non-NULL wide C string,
  // including the empty string.
  // NB: The implementations on different platforms slightly differ.
  // On windows, this method uses _wcsicmp which compares according to LC_CTYPE
  // environment variable. On GNU platform this method uses wcscasecmp
  // which compares according to LC_CTYPE category of the current locale.
  // On MacOS X, it uses towlower, which also uses LC_CTYPE category of the
  // current locale.
  static bool CaseInsensitiveWideCStringEquals(const wchar_t* lhs,
                                               const wchar_t* rhs);

  // Formats a list of arguments to a String, using the same format
  // spec string as for printf.
  //
  // We do not use the StringPrintf class as it is not universally
  // available.
  //
  // The result is limited to 4096 characters (including the tailing
  // 0).  If 4096 characters are not enough to format the input,
  // "<buffer exceeded>" is returned.
  static String Format(const char* format, ...);

  // C'tors

  // The default c'tor constructs a NULL string.
  String() : c_str_(NULL), length_(0) {}

  // Constructs a String by cloning a 0-terminated C string.
  String(const char* a_c_str) {  // NOLINT
    if (a_c_str == NULL) {
      c_str_ = NULL;
      length_ = 0;
    } else {
      ConstructNonNull(a_c_str, strlen(a_c_str));
    }
  }

  // Constructs a String by copying a given number of chars from a
  // buffer.  E.g. String("hello", 3) creates the string "hel",
  // String("a\0bcd", 4) creates "a\0bc", String(NULL, 0) creates "",
  // and String(NULL, 1) results in access violation.
  String(const char* buffer, size_t a_length) {
    ConstructNonNull(buffer, a_length);
  }

  // The copy c'tor creates a new copy of the string.  The two
  // String objects do not share content.
  String(const String& str) : c_str_(NULL), length_(0) { *this = str; }

  // D'tor.  String is intended to be a final class, so the d'tor
  // doesn't need to be virtual.
  ~String() { delete[] c_str_; }

  // Allows a String to be implicitly converted to an ::std::string or
  // ::string, and vice versa.  Converting a String containing a NULL
  // pointer to ::std::string or ::string is undefined behavior.
  // Converting a ::std::string or ::string containing an embedded NUL
  // character to a String will result in the prefix up to the first
  // NUL character.
  String(const ::std::string& str) {
    ConstructNonNull(str.c_str(), str.length());
  }

  operator ::std::string() const { return ::std::string(c_str(), length()); }

#if GTEST_HAS_GLOBAL_STRING
  String(const ::string& str) {
    ConstructNonNull(str.c_str(), str.length());
  }

  operator ::string() const { return ::string(c_str(), length()); }
#endif  // GTEST_HAS_GLOBAL_STRING

  // Returns true iff this is an empty string (i.e. "").
  bool empty() const { return (c_str() != NULL) && (length() == 0); }

  // Compares this with another String.
  // Returns < 0 if this is less than rhs, 0 if this is equal to rhs, or > 0
  // if this is greater than rhs.
  int Compare(const String& rhs) const;

  // Returns true iff this String equals the given C string.  A NULL
  // string and a non-NULL string are considered not equal.
  bool operator==(const char* a_c_str) const { return Compare(a_c_str) == 0; }

  // Returns true iff this String is less than the given String.  A
  // NULL string is considered less than "".
  bool operator<(const String& rhs) const { return Compare(rhs) < 0; }

  // Returns true iff this String doesn't equal the given C string.  A NULL
  // string and a non-NULL string are considered not equal.
  bool operator!=(const char* a_c_str) const { return !(*this == a_c_str); }

  // Returns true iff this String ends with the given suffix.  *Any*
  // String is considered to end with a NULL or empty suffix.
  bool EndsWith(const char* suffix) const;

  // Returns true iff this String ends with the given suffix, not considering
  // case. Any String is considered to end with a NULL or empty suffix.
  bool EndsWithCaseInsensitive(const char* suffix) const;

  // Returns the length of the encapsulated string, or 0 if the
  // string is NULL.
  size_t length() const { return length_; }

  // Gets the 0-terminated C string this String object represents.
  // The String object still owns the string.  Therefore the caller
  // should NOT delete the return value.
  const char* c_str() const { return c_str_; }

  // Assigns a C string to this object.  Self-assignment works.
  const String& operator=(const char* a_c_str) {
    return *this = String(a_c_str);
  }

  // Assigns a String object to this object.  Self-assignment works.
  const String& operator=(const String& rhs) {
    if (this != &rhs) {
      delete[] c_str_;
      if (rhs.c_str() == NULL) {
        c_str_ = NULL;
        length_ = 0;
      } else {
        ConstructNonNull(rhs.c_str(), rhs.length());
      }
    }

    return *this;
  }

 private:
  // Constructs a non-NULL String from the given content.  This
  // function can only be called when c_str_ has not been allocated.
  // ConstructNonNull(NULL, 0) results in an empty string ("").
  // ConstructNonNull(NULL, non_zero) is undefined behavior.
  void ConstructNonNull(const char* buffer, size_t a_length) {
    char* const str = new char[a_length + 1];
    memcpy(str, buffer, a_length);
    str[a_length] = '\0';
    c_str_ = str;
    length_ = a_length;
  }

  const char* c_str_;
  size_t length_;
};  // class String

// Streams a String to an ostream.  Each '\0' character in the String
// is replaced with "\\0".
inline ::std::ostream& operator<<(::std::ostream& os, const String& str) {
  if (str.c_str() == NULL) {
    os << "(null)";
  } else {
    const char* const c_str = str.c_str();
    for (size_t i = 0; i != str.length(); i++) {
      if (c_str[i] == '\0') {
        os << "\\0";
      } else {
        os << c_str[i];
      }
    }
  }
  return os;
}

// Gets the content of the stringstream's buffer as a String.  Each '\0'
// character in the buffer is replaced with "\\0".
GTEST_API_ String StringStreamToString(::std::stringstream* stream);

// Converts a streamable value to a String.  A NULL pointer is
// converted to "(null)".  When the input value is a ::string,
// ::std::string, ::wstring, or ::std::wstring object, each NUL
// character in it is replaced with "\\0".

// Declared here but defined in gtest.h, so that it has access
// to the definition of the Message class, required by the ARM
// compiler.
template <typename T>
String StreamableToString(const T& streamable);

}  // namespace internal
}  // namespace testing

#endif  // GTEST_INCLUDE_GTEST_INTERNAL_GTEST_STRING_H_
// Copyright 2008, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Author: keith.ray@gmail.com (Keith Ray)
//
// Google Test filepath utilities
//
// This header file declares classes and functions used internally by
// Google Test.  They are subject to change without notice.
//
// This file is #included in <gtest/internal/gtest-internal.h>.
// Do not include this header file separately!

#ifndef GTEST_INCLUDE_GTEST_INTERNAL_GTEST_FILEPATH_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_GTEST_FILEPATH_H_


namespace testing {
namespace internal {

// FilePath - a class for file and directory pathname manipulation which
// handles platform-specific conventions (like the pathname separator).
// Used for helper functions for naming files in a directory for xml output.
// Except for Set methods, all methods are const or static, which provides an
// "immutable value object" -- useful for peace of mind.
// A FilePath with a value ending in a path separator ("like/this/") represents
// a directory, otherwise it is assumed to represent a file. In either case,
// it may or may not represent an actual file or directory in the file system.
// Names are NOT checked for syntax correctness -- no checking for illegal
// characters, malformed paths, etc.

class GTEST_API_ FilePath {
 public:
  FilePath() : pathname_("") { }
  FilePath(const FilePath& rhs) : pathname_(rhs.pathname_) { }

  explicit FilePath(const char* pathname) : pathname_(pathname) {
    Normalize();
  }

  explicit FilePath(const String& pathname) : pathname_(pathname) {
    Normalize();
  }

  FilePath& operator=(const FilePath& rhs) {
    Set(rhs);
    return *this;
  }

  void Set(const FilePath& rhs) {
    pathname_ = rhs.pathname_;
  }

  String ToString() const { return pathname_; }
  const char* c_str() const { return pathname_.c_str(); }

  // Returns the current working directory, or "" if unsuccessful.
  static FilePath GetCurrentDir();

  // Given directory = "dir", base_name = "test", number = 0,
  // extension = "xml", returns "dir/test.xml". If number is greater
  // than zero (e.g., 12), returns "dir/test_12.xml".
  // On Windows platform, uses \ as the separator rather than /.
  static FilePath MakeFileName(const FilePath& directory,
                               const FilePath& base_name,
                               int number,
                               const char* extension);

  // Given directory = "dir", relative_path = "test.xml",
  // returns "dir/test.xml".
  // On Windows, uses \ as the separator rather than /.
  static FilePath ConcatPaths(const FilePath& directory,
                              const FilePath& relative_path);

  // Returns a pathname for a file that does not currently exist. The pathname
  // will be directory/base_name.extension or
  // directory/base_name_<number>.extension if directory/base_name.extension
  // already exists. The number will be incremented until a pathname is found
  // that does not already exist.
  // Examples: 'dir/foo_test.xml' or 'dir/foo_test_1.xml'.
  // There could be a race condition if two or more processes are calling this
  // function at the same time -- they could both pick the same filename.
  static FilePath GenerateUniqueFileName(const FilePath& directory,
                                         const FilePath& base_name,
                                         const char* extension);

  // Returns true iff the path is NULL or "".
  bool IsEmpty() const { return c_str() == NULL || *c_str() == '\0'; }

  // If input name has a trailing separator character, removes it and returns
  // the name, otherwise return the name string unmodified.
  // On Windows platform, uses \ as the separator, other platforms use /.
  FilePath RemoveTrailingPathSeparator() const;

  // Returns a copy of the FilePath with the directory part removed.
  // Example: FilePath("path/to/file").RemoveDirectoryName() returns
  // FilePath("file"). If there is no directory part ("just_a_file"), it returns
  // the FilePath unmodified. If there is no file part ("just_a_dir/") it
  // returns an empty FilePath ("").
  // On Windows platform, '\' is the path separator, otherwise it is '/'.
  FilePath RemoveDirectoryName() const;

  // RemoveFileName returns the directory path with the filename removed.
  // Example: FilePath("path/to/file").RemoveFileName() returns "path/to/".
  // If the FilePath is "a_file" or "/a_file", RemoveFileName returns
  // FilePath("./") or, on Windows, FilePath(".\\"). If the filepath does
  // not have a file, like "just/a/dir/", it returns the FilePath unmodified.
  // On Windows platform, '\' is the path separator, otherwise it is '/'.
  FilePath RemoveFileName() const;

  // Returns a copy of the FilePath with the case-insensitive extension removed.
  // Example: FilePath("dir/file.exe").RemoveExtension("EXE") returns
  // FilePath("dir/file"). If a case-insensitive extension is not
  // found, returns a copy of the original FilePath.
  FilePath RemoveExtension(const char* extension) const;

  // Creates directories so that path exists. Returns true if successful or if
  // the directories already exist; returns false if unable to create
  // directories for any reason. Will also return false if the FilePath does
  // not represent a directory (that is, it doesn't end with a path separator).
  bool CreateDirectoriesRecursively() const;

  // Create the directory so that path exists. Returns true if successful or
  // if the directory already exists; returns false if unable to create the
  // directory for any reason, including if the parent directory does not
  // exist. Not named "CreateDirectory" because that's a macro on Windows.
  bool CreateFolder() const;

  // Returns true if FilePath describes something in the file-system,
  // either a file, directory, or whatever, and that something exists.
  bool FileOrDirectoryExists() const;

  // Returns true if pathname describes a directory in the file-system
  // that exists.
  bool DirectoryExists() const;

  // Returns true if FilePath ends with a path separator, which indicates that
  // it is intended to represent a directory. Returns false otherwise.
  // This does NOT check that a directory (or file) actually exists.
  bool IsDirectory() const;

  // Returns true if pathname describes a root directory. (Windows has one
  // root directory per disk drive.)
  bool IsRootDirectory() const;

  // Returns true if pathname describes an absolute path.
  bool IsAbsolutePath() const;

 private:
  // Replaces multiple consecutive separators with a single separator.
  // For example, "bar///foo" becomes "bar/foo". Does not eliminate other
  // redundancies that might be in a pathname involving "." or "..".
  //
  // A pathname with multiple consecutive separators may occur either through
  // user error or as a result of some scripts or APIs that generate a pathname
  // with a trailing separator. On other platforms the same API or script
  // may NOT generate a pathname with a trailing "/". Then elsewhere that
  // pathname may have another "/" and pathname components added to it,
  // without checking for the separator already being there.
  // The script language and operating system may allow paths like "foo//bar"
  // but some of the functions in FilePath will not handle that correctly. In
  // particular, RemoveTrailingPathSeparator() only removes one separator, and
  // it is called in CreateDirectoriesRecursively() assuming that it will change
  // a pathname from directory syntax (trailing separator) to filename syntax.
  //
  // On Windows this method also replaces the alternate path separator '/' with
  // the primary path separator '\\', so that for example "bar\\/\\foo" becomes
  // "bar\\foo".

  void Normalize();

  // Returns a pointer to the last occurence of a valid path separator in
  // the FilePath. On Windows, for example, both '/' and '\' are valid path
  // separators. Returns NULL if no path separator was found.
  const char* FindLastPathSeparator() const;

  String pathname_;
};  // class FilePath

}  // namespace internal
}  // namespace testing

#endif  // GTEST_INCLUDE_GTEST_INTERNAL_GTEST_FILEPATH_H_
// This file was GENERATED by command:
//     pump.py gtest-type-util.h.pump
// DO NOT EDIT BY HAND!!!

// Copyright 2008 Google Inc.
// All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Author: wan@google.com (Zhanyong Wan)

// Type utilities needed for implementing typed and type-parameterized
// tests.  This file is generated by a SCRIPT.  DO NOT EDIT BY HAND!
//
// Currently we support at most 50 types in a list, and at most 50
// type-parameterized tests in one type-parameterized test case.
// Please contact googletestframework@googlegroups.com if you need
// more.

#ifndef GTEST_INCLUDE_GTEST_INTERNAL_GTEST_TYPE_UTIL_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_GTEST_TYPE_UTIL_H_


// #ifdef __GNUC__ is too general here.  It is possible to use gcc without using
// libstdc++ (which is where cxxabi.h comes from).
# ifdef __GLIBCXX__
#  include <cxxabi.h>
# elif defined(__HP_aCC)
#  include <acxx_demangle.h>
# endif  // __GLIBCXX__

namespace testing {
namespace internal {

// GetTypeName<T>() returns a human-readable name of type T.
// NB: This function is also used in Google Mock, so don't move it inside of
// the typed-test-only section below.
template <typename T>
String GetTypeName() {
# if GTEST_HAS_RTTI

  const char* const name = typeid(T).name();
#  if defined(__GLIBCXX__) || defined(__HP_aCC)
  int status = 0;
  // gcc's implementation of typeid(T).name() mangles the type name,
  // so we have to demangle it.
#   ifdef __GLIBCXX__
  using abi::__cxa_demangle;
#   endif // __GLIBCXX__
  char* const readable_name = __cxa_demangle(name, 0, 0, &status);
  const String name_str(status == 0 ? readable_name : name);
  free(readable_name);
  return name_str;
#  else
  return name;
#  endif  // __GLIBCXX__ || __HP_aCC

# else

  return "<type>";

# endif  // GTEST_HAS_RTTI
}

#if GTEST_HAS_TYPED_TEST || GTEST_HAS_TYPED_TEST_P

// AssertyTypeEq<T1, T2>::type is defined iff T1 and T2 are the same
// type.  This can be used as a compile-time assertion to ensure that
// two types are equal.

template <typename T1, typename T2>
struct AssertTypeEq;

template <typename T>
struct AssertTypeEq<T, T> {
  typedef bool type;
};

// A unique type used as the default value for the arguments of class
// template Types.  This allows us to simulate variadic templates
// (e.g. Types<int>, Type<int, double>, and etc), which C++ doesn't
// support directly.
struct None {};

// The following family of struct and struct templates are used to
// represent type lists.  In particular, TypesN<T1, T2, ..., TN>
// represents a type list with N types (T1, T2, ..., and TN) in it.
// Except for Types0, every struct in the family has two member types:
// Head for the first type in the list, and Tail for the rest of the
// list.

// The empty type list.
struct Types0 {};

// Type lists of length 1, 2, 3, and so on.

template <typename T1>
struct Types1 {
  typedef T1 Head;
  typedef Types0 Tail;
};
template <typename T1, typename T2>
struct Types2 {
  typedef T1 Head;
  typedef Types1<T2> Tail;
};

template <typename T1, typename T2, typename T3>
struct Types3 {
  typedef T1 Head;
  typedef Types2<T2, T3> Tail;
};

template <typename T1, typename T2, typename T3, typename T4>
struct Types4 {
  typedef T1 Head;
  typedef Types3<T2, T3, T4> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5>
struct Types5 {
  typedef T1 Head;
  typedef Types4<T2, T3, T4, T5> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6>
struct Types6 {
  typedef T1 Head;
  typedef Types5<T2, T3, T4, T5, T6> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7>
struct Types7 {
  typedef T1 Head;
  typedef Types6<T2, T3, T4, T5, T6, T7> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8>
struct Types8 {
  typedef T1 Head;
  typedef Types7<T2, T3, T4, T5, T6, T7, T8> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9>
struct Types9 {
  typedef T1 Head;
  typedef Types8<T2, T3, T4, T5, T6, T7, T8, T9> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10>
struct Types10 {
  typedef T1 Head;
  typedef Types9<T2, T3, T4, T5, T6, T7, T8, T9, T10> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11>
struct Types11 {
  typedef T1 Head;
  typedef Types10<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12>
struct Types12 {
  typedef T1 Head;
  typedef Types11<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13>
struct Types13 {
  typedef T1 Head;
  typedef Types12<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14>
struct Types14 {
  typedef T1 Head;
  typedef Types13<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15>
struct Types15 {
  typedef T1 Head;
  typedef Types14<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16>
struct Types16 {
  typedef T1 Head;
  typedef Types15<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17>
struct Types17 {
  typedef T1 Head;
  typedef Types16<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18>
struct Types18 {
  typedef T1 Head;
  typedef Types17<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19>
struct Types19 {
  typedef T1 Head;
  typedef Types18<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20>
struct Types20 {
  typedef T1 Head;
  typedef Types19<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21>
struct Types21 {
  typedef T1 Head;
  typedef Types20<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22>
struct Types22 {
  typedef T1 Head;
  typedef Types21<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23>
struct Types23 {
  typedef T1 Head;
  typedef Types22<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24>
struct Types24 {
  typedef T1 Head;
  typedef Types23<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25>
struct Types25 {
  typedef T1 Head;
  typedef Types24<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26>
struct Types26 {
  typedef T1 Head;
  typedef Types25<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27>
struct Types27 {
  typedef T1 Head;
  typedef Types26<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28>
struct Types28 {
  typedef T1 Head;
  typedef Types27<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29>
struct Types29 {
  typedef T1 Head;
  typedef Types28<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
      T29> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30>
struct Types30 {
  typedef T1 Head;
  typedef Types29<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31>
struct Types31 {
  typedef T1 Head;
  typedef Types30<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32>
struct Types32 {
  typedef T1 Head;
  typedef Types31<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33>
struct Types33 {
  typedef T1 Head;
  typedef Types32<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32, T33> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34>
struct Types34 {
  typedef T1 Head;
  typedef Types33<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32, T33, T34> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35>
struct Types35 {
  typedef T1 Head;
  typedef Types34<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32, T33, T34, T35> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36>
struct Types36 {
  typedef T1 Head;
  typedef Types35<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32, T33, T34, T35, T36> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37>
struct Types37 {
  typedef T1 Head;
  typedef Types36<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32, T33, T34, T35, T36, T37> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38>
struct Types38 {
  typedef T1 Head;
  typedef Types37<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32, T33, T34, T35, T36, T37, T38> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39>
struct Types39 {
  typedef T1 Head;
  typedef Types38<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32, T33, T34, T35, T36, T37, T38, T39> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40>
struct Types40 {
  typedef T1 Head;
  typedef Types39<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41>
struct Types41 {
  typedef T1 Head;
  typedef Types40<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42>
struct Types42 {
  typedef T1 Head;
  typedef Types41<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43>
struct Types43 {
  typedef T1 Head;
  typedef Types42<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42,
      T43> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43, typename T44>
struct Types44 {
  typedef T1 Head;
  typedef Types43<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43,
      T44> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43, typename T44, typename T45>
struct Types45 {
  typedef T1 Head;
  typedef Types44<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43,
      T44, T45> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43, typename T44, typename T45,
    typename T46>
struct Types46 {
  typedef T1 Head;
  typedef Types45<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43,
      T44, T45, T46> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43, typename T44, typename T45,
    typename T46, typename T47>
struct Types47 {
  typedef T1 Head;
  typedef Types46<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43,
      T44, T45, T46, T47> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43, typename T44, typename T45,
    typename T46, typename T47, typename T48>
struct Types48 {
  typedef T1 Head;
  typedef Types47<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43,
      T44, T45, T46, T47, T48> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43, typename T44, typename T45,
    typename T46, typename T47, typename T48, typename T49>
struct Types49 {
  typedef T1 Head;
  typedef Types48<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43,
      T44, T45, T46, T47, T48, T49> Tail;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43, typename T44, typename T45,
    typename T46, typename T47, typename T48, typename T49, typename T50>
struct Types50 {
  typedef T1 Head;
  typedef Types49<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
      T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
      T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43,
      T44, T45, T46, T47, T48, T49, T50> Tail;
};


}  // namespace internal

// We don't want to require the users to write TypesN<...> directly,
// as that would require them to count the length.  Types<...> is much
// easier to write, but generates horrible messages when there is a
// compiler error, as gcc insists on printing out each template
// argument, even if it has the default value (this means Types<int>
// will appear as Types<int, None, None, ..., None> in the compiler
// errors).
//
// Our solution is to combine the best part of the two approaches: a
// user would write Types<T1, ..., TN>, and Google Test will translate
// that to TypesN<T1, ..., TN> internally to make error messages
// readable.  The translation is done by the 'type' member of the
// Types template.
template <typename T1 = internal::None, typename T2 = internal::None,
    typename T3 = internal::None, typename T4 = internal::None,
    typename T5 = internal::None, typename T6 = internal::None,
    typename T7 = internal::None, typename T8 = internal::None,
    typename T9 = internal::None, typename T10 = internal::None,
    typename T11 = internal::None, typename T12 = internal::None,
    typename T13 = internal::None, typename T14 = internal::None,
    typename T15 = internal::None, typename T16 = internal::None,
    typename T17 = internal::None, typename T18 = internal::None,
    typename T19 = internal::None, typename T20 = internal::None,
    typename T21 = internal::None, typename T22 = internal::None,
    typename T23 = internal::None, typename T24 = internal::None,
    typename T25 = internal::None, typename T26 = internal::None,
    typename T27 = internal::None, typename T28 = internal::None,
    typename T29 = internal::None, typename T30 = internal::None,
    typename T31 = internal::None, typename T32 = internal::None,
    typename T33 = internal::None, typename T34 = internal::None,
    typename T35 = internal::None, typename T36 = internal::None,
    typename T37 = internal::None, typename T38 = internal::None,
    typename T39 = internal::None, typename T40 = internal::None,
    typename T41 = internal::None, typename T42 = internal::None,
    typename T43 = internal::None, typename T44 = internal::None,
    typename T45 = internal::None, typename T46 = internal::None,
    typename T47 = internal::None, typename T48 = internal::None,
    typename T49 = internal::None, typename T50 = internal::None>
struct Types {
  typedef internal::Types50<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
      T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40,
      T41, T42, T43, T44, T45, T46, T47, T48, T49, T50> type;
};

template <>
struct Types<internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None> {
  typedef internal::Types0 type;
};
template <typename T1>
struct Types<T1, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None> {
  typedef internal::Types1<T1> type;
};
template <typename T1, typename T2>
struct Types<T1, T2, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None> {
  typedef internal::Types2<T1, T2> type;
};
template <typename T1, typename T2, typename T3>
struct Types<T1, T2, T3, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None> {
  typedef internal::Types3<T1, T2, T3> type;
};
template <typename T1, typename T2, typename T3, typename T4>
struct Types<T1, T2, T3, T4, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None> {
  typedef internal::Types4<T1, T2, T3, T4> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5>
struct Types<T1, T2, T3, T4, T5, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None> {
  typedef internal::Types5<T1, T2, T3, T4, T5> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6>
struct Types<T1, T2, T3, T4, T5, T6, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None> {
  typedef internal::Types6<T1, T2, T3, T4, T5, T6> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7>
struct Types<T1, T2, T3, T4, T5, T6, T7, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None> {
  typedef internal::Types7<T1, T2, T3, T4, T5, T6, T7> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None> {
  typedef internal::Types8<T1, T2, T3, T4, T5, T6, T7, T8> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None> {
  typedef internal::Types9<T1, T2, T3, T4, T5, T6, T7, T8, T9> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None> {
  typedef internal::Types10<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None> {
  typedef internal::Types11<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None> {
  typedef internal::Types12<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None> {
  typedef internal::Types13<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None> {
  typedef internal::Types14<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None> {
  typedef internal::Types15<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None> {
  typedef internal::Types16<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None> {
  typedef internal::Types17<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None> {
  typedef internal::Types18<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None> {
  typedef internal::Types19<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None> {
  typedef internal::Types20<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None> {
  typedef internal::Types21<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, T22, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None> {
  typedef internal::Types22<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, T22, T23, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None> {
  typedef internal::Types23<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, T22, T23, T24, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None> {
  typedef internal::Types24<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None> {
  typedef internal::Types25<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None> {
  typedef internal::Types26<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25,
      T26> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None> {
  typedef internal::Types27<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
      T27> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None> {
  typedef internal::Types28<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
      T27, T28> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None> {
  typedef internal::Types29<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
      T27, T28, T29> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None> {
  typedef internal::Types30<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
      T27, T28, T29, T30> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30,
    T31, internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None> {
  typedef internal::Types31<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
      T27, T28, T29, T30, T31> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30,
    T31, T32, internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None> {
  typedef internal::Types32<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
      T27, T28, T29, T30, T31, T32> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30,
    T31, T32, T33, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None> {
  typedef internal::Types33<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
      T27, T28, T29, T30, T31, T32, T33> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30,
    T31, T32, T33, T34, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None> {
  typedef internal::Types34<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
      T27, T28, T29, T30, T31, T32, T33, T34> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30,
    T31, T32, T33, T34, T35, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None> {
  typedef internal::Types35<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
      T27, T28, T29, T30, T31, T32, T33, T34, T35> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30,
    T31, T32, T33, T34, T35, T36, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None> {
  typedef internal::Types36<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
      T27, T28, T29, T30, T31, T32, T33, T34, T35, T36> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30,
    T31, T32, T33, T34, T35, T36, T37, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None> {
  typedef internal::Types37<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
      T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30,
    T31, T32, T33, T34, T35, T36, T37, T38, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None> {
  typedef internal::Types38<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
      T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30,
    T31, T32, T33, T34, T35, T36, T37, T38, T39, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None> {
  typedef internal::Types39<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
      T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30,
    T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None> {
  typedef internal::Types40<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
      T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39,
      T40> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30,
    T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None, internal::None> {
  typedef internal::Types41<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
      T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40,
      T41> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30,
    T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, internal::None,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None> {
  typedef internal::Types42<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
      T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40,
      T41, T42> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30,
    T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None, internal::None> {
  typedef internal::Types43<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
      T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40,
      T41, T42, T43> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43, typename T44>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30,
    T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44,
    internal::None, internal::None, internal::None, internal::None,
    internal::None, internal::None> {
  typedef internal::Types44<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
      T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40,
      T41, T42, T43, T44> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43, typename T44, typename T45>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30,
    T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45,
    internal::None, internal::None, internal::None, internal::None,
    internal::None> {
  typedef internal::Types45<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
      T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40,
      T41, T42, T43, T44, T45> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43, typename T44, typename T45,
    typename T46>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30,
    T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45,
    T46, internal::None, internal::None, internal::None, internal::None> {
  typedef internal::Types46<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
      T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40,
      T41, T42, T43, T44, T45, T46> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43, typename T44, typename T45,
    typename T46, typename T47>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30,
    T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45,
    T46, T47, internal::None, internal::None, internal::None> {
  typedef internal::Types47<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
      T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40,
      T41, T42, T43, T44, T45, T46, T47> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43, typename T44, typename T45,
    typename T46, typename T47, typename T48>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30,
    T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45,
    T46, T47, T48, internal::None, internal::None> {
  typedef internal::Types48<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
      T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40,
      T41, T42, T43, T44, T45, T46, T47, T48> type;
};
template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43, typename T44, typename T45,
    typename T46, typename T47, typename T48, typename T49>
struct Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15,
    T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30,
    T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45,
    T46, T47, T48, T49, internal::None> {
  typedef internal::Types49<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
      T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40,
      T41, T42, T43, T44, T45, T46, T47, T48, T49> type;
};

namespace internal {

# define GTEST_TEMPLATE_ template <typename T> class

// The template "selector" struct TemplateSel<Tmpl> is used to
// represent Tmpl, which must be a class template with one type
// parameter, as a type.  TemplateSel<Tmpl>::Bind<T>::type is defined
// as the type Tmpl<T>.  This allows us to actually instantiate the
// template "selected" by TemplateSel<Tmpl>.
//
// This trick is necessary for simulating typedef for class templates,
// which C++ doesn't support directly.
template <GTEST_TEMPLATE_ Tmpl>
struct TemplateSel {
  template <typename T>
  struct Bind {
    typedef Tmpl<T> type;
  };
};

# define GTEST_BIND_(TmplSel, T) \
  TmplSel::template Bind<T>::type

// A unique struct template used as the default value for the
// arguments of class template Templates.  This allows us to simulate
// variadic templates (e.g. Templates<int>, Templates<int, double>,
// and etc), which C++ doesn't support directly.
template <typename T>
struct NoneT {};

// The following family of struct and struct templates are used to
// represent template lists.  In particular, TemplatesN<T1, T2, ...,
// TN> represents a list of N templates (T1, T2, ..., and TN).  Except
// for Templates0, every struct in the family has two member types:
// Head for the selector of the first template in the list, and Tail
// for the rest of the list.

// The empty template list.
struct Templates0 {};

// Template lists of length 1, 2, 3, and so on.

template <GTEST_TEMPLATE_ T1>
struct Templates1 {
  typedef TemplateSel<T1> Head;
  typedef Templates0 Tail;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2>
struct Templates2 {
  typedef TemplateSel<T1> Head;
  typedef Templates1<T2> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3>
struct Templates3 {
  typedef TemplateSel<T1> Head;
  typedef Templates2<T2, T3> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4>
struct Templates4 {
  typedef TemplateSel<T1> Head;
  typedef Templates3<T2, T3, T4> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5>
struct Templates5 {
  typedef TemplateSel<T1> Head;
  typedef Templates4<T2, T3, T4, T5> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6>
struct Templates6 {
  typedef TemplateSel<T1> Head;
  typedef Templates5<T2, T3, T4, T5, T6> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7>
struct Templates7 {
  typedef TemplateSel<T1> Head;
  typedef Templates6<T2, T3, T4, T5, T6, T7> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8>
struct Templates8 {
  typedef TemplateSel<T1> Head;
  typedef Templates7<T2, T3, T4, T5, T6, T7, T8> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9>
struct Templates9 {
  typedef TemplateSel<T1> Head;
  typedef Templates8<T2, T3, T4, T5, T6, T7, T8, T9> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10>
struct Templates10 {
  typedef TemplateSel<T1> Head;
  typedef Templates9<T2, T3, T4, T5, T6, T7, T8, T9, T10> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11>
struct Templates11 {
  typedef TemplateSel<T1> Head;
  typedef Templates10<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12>
struct Templates12 {
  typedef TemplateSel<T1> Head;
  typedef Templates11<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13>
struct Templates13 {
  typedef TemplateSel<T1> Head;
  typedef Templates12<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14>
struct Templates14 {
  typedef TemplateSel<T1> Head;
  typedef Templates13<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15>
struct Templates15 {
  typedef TemplateSel<T1> Head;
  typedef Templates14<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16>
struct Templates16 {
  typedef TemplateSel<T1> Head;
  typedef Templates15<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17>
struct Templates17 {
  typedef TemplateSel<T1> Head;
  typedef Templates16<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18>
struct Templates18 {
  typedef TemplateSel<T1> Head;
  typedef Templates17<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19>
struct Templates19 {
  typedef TemplateSel<T1> Head;
  typedef Templates18<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20>
struct Templates20 {
  typedef TemplateSel<T1> Head;
  typedef Templates19<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21>
struct Templates21 {
  typedef TemplateSel<T1> Head;
  typedef Templates20<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22>
struct Templates22 {
  typedef TemplateSel<T1> Head;
  typedef Templates21<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23>
struct Templates23 {
  typedef TemplateSel<T1> Head;
  typedef Templates22<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22, T23> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24>
struct Templates24 {
  typedef TemplateSel<T1> Head;
  typedef Templates23<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22, T23, T24> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25>
struct Templates25 {
  typedef TemplateSel<T1> Head;
  typedef Templates24<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26>
struct Templates26 {
  typedef TemplateSel<T1> Head;
  typedef Templates25<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27>
struct Templates27 {
  typedef TemplateSel<T1> Head;
  typedef Templates26<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28>
struct Templates28 {
  typedef TemplateSel<T1> Head;
  typedef Templates27<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
      T28> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29>
struct Templates29 {
  typedef TemplateSel<T1> Head;
  typedef Templates28<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
      T29> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30>
struct Templates30 {
  typedef TemplateSel<T1> Head;
  typedef Templates29<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
      T29, T30> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31>
struct Templates31 {
  typedef TemplateSel<T1> Head;
  typedef Templates30<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
      T29, T30, T31> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32>
struct Templates32 {
  typedef TemplateSel<T1> Head;
  typedef Templates31<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
      T29, T30, T31, T32> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33>
struct Templates33 {
  typedef TemplateSel<T1> Head;
  typedef Templates32<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
      T29, T30, T31, T32, T33> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34>
struct Templates34 {
  typedef TemplateSel<T1> Head;
  typedef Templates33<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
      T29, T30, T31, T32, T33, T34> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35>
struct Templates35 {
  typedef TemplateSel<T1> Head;
  typedef Templates34<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
      T29, T30, T31, T32, T33, T34, T35> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36>
struct Templates36 {
  typedef TemplateSel<T1> Head;
  typedef Templates35<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
      T29, T30, T31, T32, T33, T34, T35, T36> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36,
    GTEST_TEMPLATE_ T37>
struct Templates37 {
  typedef TemplateSel<T1> Head;
  typedef Templates36<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
      T29, T30, T31, T32, T33, T34, T35, T36, T37> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36,
    GTEST_TEMPLATE_ T37, GTEST_TEMPLATE_ T38>
struct Templates38 {
  typedef TemplateSel<T1> Head;
  typedef Templates37<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
      T29, T30, T31, T32, T33, T34, T35, T36, T37, T38> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36,
    GTEST_TEMPLATE_ T37, GTEST_TEMPLATE_ T38, GTEST_TEMPLATE_ T39>
struct Templates39 {
  typedef TemplateSel<T1> Head;
  typedef Templates38<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
      T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36,
    GTEST_TEMPLATE_ T37, GTEST_TEMPLATE_ T38, GTEST_TEMPLATE_ T39,
    GTEST_TEMPLATE_ T40>
struct Templates40 {
  typedef TemplateSel<T1> Head;
  typedef Templates39<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
      T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36,
    GTEST_TEMPLATE_ T37, GTEST_TEMPLATE_ T38, GTEST_TEMPLATE_ T39,
    GTEST_TEMPLATE_ T40, GTEST_TEMPLATE_ T41>
struct Templates41 {
  typedef TemplateSel<T1> Head;
  typedef Templates40<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
      T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36,
    GTEST_TEMPLATE_ T37, GTEST_TEMPLATE_ T38, GTEST_TEMPLATE_ T39,
    GTEST_TEMPLATE_ T40, GTEST_TEMPLATE_ T41, GTEST_TEMPLATE_ T42>
struct Templates42 {
  typedef TemplateSel<T1> Head;
  typedef Templates41<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
      T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41,
      T42> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36,
    GTEST_TEMPLATE_ T37, GTEST_TEMPLATE_ T38, GTEST_TEMPLATE_ T39,
    GTEST_TEMPLATE_ T40, GTEST_TEMPLATE_ T41, GTEST_TEMPLATE_ T42,
    GTEST_TEMPLATE_ T43>
struct Templates43 {
  typedef TemplateSel<T1> Head;
  typedef Templates42<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
      T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42,
      T43> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36,
    GTEST_TEMPLATE_ T37, GTEST_TEMPLATE_ T38, GTEST_TEMPLATE_ T39,
    GTEST_TEMPLATE_ T40, GTEST_TEMPLATE_ T41, GTEST_TEMPLATE_ T42,
    GTEST_TEMPLATE_ T43, GTEST_TEMPLATE_ T44>
struct Templates44 {
  typedef TemplateSel<T1> Head;
  typedef Templates43<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
      T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42,
      T43, T44> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36,
    GTEST_TEMPLATE_ T37, GTEST_TEMPLATE_ T38, GTEST_TEMPLATE_ T39,
    GTEST_TEMPLATE_ T40, GTEST_TEMPLATE_ T41, GTEST_TEMPLATE_ T42,
    GTEST_TEMPLATE_ T43, GTEST_TEMPLATE_ T44, GTEST_TEMPLATE_ T45>
struct Templates45 {
  typedef TemplateSel<T1> Head;
  typedef Templates44<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
      T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42,
      T43, T44, T45> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36,
    GTEST_TEMPLATE_ T37, GTEST_TEMPLATE_ T38, GTEST_TEMPLATE_ T39,
    GTEST_TEMPLATE_ T40, GTEST_TEMPLATE_ T41, GTEST_TEMPLATE_ T42,
    GTEST_TEMPLATE_ T43, GTEST_TEMPLATE_ T44, GTEST_TEMPLATE_ T45,
    GTEST_TEMPLATE_ T46>
struct Templates46 {
  typedef TemplateSel<T1> Head;
  typedef Templates45<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
      T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42,
      T43, T44, T45, T46> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36,
    GTEST_TEMPLATE_ T37, GTEST_TEMPLATE_ T38, GTEST_TEMPLATE_ T39,
    GTEST_TEMPLATE_ T40, GTEST_TEMPLATE_ T41, GTEST_TEMPLATE_ T42,
    GTEST_TEMPLATE_ T43, GTEST_TEMPLATE_ T44, GTEST_TEMPLATE_ T45,
    GTEST_TEMPLATE_ T46, GTEST_TEMPLATE_ T47>
struct Templates47 {
  typedef TemplateSel<T1> Head;
  typedef Templates46<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
      T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42,
      T43, T44, T45, T46, T47> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36,
    GTEST_TEMPLATE_ T37, GTEST_TEMPLATE_ T38, GTEST_TEMPLATE_ T39,
    GTEST_TEMPLATE_ T40, GTEST_TEMPLATE_ T41, GTEST_TEMPLATE_ T42,
    GTEST_TEMPLATE_ T43, GTEST_TEMPLATE_ T44, GTEST_TEMPLATE_ T45,
    GTEST_TEMPLATE_ T46, GTEST_TEMPLATE_ T47, GTEST_TEMPLATE_ T48>
struct Templates48 {
  typedef TemplateSel<T1> Head;
  typedef Templates47<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
      T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42,
      T43, T44, T45, T46, T47, T48> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36,
    GTEST_TEMPLATE_ T37, GTEST_TEMPLATE_ T38, GTEST_TEMPLATE_ T39,
    GTEST_TEMPLATE_ T40, GTEST_TEMPLATE_ T41, GTEST_TEMPLATE_ T42,
    GTEST_TEMPLATE_ T43, GTEST_TEMPLATE_ T44, GTEST_TEMPLATE_ T45,
    GTEST_TEMPLATE_ T46, GTEST_TEMPLATE_ T47, GTEST_TEMPLATE_ T48,
    GTEST_TEMPLATE_ T49>
struct Templates49 {
  typedef TemplateSel<T1> Head;
  typedef Templates48<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
      T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42,
      T43, T44, T45, T46, T47, T48, T49> Tail;
};

template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36,
    GTEST_TEMPLATE_ T37, GTEST_TEMPLATE_ T38, GTEST_TEMPLATE_ T39,
    GTEST_TEMPLATE_ T40, GTEST_TEMPLATE_ T41, GTEST_TEMPLATE_ T42,
    GTEST_TEMPLATE_ T43, GTEST_TEMPLATE_ T44, GTEST_TEMPLATE_ T45,
    GTEST_TEMPLATE_ T46, GTEST_TEMPLATE_ T47, GTEST_TEMPLATE_ T48,
    GTEST_TEMPLATE_ T49, GTEST_TEMPLATE_ T50>
struct Templates50 {
  typedef TemplateSel<T1> Head;
  typedef Templates49<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
      T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
      T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42,
      T43, T44, T45, T46, T47, T48, T49, T50> Tail;
};


// We don't want to require the users to write TemplatesN<...> directly,
// as that would require them to count the length.  Templates<...> is much
// easier to write, but generates horrible messages when there is a
// compiler error, as gcc insists on printing out each template
// argument, even if it has the default value (this means Templates<list>
// will appear as Templates<list, NoneT, NoneT, ..., NoneT> in the compiler
// errors).
//
// Our solution is to combine the best part of the two approaches: a
// user would write Templates<T1, ..., TN>, and Google Test will translate
// that to TemplatesN<T1, ..., TN> internally to make error messages
// readable.  The translation is done by the 'type' member of the
// Templates template.
template <GTEST_TEMPLATE_ T1 = NoneT, GTEST_TEMPLATE_ T2 = NoneT,
    GTEST_TEMPLATE_ T3 = NoneT, GTEST_TEMPLATE_ T4 = NoneT,
    GTEST_TEMPLATE_ T5 = NoneT, GTEST_TEMPLATE_ T6 = NoneT,
    GTEST_TEMPLATE_ T7 = NoneT, GTEST_TEMPLATE_ T8 = NoneT,
    GTEST_TEMPLATE_ T9 = NoneT, GTEST_TEMPLATE_ T10 = NoneT,
    GTEST_TEMPLATE_ T11 = NoneT, GTEST_TEMPLATE_ T12 = NoneT,
    GTEST_TEMPLATE_ T13 = NoneT, GTEST_TEMPLATE_ T14 = NoneT,
    GTEST_TEMPLATE_ T15 = NoneT, GTEST_TEMPLATE_ T16 = NoneT,
    GTEST_TEMPLATE_ T17 = NoneT, GTEST_TEMPLATE_ T18 = NoneT,
    GTEST_TEMPLATE_ T19 = NoneT, GTEST_TEMPLATE_ T20 = NoneT,
    GTEST_TEMPLATE_ T21 = NoneT, GTEST_TEMPLATE_ T22 = NoneT,
    GTEST_TEMPLATE_ T23 = NoneT, GTEST_TEMPLATE_ T24 = NoneT,
    GTEST_TEMPLATE_ T25 = NoneT, GTEST_TEMPLATE_ T26 = NoneT,
    GTEST_TEMPLATE_ T27 = NoneT, GTEST_TEMPLATE_ T28 = NoneT,
    GTEST_TEMPLATE_ T29 = NoneT, GTEST_TEMPLATE_ T30 = NoneT,
    GTEST_TEMPLATE_ T31 = NoneT, GTEST_TEMPLATE_ T32 = NoneT,
    GTEST_TEMPLATE_ T33 = NoneT, GTEST_TEMPLATE_ T34 = NoneT,
    GTEST_TEMPLATE_ T35 = NoneT, GTEST_TEMPLATE_ T36 = NoneT,
    GTEST_TEMPLATE_ T37 = NoneT, GTEST_TEMPLATE_ T38 = NoneT,
    GTEST_TEMPLATE_ T39 = NoneT, GTEST_TEMPLATE_ T40 = NoneT,
    GTEST_TEMPLATE_ T41 = NoneT, GTEST_TEMPLATE_ T42 = NoneT,
    GTEST_TEMPLATE_ T43 = NoneT, GTEST_TEMPLATE_ T44 = NoneT,
    GTEST_TEMPLATE_ T45 = NoneT, GTEST_TEMPLATE_ T46 = NoneT,
    GTEST_TEMPLATE_ T47 = NoneT, GTEST_TEMPLATE_ T48 = NoneT,
    GTEST_TEMPLATE_ T49 = NoneT, GTEST_TEMPLATE_ T50 = NoneT>
struct Templates {
  typedef Templates50<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
      T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41,
      T42, T43, T44, T45, T46, T47, T48, T49, T50> type;
};

template <>
struct Templates<NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT> {
  typedef Templates0 type;
};
template <GTEST_TEMPLATE_ T1>
struct Templates<T1, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT> {
  typedef Templates1<T1> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2>
struct Templates<T1, T2, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT> {
  typedef Templates2<T1, T2> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3>
struct Templates<T1, T2, T3, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates3<T1, T2, T3> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4>
struct Templates<T1, T2, T3, T4, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates4<T1, T2, T3, T4> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5>
struct Templates<T1, T2, T3, T4, T5, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates5<T1, T2, T3, T4, T5> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6>
struct Templates<T1, T2, T3, T4, T5, T6, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates6<T1, T2, T3, T4, T5, T6> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7>
struct Templates<T1, T2, T3, T4, T5, T6, T7, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates7<T1, T2, T3, T4, T5, T6, T7> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates8<T1, T2, T3, T4, T5, T6, T7, T8> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates9<T1, T2, T3, T4, T5, T6, T7, T8, T9> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates10<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates11<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates12<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates13<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates14<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates15<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates16<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates17<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT> {
  typedef Templates18<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT> {
  typedef Templates19<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT> {
  typedef Templates20<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT> {
  typedef Templates21<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, T22, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT> {
  typedef Templates22<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, T22, T23, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT> {
  typedef Templates23<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22, T23> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT> {
  typedef Templates24<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT> {
  typedef Templates25<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT> {
  typedef Templates26<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT> {
  typedef Templates27<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
      T27> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT> {
  typedef Templates28<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
      T28> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT> {
  typedef Templates29<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
      T28, T29> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
    T30, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates30<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
      T28, T29, T30> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
    T30, T31, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates31<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
      T28, T29, T30, T31> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
    T30, T31, T32, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates32<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
      T28, T29, T30, T31, T32> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
    T30, T31, T32, T33, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates33<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
      T28, T29, T30, T31, T32, T33> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
    T30, T31, T32, T33, T34, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates34<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
      T28, T29, T30, T31, T32, T33, T34> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
    T30, T31, T32, T33, T34, T35, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates35<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
      T28, T29, T30, T31, T32, T33, T34, T35> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
    T30, T31, T32, T33, T34, T35, T36, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates36<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
      T28, T29, T30, T31, T32, T33, T34, T35, T36> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36,
    GTEST_TEMPLATE_ T37>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
    T30, T31, T32, T33, T34, T35, T36, T37, NoneT, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates37<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
      T28, T29, T30, T31, T32, T33, T34, T35, T36, T37> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36,
    GTEST_TEMPLATE_ T37, GTEST_TEMPLATE_ T38>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
    T30, T31, T32, T33, T34, T35, T36, T37, T38, NoneT, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates38<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
      T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36,
    GTEST_TEMPLATE_ T37, GTEST_TEMPLATE_ T38, GTEST_TEMPLATE_ T39>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
    T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates39<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
      T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36,
    GTEST_TEMPLATE_ T37, GTEST_TEMPLATE_ T38, GTEST_TEMPLATE_ T39,
    GTEST_TEMPLATE_ T40>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
    T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, NoneT, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates40<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
      T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36,
    GTEST_TEMPLATE_ T37, GTEST_TEMPLATE_ T38, GTEST_TEMPLATE_ T39,
    GTEST_TEMPLATE_ T40, GTEST_TEMPLATE_ T41>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
    T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, NoneT, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates41<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
      T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40,
      T41> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36,
    GTEST_TEMPLATE_ T37, GTEST_TEMPLATE_ T38, GTEST_TEMPLATE_ T39,
    GTEST_TEMPLATE_ T40, GTEST_TEMPLATE_ T41, GTEST_TEMPLATE_ T42>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
    T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, NoneT,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates42<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
      T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41,
      T42> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36,
    GTEST_TEMPLATE_ T37, GTEST_TEMPLATE_ T38, GTEST_TEMPLATE_ T39,
    GTEST_TEMPLATE_ T40, GTEST_TEMPLATE_ T41, GTEST_TEMPLATE_ T42,
    GTEST_TEMPLATE_ T43>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
    T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates43<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
      T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41,
      T42, T43> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36,
    GTEST_TEMPLATE_ T37, GTEST_TEMPLATE_ T38, GTEST_TEMPLATE_ T39,
    GTEST_TEMPLATE_ T40, GTEST_TEMPLATE_ T41, GTEST_TEMPLATE_ T42,
    GTEST_TEMPLATE_ T43, GTEST_TEMPLATE_ T44>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
    T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44,
    NoneT, NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates44<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
      T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41,
      T42, T43, T44> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36,
    GTEST_TEMPLATE_ T37, GTEST_TEMPLATE_ T38, GTEST_TEMPLATE_ T39,
    GTEST_TEMPLATE_ T40, GTEST_TEMPLATE_ T41, GTEST_TEMPLATE_ T42,
    GTEST_TEMPLATE_ T43, GTEST_TEMPLATE_ T44, GTEST_TEMPLATE_ T45>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
    T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44,
    T45, NoneT, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates45<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
      T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41,
      T42, T43, T44, T45> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36,
    GTEST_TEMPLATE_ T37, GTEST_TEMPLATE_ T38, GTEST_TEMPLATE_ T39,
    GTEST_TEMPLATE_ T40, GTEST_TEMPLATE_ T41, GTEST_TEMPLATE_ T42,
    GTEST_TEMPLATE_ T43, GTEST_TEMPLATE_ T44, GTEST_TEMPLATE_ T45,
    GTEST_TEMPLATE_ T46>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
    T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44,
    T45, T46, NoneT, NoneT, NoneT, NoneT> {
  typedef Templates46<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
      T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41,
      T42, T43, T44, T45, T46> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36,
    GTEST_TEMPLATE_ T37, GTEST_TEMPLATE_ T38, GTEST_TEMPLATE_ T39,
    GTEST_TEMPLATE_ T40, GTEST_TEMPLATE_ T41, GTEST_TEMPLATE_ T42,
    GTEST_TEMPLATE_ T43, GTEST_TEMPLATE_ T44, GTEST_TEMPLATE_ T45,
    GTEST_TEMPLATE_ T46, GTEST_TEMPLATE_ T47>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
    T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44,
    T45, T46, T47, NoneT, NoneT, NoneT> {
  typedef Templates47<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
      T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41,
      T42, T43, T44, T45, T46, T47> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36,
    GTEST_TEMPLATE_ T37, GTEST_TEMPLATE_ T38, GTEST_TEMPLATE_ T39,
    GTEST_TEMPLATE_ T40, GTEST_TEMPLATE_ T41, GTEST_TEMPLATE_ T42,
    GTEST_TEMPLATE_ T43, GTEST_TEMPLATE_ T44, GTEST_TEMPLATE_ T45,
    GTEST_TEMPLATE_ T46, GTEST_TEMPLATE_ T47, GTEST_TEMPLATE_ T48>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
    T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44,
    T45, T46, T47, T48, NoneT, NoneT> {
  typedef Templates48<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
      T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41,
      T42, T43, T44, T45, T46, T47, T48> type;
};
template <GTEST_TEMPLATE_ T1, GTEST_TEMPLATE_ T2, GTEST_TEMPLATE_ T3,
    GTEST_TEMPLATE_ T4, GTEST_TEMPLATE_ T5, GTEST_TEMPLATE_ T6,
    GTEST_TEMPLATE_ T7, GTEST_TEMPLATE_ T8, GTEST_TEMPLATE_ T9,
    GTEST_TEMPLATE_ T10, GTEST_TEMPLATE_ T11, GTEST_TEMPLATE_ T12,
    GTEST_TEMPLATE_ T13, GTEST_TEMPLATE_ T14, GTEST_TEMPLATE_ T15,
    GTEST_TEMPLATE_ T16, GTEST_TEMPLATE_ T17, GTEST_TEMPLATE_ T18,
    GTEST_TEMPLATE_ T19, GTEST_TEMPLATE_ T20, GTEST_TEMPLATE_ T21,
    GTEST_TEMPLATE_ T22, GTEST_TEMPLATE_ T23, GTEST_TEMPLATE_ T24,
    GTEST_TEMPLATE_ T25, GTEST_TEMPLATE_ T26, GTEST_TEMPLATE_ T27,
    GTEST_TEMPLATE_ T28, GTEST_TEMPLATE_ T29, GTEST_TEMPLATE_ T30,
    GTEST_TEMPLATE_ T31, GTEST_TEMPLATE_ T32, GTEST_TEMPLATE_ T33,
    GTEST_TEMPLATE_ T34, GTEST_TEMPLATE_ T35, GTEST_TEMPLATE_ T36,
    GTEST_TEMPLATE_ T37, GTEST_TEMPLATE_ T38, GTEST_TEMPLATE_ T39,
    GTEST_TEMPLATE_ T40, GTEST_TEMPLATE_ T41, GTEST_TEMPLATE_ T42,
    GTEST_TEMPLATE_ T43, GTEST_TEMPLATE_ T44, GTEST_TEMPLATE_ T45,
    GTEST_TEMPLATE_ T46, GTEST_TEMPLATE_ T47, GTEST_TEMPLATE_ T48,
    GTEST_TEMPLATE_ T49>
struct Templates<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
    T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29,
    T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44,
    T45, T46, T47, T48, T49, NoneT> {
  typedef Templates49<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
      T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
      T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41,
      T42, T43, T44, T45, T46, T47, T48, T49> type;
};

// The TypeList template makes it possible to use either a single type
// or a Types<...> list in TYPED_TEST_CASE() and
// INSTANTIATE_TYPED_TEST_CASE_P().

template <typename T>
struct TypeList { typedef Types1<T> type; };

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43, typename T44, typename T45,
    typename T46, typename T47, typename T48, typename T49, typename T50>
struct TypeList<Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
    T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43,
    T44, T45, T46, T47, T48, T49, T50> > {
  typedef typename Types<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
      T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
      T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40,
      T41, T42, T43, T44, T45, T46, T47, T48, T49, T50>::type type;
};

#endif  // GTEST_HAS_TYPED_TEST || GTEST_HAS_TYPED_TEST_P

}  // namespace internal
}  // namespace testing

#endif  // GTEST_INCLUDE_GTEST_INTERNAL_GTEST_TYPE_UTIL_H_

// Due to C++ preprocessor weirdness, we need double indirection to
// concatenate two tokens when one of them is __LINE__.  Writing
//
//   foo ## __LINE__
//
// will result in the token foo__LINE__, instead of foo followed by
// the current line number.  For more details, see
// http://www.parashift.com/c++-faq-lite/misc-technical-issues.html#faq-39.6
#define GTEST_CONCAT_TOKEN_(foo, bar) GTEST_CONCAT_TOKEN_IMPL_(foo, bar)
#define GTEST_CONCAT_TOKEN_IMPL_(foo, bar) foo ## bar

// Google Test defines the testing::Message class to allow construction of
// test messages via the << operator.  The idea is that anything
// streamable to std::ostream can be streamed to a testing::Message.
// This allows a user to use his own types in Google Test assertions by
// overloading the << operator.
//
// util/gtl/stl_logging-inl.h overloads << for STL containers.  These
// overloads cannot be defined in the std namespace, as that will be
// undefined behavior.  Therefore, they are defined in the global
// namespace instead.
//
// C++'s symbol lookup rule (i.e. Koenig lookup) says that these
// overloads are visible in either the std namespace or the global
// namespace, but not other namespaces, including the testing
// namespace which Google Test's Message class is in.
//
// To allow STL containers (and other types that has a << operator
// defined in the global namespace) to be used in Google Test assertions,
// testing::Message must access the custom << operator from the global
// namespace.  Hence this helper function.
//
// Note: Jeffrey Yasskin suggested an alternative fix by "using
// ::operator<<;" in the definition of Message's operator<<.  That fix
// doesn't require a helper function, but unfortunately doesn't
// compile with MSVC.
template <typename T>
inline void GTestStreamToHelper(std::ostream* os, const T& val) {
  *os << val;
}

class ProtocolMessage;
namespace proto2 { class Message; }

namespace testing {

// Forward declarations.

class AssertionResult;                 // Result of an assertion.
class Message;                         // Represents a failure message.
class Test;                            // Represents a test.
class TestInfo;                        // Information about a test.
class TestPartResult;                  // Result of a test part.
class UnitTest;                        // A collection of test cases.

template <typename T>
::std::string PrintToString(const T& value);

namespace internal {

struct TraceInfo;                      // Information about a trace point.
class ScopedTrace;                     // Implements scoped trace.
class TestInfoImpl;                    // Opaque implementation of TestInfo
class UnitTestImpl;                    // Opaque implementation of UnitTest

// How many times InitGoogleTest() has been called.
extern int g_init_gtest_count;

// The text used in failure messages to indicate the start of the
// stack trace.
GTEST_API_ extern const char kStackTraceMarker[];

// A secret type that Google Test users don't know about.  It has no
// definition on purpose.  Therefore it's impossible to create a
// Secret object, which is what we want.
class Secret;

// Two overloaded helpers for checking at compile time whether an
// expression is a null pointer literal (i.e. NULL or any 0-valued
// compile-time integral constant).  Their return values have
// different sizes, so we can use sizeof() to test which version is
// picked by the compiler.  These helpers have no implementations, as
// we only need their signatures.
//
// Given IsNullLiteralHelper(x), the compiler will pick the first
// version if x can be implicitly converted to Secret*, and pick the
// second version otherwise.  Since Secret is a secret and incomplete
// type, the only expression a user can write that has type Secret* is
// a null pointer literal.  Therefore, we know that x is a null
// pointer literal if and only if the first version is picked by the
// compiler.
char IsNullLiteralHelper(Secret* p);
char (&IsNullLiteralHelper(...))[2];  // NOLINT

// A compile-time bool constant that is true if and only if x is a
// null pointer literal (i.e. NULL or any 0-valued compile-time
// integral constant).
#ifdef GTEST_ELLIPSIS_NEEDS_POD_
// We lose support for NULL detection where the compiler doesn't like
// passing non-POD classes through ellipsis (...).
# define GTEST_IS_NULL_LITERAL_(x) false
#else
# define GTEST_IS_NULL_LITERAL_(x) \
    (sizeof(::testing::internal::IsNullLiteralHelper(x)) == 1)
#endif  // GTEST_ELLIPSIS_NEEDS_POD_

// Appends the user-supplied message to the Google-Test-generated message.
GTEST_API_ String AppendUserMessage(const String& gtest_msg,
                                    const Message& user_msg);

// A helper class for creating scoped traces in user programs.
class GTEST_API_ ScopedTrace {
 public:
  // The c'tor pushes the given source file location and message onto
  // a trace stack maintained by Google Test.
  ScopedTrace(const char* file, int line, const Message& message);

  // The d'tor pops the info pushed by the c'tor.
  //
  // Note that the d'tor is not virtual in order to be efficient.
  // Don't inherit from ScopedTrace!
  ~ScopedTrace();

 private:
  GTEST_DISALLOW_COPY_AND_ASSIGN_(ScopedTrace);
} GTEST_ATTRIBUTE_UNUSED_;  // A ScopedTrace object does its job in its
                            // c'tor and d'tor.  Therefore it doesn't
                            // need to be used otherwise.

// Converts a streamable value to a String.  A NULL pointer is
// converted to "(null)".  When the input value is a ::string,
// ::std::string, ::wstring, or ::std::wstring object, each NUL
// character in it is replaced with "\\0".
// Declared here but defined in gtest.h, so that it has access
// to the definition of the Message class, required by the ARM
// compiler.
template <typename T>
String StreamableToString(const T& streamable);

// The Symbian compiler has a bug that prevents it from selecting the
// correct overload of FormatForComparisonFailureMessage (see below)
// unless we pass the first argument by reference.  If we do that,
// however, Visual Age C++ 10.1 generates a compiler error.  Therefore
// we only apply the work-around for Symbian.
#if defined(__SYMBIAN32__)
# define GTEST_CREF_WORKAROUND_ const&
#else
# define GTEST_CREF_WORKAROUND_
#endif

// When this operand is a const char* or char*, if the other operand
// is a ::std::string or ::string, we print this operand as a C string
// rather than a pointer (we do the same for wide strings); otherwise
// we print it as a pointer to be safe.

// This internal macro is used to avoid duplicated code.
#define GTEST_FORMAT_IMPL_(operand2_type, operand1_printer)\
inline String FormatForComparisonFailureMessage(\
    operand2_type::value_type* GTEST_CREF_WORKAROUND_ str, \
    const operand2_type& /*operand2*/) {\
  return operand1_printer(str);\
}\
inline String FormatForComparisonFailureMessage(\
    const operand2_type::value_type* GTEST_CREF_WORKAROUND_ str, \
    const operand2_type& /*operand2*/) {\
  return operand1_printer(str);\
}

GTEST_FORMAT_IMPL_(::std::string, String::ShowCStringQuoted)
#if GTEST_HAS_STD_WSTRING
GTEST_FORMAT_IMPL_(::std::wstring, String::ShowWideCStringQuoted)
#endif  // GTEST_HAS_STD_WSTRING

#if GTEST_HAS_GLOBAL_STRING
GTEST_FORMAT_IMPL_(::string, String::ShowCStringQuoted)
#endif  // GTEST_HAS_GLOBAL_STRING
#if GTEST_HAS_GLOBAL_WSTRING
GTEST_FORMAT_IMPL_(::wstring, String::ShowWideCStringQuoted)
#endif  // GTEST_HAS_GLOBAL_WSTRING

#undef GTEST_FORMAT_IMPL_

// The next four overloads handle the case where the operand being
// printed is a char/wchar_t pointer and the other operand is not a
// string/wstring object.  In such cases, we just print the operand as
// a pointer to be safe.
#define GTEST_FORMAT_CHAR_PTR_IMPL_(CharType)                       \
  template <typename T>                                             \
  String FormatForComparisonFailureMessage(CharType* GTEST_CREF_WORKAROUND_ p, \
                                           const T&) { \
    return PrintToString(static_cast<const void*>(p));              \
  }

GTEST_FORMAT_CHAR_PTR_IMPL_(char)
GTEST_FORMAT_CHAR_PTR_IMPL_(const char)
GTEST_FORMAT_CHAR_PTR_IMPL_(wchar_t)
GTEST_FORMAT_CHAR_PTR_IMPL_(const wchar_t)

#undef GTEST_FORMAT_CHAR_PTR_IMPL_

// Constructs and returns the message for an equality assertion
// (e.g. ASSERT_EQ, EXPECT_STREQ, etc) failure.
//
// The first four parameters are the expressions used in the assertion
// and their values, as strings.  For example, for ASSERT_EQ(foo, bar)
// where foo is 5 and bar is 6, we have:
//
//   expected_expression: "foo"
//   actual_expression:   "bar"
//   expected_value:      "5"
//   actual_value:        "6"
//
// The ignoring_case parameter is true iff the assertion is a
// *_STRCASEEQ*.  When it's true, the string " (ignoring case)" will
// be inserted into the message.
GTEST_API_ AssertionResult EqFailure(const char* expected_expression,
                                     const char* actual_expression,
                                     const String& expected_value,
                                     const String& actual_value,
                                     bool ignoring_case);

// Constructs a failure message for Boolean assertions such as EXPECT_TRUE.
GTEST_API_ String GetBoolAssertionFailureMessage(
    const AssertionResult& assertion_result,
    const char* expression_text,
    const char* actual_predicate_value,
    const char* expected_predicate_value);

// This template class represents an IEEE floating-point number
// (either single-precision or double-precision, depending on the
// template parameters).
//
// The purpose of this class is to do more sophisticated number
// comparison.  (Due to round-off error, etc, it's very unlikely that
// two floating-points will be equal exactly.  Hence a naive
// comparison by the == operation often doesn't work.)
//
// Format of IEEE floating-point:
//
//   The most-significant bit being the leftmost, an IEEE
//   floating-point looks like
//
//     sign_bit exponent_bits fraction_bits
//
//   Here, sign_bit is a single bit that designates the sign of the
//   number.
//
//   For float, there are 8 exponent bits and 23 fraction bits.
//
//   For double, there are 11 exponent bits and 52 fraction bits.
//
//   More details can be found at
//   http://en.wikipedia.org/wiki/IEEE_floating-point_standard.
//
// Template parameter:
//
//   RawType: the raw floating-point type (either float or double)
template <typename RawType>
class FloatingPoint {
 public:
  // Defines the unsigned integer type that has the same size as the
  // floating point number.
  typedef typename TypeWithSize<sizeof(RawType)>::UInt Bits;

  // Constants.

  // # of bits in a number.
  static const size_t kBitCount = 8*sizeof(RawType);

  // # of fraction bits in a number.
  static const size_t kFractionBitCount =
    std::numeric_limits<RawType>::digits - 1;

  // # of exponent bits in a number.
  static const size_t kExponentBitCount = kBitCount - 1 - kFractionBitCount;

  // The mask for the sign bit.
  static const Bits kSignBitMask = static_cast<Bits>(1) << (kBitCount - 1);

  // The mask for the fraction bits.
  static const Bits kFractionBitMask =
    ~static_cast<Bits>(0) >> (kExponentBitCount + 1);

  // The mask for the exponent bits.
  static const Bits kExponentBitMask = ~(kSignBitMask | kFractionBitMask);

  // How many ULP's (Units in the Last Place) we want to tolerate when
  // comparing two numbers.  The larger the value, the more error we
  // allow.  A 0 value means that two numbers must be exactly the same
  // to be considered equal.
  //
  // The maximum error of a single floating-point operation is 0.5
  // units in the last place.  On Intel CPU's, all floating-point
  // calculations are done with 80-bit precision, while double has 64
  // bits.  Therefore, 4 should be enough for ordinary use.
  //
  // See the following article for more details on ULP:
  // http://www.cygnus-software.com/papers/comparingfloats/comparingfloats.htm.
  static const size_t kMaxUlps = 4;

  // Constructs a FloatingPoint from a raw floating-point number.
  //
  // On an Intel CPU, passing a non-normalized NAN (Not a Number)
  // around may change its bits, although the new value is guaranteed
  // to be also a NAN.  Therefore, don't expect this constructor to
  // preserve the bits in x when x is a NAN.
  explicit FloatingPoint(const RawType& x) { u_.value_ = x; }

  // Static methods

  // Reinterprets a bit pattern as a floating-point number.
  //
  // This function is needed to test the AlmostEquals() method.
  static RawType ReinterpretBits(const Bits bits) {
    FloatingPoint fp(0);
    fp.u_.bits_ = bits;
    return fp.u_.value_;
  }

  // Returns the floating-point number that represent positive infinity.
  static RawType Infinity() {
    return ReinterpretBits(kExponentBitMask);
  }

  // Non-static methods

  // Returns the bits that represents this number.
  const Bits &bits() const { return u_.bits_; }

  // Returns the exponent bits of this number.
  Bits exponent_bits() const { return kExponentBitMask & u_.bits_; }

  // Returns the fraction bits of this number.
  Bits fraction_bits() const { return kFractionBitMask & u_.bits_; }

  // Returns the sign bit of this number.
  Bits sign_bit() const { return kSignBitMask & u_.bits_; }

  // Returns true iff this is NAN (not a number).
  bool is_nan() const {
    // It's a NAN if the exponent bits are all ones and the fraction
    // bits are not entirely zeros.
    return (exponent_bits() == kExponentBitMask) && (fraction_bits() != 0);
  }

  // Returns true iff this number is at most kMaxUlps ULP's away from
  // rhs.  In particular, this function:
  //
  //   - returns false if either number is (or both are) NAN.
  //   - treats really large numbers as almost equal to infinity.
  //   - thinks +0.0 and -0.0 are 0 DLP's apart.
  bool AlmostEquals(const FloatingPoint& rhs) const {
    // The IEEE standard says that any comparison operation involving
    // a NAN must return false.
    if (is_nan() || rhs.is_nan()) return false;

    return DistanceBetweenSignAndMagnitudeNumbers(u_.bits_, rhs.u_.bits_)
        <= kMaxUlps;
  }

 private:
  // The data type used to store the actual floating-point number.
  union FloatingPointUnion {
    RawType value_;  // The raw floating-point number.
    Bits bits_;      // The bits that represent the number.
  };

  // Converts an integer from the sign-and-magnitude representation to
  // the biased representation.  More precisely, let N be 2 to the
  // power of (kBitCount - 1), an integer x is represented by the
  // unsigned number x + N.
  //
  // For instance,
  //
  //   -N + 1 (the most negative number representable using
  //          sign-and-magnitude) is represented by 1;
  //   0      is represented by N; and
  //   N - 1  (the biggest number representable using
  //          sign-and-magnitude) is represented by 2N - 1.
  //
  // Read http://en.wikipedia.org/wiki/Signed_number_representations
  // for more details on signed number representations.
  static Bits SignAndMagnitudeToBiased(const Bits &sam) {
    if (kSignBitMask & sam) {
      // sam represents a negative number.
      return ~sam + 1;
    } else {
      // sam represents a positive number.
      return kSignBitMask | sam;
    }
  }

  // Given two numbers in the sign-and-magnitude representation,
  // returns the distance between them as an unsigned number.
  static Bits DistanceBetweenSignAndMagnitudeNumbers(const Bits &sam1,
                                                     const Bits &sam2) {
    const Bits biased1 = SignAndMagnitudeToBiased(sam1);
    const Bits biased2 = SignAndMagnitudeToBiased(sam2);
    return (biased1 >= biased2) ? (biased1 - biased2) : (biased2 - biased1);
  }

  FloatingPointUnion u_;
};

// Typedefs the instances of the FloatingPoint template class that we
// care to use.
typedef FloatingPoint<float> Float;
typedef FloatingPoint<double> Double;

// In order to catch the mistake of putting tests that use different
// test fixture classes in the same test case, we need to assign
// unique IDs to fixture classes and compare them.  The TypeId type is
// used to hold such IDs.  The user should treat TypeId as an opaque
// type: the only operation allowed on TypeId values is to compare
// them for equality using the == operator.
typedef const void* TypeId;

template <typename T>
class TypeIdHelper {
 public:
  // dummy_ must not have a const type.  Otherwise an overly eager
  // compiler (e.g. MSVC 7.1 & 8.0) may try to merge
  // TypeIdHelper<T>::dummy_ for different Ts as an "optimization".
  static bool dummy_;
};

template <typename T>
bool TypeIdHelper<T>::dummy_ = false;

// GetTypeId<T>() returns the ID of type T.  Different values will be
// returned for different types.  Calling the function twice with the
// same type argument is guaranteed to return the same ID.
template <typename T>
TypeId GetTypeId() {
  // The compiler is required to allocate a different
  // TypeIdHelper<T>::dummy_ variable for each T used to instantiate
  // the template.  Therefore, the address of dummy_ is guaranteed to
  // be unique.
  return &(TypeIdHelper<T>::dummy_);
}

// Returns the type ID of ::testing::Test.  Always call this instead
// of GetTypeId< ::testing::Test>() to get the type ID of
// ::testing::Test, as the latter may give the wrong result due to a
// suspected linker bug when compiling Google Test as a Mac OS X
// framework.
GTEST_API_ TypeId GetTestTypeId();

// Defines the abstract factory interface that creates instances
// of a Test object.
class TestFactoryBase {
 public:
  virtual ~TestFactoryBase() {}

  // Creates a test instance to run. The instance is both created and destroyed
  // within TestInfoImpl::Run()
  virtual Test* CreateTest() = 0;

 protected:
  TestFactoryBase() {}

 private:
  GTEST_DISALLOW_COPY_AND_ASSIGN_(TestFactoryBase);
};

// This class provides implementation of TeastFactoryBase interface.
// It is used in TEST and TEST_F macros.
template <class TestClass>
class TestFactoryImpl : public TestFactoryBase {
 public:
  virtual Test* CreateTest() { return new TestClass; }
};

#if GTEST_OS_WINDOWS

// Predicate-formatters for implementing the HRESULT checking macros
// {ASSERT|EXPECT}_HRESULT_{SUCCEEDED|FAILED}
// We pass a long instead of HRESULT to avoid causing an
// include dependency for the HRESULT type.
GTEST_API_ AssertionResult IsHRESULTSuccess(const char* expr,
                                            long hr);  // NOLINT
GTEST_API_ AssertionResult IsHRESULTFailure(const char* expr,
                                            long hr);  // NOLINT

#endif  // GTEST_OS_WINDOWS

// Types of SetUpTestCase() and TearDownTestCase() functions.
typedef void (*SetUpTestCaseFunc)();
typedef void (*TearDownTestCaseFunc)();

// Creates a new TestInfo object and registers it with Google Test;
// returns the created object.
//
// Arguments:
//
//   test_case_name:   name of the test case
//   name:             name of the test
//   type_param        the name of the test's type parameter, or NULL if
//                     this is not  a typed or a type-parameterized test.
//   value_param       text representation of the test's value parameter,
//                     or NULL if this is not a type-parameterized test.
//   fixture_class_id: ID of the test fixture class
//   set_up_tc:        pointer to the function that sets up the test case
//   tear_down_tc:     pointer to the function that tears down the test case
//   factory:          pointer to the factory that creates a test object.
//                     The newly created TestInfo instance will assume
//                     ownership of the factory object.
GTEST_API_ TestInfo* MakeAndRegisterTestInfo(
    const char* test_case_name, const char* name,
    const char* type_param,
    const char* value_param,
    TypeId fixture_class_id,
    SetUpTestCaseFunc set_up_tc,
    TearDownTestCaseFunc tear_down_tc,
    TestFactoryBase* factory);

// If *pstr starts with the given prefix, modifies *pstr to be right
// past the prefix and returns true; otherwise leaves *pstr unchanged
// and returns false.  None of pstr, *pstr, and prefix can be NULL.
GTEST_API_ bool SkipPrefix(const char* prefix, const char** pstr);

#if GTEST_HAS_TYPED_TEST || GTEST_HAS_TYPED_TEST_P

// State of the definition of a type-parameterized test case.
class GTEST_API_ TypedTestCasePState {
 public:
  TypedTestCasePState() : registered_(false) {}

  // Adds the given test name to defined_test_names_ and return true
  // if the test case hasn't been registered; otherwise aborts the
  // program.
  bool AddTestName(const char* file, int line, const char* case_name,
                   const char* test_name) {
    if (registered_) {
      fprintf(stderr, "%s Test %s must be defined before "
              "REGISTER_TYPED_TEST_CASE_P(%s, ...).\n",
              FormatFileLocation(file, line).c_str(), test_name, case_name);
      fflush(stderr);
      posix::Abort();
    }
    defined_test_names_.insert(test_name);
    return true;
  }

  // Verifies that registered_tests match the test names in
  // defined_test_names_; returns registered_tests if successful, or
  // aborts the program otherwise.
  const char* VerifyRegisteredTestNames(
      const char* file, int line, const char* registered_tests);

 private:
  bool registered_;
  ::std::set<const char*> defined_test_names_;
};

// Skips to the first non-space char after the first comma in 'str';
// returns NULL if no comma is found in 'str'.
inline const char* SkipComma(const char* str) {
  const char* comma = strchr(str, ',');
  if (comma == NULL) {
    return NULL;
  }
  while (IsSpace(*(++comma))) {}
  return comma;
}

// Returns the prefix of 'str' before the first comma in it; returns
// the entire string if it contains no comma.
inline String GetPrefixUntilComma(const char* str) {
  const char* comma = strchr(str, ',');
  return comma == NULL ? String(str) : String(str, comma - str);
}

// TypeParameterizedTest<Fixture, TestSel, Types>::Register()
// registers a list of type-parameterized tests with Google Test.  The
// return value is insignificant - we just need to return something
// such that we can call this function in a namespace scope.
//
// Implementation note: The GTEST_TEMPLATE_ macro declares a template
// template parameter.  It's defined in gtest-type-util.h.
template <GTEST_TEMPLATE_ Fixture, class TestSel, typename Types>
class TypeParameterizedTest {
 public:
  // 'index' is the index of the test in the type list 'Types'
  // specified in INSTANTIATE_TYPED_TEST_CASE_P(Prefix, TestCase,
  // Types).  Valid values for 'index' are [0, N - 1] where N is the
  // length of Types.
  static bool Register(const char* prefix, const char* case_name,
                       const char* test_names, int index) {
    typedef typename Types::Head Type;
    typedef Fixture<Type> FixtureClass;
    typedef typename GTEST_BIND_(TestSel, Type) TestClass;

    // First, registers the first type-parameterized test in the type
    // list.
    MakeAndRegisterTestInfo(
        String::Format("%s%s%s/%d", prefix, prefix[0] == '\0' ? "" : "/",
                       case_name, index).c_str(),
        GetPrefixUntilComma(test_names).c_str(),
        GetTypeName<Type>().c_str(),
        NULL,  // No value parameter.
        GetTypeId<FixtureClass>(),
        TestClass::SetUpTestCase,
        TestClass::TearDownTestCase,
        new TestFactoryImpl<TestClass>);

    // Next, recurses (at compile time) with the tail of the type list.
    return TypeParameterizedTest<Fixture, TestSel, typename Types::Tail>
        ::Register(prefix, case_name, test_names, index + 1);
  }
};

// The base case for the compile time recursion.
template <GTEST_TEMPLATE_ Fixture, class TestSel>
class TypeParameterizedTest<Fixture, TestSel, Types0> {
 public:
  static bool Register(const char* /*prefix*/, const char* /*case_name*/,
                       const char* /*test_names*/, int /*index*/) {
    return true;
  }
};

// TypeParameterizedTestCase<Fixture, Tests, Types>::Register()
// registers *all combinations* of 'Tests' and 'Types' with Google
// Test.  The return value is insignificant - we just need to return
// something such that we can call this function in a namespace scope.
template <GTEST_TEMPLATE_ Fixture, typename Tests, typename Types>
class TypeParameterizedTestCase {
 public:
  static bool Register(const char* prefix, const char* case_name,
                       const char* test_names) {
    typedef typename Tests::Head Head;

    // First, register the first test in 'Test' for each type in 'Types'.
    TypeParameterizedTest<Fixture, Head, Types>::Register(
        prefix, case_name, test_names, 0);

    // Next, recurses (at compile time) with the tail of the test list.
    return TypeParameterizedTestCase<Fixture, typename Tests::Tail, Types>
        ::Register(prefix, case_name, SkipComma(test_names));
  }
};

// The base case for the compile time recursion.
template <GTEST_TEMPLATE_ Fixture, typename Types>
class TypeParameterizedTestCase<Fixture, Templates0, Types> {
 public:
  static bool Register(const char* /*prefix*/, const char* /*case_name*/,
                       const char* /*test_names*/) {
    return true;
  }
};

#endif  // GTEST_HAS_TYPED_TEST || GTEST_HAS_TYPED_TEST_P

// Returns the current OS stack trace as a String.
//
// The maximum number of stack frames to be included is specified by
// the gtest_stack_trace_depth flag.  The skip_count parameter
// specifies the number of top frames to be skipped, which doesn't
// count against the number of frames to be included.
//
// For example, if Foo() calls Bar(), which in turn calls
// GetCurrentOsStackTraceExceptTop(..., 1), Foo() will be included in
// the trace but Bar() and GetCurrentOsStackTraceExceptTop() won't.
GTEST_API_ String GetCurrentOsStackTraceExceptTop(UnitTest* unit_test,
                                                  int skip_count);

// Helpers for suppressing warnings on unreachable code or constant
// condition.

// Always returns true.
GTEST_API_ bool AlwaysTrue();

// Always returns false.
inline bool AlwaysFalse() { return !AlwaysTrue(); }

// Helper for suppressing false warning from Clang on a const char*
// variable declared in a conditional expression always being NULL in
// the else branch.
struct GTEST_API_ ConstCharPtr {
  ConstCharPtr(const char* str) : value(str) {}
  operator bool() const { return true; }
  const char* value;
};

// A simple Linear Congruential Generator for generating random
// numbers with a uniform distribution.  Unlike rand() and srand(), it
// doesn't use global state (and therefore can't interfere with user
// code).  Unlike rand_r(), it's portable.  An LCG isn't very random,
// but it's good enough for our purposes.
class GTEST_API_ Random {
 public:
  static const UInt32 kMaxRange = 1u << 31;

  explicit Random(UInt32 seed) : state_(seed) {}

  void Reseed(UInt32 seed) { state_ = seed; }

  // Generates a random number from [0, range).  Crashes if 'range' is
  // 0 or greater than kMaxRange.
  UInt32 Generate(UInt32 range);

 private:
  UInt32 state_;
  GTEST_DISALLOW_COPY_AND_ASSIGN_(Random);
};

// Defining a variable of type CompileAssertTypesEqual<T1, T2> will cause a
// compiler error iff T1 and T2 are different types.
template <typename T1, typename T2>
struct CompileAssertTypesEqual;

template <typename T>
struct CompileAssertTypesEqual<T, T> {
};

// Removes the reference from a type if it is a reference type,
// otherwise leaves it unchanged.  This is the same as
// tr1::remove_reference, which is not widely available yet.
template <typename T>
struct RemoveReference { typedef T type; };  // NOLINT
template <typename T>
struct RemoveReference<T&> { typedef T type; };  // NOLINT

// A handy wrapper around RemoveReference that works when the argument
// T depends on template parameters.
#define GTEST_REMOVE_REFERENCE_(T) \
    typename ::testing::internal::RemoveReference<T>::type

// Removes const from a type if it is a const type, otherwise leaves
// it unchanged.  This is the same as tr1::remove_const, which is not
// widely available yet.
template <typename T>
struct RemoveConst { typedef T type; };  // NOLINT
template <typename T>
struct RemoveConst<const T> { typedef T type; };  // NOLINT

// MSVC 8.0, Sun C++, and IBM XL C++ have a bug which causes the above
// definition to fail to remove the const in 'const int[3]' and 'const
// char[3][4]'.  The following specialization works around the bug.
// However, it causes trouble with GCC and thus needs to be
// conditionally compiled.
#if defined(_MSC_VER) || defined(__SUNPRO_CC) || defined(__IBMCPP__)
template <typename T, size_t N>
struct RemoveConst<const T[N]> {
  typedef typename RemoveConst<T>::type type[N];
};
#endif

// A handy wrapper around RemoveConst that works when the argument
// T depends on template parameters.
#define GTEST_REMOVE_CONST_(T) \
    typename ::testing::internal::RemoveConst<T>::type

// Turns const U&, U&, const U, and U all into U.
#define GTEST_REMOVE_REFERENCE_AND_CONST_(T) \
    GTEST_REMOVE_CONST_(GTEST_REMOVE_REFERENCE_(T))

// Adds reference to a type if it is not a reference type,
// otherwise leaves it unchanged.  This is the same as
// tr1::add_reference, which is not widely available yet.
template <typename T>
struct AddReference { typedef T& type; };  // NOLINT
template <typename T>
struct AddReference<T&> { typedef T& type; };  // NOLINT

// A handy wrapper around AddReference that works when the argument T
// depends on template parameters.
#define GTEST_ADD_REFERENCE_(T) \
    typename ::testing::internal::AddReference<T>::type

// Adds a reference to const on top of T as necessary.  For example,
// it transforms
//
//   char         ==> const char&
//   const char   ==> const char&
//   char&        ==> const char&
//   const char&  ==> const char&
//
// The argument T must depend on some template parameters.
#define GTEST_REFERENCE_TO_CONST_(T) \
    GTEST_ADD_REFERENCE_(const GTEST_REMOVE_REFERENCE_(T))

// ImplicitlyConvertible<From, To>::value is a compile-time bool
// constant that's true iff type From can be implicitly converted to
// type To.
template <typename From, typename To>
class ImplicitlyConvertible {
 private:
  // We need the following helper functions only for their types.
  // They have no implementations.

  // MakeFrom() is an expression whose type is From.  We cannot simply
  // use From(), as the type From may not have a public default
  // constructor.
  static From MakeFrom();

  // These two functions are overloaded.  Given an expression
  // Helper(x), the compiler will pick the first version if x can be
  // implicitly converted to type To; otherwise it will pick the
  // second version.
  //
  // The first version returns a value of size 1, and the second
  // version returns a value of size 2.  Therefore, by checking the
  // size of Helper(x), which can be done at compile time, we can tell
  // which version of Helper() is used, and hence whether x can be
  // implicitly converted to type To.
  static char Helper(To);
  static char (&Helper(...))[2];  // NOLINT

  // We have to put the 'public' section after the 'private' section,
  // or MSVC refuses to compile the code.
 public:
  // MSVC warns about implicitly converting from double to int for
  // possible loss of data, so we need to temporarily disable the
  // warning.
#ifdef _MSC_VER
# pragma warning(push)          // Saves the current warning state.
# pragma warning(disable:4244)  // Temporarily disables warning 4244.

  static const bool value =
      sizeof(Helper(ImplicitlyConvertible::MakeFrom())) == 1;
# pragma warning(pop)           // Restores the warning state.
#elif defined(__BORLANDC__)
  // C++Builder cannot use member overload resolution during template
  // instantiation.  The simplest workaround is to use its C++0x type traits
  // functions (C++Builder 2009 and above only).
  static const bool value = __is_convertible(From, To);
#else
  static const bool value =
      sizeof(Helper(ImplicitlyConvertible::MakeFrom())) == 1;
#endif  // _MSV_VER
};
template <typename From, typename To>
const bool ImplicitlyConvertible<From, To>::value;

// IsAProtocolMessage<T>::value is a compile-time bool constant that's
// true iff T is type ProtocolMessage, proto2::Message, or a subclass
// of those.
template <typename T>
struct IsAProtocolMessage
    : public bool_constant<
  ImplicitlyConvertible<const T*, const ::ProtocolMessage*>::value ||
  ImplicitlyConvertible<const T*, const ::proto2::Message*>::value> {
};

// When the compiler sees expression IsContainerTest<C>(0), if C is an
// STL-style container class, the first overload of IsContainerTest
// will be viable (since both C::iterator* and C::const_iterator* are
// valid types and NULL can be implicitly converted to them).  It will
// be picked over the second overload as 'int' is a perfect match for
// the type of argument 0.  If C::iterator or C::const_iterator is not
// a valid type, the first overload is not viable, and the second
// overload will be picked.  Therefore, we can determine whether C is
// a container class by checking the type of IsContainerTest<C>(0).
// The value of the expression is insignificant.
//
// Note that we look for both C::iterator and C::const_iterator.  The
// reason is that C++ injects the name of a class as a member of the
// class itself (e.g. you can refer to class iterator as either
// 'iterator' or 'iterator::iterator').  If we look for C::iterator
// only, for example, we would mistakenly think that a class named
// iterator is an STL container.
//
// Also note that the simpler approach of overloading
// IsContainerTest(typename C::const_iterator*) and
// IsContainerTest(...) doesn't work with Visual Age C++ and Sun C++.
typedef int IsContainer;
template <class C>
IsContainer IsContainerTest(int /* dummy */,
                            typename C::iterator* /* it */ = NULL,
                            typename C::const_iterator* /* const_it */ = NULL) {
  return 0;
}

typedef char IsNotContainer;
template <class C>
IsNotContainer IsContainerTest(long /* dummy */) { return '\0'; }

// EnableIf<condition>::type is void when 'Cond' is true, and
// undefined when 'Cond' is false.  To use SFINAE to make a function
// overload only apply when a particular expression is true, add
// "typename EnableIf<expression>::type* = 0" as the last parameter.
template<bool> struct EnableIf;
template<> struct EnableIf<true> { typedef void type; };  // NOLINT

// Utilities for native arrays.

// ArrayEq() compares two k-dimensional native arrays using the
// elements' operator==, where k can be any integer >= 0.  When k is
// 0, ArrayEq() degenerates into comparing a single pair of values.

template <typename T, typename U>
bool ArrayEq(const T* lhs, size_t size, const U* rhs);

// This generic version is used when k is 0.
template <typename T, typename U>
inline bool ArrayEq(const T& lhs, const U& rhs) { return lhs == rhs; }

// This overload is used when k >= 1.
template <typename T, typename U, size_t N>
inline bool ArrayEq(const T(&lhs)[N], const U(&rhs)[N]) {
  return internal::ArrayEq(lhs, N, rhs);
}

// This helper reduces code bloat.  If we instead put its logic inside
// the previous ArrayEq() function, arrays with different sizes would
// lead to different copies of the template code.
template <typename T, typename U>
bool ArrayEq(const T* lhs, size_t size, const U* rhs) {
  for (size_t i = 0; i != size; i++) {
    if (!internal::ArrayEq(lhs[i], rhs[i]))
      return false;
  }
  return true;
}

// Finds the first element in the iterator range [begin, end) that
// equals elem.  Element may be a native array type itself.
template <typename Iter, typename Element>
Iter ArrayAwareFind(Iter begin, Iter end, const Element& elem) {
  for (Iter it = begin; it != end; ++it) {
    if (internal::ArrayEq(*it, elem))
      return it;
  }
  return end;
}

// CopyArray() copies a k-dimensional native array using the elements'
// operator=, where k can be any integer >= 0.  When k is 0,
// CopyArray() degenerates into copying a single value.

template <typename T, typename U>
void CopyArray(const T* from, size_t size, U* to);

// This generic version is used when k is 0.
template <typename T, typename U>
inline void CopyArray(const T& from, U* to) { *to = from; }

// This overload is used when k >= 1.
template <typename T, typename U, size_t N>
inline void CopyArray(const T(&from)[N], U(*to)[N]) {
  internal::CopyArray(from, N, *to);
}

// This helper reduces code bloat.  If we instead put its logic inside
// the previous CopyArray() function, arrays with different sizes
// would lead to different copies of the template code.
template <typename T, typename U>
void CopyArray(const T* from, size_t size, U* to) {
  for (size_t i = 0; i != size; i++) {
    internal::CopyArray(from[i], to + i);
  }
}

// The relation between an NativeArray object (see below) and the
// native array it represents.
enum RelationToSource {
  kReference,  // The NativeArray references the native array.
  kCopy        // The NativeArray makes a copy of the native array and
               // owns the copy.
};

// Adapts a native array to a read-only STL-style container.  Instead
// of the complete STL container concept, this adaptor only implements
// members useful for Google Mock's container matchers.  New members
// should be added as needed.  To simplify the implementation, we only
// support Element being a raw type (i.e. having no top-level const or
// reference modifier).  It's the client's responsibility to satisfy
// this requirement.  Element can be an array type itself (hence
// multi-dimensional arrays are supported).
template <typename Element>
class NativeArray {
 public:
  // STL-style container typedefs.
  typedef Element value_type;
  typedef Element* iterator;
  typedef const Element* const_iterator;

  // Constructs from a native array.
  NativeArray(const Element* array, size_t count, RelationToSource relation) {
    Init(array, count, relation);
  }

  // Copy constructor.
  NativeArray(const NativeArray& rhs) {
    Init(rhs.array_, rhs.size_, rhs.relation_to_source_);
  }

  ~NativeArray() {
    // Ensures that the user doesn't instantiate NativeArray with a
    // const or reference type.
    static_cast<void>(StaticAssertTypeEqHelper<Element,
        GTEST_REMOVE_REFERENCE_AND_CONST_(Element)>());
    if (relation_to_source_ == kCopy)
      delete[] array_;
  }

  // STL-style container methods.
  size_t size() const { return size_; }
  const_iterator begin() const { return array_; }
  const_iterator end() const { return array_ + size_; }
  bool operator==(const NativeArray& rhs) const {
    return size() == rhs.size() &&
        ArrayEq(begin(), size(), rhs.begin());
  }

 private:
  // Initializes this object; makes a copy of the input array if
  // 'relation' is kCopy.
  void Init(const Element* array, size_t a_size, RelationToSource relation) {
    if (relation == kReference) {
      array_ = array;
    } else {
      Element* const copy = new Element[a_size];
      CopyArray(array, a_size, copy);
      array_ = copy;
    }
    size_ = a_size;
    relation_to_source_ = relation;
  }

  const Element* array_;
  size_t size_;
  RelationToSource relation_to_source_;

  GTEST_DISALLOW_ASSIGN_(NativeArray);
};

}  // namespace internal
}  // namespace testing

#define GTEST_MESSAGE_AT_(file, line, message, result_type) \
  ::testing::internal::AssertHelper(result_type, file, line, message) \
    = ::testing::Message()

#define GTEST_MESSAGE_(message, result_type) \
  GTEST_MESSAGE_AT_(__FILE__, __LINE__, message, result_type)

#define GTEST_FATAL_FAILURE_(message) \
  return GTEST_MESSAGE_(message, ::testing::TestPartResult::kFatalFailure)

#define GTEST_NONFATAL_FAILURE_(message) \
  GTEST_MESSAGE_(message, ::testing::TestPartResult::kNonFatalFailure)

#define GTEST_SUCCESS_(message) \
  GTEST_MESSAGE_(message, ::testing::TestPartResult::kSuccess)

// Suppresses MSVC warnings 4072 (unreachable code) for the code following
// statement if it returns or throws (or doesn't return or throw in some
// situations).
#define GTEST_SUPPRESS_UNREACHABLE_CODE_WARNING_BELOW_(statement) \
  if (::testing::internal::AlwaysTrue()) { statement; }

#define GTEST_TEST_THROW_(statement, expected_exception, fail) \
  GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
  if (::testing::internal::ConstCharPtr gtest_msg = "") { \
    bool gtest_caught_expected = false; \
    try { \
      GTEST_SUPPRESS_UNREACHABLE_CODE_WARNING_BELOW_(statement); \
    } \
    catch (expected_exception const&) { \
      gtest_caught_expected = true; \
    } \
    catch (...) { \
      gtest_msg.value = \
          "Expected: " #statement " throws an exception of type " \
          #expected_exception ".\n  Actual: it throws a different type."; \
      goto GTEST_CONCAT_TOKEN_(gtest_label_testthrow_, __LINE__); \
    } \
    if (!gtest_caught_expected) { \
      gtest_msg.value = \
          "Expected: " #statement " throws an exception of type " \
          #expected_exception ".\n  Actual: it throws nothing."; \
      goto GTEST_CONCAT_TOKEN_(gtest_label_testthrow_, __LINE__); \
    } \
  } else \
    GTEST_CONCAT_TOKEN_(gtest_label_testthrow_, __LINE__): \
      fail(gtest_msg.value)

#define GTEST_TEST_NO_THROW_(statement, fail) \
  GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
  if (::testing::internal::AlwaysTrue()) { \
    try { \
      GTEST_SUPPRESS_UNREACHABLE_CODE_WARNING_BELOW_(statement); \
    } \
    catch (...) { \
      goto GTEST_CONCAT_TOKEN_(gtest_label_testnothrow_, __LINE__); \
    } \
  } else \
    GTEST_CONCAT_TOKEN_(gtest_label_testnothrow_, __LINE__): \
      fail("Expected: " #statement " doesn't throw an exception.\n" \
           "  Actual: it throws.")

#define GTEST_TEST_ANY_THROW_(statement, fail) \
  GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
  if (::testing::internal::AlwaysTrue()) { \
    bool gtest_caught_any = false; \
    try { \
      GTEST_SUPPRESS_UNREACHABLE_CODE_WARNING_BELOW_(statement); \
    } \
    catch (...) { \
      gtest_caught_any = true; \
    } \
    if (!gtest_caught_any) { \
      goto GTEST_CONCAT_TOKEN_(gtest_label_testanythrow_, __LINE__); \
    } \
  } else \
    GTEST_CONCAT_TOKEN_(gtest_label_testanythrow_, __LINE__): \
      fail("Expected: " #statement " throws an exception.\n" \
           "  Actual: it doesn't.")


// Implements Boolean test assertions such as EXPECT_TRUE. expression can be
// either a boolean expression or an AssertionResult. text is a textual
// represenation of expression as it was passed into the EXPECT_TRUE.
#define GTEST_TEST_BOOLEAN_(expression, text, actual, expected, fail) \
  GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
  if (const ::testing::AssertionResult gtest_ar_ = \
      ::testing::AssertionResult(expression)) \
    ; \
  else \
    fail(::testing::internal::GetBoolAssertionFailureMessage(\
        gtest_ar_, text, #actual, #expected).c_str())

#define GTEST_TEST_NO_FATAL_FAILURE_(statement, fail) \
  GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
  if (::testing::internal::AlwaysTrue()) { \
    ::testing::internal::HasNewFatalFailureHelper gtest_fatal_failure_checker; \
    GTEST_SUPPRESS_UNREACHABLE_CODE_WARNING_BELOW_(statement); \
    if (gtest_fatal_failure_checker.has_new_fatal_failure()) { \
      goto GTEST_CONCAT_TOKEN_(gtest_label_testnofatal_, __LINE__); \
    } \
  } else \
    GTEST_CONCAT_TOKEN_(gtest_label_testnofatal_, __LINE__): \
      fail("Expected: " #statement " doesn't generate new fatal " \
           "failures in the current thread.\n" \
           "  Actual: it does.")

// Expands to the name of the class that implements the given test.
#define GTEST_TEST_CLASS_NAME_(test_case_name, test_name) \
  test_case_name##_##test_name##_Test

// Helper macro for defining tests.
#define GTEST_TEST_(test_case_name, test_name, parent_class, parent_id)\
class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) : public parent_class {\
 public:\
  GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() {}\
 private:\
  virtual void TestBody();\
  static ::testing::TestInfo* const test_info_ GTEST_ATTRIBUTE_UNUSED_;\
  GTEST_DISALLOW_COPY_AND_ASSIGN_(\
      GTEST_TEST_CLASS_NAME_(test_case_name, test_name));\
};\
\
::testing::TestInfo* const GTEST_TEST_CLASS_NAME_(test_case_name, test_name)\
  ::test_info_ =\
    ::testing::internal::MakeAndRegisterTestInfo(\
        #test_case_name, #test_name, NULL, NULL, \
        (parent_id), \
        parent_class::SetUpTestCase, \
        parent_class::TearDownTestCase, \
        new ::testing::internal::TestFactoryImpl<\
            GTEST_TEST_CLASS_NAME_(test_case_name, test_name)>);\
void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::TestBody()

#endif  // GTEST_INCLUDE_GTEST_INTERNAL_GTEST_INTERNAL_H_
// Copyright 2005, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Author: wan@google.com (Zhanyong Wan)
//
// The Google C++ Testing Framework (Google Test)
//
// This header file defines the public API for death tests.  It is
// #included by gtest.h so a user doesn't need to include this
// directly.

#ifndef GTEST_INCLUDE_GTEST_GTEST_DEATH_TEST_H_
#define GTEST_INCLUDE_GTEST_GTEST_DEATH_TEST_H_

// Copyright 2005, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Authors: wan@google.com (Zhanyong Wan), eefacm@gmail.com (Sean Mcafee)
//
// The Google C++ Testing Framework (Google Test)
//
// This header file defines internal utilities needed for implementing
// death tests.  They are subject to change without notice.

#ifndef GTEST_INCLUDE_GTEST_INTERNAL_GTEST_DEATH_TEST_INTERNAL_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_GTEST_DEATH_TEST_INTERNAL_H_


#include <stdio.h>

namespace testing {
namespace internal {

GTEST_DECLARE_string_(internal_run_death_test);

// Names of the flags (needed for parsing Google Test flags).
const char kDeathTestStyleFlag[] = "death_test_style";
const char kDeathTestUseFork[] = "death_test_use_fork";
const char kInternalRunDeathTestFlag[] = "internal_run_death_test";

#if GTEST_HAS_DEATH_TEST

// DeathTest is a class that hides much of the complexity of the
// GTEST_DEATH_TEST_ macro.  It is abstract; its static Create method
// returns a concrete class that depends on the prevailing death test
// style, as defined by the --gtest_death_test_style and/or
// --gtest_internal_run_death_test flags.

// In describing the results of death tests, these terms are used with
// the corresponding definitions:
//
// exit status:  The integer exit information in the format specified
//               by wait(2)
// exit code:    The integer code passed to exit(3), _exit(2), or
//               returned from main()
class GTEST_API_ DeathTest {
 public:
  // Create returns false if there was an error determining the
  // appropriate action to take for the current death test; for example,
  // if the gtest_death_test_style flag is set to an invalid value.
  // The LastMessage method will return a more detailed message in that
  // case.  Otherwise, the DeathTest pointer pointed to by the "test"
  // argument is set.  If the death test should be skipped, the pointer
  // is set to NULL; otherwise, it is set to the address of a new concrete
  // DeathTest object that controls the execution of the current test.
  static bool Create(const char* statement, const RE* regex,
                     const char* file, int line, DeathTest** test);
  DeathTest();
  virtual ~DeathTest() { }

  // A helper class that aborts a death test when it's deleted.
  class ReturnSentinel {
   public:
    explicit ReturnSentinel(DeathTest* test) : test_(test) { }
    ~ReturnSentinel() { test_->Abort(TEST_ENCOUNTERED_RETURN_STATEMENT); }
   private:
    DeathTest* const test_;
    GTEST_DISALLOW_COPY_AND_ASSIGN_(ReturnSentinel);
  } GTEST_ATTRIBUTE_UNUSED_;

  // An enumeration of possible roles that may be taken when a death
  // test is encountered.  EXECUTE means that the death test logic should
  // be executed immediately.  OVERSEE means that the program should prepare
  // the appropriate environment for a child process to execute the death
  // test, then wait for it to complete.
  enum TestRole { OVERSEE_TEST, EXECUTE_TEST };

  // An enumeration of the three reasons that a test might be aborted.
  enum AbortReason {
    TEST_ENCOUNTERED_RETURN_STATEMENT,
    TEST_THREW_EXCEPTION,
    TEST_DID_NOT_DIE
  };

  // Assumes one of the above roles.
  virtual TestRole AssumeRole() = 0;

  // Waits for the death test to finish and returns its status.
  virtual int Wait() = 0;

  // Returns true if the death test passed; that is, the test process
  // exited during the test, its exit status matches a user-supplied
  // predicate, and its stderr output matches a user-supplied regular
  // expression.
  // The user-supplied predicate may be a macro expression rather
  // than a function pointer or functor, or else Wait and Passed could
  // be combined.
  virtual bool Passed(bool exit_status_ok) = 0;

  // Signals that the death test did not die as expected.
  virtual void Abort(AbortReason reason) = 0;

  // Returns a human-readable outcome message regarding the outcome of
  // the last death test.
  static const char* LastMessage();

  static void set_last_death_test_message(const String& message);

 private:
  // A string containing a description of the outcome of the last death test.
  static String last_death_test_message_;

  GTEST_DISALLOW_COPY_AND_ASSIGN_(DeathTest);
};

// Factory interface for death tests.  May be mocked out for testing.
class DeathTestFactory {
 public:
  virtual ~DeathTestFactory() { }
  virtual bool Create(const char* statement, const RE* regex,
                      const char* file, int line, DeathTest** test) = 0;
};

// A concrete DeathTestFactory implementation for normal use.
class DefaultDeathTestFactory : public DeathTestFactory {
 public:
  virtual bool Create(const char* statement, const RE* regex,
                      const char* file, int line, DeathTest** test);
};

// Returns true if exit_status describes a process that was terminated
// by a signal, or exited normally with a nonzero exit code.
GTEST_API_ bool ExitedUnsuccessfully(int exit_status);

// Traps C++ exceptions escaping statement and reports them as test
// failures. Note that trapping SEH exceptions is not implemented here.
# if GTEST_HAS_EXCEPTIONS
#  define GTEST_EXECUTE_DEATH_TEST_STATEMENT_(statement, death_test) \
  try { \
    GTEST_SUPPRESS_UNREACHABLE_CODE_WARNING_BELOW_(statement); \
  } catch (const ::std::exception& gtest_exception) { \
    fprintf(\
        stderr, \
        "\n%s: Caught std::exception-derived exception escaping the " \
        "death test statement. Exception message: %s\n", \
        ::testing::internal::FormatFileLocation(__FILE__, __LINE__).c_str(), \
        gtest_exception.what()); \
    fflush(stderr); \
    death_test->Abort(::testing::internal::DeathTest::TEST_THREW_EXCEPTION); \
  } catch (...) { \
    death_test->Abort(::testing::internal::DeathTest::TEST_THREW_EXCEPTION); \
  }

# else
#  define GTEST_EXECUTE_DEATH_TEST_STATEMENT_(statement, death_test) \
  GTEST_SUPPRESS_UNREACHABLE_CODE_WARNING_BELOW_(statement)

# endif

// This macro is for implementing ASSERT_DEATH*, EXPECT_DEATH*,
// ASSERT_EXIT*, and EXPECT_EXIT*.
# define GTEST_DEATH_TEST_(statement, predicate, regex, fail) \
  GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
  if (::testing::internal::AlwaysTrue()) { \
    const ::testing::internal::RE& gtest_regex = (regex); \
    ::testing::internal::DeathTest* gtest_dt; \
    if (!::testing::internal::DeathTest::Create(#statement, &gtest_regex, \
        __FILE__, __LINE__, &gtest_dt)) { \
      goto GTEST_CONCAT_TOKEN_(gtest_label_, __LINE__); \
    } \
    if (gtest_dt != NULL) { \
      ::testing::internal::scoped_ptr< ::testing::internal::DeathTest> \
          gtest_dt_ptr(gtest_dt); \
      switch (gtest_dt->AssumeRole()) { \
        case ::testing::internal::DeathTest::OVERSEE_TEST: \
          if (!gtest_dt->Passed(predicate(gtest_dt->Wait()))) { \
            goto GTEST_CONCAT_TOKEN_(gtest_label_, __LINE__); \
          } \
          break; \
        case ::testing::internal::DeathTest::EXECUTE_TEST: { \
          ::testing::internal::DeathTest::ReturnSentinel \
              gtest_sentinel(gtest_dt); \
          GTEST_EXECUTE_DEATH_TEST_STATEMENT_(statement, gtest_dt); \
          gtest_dt->Abort(::testing::internal::DeathTest::TEST_DID_NOT_DIE); \
          break; \
        } \
        default: \
          break; \
      } \
    } \
  } else \
    GTEST_CONCAT_TOKEN_(gtest_label_, __LINE__): \
      fail(::testing::internal::DeathTest::LastMessage())
// The symbol "fail" here expands to something into which a message
// can be streamed.

// A class representing the parsed contents of the
// --gtest_internal_run_death_test flag, as it existed when
// RUN_ALL_TESTS was called.
class InternalRunDeathTestFlag {
 public:
  InternalRunDeathTestFlag(const String& a_file,
                           int a_line,
                           int an_index,
                           int a_write_fd)
      : file_(a_file), line_(a_line), index_(an_index),
        write_fd_(a_write_fd) {}

  ~InternalRunDeathTestFlag() {
    if (write_fd_ >= 0)
      posix::Close(write_fd_);
  }

  String file() const { return file_; }
  int line() const { return line_; }
  int index() const { return index_; }
  int write_fd() const { return write_fd_; }

 private:
  String file_;
  int line_;
  int index_;
  int write_fd_;

  GTEST_DISALLOW_COPY_AND_ASSIGN_(InternalRunDeathTestFlag);
};

// Returns a newly created InternalRunDeathTestFlag object with fields
// initialized from the GTEST_FLAG(internal_run_death_test) flag if
// the flag is specified; otherwise returns NULL.
InternalRunDeathTestFlag* ParseInternalRunDeathTestFlag();

#else  // GTEST_HAS_DEATH_TEST

// This macro is used for implementing macros such as
// EXPECT_DEATH_IF_SUPPORTED and ASSERT_DEATH_IF_SUPPORTED on systems where
// death tests are not supported. Those macros must compile on such systems
// iff EXPECT_DEATH and ASSERT_DEATH compile with the same parameters on
// systems that support death tests. This allows one to write such a macro
// on a system that does not support death tests and be sure that it will
// compile on a death-test supporting system.
//
// Parameters:
//   statement -  A statement that a macro such as EXPECT_DEATH would test
//                for program termination. This macro has to make sure this
//                statement is compiled but not executed, to ensure that
//                EXPECT_DEATH_IF_SUPPORTED compiles with a certain
//                parameter iff EXPECT_DEATH compiles with it.
//   regex     -  A regex that a macro such as EXPECT_DEATH would use to test
//                the output of statement.  This parameter has to be
//                compiled but not evaluated by this macro, to ensure that
//                this macro only accepts expressions that a macro such as
//                EXPECT_DEATH would accept.
//   terminator - Must be an empty statement for EXPECT_DEATH_IF_SUPPORTED
//                and a return statement for ASSERT_DEATH_IF_SUPPORTED.
//                This ensures that ASSERT_DEATH_IF_SUPPORTED will not
//                compile inside functions where ASSERT_DEATH doesn't
//                compile.
//
//  The branch that has an always false condition is used to ensure that
//  statement and regex are compiled (and thus syntactically correct) but
//  never executed. The unreachable code macro protects the terminator
//  statement from generating an 'unreachable code' warning in case
//  statement unconditionally returns or throws. The Message constructor at
//  the end allows the syntax of streaming additional messages into the
//  macro, for compilational compatibility with EXPECT_DEATH/ASSERT_DEATH.
# define GTEST_UNSUPPORTED_DEATH_TEST_(statement, regex, terminator) \
    GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
    if (::testing::internal::AlwaysTrue()) { \
      GTEST_LOG_(WARNING) \
          << "Death tests are not supported on this platform.\n" \
          << "Statement '" #statement "' cannot be verified."; \
    } else if (::testing::internal::AlwaysFalse()) { \
      ::testing::internal::RE::PartialMatch(".*", (regex)); \
      GTEST_SUPPRESS_UNREACHABLE_CODE_WARNING_BELOW_(statement); \
      terminator; \
    } else \
      ::testing::Message()

#endif  // GTEST_HAS_DEATH_TEST

}  // namespace internal
}  // namespace testing

#endif  // GTEST_INCLUDE_GTEST_INTERNAL_GTEST_DEATH_TEST_INTERNAL_H_

namespace testing {

// This flag controls the style of death tests.  Valid values are "threadsafe",
// meaning that the death test child process will re-execute the test binary
// from the start, running only a single death test, or "fast",
// meaning that the child process will execute the test logic immediately
// after forking.
GTEST_DECLARE_string_(death_test_style);

#if GTEST_HAS_DEATH_TEST

// The following macros are useful for writing death tests.

// Here's what happens when an ASSERT_DEATH* or EXPECT_DEATH* is
// executed:
//
//   1. It generates a warning if there is more than one active
//   thread.  This is because it's safe to fork() or clone() only
//   when there is a single thread.
//
//   2. The parent process clone()s a sub-process and runs the death
//   test in it; the sub-process exits with code 0 at the end of the
//   death test, if it hasn't exited already.
//
//   3. The parent process waits for the sub-process to terminate.
//
//   4. The parent process checks the exit code and error message of
//   the sub-process.
//
// Examples:
//
//   ASSERT_DEATH(server.SendMessage(56, "Hello"), "Invalid port number");
//   for (int i = 0; i < 5; i++) {
//     EXPECT_DEATH(server.ProcessRequest(i),
//                  "Invalid request .* in ProcessRequest()")
//         << "Failed to die on request " << i);
//   }
//
//   ASSERT_EXIT(server.ExitNow(), ::testing::ExitedWithCode(0), "Exiting");
//
//   bool KilledBySIGHUP(int exit_code) {
//     return WIFSIGNALED(exit_code) && WTERMSIG(exit_code) == SIGHUP;
//   }
//
//   ASSERT_EXIT(client.HangUpServer(), KilledBySIGHUP, "Hanging up!");
//
// On the regular expressions used in death tests:
//
//   On POSIX-compliant systems (*nix), we use the <regex.h> library,
//   which uses the POSIX extended regex syntax.
//
//   On other platforms (e.g. Windows), we only support a simple regex
//   syntax implemented as part of Google Test.  This limited
//   implementation should be enough most of the time when writing
//   death tests; though it lacks many features you can find in PCRE
//   or POSIX extended regex syntax.  For example, we don't support
//   union ("x|y"), grouping ("(xy)"), brackets ("[xy]"), and
//   repetition count ("x{5,7}"), among others.
//
//   Below is the syntax that we do support.  We chose it to be a
//   subset of both PCRE and POSIX extended regex, so it's easy to
//   learn wherever you come from.  In the following: 'A' denotes a
//   literal character, period (.), or a single \\ escape sequence;
//   'x' and 'y' denote regular expressions; 'm' and 'n' are for
//   natural numbers.
//
//     c     matches any literal character c
//     \\d   matches any decimal digit
//     \\D   matches any character that's not a decimal digit
//     \\f   matches \f
//     \\n   matches \n
//     \\r   matches \r
//     \\s   matches any ASCII whitespace, including \n
//     \\S   matches any character that's not a whitespace
//     \\t   matches \t
//     \\v   matches \v
//     \\w   matches any letter, _, or decimal digit
//     \\W   matches any character that \\w doesn't match
//     \\c   matches any literal character c, which must be a punctuation
//     .     matches any single character except \n
//     A?    matches 0 or 1 occurrences of A
//     A*    matches 0 or many occurrences of A
//     A+    matches 1 or many occurrences of A
//     ^     matches the beginning of a string (not that of each line)
//     $     matches the end of a string (not that of each line)
//     xy    matches x followed by y
//
//   If you accidentally use PCRE or POSIX extended regex features
//   not implemented by us, you will get a run-time failure.  In that
//   case, please try to rewrite your regular expression within the
//   above syntax.
//
//   This implementation is *not* meant to be as highly tuned or robust
//   as a compiled regex library, but should perform well enough for a
//   death test, which already incurs significant overhead by launching
//   a child process.
//
// Known caveats:
//
//   A "threadsafe" style death test obtains the path to the test
//   program from argv[0] and re-executes it in the sub-process.  For
//   simplicity, the current implementation doesn't search the PATH
//   when launching the sub-process.  This means that the user must
//   invoke the test program via a path that contains at least one
//   path separator (e.g. path/to/foo_test and
//   /absolute/path/to/bar_test are fine, but foo_test is not).  This
//   is rarely a problem as people usually don't put the test binary
//   directory in PATH.
//
// TODO(wan@google.com): make thread-safe death tests search the PATH.

// Asserts that a given statement causes the program to exit, with an
// integer exit status that satisfies predicate, and emitting error output
// that matches regex.
# define ASSERT_EXIT(statement, predicate, regex) \
    GTEST_DEATH_TEST_(statement, predicate, regex, GTEST_FATAL_FAILURE_)

// Like ASSERT_EXIT, but continues on to successive tests in the
// test case, if any:
# define EXPECT_EXIT(statement, predicate, regex) \
    GTEST_DEATH_TEST_(statement, predicate, regex, GTEST_NONFATAL_FAILURE_)

// Asserts that a given statement causes the program to exit, either by
// explicitly exiting with a nonzero exit code or being killed by a
// signal, and emitting error output that matches regex.
# define ASSERT_DEATH(statement, regex) \
    ASSERT_EXIT(statement, ::testing::internal::ExitedUnsuccessfully, regex)

// Like ASSERT_DEATH, but continues on to successive tests in the
// test case, if any:
# define EXPECT_DEATH(statement, regex) \
    EXPECT_EXIT(statement, ::testing::internal::ExitedUnsuccessfully, regex)

// Two predicate classes that can be used in {ASSERT,EXPECT}_EXIT*:

// Tests that an exit code describes a normal exit with a given exit code.
class GTEST_API_ ExitedWithCode {
 public:
  explicit ExitedWithCode(int exit_code);
  bool operator()(int exit_status) const;
 private:
  // No implementation - assignment is unsupported.
  void operator=(const ExitedWithCode& other);

  const int exit_code_;
};

# if !GTEST_OS_WINDOWS
// Tests that an exit code describes an exit due to termination by a
// given signal.
class GTEST_API_ KilledBySignal {
 public:
  explicit KilledBySignal(int signum);
  bool operator()(int exit_status) const;
 private:
  const int signum_;
};
# endif  // !GTEST_OS_WINDOWS

// EXPECT_DEBUG_DEATH asserts that the given statements die in debug mode.
// The death testing framework causes this to have interesting semantics,
// since the sideeffects of the call are only visible in opt mode, and not
// in debug mode.
//
// In practice, this can be used to test functions that utilize the
// LOG(DFATAL) macro using the following style:
//
// int DieInDebugOr12(int* sideeffect) {
//   if (sideeffect) {
//     *sideeffect = 12;
//   }
//   LOG(DFATAL) << "death";
//   return 12;
// }
//
// TEST(TestCase, TestDieOr12WorksInDgbAndOpt) {
//   int sideeffect = 0;
//   // Only asserts in dbg.
//   EXPECT_DEBUG_DEATH(DieInDebugOr12(&sideeffect), "death");
//
// #ifdef NDEBUG
//   // opt-mode has sideeffect visible.
//   EXPECT_EQ(12, sideeffect);
// #else
//   // dbg-mode no visible sideeffect.
//   EXPECT_EQ(0, sideeffect);
// #endif
// }
//
// This will assert that DieInDebugReturn12InOpt() crashes in debug
// mode, usually due to a DCHECK or LOG(DFATAL), but returns the
// appropriate fallback value (12 in this case) in opt mode. If you
// need to test that a function has appropriate side-effects in opt
// mode, include assertions against the side-effects.  A general
// pattern for this is:
//
// EXPECT_DEBUG_DEATH({
//   // Side-effects here will have an effect after this statement in
//   // opt mode, but none in debug mode.
//   EXPECT_EQ(12, DieInDebugOr12(&sideeffect));
// }, "death");
//
# ifdef NDEBUG

#  define EXPECT_DEBUG_DEATH(statement, regex) \
  do { statement; } while (::testing::internal::AlwaysFalse())

#  define ASSERT_DEBUG_DEATH(statement, regex) \
  do { statement; } while (::testing::internal::AlwaysFalse())

# else

#  define EXPECT_DEBUG_DEATH(statement, regex) \
  EXPECT_DEATH(statement, regex)

#  define ASSERT_DEBUG_DEATH(statement, regex) \
  ASSERT_DEATH(statement, regex)

# endif  // NDEBUG for EXPECT_DEBUG_DEATH
#endif  // GTEST_HAS_DEATH_TEST

// EXPECT_DEATH_IF_SUPPORTED(statement, regex) and
// ASSERT_DEATH_IF_SUPPORTED(statement, regex) expand to real death tests if
// death tests are supported; otherwise they just issue a warning.  This is
// useful when you are combining death test assertions with normal test
// assertions in one test.
#if GTEST_HAS_DEATH_TEST
# define EXPECT_DEATH_IF_SUPPORTED(statement, regex) \
    EXPECT_DEATH(statement, regex)
# define ASSERT_DEATH_IF_SUPPORTED(statement, regex) \
    ASSERT_DEATH(statement, regex)
#else
# define EXPECT_DEATH_IF_SUPPORTED(statement, regex) \
    GTEST_UNSUPPORTED_DEATH_TEST_(statement, regex, )
# define ASSERT_DEATH_IF_SUPPORTED(statement, regex) \
    GTEST_UNSUPPORTED_DEATH_TEST_(statement, regex, return)
#endif

}  // namespace testing

#endif  // GTEST_INCLUDE_GTEST_GTEST_DEATH_TEST_H_
// Copyright 2005, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Author: wan@google.com (Zhanyong Wan)
//
// The Google C++ Testing Framework (Google Test)
//
// This header file defines the Message class.
//
// IMPORTANT NOTE: Due to limitation of the C++ language, we have to
// leave some internal implementation details in this header file.
// They are clearly marked by comments like this:
//
//   // INTERNAL IMPLEMENTATION - DO NOT USE IN A USER PROGRAM.
//
// Such code is NOT meant to be used by a user directly, and is subject
// to CHANGE WITHOUT NOTICE.  Therefore DO NOT DEPEND ON IT in a user
// program!

#ifndef GTEST_INCLUDE_GTEST_GTEST_MESSAGE_H_
#define GTEST_INCLUDE_GTEST_GTEST_MESSAGE_H_

#include <limits>


namespace testing {

// The Message class works like an ostream repeater.
//
// Typical usage:
//
//   1. You stream a bunch of values to a Message object.
//      It will remember the text in a stringstream.
//   2. Then you stream the Message object to an ostream.
//      This causes the text in the Message to be streamed
//      to the ostream.
//
// For example;
//
//   testing::Message foo;
//   foo << 1 << " != " << 2;
//   std::cout << foo;
//
// will print "1 != 2".
//
// Message is not intended to be inherited from.  In particular, its
// destructor is not virtual.
//
// Note that stringstream behaves differently in gcc and in MSVC.  You
// can stream a NULL char pointer to it in the former, but not in the
// latter (it causes an access violation if you do).  The Message
// class hides this difference by treating a NULL char pointer as
// "(null)".
class GTEST_API_ Message {
 private:
  // The type of basic IO manipulators (endl, ends, and flush) for
  // narrow streams.
  typedef std::ostream& (*BasicNarrowIoManip)(std::ostream&);

 public:
  // Constructs an empty Message.
  // We allocate the stringstream separately because otherwise each use of
  // ASSERT/EXPECT in a procedure adds over 200 bytes to the procedure's
  // stack frame leading to huge stack frames in some cases; gcc does not reuse
  // the stack space.
  Message() : ss_(new ::std::stringstream) {
    // By default, we want there to be enough precision when printing
    // a double to a Message.
    *ss_ << std::setprecision(std::numeric_limits<double>::digits10 + 2);
  }

  // Copy constructor.
  Message(const Message& msg) : ss_(new ::std::stringstream) {  // NOLINT
    *ss_ << msg.GetString();
  }

  // Constructs a Message from a C-string.
  explicit Message(const char* str) : ss_(new ::std::stringstream) {
    *ss_ << str;
  }

#if GTEST_OS_SYMBIAN
  // Streams a value (either a pointer or not) to this object.
  template <typename T>
  inline Message& operator <<(const T& value) {
    StreamHelper(typename internal::is_pointer<T>::type(), value);
    return *this;
  }
#else
  // Streams a non-pointer value to this object.
  template <typename T>
  inline Message& operator <<(const T& val) {
    ::GTestStreamToHelper(ss_.get(), val);
    return *this;
  }

  // Streams a pointer value to this object.
  //
  // This function is an overload of the previous one.  When you
  // stream a pointer to a Message, this definition will be used as it
  // is more specialized.  (The C++ Standard, section
  // [temp.func.order].)  If you stream a non-pointer, then the
  // previous definition will be used.
  //
  // The reason for this overload is that streaming a NULL pointer to
  // ostream is undefined behavior.  Depending on the compiler, you
  // may get "0", "(nil)", "(null)", or an access violation.  To
  // ensure consistent result across compilers, we always treat NULL
  // as "(null)".
  template <typename T>
  inline Message& operator <<(T* const& pointer) {  // NOLINT
    if (pointer == NULL) {
      *ss_ << "(null)";
    } else {
      ::GTestStreamToHelper(ss_.get(), pointer);
    }
    return *this;
  }
#endif  // GTEST_OS_SYMBIAN

  // Since the basic IO manipulators are overloaded for both narrow
  // and wide streams, we have to provide this specialized definition
  // of operator <<, even though its body is the same as the
  // templatized version above.  Without this definition, streaming
  // endl or other basic IO manipulators to Message will confuse the
  // compiler.
  Message& operator <<(BasicNarrowIoManip val) {
    *ss_ << val;
    return *this;
  }

  // Instead of 1/0, we want to see true/false for bool values.
  Message& operator <<(bool b) {
    return *this << (b ? "true" : "false");
  }

  // These two overloads allow streaming a wide C string to a Message
  // using the UTF-8 encoding.
  Message& operator <<(const wchar_t* wide_c_str) {
    return *this << internal::String::ShowWideCString(wide_c_str);
  }
  Message& operator <<(wchar_t* wide_c_str) {
    return *this << internal::String::ShowWideCString(wide_c_str);
  }

#if GTEST_HAS_STD_WSTRING
  // Converts the given wide string to a narrow string using the UTF-8
  // encoding, and streams the result to this Message object.
  Message& operator <<(const ::std::wstring& wstr);
#endif  // GTEST_HAS_STD_WSTRING

#if GTEST_HAS_GLOBAL_WSTRING
  // Converts the given wide string to a narrow string using the UTF-8
  // encoding, and streams the result to this Message object.
  Message& operator <<(const ::wstring& wstr);
#endif  // GTEST_HAS_GLOBAL_WSTRING

  // Gets the text streamed to this object so far as a String.
  // Each '\0' character in the buffer is replaced with "\\0".
  //
  // INTERNAL IMPLEMENTATION - DO NOT USE IN A USER PROGRAM.
  internal::String GetString() const {
    return internal::StringStreamToString(ss_.get());
  }

 private:

#if GTEST_OS_SYMBIAN
  // These are needed as the Nokia Symbian Compiler cannot decide between
  // const T& and const T* in a function template. The Nokia compiler _can_
  // decide between class template specializations for T and T*, so a
  // tr1::type_traits-like is_pointer works, and we can overload on that.
  template <typename T>
  inline void StreamHelper(internal::true_type /*dummy*/, T* pointer) {
    if (pointer == NULL) {
      *ss_ << "(null)";
    } else {
      ::GTestStreamToHelper(ss_.get(), pointer);
    }
  }
  template <typename T>
  inline void StreamHelper(internal::false_type /*dummy*/, const T& value) {
    ::GTestStreamToHelper(ss_.get(), value);
  }
#endif  // GTEST_OS_SYMBIAN

  // We'll hold the text streamed to this object here.
  const internal::scoped_ptr< ::std::stringstream> ss_;

  // We declare (but don't implement) this to prevent the compiler
  // from implementing the assignment operator.
  void operator=(const Message&);
};

// Streams a Message to an ostream.
inline std::ostream& operator <<(std::ostream& os, const Message& sb) {
  return os << sb.GetString();
}

}  // namespace testing

#endif  // GTEST_INCLUDE_GTEST_GTEST_MESSAGE_H_
// This file was GENERATED by command:
//     pump.py gtest-param-test.h.pump
// DO NOT EDIT BY HAND!!!

// Copyright 2008, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Authors: vladl@google.com (Vlad Losev)
//
// Macros and functions for implementing parameterized tests
// in Google C++ Testing Framework (Google Test)
//
// This file is generated by a SCRIPT.  DO NOT EDIT BY HAND!
//
#ifndef GTEST_INCLUDE_GTEST_GTEST_PARAM_TEST_H_
#define GTEST_INCLUDE_GTEST_GTEST_PARAM_TEST_H_


// Value-parameterized tests allow you to test your code with different
// parameters without writing multiple copies of the same test.
//
// Here is how you use value-parameterized tests:

#if 0

// To write value-parameterized tests, first you should define a fixture
// class. It is usually derived from testing::TestWithParam<T> (see below for
// another inheritance scheme that's sometimes useful in more complicated
// class hierarchies), where the type of your parameter values.
// TestWithParam<T> is itself derived from testing::Test. T can be any
// copyable type. If it's a raw pointer, you are responsible for managing the
// lifespan of the pointed values.

class FooTest : public ::testing::TestWithParam<const char*> {
  // You can implement all the usual class fixture members here.
};

// Then, use the TEST_P macro to define as many parameterized tests
// for this fixture as you want. The _P suffix is for "parameterized"
// or "pattern", whichever you prefer to think.

TEST_P(FooTest, DoesBlah) {
  // Inside a test, access the test parameter with the GetParam() method
  // of the TestWithParam<T> class:
  EXPECT_TRUE(foo.Blah(GetParam()));
  ...
}

TEST_P(FooTest, HasBlahBlah) {
  ...
}

// Finally, you can use INSTANTIATE_TEST_CASE_P to instantiate the test
// case with any set of parameters you want. Google Test defines a number
// of functions for generating test parameters. They return what we call
// (surprise!) parameter generators. Here is a  summary of them, which
// are all in the testing namespace:
//
//
//  Range(begin, end [, step]) - Yields values {begin, begin+step,
//                               begin+step+step, ...}. The values do not
//                               include end. step defaults to 1.
//  Values(v1, v2, ..., vN)    - Yields values {v1, v2, ..., vN}.
//  ValuesIn(container)        - Yields values from a C-style array, an STL
//  ValuesIn(begin,end)          container, or an iterator range [begin, end).
//  Bool()                     - Yields sequence {false, true}.
//  Combine(g1, g2, ..., gN)   - Yields all combinations (the Cartesian product
//                               for the math savvy) of the values generated
//                               by the N generators.
//
// For more details, see comments at the definitions of these functions below
// in this file.
//
// The following statement will instantiate tests from the FooTest test case
// each with parameter values "meeny", "miny", and "moe".

INSTANTIATE_TEST_CASE_P(InstantiationName,
                        FooTest,
                        Values("meeny", "miny", "moe"));

// To distinguish different instances of the pattern, (yes, you
// can instantiate it more then once) the first argument to the
// INSTANTIATE_TEST_CASE_P macro is a prefix that will be added to the
// actual test case name. Remember to pick unique prefixes for different
// instantiations. The tests from the instantiation above will have
// these names:
//
//    * InstantiationName/FooTest.DoesBlah/0 for "meeny"
//    * InstantiationName/FooTest.DoesBlah/1 for "miny"
//    * InstantiationName/FooTest.DoesBlah/2 for "moe"
//    * InstantiationName/FooTest.HasBlahBlah/0 for "meeny"
//    * InstantiationName/FooTest.HasBlahBlah/1 for "miny"
//    * InstantiationName/FooTest.HasBlahBlah/2 for "moe"
//
// You can use these names in --gtest_filter.
//
// This statement will instantiate all tests from FooTest again, each
// with parameter values "cat" and "dog":

const char* pets[] = {"cat", "dog"};
INSTANTIATE_TEST_CASE_P(AnotherInstantiationName, FooTest, ValuesIn(pets));

// The tests from the instantiation above will have these names:
//
//    * AnotherInstantiationName/FooTest.DoesBlah/0 for "cat"
//    * AnotherInstantiationName/FooTest.DoesBlah/1 for "dog"
//    * AnotherInstantiationName/FooTest.HasBlahBlah/0 for "cat"
//    * AnotherInstantiationName/FooTest.HasBlahBlah/1 for "dog"
//
// Please note that INSTANTIATE_TEST_CASE_P will instantiate all tests
// in the given test case, whether their definitions come before or
// AFTER the INSTANTIATE_TEST_CASE_P statement.
//
// Please also note that generator expressions (including parameters to the
// generators) are evaluated in InitGoogleTest(), after main() has started.
// This allows the user on one hand, to adjust generator parameters in order
// to dynamically determine a set of tests to run and on the other hand,
// give the user a chance to inspect the generated tests with Google Test
// reflection API before RUN_ALL_TESTS() is executed.
//
// You can see samples/sample7_unittest.cc and samples/sample8_unittest.cc
// for more examples.
//
// In the future, we plan to publish the API for defining new parameter
// generators. But for now this interface remains part of the internal
// implementation and is subject to change.
//
//
// A parameterized test fixture must be derived from testing::Test and from
// testing::WithParamInterface<T>, where T is the type of the parameter
// values. Inheriting from TestWithParam<T> satisfies that requirement because
// TestWithParam<T> inherits from both Test and WithParamInterface. In more
// complicated hierarchies, however, it is occasionally useful to inherit
// separately from Test and WithParamInterface. For example:

class BaseTest : public ::testing::Test {
  // You can inherit all the usual members for a non-parameterized test
  // fixture here.
};

class DerivedTest : public BaseTest, public ::testing::WithParamInterface<int> {
  // The usual test fixture members go here too.
};

TEST_F(BaseTest, HasFoo) {
  // This is an ordinary non-parameterized test.
}

TEST_P(DerivedTest, DoesBlah) {
  // GetParam works just the same here as if you inherit from TestWithParam.
  EXPECT_TRUE(foo.Blah(GetParam()));
}

#endif  // 0


#if !GTEST_OS_SYMBIAN
# include <utility>
#endif

// scripts/fuse_gtest.py depends on gtest's own header being #included
// *unconditionally*.  Therefore these #includes cannot be moved
// inside #if GTEST_HAS_PARAM_TEST.
// Copyright 2008 Google Inc.
// All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Author: vladl@google.com (Vlad Losev)

// Type and function utilities for implementing parameterized tests.

#ifndef GTEST_INCLUDE_GTEST_INTERNAL_GTEST_PARAM_UTIL_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_GTEST_PARAM_UTIL_H_

#include <iterator>
#include <utility>
#include <vector>

// scripts/fuse_gtest.py depends on gtest's own header being #included
// *unconditionally*.  Therefore these #includes cannot be moved
// inside #if GTEST_HAS_PARAM_TEST.
// Copyright 2003 Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Authors: Dan Egnor (egnor@google.com)
//
// A "smart" pointer type with reference tracking.  Every pointer to a
// particular object is kept on a circular linked list.  When the last pointer
// to an object is destroyed or reassigned, the object is deleted.
//
// Used properly, this deletes the object when the last reference goes away.
// There are several caveats:
// - Like all reference counting schemes, cycles lead to leaks.
// - Each smart pointer is actually two pointers (8 bytes instead of 4).
// - Every time a pointer is assigned, the entire list of pointers to that
//   object is traversed.  This class is therefore NOT SUITABLE when there
//   will often be more than two or three pointers to a particular object.
// - References are only tracked as long as linked_ptr<> objects are copied.
//   If a linked_ptr<> is converted to a raw pointer and back, BAD THINGS
//   will happen (double deletion).
//
// A good use of this class is storing object references in STL containers.
// You can safely put linked_ptr<> in a vector<>.
// Other uses may not be as good.
//
// Note: If you use an incomplete type with linked_ptr<>, the class
// *containing* linked_ptr<> must have a constructor and destructor (even
// if they do nothing!).
//
// Bill Gibbons suggested we use something like this.
//
// Thread Safety:
//   Unlike other linked_ptr implementations, in this implementation
//   a linked_ptr object is thread-safe in the sense that:
//     - it's safe to copy linked_ptr objects concurrently,
//     - it's safe to copy *from* a linked_ptr and read its underlying
//       raw pointer (e.g. via get()) concurrently, and
//     - it's safe to write to two linked_ptrs that point to the same
//       shared object concurrently.
// TODO(wan@google.com): rename this to safe_linked_ptr to avoid
// confusion with normal linked_ptr.

#ifndef GTEST_INCLUDE_GTEST_INTERNAL_GTEST_LINKED_PTR_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_GTEST_LINKED_PTR_H_

#include <stdlib.h>
#include <assert.h>


namespace testing {
namespace internal {

// Protects copying of all linked_ptr objects.
GTEST_API_ GTEST_DECLARE_STATIC_MUTEX_(g_linked_ptr_mutex);

// This is used internally by all instances of linked_ptr<>.  It needs to be
// a non-template class because different types of linked_ptr<> can refer to
// the same object (linked_ptr<Superclass>(obj) vs linked_ptr<Subclass>(obj)).
// So, it needs to be possible for different types of linked_ptr to participate
// in the same circular linked list, so we need a single class type here.
//
// DO NOT USE THIS CLASS DIRECTLY YOURSELF.  Use linked_ptr<T>.
class linked_ptr_internal {
 public:
  // Create a new circle that includes only this instance.
  void join_new() {
    next_ = this;
  }

  // Many linked_ptr operations may change p.link_ for some linked_ptr
  // variable p in the same circle as this object.  Therefore we need
  // to prevent two such operations from occurring concurrently.
  //
  // Note that different types of linked_ptr objects can coexist in a
  // circle (e.g. linked_ptr<Base>, linked_ptr<Derived1>, and
  // linked_ptr<Derived2>).  Therefore we must use a single mutex to
  // protect all linked_ptr objects.  This can create serious
  // contention in production code, but is acceptable in a testing
  // framework.

  // Join an existing circle.
  // L < g_linked_ptr_mutex
  void join(linked_ptr_internal const* ptr) {
    MutexLock lock(&g_linked_ptr_mutex);

    linked_ptr_internal const* p = ptr;
    while (p->next_ != ptr) p = p->next_;
    p->next_ = this;
    next_ = ptr;
  }

  // Leave whatever circle we're part of.  Returns true if we were the
  // last member of the circle.  Once this is done, you can join() another.
  // L < g_linked_ptr_mutex
  bool depart() {
    MutexLock lock(&g_linked_ptr_mutex);

    if (next_ == this) return true;
    linked_ptr_internal const* p = next_;
    while (p->next_ != this) p = p->next_;
    p->next_ = next_;
    return false;
  }

 private:
  mutable linked_ptr_internal const* next_;
};

template <typename T>
class linked_ptr {
 public:
  typedef T element_type;

  // Take over ownership of a raw pointer.  This should happen as soon as
  // possible after the object is created.
  explicit linked_ptr(T* ptr = NULL) { capture(ptr); }
  ~linked_ptr() { depart(); }

  // Copy an existing linked_ptr<>, adding ourselves to the list of references.
  template <typename U> linked_ptr(linked_ptr<U> const& ptr) { copy(&ptr); }
  linked_ptr(linked_ptr const& ptr) {  // NOLINT
    assert(&ptr != this);
    copy(&ptr);
  }

  // Assignment releases the old value and acquires the new.
  template <typename U> linked_ptr& operator=(linked_ptr<U> const& ptr) {
    depart();
    copy(&ptr);
    return *this;
  }

  linked_ptr& operator=(linked_ptr const& ptr) {
    if (&ptr != this) {
      depart();
      copy(&ptr);
    }
    return *this;
  }

  // Smart pointer members.
  void reset(T* ptr = NULL) {
    depart();
    capture(ptr);
  }
  T* get() const { return value_; }
  T* operator->() const { return value_; }
  T& operator*() const { return *value_; }

  bool operator==(T* p) const { return value_ == p; }
  bool operator!=(T* p) const { return value_ != p; }
  template <typename U>
  bool operator==(linked_ptr<U> const& ptr) const {
    return value_ == ptr.get();
  }
  template <typename U>
  bool operator!=(linked_ptr<U> const& ptr) const {
    return value_ != ptr.get();
  }

 private:
  template <typename U>
  friend class linked_ptr;

  T* value_;
  linked_ptr_internal link_;

  void depart() {
    if (link_.depart()) delete value_;
  }

  void capture(T* ptr) {
    value_ = ptr;
    link_.join_new();
  }

  template <typename U> void copy(linked_ptr<U> const* ptr) {
    value_ = ptr->get();
    if (value_)
      link_.join(&ptr->link_);
    else
      link_.join_new();
  }
};

template<typename T> inline
bool operator==(T* ptr, const linked_ptr<T>& x) {
  return ptr == x.get();
}

template<typename T> inline
bool operator!=(T* ptr, const linked_ptr<T>& x) {
  return ptr != x.get();
}

// A function to convert T* into linked_ptr<T>
// Doing e.g. make_linked_ptr(new FooBarBaz<type>(arg)) is a shorter notation
// for linked_ptr<FooBarBaz<type> >(new FooBarBaz<type>(arg))
template <typename T>
linked_ptr<T> make_linked_ptr(T* ptr) {
  return linked_ptr<T>(ptr);
}

}  // namespace internal
}  // namespace testing

#endif  // GTEST_INCLUDE_GTEST_INTERNAL_GTEST_LINKED_PTR_H_
// Copyright 2007, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Author: wan@google.com (Zhanyong Wan)

// Google Test - The Google C++ Testing Framework
//
// This file implements a universal value printer that can print a
// value of any type T:
//
//   void ::testing::internal::UniversalPrinter<T>::Print(value, ostream_ptr);
//
// A user can teach this function how to print a class type T by
// defining either operator<<() or PrintTo() in the namespace that
// defines T.  More specifically, the FIRST defined function in the
// following list will be used (assuming T is defined in namespace
// foo):
//
//   1. foo::PrintTo(const T&, ostream*)
//   2. operator<<(ostream&, const T&) defined in either foo or the
//      global namespace.
//
// If none of the above is defined, it will print the debug string of
// the value if it is a protocol buffer, or print the raw bytes in the
// value otherwise.
//
// To aid debugging: when T is a reference type, the address of the
// value is also printed; when T is a (const) char pointer, both the
// pointer value and the NUL-terminated string it points to are
// printed.
//
// We also provide some convenient wrappers:
//
//   // Prints a value to a string.  For a (const or not) char
//   // pointer, the NUL-terminated string (but not the pointer) is
//   // printed.
//   std::string ::testing::PrintToString(const T& value);
//
//   // Prints a value tersely: for a reference type, the referenced
//   // value (but not the address) is printed; for a (const or not) char
//   // pointer, the NUL-terminated string (but not the pointer) is
//   // printed.
//   void ::testing::internal::UniversalTersePrint(const T& value, ostream*);
//
//   // Prints value using the type inferred by the compiler.  The difference
//   // from UniversalTersePrint() is that this function prints both the
//   // pointer and the NUL-terminated string for a (const or not) char pointer.
//   void ::testing::internal::UniversalPrint(const T& value, ostream*);
//
//   // Prints the fields of a tuple tersely to a string vector, one
//   // element for each field. Tuple support must be enabled in
//   // gtest-port.h.
//   std::vector<string> UniversalTersePrintTupleFieldsToStrings(
//       const Tuple& value);
//
// Known limitation:
//
// The print primitives print the elements of an STL-style container
// using the compiler-inferred type of *iter where iter is a
// const_iterator of the container.  When const_iterator is an input
// iterator but not a forward iterator, this inferred type may not
// match value_type, and the print output may be incorrect.  In
// practice, this is rarely a problem as for most containers
// const_iterator is a forward iterator.  We'll fix this if there's an
// actual need for it.  Note that this fix cannot rely on value_type
// being defined as many user-defined container types don't have
// value_type.

#ifndef GTEST_INCLUDE_GTEST_GTEST_PRINTERS_H_
#define GTEST_INCLUDE_GTEST_GTEST_PRINTERS_H_

#include <ostream>  // NOLINT
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace testing {

// Definitions in the 'internal' and 'internal2' name spaces are
// subject to change without notice.  DO NOT USE THEM IN USER CODE!
namespace internal2 {

// Prints the given number of bytes in the given object to the given
// ostream.
GTEST_API_ void PrintBytesInObjectTo(const unsigned char* obj_bytes,
                                     size_t count,
                                     ::std::ostream* os);

// For selecting which printer to use when a given type has neither <<
// nor PrintTo().
enum TypeKind {
  kProtobuf,              // a protobuf type
  kConvertibleToInteger,  // a type implicitly convertible to BiggestInt
                          // (e.g. a named or unnamed enum type)
  kOtherType              // anything else
};

// TypeWithoutFormatter<T, kTypeKind>::PrintValue(value, os) is called
// by the universal printer to print a value of type T when neither
// operator<< nor PrintTo() is defined for T, where kTypeKind is the
// "kind" of T as defined by enum TypeKind.
template <typename T, TypeKind kTypeKind>
class TypeWithoutFormatter {
 public:
  // This default version is called when kTypeKind is kOtherType.
  static void PrintValue(const T& value, ::std::ostream* os) {
    PrintBytesInObjectTo(reinterpret_cast<const unsigned char*>(&value),
                         sizeof(value), os);
  }
};

// We print a protobuf using its ShortDebugString() when the string
// doesn't exceed this many characters; otherwise we print it using
// DebugString() for better readability.
const size_t kProtobufOneLinerMaxLength = 50;

template <typename T>
class TypeWithoutFormatter<T, kProtobuf> {
 public:
  static void PrintValue(const T& value, ::std::ostream* os) {
    const ::testing::internal::string short_str = value.ShortDebugString();
    const ::testing::internal::string pretty_str =
        short_str.length() <= kProtobufOneLinerMaxLength ?
        short_str : ("\n" + value.DebugString());
    *os << ("<" + pretty_str + ">");
  }
};

template <typename T>
class TypeWithoutFormatter<T, kConvertibleToInteger> {
 public:
  // Since T has no << operator or PrintTo() but can be implicitly
  // converted to BiggestInt, we print it as a BiggestInt.
  //
  // Most likely T is an enum type (either named or unnamed), in which
  // case printing it as an integer is the desired behavior.  In case
  // T is not an enum, printing it as an integer is the best we can do
  // given that it has no user-defined printer.
  static void PrintValue(const T& value, ::std::ostream* os) {
    const internal::BiggestInt kBigInt = value;
    *os << kBigInt;
  }
};

// Prints the given value to the given ostream.  If the value is a
// protocol message, its debug string is printed; if it's an enum or
// of a type implicitly convertible to BiggestInt, it's printed as an
// integer; otherwise the bytes in the value are printed.  This is
// what UniversalPrinter<T>::Print() does when it knows nothing about
// type T and T has neither << operator nor PrintTo().
//
// A user can override this behavior for a class type Foo by defining
// a << operator in the namespace where Foo is defined.
//
// We put this operator in namespace 'internal2' instead of 'internal'
// to simplify the implementation, as much code in 'internal' needs to
// use << in STL, which would conflict with our own << were it defined
// in 'internal'.
//
// Note that this operator<< takes a generic std::basic_ostream<Char,
// CharTraits> type instead of the more restricted std::ostream.  If
// we define it to take an std::ostream instead, we'll get an
// "ambiguous overloads" compiler error when trying to print a type
// Foo that supports streaming to std::basic_ostream<Char,
// CharTraits>, as the compiler cannot tell whether
// operator<<(std::ostream&, const T&) or
// operator<<(std::basic_stream<Char, CharTraits>, const Foo&) is more
// specific.
template <typename Char, typename CharTraits, typename T>
::std::basic_ostream<Char, CharTraits>& operator<<(
    ::std::basic_ostream<Char, CharTraits>& os, const T& x) {
  TypeWithoutFormatter<T,
      (internal::IsAProtocolMessage<T>::value ? kProtobuf :
       internal::ImplicitlyConvertible<const T&, internal::BiggestInt>::value ?
       kConvertibleToInteger : kOtherType)>::PrintValue(x, &os);
  return os;
}

}  // namespace internal2
}  // namespace testing

// This namespace MUST NOT BE NESTED IN ::testing, or the name look-up
// magic needed for implementing UniversalPrinter won't work.
namespace testing_internal {

// Used to print a value that is not an STL-style container when the
// user doesn't define PrintTo() for it.
template <typename T>
void DefaultPrintNonContainerTo(const T& value, ::std::ostream* os) {
  // With the following statement, during unqualified name lookup,
  // testing::internal2::operator<< appears as if it was declared in
  // the nearest enclosing namespace that contains both
  // ::testing_internal and ::testing::internal2, i.e. the global
  // namespace.  For more details, refer to the C++ Standard section
  // 7.3.4-1 [namespace.udir].  This allows us to fall back onto
  // testing::internal2::operator<< in case T doesn't come with a <<
  // operator.
  //
  // We cannot write 'using ::testing::internal2::operator<<;', which
  // gcc 3.3 fails to compile due to a compiler bug.
  using namespace ::testing::internal2;  // NOLINT

  // Assuming T is defined in namespace foo, in the next statement,
  // the compiler will consider all of:
  //
  //   1. foo::operator<< (thanks to Koenig look-up),
  //   2. ::operator<< (as the current namespace is enclosed in ::),
  //   3. testing::internal2::operator<< (thanks to the using statement above).
  //
  // The operator<< whose type matches T best will be picked.
  //
  // We deliberately allow #2 to be a candidate, as sometimes it's
  // impossible to define #1 (e.g. when foo is ::std, defining
  // anything in it is undefined behavior unless you are a compiler
  // vendor.).
  *os << value;
}

}  // namespace testing_internal

namespace testing {
namespace internal {

// UniversalPrinter<T>::Print(value, ostream_ptr) prints the given
// value to the given ostream.  The caller must ensure that
// 'ostream_ptr' is not NULL, or the behavior is undefined.
//
// We define UniversalPrinter as a class template (as opposed to a
// function template), as we need to partially specialize it for
// reference types, which cannot be done with function templates.
template <typename T>
class UniversalPrinter;

template <typename T>
void UniversalPrint(const T& value, ::std::ostream* os);

// Used to print an STL-style container when the user doesn't define
// a PrintTo() for it.
template <typename C>
void DefaultPrintTo(IsContainer /* dummy */,
                    false_type /* is not a pointer */,
                    const C& container, ::std::ostream* os) {
  const size_t kMaxCount = 32;  // The maximum number of elements to print.
  *os << '{';
  size_t count = 0;
  for (typename C::const_iterator it = container.begin();
       it != container.end(); ++it, ++count) {
    if (count > 0) {
      *os << ',';
      if (count == kMaxCount) {  // Enough has been printed.
        *os << " ...";
        break;
      }
    }
    *os << ' ';
    // We cannot call PrintTo(*it, os) here as PrintTo() doesn't
    // handle *it being a native array.
    internal::UniversalPrint(*it, os);
  }

  if (count > 0) {
    *os << ' ';
  }
  *os << '}';
}

// Used to print a pointer that is neither a char pointer nor a member
// pointer, when the user doesn't define PrintTo() for it.  (A member
// variable pointer or member function pointer doesn't really point to
// a location in the address space.  Their representation is
// implementation-defined.  Therefore they will be printed as raw
// bytes.)
template <typename T>
void DefaultPrintTo(IsNotContainer /* dummy */,
                    true_type /* is a pointer */,
                    T* p, ::std::ostream* os) {
  if (p == NULL) {
    *os << "NULL";
  } else {
    // C++ doesn't allow casting from a function pointer to any object
    // pointer.
    //
    // IsTrue() silences warnings: "Condition is always true",
    // "unreachable code".
    if (IsTrue(ImplicitlyConvertible<T*, const void*>::value)) {
      // T is not a function type.  We just call << to print p,
      // relying on ADL to pick up user-defined << for their pointer
      // types, if any.
      *os << p;
    } else {
      // T is a function type, so '*os << p' doesn't do what we want
      // (it just prints p as bool).  We want to print p as a const
      // void*.  However, we cannot cast it to const void* directly,
      // even using reinterpret_cast, as earlier versions of gcc
      // (e.g. 3.4.5) cannot compile the cast when p is a function
      // pointer.  Casting to UInt64 first solves the problem.
      *os << reinterpret_cast<const void*>(
          reinterpret_cast<internal::UInt64>(p));
    }
  }
}

// Used to print a non-container, non-pointer value when the user
// doesn't define PrintTo() for it.
template <typename T>
void DefaultPrintTo(IsNotContainer /* dummy */,
                    false_type /* is not a pointer */,
                    const T& value, ::std::ostream* os) {
  ::testing_internal::DefaultPrintNonContainerTo(value, os);
}

// Prints the given value using the << operator if it has one;
// otherwise prints the bytes in it.  This is what
// UniversalPrinter<T>::Print() does when PrintTo() is not specialized
// or overloaded for type T.
//
// A user can override this behavior for a class type Foo by defining
// an overload of PrintTo() in the namespace where Foo is defined.  We
// give the user this option as sometimes defining a << operator for
// Foo is not desirable (e.g. the coding style may prevent doing it,
// or there is already a << operator but it doesn't do what the user
// wants).
template <typename T>
void PrintTo(const T& value, ::std::ostream* os) {
  // DefaultPrintTo() is overloaded.  The type of its first two
  // arguments determine which version will be picked.  If T is an
  // STL-style container, the version for container will be called; if
  // T is a pointer, the pointer version will be called; otherwise the
  // generic version will be called.
  //
  // Note that we check for container types here, prior to we check
  // for protocol message types in our operator<<.  The rationale is:
  //
  // For protocol messages, we want to give people a chance to
  // override Google Mock's format by defining a PrintTo() or
  // operator<<.  For STL containers, other formats can be
  // incompatible with Google Mock's format for the container
  // elements; therefore we check for container types here to ensure
  // that our format is used.
  //
  // The second argument of DefaultPrintTo() is needed to bypass a bug
  // in Symbian's C++ compiler that prevents it from picking the right
  // overload between:
  //
  //   PrintTo(const T& x, ...);
  //   PrintTo(T* x, ...);
  DefaultPrintTo(IsContainerTest<T>(0), is_pointer<T>(), value, os);
}

// The following list of PrintTo() overloads tells
// UniversalPrinter<T>::Print() how to print standard types (built-in
// types, strings, plain arrays, and pointers).

// Overloads for various char types.
GTEST_API_ void PrintTo(unsigned char c, ::std::ostream* os);
GTEST_API_ void PrintTo(signed char c, ::std::ostream* os);
inline void PrintTo(char c, ::std::ostream* os) {
  // When printing a plain char, we always treat it as unsigned.  This
  // way, the output won't be affected by whether the compiler thinks
  // char is signed or not.
  PrintTo(static_cast<unsigned char>(c), os);
}

// Overloads for other simple built-in types.
inline void PrintTo(bool x, ::std::ostream* os) {
  *os << (x ? "true" : "false");
}

// Overload for wchar_t type.
// Prints a wchar_t as a symbol if it is printable or as its internal
// code otherwise and also as its decimal code (except for L'\0').
// The L'\0' char is printed as "L'\\0'". The decimal code is printed
// as signed integer when wchar_t is implemented by the compiler
// as a signed type and is printed as an unsigned integer when wchar_t
// is implemented as an unsigned type.
GTEST_API_ void PrintTo(wchar_t wc, ::std::ostream* os);

// Overloads for C strings.
GTEST_API_ void PrintTo(const char* s, ::std::ostream* os);
inline void PrintTo(char* s, ::std::ostream* os) {
  PrintTo(ImplicitCast_<const char*>(s), os);
}

// signed/unsigned char is often used for representing binary data, so
// we print pointers to it as void* to be safe.
inline void PrintTo(const signed char* s, ::std::ostream* os) {
  PrintTo(ImplicitCast_<const void*>(s), os);
}
inline void PrintTo(signed char* s, ::std::ostream* os) {
  PrintTo(ImplicitCast_<const void*>(s), os);
}
inline void PrintTo(const unsigned char* s, ::std::ostream* os) {
  PrintTo(ImplicitCast_<const void*>(s), os);
}
inline void PrintTo(unsigned char* s, ::std::ostream* os) {
  PrintTo(ImplicitCast_<const void*>(s), os);
}

// MSVC can be configured to define wchar_t as a typedef of unsigned
// short.  It defines _NATIVE_WCHAR_T_DEFINED when wchar_t is a native
// type.  When wchar_t is a typedef, defining an overload for const
// wchar_t* would cause unsigned short* be printed as a wide string,
// possibly causing invalid memory accesses.
#if !defined(_MSC_VER) || defined(_NATIVE_WCHAR_T_DEFINED)
// Overloads for wide C strings
GTEST_API_ void PrintTo(const wchar_t* s, ::std::ostream* os);
inline void PrintTo(wchar_t* s, ::std::ostream* os) {
  PrintTo(ImplicitCast_<const wchar_t*>(s), os);
}
#endif

// Overload for C arrays.  Multi-dimensional arrays are printed
// properly.

// Prints the given number of elements in an array, without printing
// the curly braces.
template <typename T>
void PrintRawArrayTo(const T a[], size_t count, ::std::ostream* os) {
  UniversalPrint(a[0], os);
  for (size_t i = 1; i != count; i++) {
    *os << ", ";
    UniversalPrint(a[i], os);
  }
}

// Overloads for ::string and ::std::string.
#if GTEST_HAS_GLOBAL_STRING
GTEST_API_ void PrintStringTo(const ::string&s, ::std::ostream* os);
inline void PrintTo(const ::string& s, ::std::ostream* os) {
  PrintStringTo(s, os);
}
#endif  // GTEST_HAS_GLOBAL_STRING

GTEST_API_ void PrintStringTo(const ::std::string&s, ::std::ostream* os);
inline void PrintTo(const ::std::string& s, ::std::ostream* os) {
  PrintStringTo(s, os);
}

// Overloads for ::wstring and ::std::wstring.
#if GTEST_HAS_GLOBAL_WSTRING
GTEST_API_ void PrintWideStringTo(const ::wstring&s, ::std::ostream* os);
inline void PrintTo(const ::wstring& s, ::std::ostream* os) {
  PrintWideStringTo(s, os);
}
#endif  // GTEST_HAS_GLOBAL_WSTRING

#if GTEST_HAS_STD_WSTRING
GTEST_API_ void PrintWideStringTo(const ::std::wstring&s, ::std::ostream* os);
inline void PrintTo(const ::std::wstring& s, ::std::ostream* os) {
  PrintWideStringTo(s, os);
}
#endif  // GTEST_HAS_STD_WSTRING

#if GTEST_HAS_TR1_TUPLE
// Overload for ::std::tr1::tuple.  Needed for printing function arguments,
// which are packed as tuples.

// Helper function for printing a tuple.  T must be instantiated with
// a tuple type.
template <typename T>
void PrintTupleTo(const T& t, ::std::ostream* os);

// Overloaded PrintTo() for tuples of various arities.  We support
// tuples of up-to 10 fields.  The following implementation works
// regardless of whether tr1::tuple is implemented using the
// non-standard variadic template feature or not.

inline void PrintTo(const ::std::tr1::tuple<>& t, ::std::ostream* os) {
  PrintTupleTo(t, os);
}

template <typename T1>
void PrintTo(const ::std::tr1::tuple<T1>& t, ::std::ostream* os) {
  PrintTupleTo(t, os);
}

template <typename T1, typename T2>
void PrintTo(const ::std::tr1::tuple<T1, T2>& t, ::std::ostream* os) {
  PrintTupleTo(t, os);
}

template <typename T1, typename T2, typename T3>
void PrintTo(const ::std::tr1::tuple<T1, T2, T3>& t, ::std::ostream* os) {
  PrintTupleTo(t, os);
}

template <typename T1, typename T2, typename T3, typename T4>
void PrintTo(const ::std::tr1::tuple<T1, T2, T3, T4>& t, ::std::ostream* os) {
  PrintTupleTo(t, os);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
void PrintTo(const ::std::tr1::tuple<T1, T2, T3, T4, T5>& t,
             ::std::ostream* os) {
  PrintTupleTo(t, os);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
          typename T6>
void PrintTo(const ::std::tr1::tuple<T1, T2, T3, T4, T5, T6>& t,
             ::std::ostream* os) {
  PrintTupleTo(t, os);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
          typename T6, typename T7>
void PrintTo(const ::std::tr1::tuple<T1, T2, T3, T4, T5, T6, T7>& t,
             ::std::ostream* os) {
  PrintTupleTo(t, os);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
          typename T6, typename T7, typename T8>
void PrintTo(const ::std::tr1::tuple<T1, T2, T3, T4, T5, T6, T7, T8>& t,
             ::std::ostream* os) {
  PrintTupleTo(t, os);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
          typename T6, typename T7, typename T8, typename T9>
void PrintTo(const ::std::tr1::tuple<T1, T2, T3, T4, T5, T6, T7, T8, T9>& t,
             ::std::ostream* os) {
  PrintTupleTo(t, os);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
          typename T6, typename T7, typename T8, typename T9, typename T10>
void PrintTo(
    const ::std::tr1::tuple<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>& t,
    ::std::ostream* os) {
  PrintTupleTo(t, os);
}
#endif  // GTEST_HAS_TR1_TUPLE

// Overload for std::pair.
template <typename T1, typename T2>
void PrintTo(const ::std::pair<T1, T2>& value, ::std::ostream* os) {
  *os << '(';
  // We cannot use UniversalPrint(value.first, os) here, as T1 may be
  // a reference type.  The same for printing value.second.
  UniversalPrinter<T1>::Print(value.first, os);
  *os << ", ";
  UniversalPrinter<T2>::Print(value.second, os);
  *os << ')';
}

// Implements printing a non-reference type T by letting the compiler
// pick the right overload of PrintTo() for T.
template <typename T>
class UniversalPrinter {
 public:
  // MSVC warns about adding const to a function type, so we want to
  // disable the warning.
#ifdef _MSC_VER
# pragma warning(push)          // Saves the current warning state.
# pragma warning(disable:4180)  // Temporarily disables warning 4180.
#endif  // _MSC_VER

  // Note: we deliberately don't call this PrintTo(), as that name
  // conflicts with ::testing::internal::PrintTo in the body of the
  // function.
  static void Print(const T& value, ::std::ostream* os) {
    // By default, ::testing::internal::PrintTo() is used for printing
    // the value.
    //
    // Thanks to Koenig look-up, if T is a class and has its own
    // PrintTo() function defined in its namespace, that function will
    // be visible here.  Since it is more specific than the generic ones
    // in ::testing::internal, it will be picked by the compiler in the
    // following statement - exactly what we want.
    PrintTo(value, os);
  }

#ifdef _MSC_VER
# pragma warning(pop)           // Restores the warning state.
#endif  // _MSC_VER
};

// UniversalPrintArray(begin, len, os) prints an array of 'len'
// elements, starting at address 'begin'.
template <typename T>
void UniversalPrintArray(const T* begin, size_t len, ::std::ostream* os) {
  if (len == 0) {
    *os << "{}";
  } else {
    *os << "{ ";
    const size_t kThreshold = 18;
    const size_t kChunkSize = 8;
    // If the array has more than kThreshold elements, we'll have to
    // omit some details by printing only the first and the last
    // kChunkSize elements.
    // TODO(wan@google.com): let the user control the threshold using a flag.
    if (len <= kThreshold) {
      PrintRawArrayTo(begin, len, os);
    } else {
      PrintRawArrayTo(begin, kChunkSize, os);
      *os << ", ..., ";
      PrintRawArrayTo(begin + len - kChunkSize, kChunkSize, os);
    }
    *os << " }";
  }
}
// This overload prints a (const) char array compactly.
GTEST_API_ void UniversalPrintArray(const char* begin,
                                    size_t len,
                                    ::std::ostream* os);

// Implements printing an array type T[N].
template <typename T, size_t N>
class UniversalPrinter<T[N]> {
 public:
  // Prints the given array, omitting some elements when there are too
  // many.
  static void Print(const T (&a)[N], ::std::ostream* os) {
    UniversalPrintArray(a, N, os);
  }
};

// Implements printing a reference type T&.
template <typename T>
class UniversalPrinter<T&> {
 public:
  // MSVC warns about adding const to a function type, so we want to
  // disable the warning.
#ifdef _MSC_VER
# pragma warning(push)          // Saves the current warning state.
# pragma warning(disable:4180)  // Temporarily disables warning 4180.
#endif  // _MSC_VER

  static void Print(const T& value, ::std::ostream* os) {
    // Prints the address of the value.  We use reinterpret_cast here
    // as static_cast doesn't compile when T is a function type.
    *os << "@" << reinterpret_cast<const void*>(&value) << " ";

    // Then prints the value itself.
    UniversalPrint(value, os);
  }

#ifdef _MSC_VER
# pragma warning(pop)           // Restores the warning state.
#endif  // _MSC_VER
};

// Prints a value tersely: for a reference type, the referenced value
// (but not the address) is printed; for a (const) char pointer, the
// NUL-terminated string (but not the pointer) is printed.
template <typename T>
void UniversalTersePrint(const T& value, ::std::ostream* os) {
  UniversalPrint(value, os);
}
inline void UniversalTersePrint(const char* str, ::std::ostream* os) {
  if (str == NULL) {
    *os << "NULL";
  } else {
    UniversalPrint(string(str), os);
  }
}
inline void UniversalTersePrint(char* str, ::std::ostream* os) {
  UniversalTersePrint(static_cast<const char*>(str), os);
}

// Prints a value using the type inferred by the compiler.  The
// difference between this and UniversalTersePrint() is that for a
// (const) char pointer, this prints both the pointer and the
// NUL-terminated string.
template <typename T>
void UniversalPrint(const T& value, ::std::ostream* os) {
  UniversalPrinter<T>::Print(value, os);
}

#if GTEST_HAS_TR1_TUPLE
typedef ::std::vector<string> Strings;

// This helper template allows PrintTo() for tuples and
// UniversalTersePrintTupleFieldsToStrings() to be defined by
// induction on the number of tuple fields.  The idea is that
// TuplePrefixPrinter<N>::PrintPrefixTo(t, os) prints the first N
// fields in tuple t, and can be defined in terms of
// TuplePrefixPrinter<N - 1>.

// The inductive case.
template <size_t N>
struct TuplePrefixPrinter {
  // Prints the first N fields of a tuple.
  template <typename Tuple>
  static void PrintPrefixTo(const Tuple& t, ::std::ostream* os) {
    TuplePrefixPrinter<N - 1>::PrintPrefixTo(t, os);
    *os << ", ";
    UniversalPrinter<typename ::std::tr1::tuple_element<N - 1, Tuple>::type>
        ::Print(::std::tr1::get<N - 1>(t), os);
  }

  // Tersely prints the first N fields of a tuple to a string vector,
  // one element for each field.
  template <typename Tuple>
  static void TersePrintPrefixToStrings(const Tuple& t, Strings* strings) {
    TuplePrefixPrinter<N - 1>::TersePrintPrefixToStrings(t, strings);
    ::std::stringstream ss;
    UniversalTersePrint(::std::tr1::get<N - 1>(t), &ss);
    strings->push_back(ss.str());
  }
};

// Base cases.
template <>
struct TuplePrefixPrinter<0> {
  template <typename Tuple>
  static void PrintPrefixTo(const Tuple&, ::std::ostream*) {}

  template <typename Tuple>
  static void TersePrintPrefixToStrings(const Tuple&, Strings*) {}
};
// We have to specialize the entire TuplePrefixPrinter<> class
// template here, even though the definition of
// TersePrintPrefixToStrings() is the same as the generic version, as
// Embarcadero (formerly CodeGear, formerly Borland) C++ doesn't
// support specializing a method template of a class template.
template <>
struct TuplePrefixPrinter<1> {
  template <typename Tuple>
  static void PrintPrefixTo(const Tuple& t, ::std::ostream* os) {
    UniversalPrinter<typename ::std::tr1::tuple_element<0, Tuple>::type>::
        Print(::std::tr1::get<0>(t), os);
  }

  template <typename Tuple>
  static void TersePrintPrefixToStrings(const Tuple& t, Strings* strings) {
    ::std::stringstream ss;
    UniversalTersePrint(::std::tr1::get<0>(t), &ss);
    strings->push_back(ss.str());
  }
};

// Helper function for printing a tuple.  T must be instantiated with
// a tuple type.
template <typename T>
void PrintTupleTo(const T& t, ::std::ostream* os) {
  *os << "(";
  TuplePrefixPrinter< ::std::tr1::tuple_size<T>::value>::
      PrintPrefixTo(t, os);
  *os << ")";
}

// Prints the fields of a tuple tersely to a string vector, one
// element for each field.  See the comment before
// UniversalTersePrint() for how we define "tersely".
template <typename Tuple>
Strings UniversalTersePrintTupleFieldsToStrings(const Tuple& value) {
  Strings result;
  TuplePrefixPrinter< ::std::tr1::tuple_size<Tuple>::value>::
      TersePrintPrefixToStrings(value, &result);
  return result;
}
#endif  // GTEST_HAS_TR1_TUPLE

}  // namespace internal

template <typename T>
::std::string PrintToString(const T& value) {
  ::std::stringstream ss;
  internal::UniversalTersePrint(value, &ss);
  return ss.str();
}

}  // namespace testing

#endif  // GTEST_INCLUDE_GTEST_GTEST_PRINTERS_H_

#if GTEST_HAS_PARAM_TEST

namespace testing {
namespace internal {

// INTERNAL IMPLEMENTATION - DO NOT USE IN USER CODE.
//
// Outputs a message explaining invalid registration of different
// fixture class for the same test case. This may happen when
// TEST_P macro is used to define two tests with the same name
// but in different namespaces.
GTEST_API_ void ReportInvalidTestCaseType(const char* test_case_name,
                                          const char* file, int line);

template <typename> class ParamGeneratorInterface;
template <typename> class ParamGenerator;

// Interface for iterating over elements provided by an implementation
// of ParamGeneratorInterface<T>.
template <typename T>
class ParamIteratorInterface {
 public:
  virtual ~ParamIteratorInterface() {}
  // A pointer to the base generator instance.
  // Used only for the purposes of iterator comparison
  // to make sure that two iterators belong to the same generator.
  virtual const ParamGeneratorInterface<T>* BaseGenerator() const = 0;
  // Advances iterator to point to the next element
  // provided by the generator. The caller is responsible
  // for not calling Advance() on an iterator equal to
  // BaseGenerator()->End().
  virtual void Advance() = 0;
  // Clones the iterator object. Used for implementing copy semantics
  // of ParamIterator<T>.
  virtual ParamIteratorInterface* Clone() const = 0;
  // Dereferences the current iterator and provides (read-only) access
  // to the pointed value. It is the caller's responsibility not to call
  // Current() on an iterator equal to BaseGenerator()->End().
  // Used for implementing ParamGenerator<T>::operator*().
  virtual const T* Current() const = 0;
  // Determines whether the given iterator and other point to the same
  // element in the sequence generated by the generator.
  // Used for implementing ParamGenerator<T>::operator==().
  virtual bool Equals(const ParamIteratorInterface& other) const = 0;
};

// Class iterating over elements provided by an implementation of
// ParamGeneratorInterface<T>. It wraps ParamIteratorInterface<T>
// and implements the const forward iterator concept.
template <typename T>
class ParamIterator {
 public:
  typedef T value_type;
  typedef const T& reference;
  typedef ptrdiff_t difference_type;

  // ParamIterator assumes ownership of the impl_ pointer.
  ParamIterator(const ParamIterator& other) : impl_(other.impl_->Clone()) {}
  ParamIterator& operator=(const ParamIterator& other) {
    if (this != &other)
      impl_.reset(other.impl_->Clone());
    return *this;
  }

  const T& operator*() const { return *impl_->Current(); }
  const T* operator->() const { return impl_->Current(); }
  // Prefix version of operator++.
  ParamIterator& operator++() {
    impl_->Advance();
    return *this;
  }
  // Postfix version of operator++.
  ParamIterator operator++(int /*unused*/) {
    ParamIteratorInterface<T>* clone = impl_->Clone();
    impl_->Advance();
    return ParamIterator(clone);
  }
  bool operator==(const ParamIterator& other) const {
    return impl_.get() == other.impl_.get() || impl_->Equals(*other.impl_);
  }
  bool operator!=(const ParamIterator& other) const {
    return !(*this == other);
  }

 private:
  friend class ParamGenerator<T>;
  explicit ParamIterator(ParamIteratorInterface<T>* impl) : impl_(impl) {}
  scoped_ptr<ParamIteratorInterface<T> > impl_;
};

// ParamGeneratorInterface<T> is the binary interface to access generators
// defined in other translation units.
template <typename T>
class ParamGeneratorInterface {
 public:
  typedef T ParamType;

  virtual ~ParamGeneratorInterface() {}

  // Generator interface definition
  virtual ParamIteratorInterface<T>* Begin() const = 0;
  virtual ParamIteratorInterface<T>* End() const = 0;
};

// Wraps ParamGeneratorInterface<T> and provides general generator syntax
// compatible with the STL Container concept.
// This class implements copy initialization semantics and the contained
// ParamGeneratorInterface<T> instance is shared among all copies
// of the original object. This is possible because that instance is immutable.
template<typename T>
class ParamGenerator {
 public:
  typedef ParamIterator<T> iterator;

  explicit ParamGenerator(ParamGeneratorInterface<T>* impl) : impl_(impl) {}
  ParamGenerator(const ParamGenerator& other) : impl_(other.impl_) {}

  ParamGenerator& operator=(const ParamGenerator& other) {
    impl_ = other.impl_;
    return *this;
  }

  iterator begin() const { return iterator(impl_->Begin()); }
  iterator end() const { return iterator(impl_->End()); }

 private:
  linked_ptr<const ParamGeneratorInterface<T> > impl_;
};

// Generates values from a range of two comparable values. Can be used to
// generate sequences of user-defined types that implement operator+() and
// operator<().
// This class is used in the Range() function.
template <typename T, typename IncrementT>
class RangeGenerator : public ParamGeneratorInterface<T> {
 public:
  RangeGenerator(T begin, T end, IncrementT step)
      : begin_(begin), end_(end),
        step_(step), end_index_(CalculateEndIndex(begin, end, step)) {}
  virtual ~RangeGenerator() {}

  virtual ParamIteratorInterface<T>* Begin() const {
    return new Iterator(this, begin_, 0, step_);
  }
  virtual ParamIteratorInterface<T>* End() const {
    return new Iterator(this, end_, end_index_, step_);
  }

 private:
  class Iterator : public ParamIteratorInterface<T> {
   public:
    Iterator(const ParamGeneratorInterface<T>* base, T value, int index,
             IncrementT step)
        : base_(base), value_(value), index_(index), step_(step) {}
    virtual ~Iterator() {}

    virtual const ParamGeneratorInterface<T>* BaseGenerator() const {
      return base_;
    }
    virtual void Advance() {
      value_ = value_ + step_;
      index_++;
    }
    virtual ParamIteratorInterface<T>* Clone() const {
      return new Iterator(*this);
    }
    virtual const T* Current() const { return &value_; }
    virtual bool Equals(const ParamIteratorInterface<T>& other) const {
      // Having the same base generator guarantees that the other
      // iterator is of the same type and we can downcast.
      GTEST_CHECK_(BaseGenerator() == other.BaseGenerator())
          << "The program attempted to compare iterators "
          << "from different generators." << std::endl;
      const int other_index =
          CheckedDowncastToActualType<const Iterator>(&other)->index_;
      return index_ == other_index;
    }

   private:
    Iterator(const Iterator& other)
        : ParamIteratorInterface<T>(),
          base_(other.base_), value_(other.value_), index_(other.index_),
          step_(other.step_) {}

    // No implementation - assignment is unsupported.
    void operator=(const Iterator& other);

    const ParamGeneratorInterface<T>* const base_;
    T value_;
    int index_;
    const IncrementT step_;
  };  // class RangeGenerator::Iterator

  static int CalculateEndIndex(const T& begin,
                               const T& end,
                               const IncrementT& step) {
    int end_index = 0;
    for (T i = begin; i < end; i = i + step)
      end_index++;
    return end_index;
  }

  // No implementation - assignment is unsupported.
  void operator=(const RangeGenerator& other);

  const T begin_;
  const T end_;
  const IncrementT step_;
  // The index for the end() iterator. All the elements in the generated
  // sequence are indexed (0-based) to aid iterator comparison.
  const int end_index_;
};  // class RangeGenerator


// Generates values from a pair of STL-style iterators. Used in the
// ValuesIn() function. The elements are copied from the source range
// since the source can be located on the stack, and the generator
// is likely to persist beyond that stack frame.
template <typename T>
class ValuesInIteratorRangeGenerator : public ParamGeneratorInterface<T> {
 public:
  template <typename ForwardIterator>
  ValuesInIteratorRangeGenerator(ForwardIterator begin, ForwardIterator end)
      : container_(begin, end) {}
  virtual ~ValuesInIteratorRangeGenerator() {}

  virtual ParamIteratorInterface<T>* Begin() const {
    return new Iterator(this, container_.begin());
  }
  virtual ParamIteratorInterface<T>* End() const {
    return new Iterator(this, container_.end());
  }

 private:
  typedef typename ::std::vector<T> ContainerType;

  class Iterator : public ParamIteratorInterface<T> {
   public:
    Iterator(const ParamGeneratorInterface<T>* base,
             typename ContainerType::const_iterator iterator)
        : base_(base), iterator_(iterator) {}
    virtual ~Iterator() {}

    virtual const ParamGeneratorInterface<T>* BaseGenerator() const {
      return base_;
    }
    virtual void Advance() {
      ++iterator_;
      value_.reset();
    }
    virtual ParamIteratorInterface<T>* Clone() const {
      return new Iterator(*this);
    }
    // We need to use cached value referenced by iterator_ because *iterator_
    // can return a temporary object (and of type other then T), so just
    // having "return &*iterator_;" doesn't work.
    // value_ is updated here and not in Advance() because Advance()
    // can advance iterator_ beyond the end of the range, and we cannot
    // detect that fact. The client code, on the other hand, is
    // responsible for not calling Current() on an out-of-range iterator.
    virtual const T* Current() const {
      if (value_.get() == NULL)
        value_.reset(new T(*iterator_));
      return value_.get();
    }
    virtual bool Equals(const ParamIteratorInterface<T>& other) const {
      // Having the same base generator guarantees that the other
      // iterator is of the same type and we can downcast.
      GTEST_CHECK_(BaseGenerator() == other.BaseGenerator())
          << "The program attempted to compare iterators "
          << "from different generators." << std::endl;
      return iterator_ ==
          CheckedDowncastToActualType<const Iterator>(&other)->iterator_;
    }

   private:
    Iterator(const Iterator& other)
          // The explicit constructor call suppresses a false warning
          // emitted by gcc when supplied with the -Wextra option.
        : ParamIteratorInterface<T>(),
          base_(other.base_),
          iterator_(other.iterator_) {}

    const ParamGeneratorInterface<T>* const base_;
    typename ContainerType::const_iterator iterator_;
    // A cached value of *iterator_. We keep it here to allow access by
    // pointer in the wrapping iterator's operator->().
    // value_ needs to be mutable to be accessed in Current().
    // Use of scoped_ptr helps manage cached value's lifetime,
    // which is bound by the lifespan of the iterator itself.
    mutable scoped_ptr<const T> value_;
  };  // class ValuesInIteratorRangeGenerator::Iterator

  // No implementation - assignment is unsupported.
  void operator=(const ValuesInIteratorRangeGenerator& other);

  const ContainerType container_;
};  // class ValuesInIteratorRangeGenerator

// INTERNAL IMPLEMENTATION - DO NOT USE IN USER CODE.
//
// Stores a parameter value and later creates tests parameterized with that
// value.
template <class TestClass>
class ParameterizedTestFactory : public TestFactoryBase {
 public:
  typedef typename TestClass::ParamType ParamType;
  explicit ParameterizedTestFactory(ParamType parameter) :
      parameter_(parameter) {}
  virtual Test* CreateTest() {
    TestClass::SetParam(&parameter_);
    return new TestClass();
  }

 private:
  const ParamType parameter_;

  GTEST_DISALLOW_COPY_AND_ASSIGN_(ParameterizedTestFactory);
};

// INTERNAL IMPLEMENTATION - DO NOT USE IN USER CODE.
//
// TestMetaFactoryBase is a base class for meta-factories that create
// test factories for passing into MakeAndRegisterTestInfo function.
template <class ParamType>
class TestMetaFactoryBase {
 public:
  virtual ~TestMetaFactoryBase() {}

  virtual TestFactoryBase* CreateTestFactory(ParamType parameter) = 0;
};

// INTERNAL IMPLEMENTATION - DO NOT USE IN USER CODE.
//
// TestMetaFactory creates test factories for passing into
// MakeAndRegisterTestInfo function. Since MakeAndRegisterTestInfo receives
// ownership of test factory pointer, same factory object cannot be passed
// into that method twice. But ParameterizedTestCaseInfo is going to call
// it for each Test/Parameter value combination. Thus it needs meta factory
// creator class.
template <class TestCase>
class TestMetaFactory
    : public TestMetaFactoryBase<typename TestCase::ParamType> {
 public:
  typedef typename TestCase::ParamType ParamType;

  TestMetaFactory() {}

  virtual TestFactoryBase* CreateTestFactory(ParamType parameter) {
    return new ParameterizedTestFactory<TestCase>(parameter);
  }

 private:
  GTEST_DISALLOW_COPY_AND_ASSIGN_(TestMetaFactory);
};

// INTERNAL IMPLEMENTATION - DO NOT USE IN USER CODE.
//
// ParameterizedTestCaseInfoBase is a generic interface
// to ParameterizedTestCaseInfo classes. ParameterizedTestCaseInfoBase
// accumulates test information provided by TEST_P macro invocations
// and generators provided by INSTANTIATE_TEST_CASE_P macro invocations
// and uses that information to register all resulting test instances
// in RegisterTests method. The ParameterizeTestCaseRegistry class holds
// a collection of pointers to the ParameterizedTestCaseInfo objects
// and calls RegisterTests() on each of them when asked.
class ParameterizedTestCaseInfoBase {
 public:
  virtual ~ParameterizedTestCaseInfoBase() {}

  // Base part of test case name for display purposes.
  virtual const string& GetTestCaseName() const = 0;
  // Test case id to verify identity.
  virtual TypeId GetTestCaseTypeId() const = 0;
  // UnitTest class invokes this method to register tests in this
  // test case right before running them in RUN_ALL_TESTS macro.
  // This method should not be called more then once on any single
  // instance of a ParameterizedTestCaseInfoBase derived class.
  virtual void RegisterTests() = 0;

 protected:
  ParameterizedTestCaseInfoBase() {}

 private:
  GTEST_DISALLOW_COPY_AND_ASSIGN_(ParameterizedTestCaseInfoBase);
};

// INTERNAL IMPLEMENTATION - DO NOT USE IN USER CODE.
//
// ParameterizedTestCaseInfo accumulates tests obtained from TEST_P
// macro invocations for a particular test case and generators
// obtained from INSTANTIATE_TEST_CASE_P macro invocations for that
// test case. It registers tests with all values generated by all
// generators when asked.
template <class TestCase>
class ParameterizedTestCaseInfo : public ParameterizedTestCaseInfoBase {
 public:
  // ParamType and GeneratorCreationFunc are private types but are required
  // for declarations of public methods AddTestPattern() and
  // AddTestCaseInstantiation().
  typedef typename TestCase::ParamType ParamType;
  // A function that returns an instance of appropriate generator type.
  typedef ParamGenerator<ParamType>(GeneratorCreationFunc)();

  explicit ParameterizedTestCaseInfo(const char* name)
      : test_case_name_(name) {}

  // Test case base name for display purposes.
  virtual const string& GetTestCaseName() const { return test_case_name_; }
  // Test case id to verify identity.
  virtual TypeId GetTestCaseTypeId() const { return GetTypeId<TestCase>(); }
  // TEST_P macro uses AddTestPattern() to record information
  // about a single test in a LocalTestInfo structure.
  // test_case_name is the base name of the test case (without invocation
  // prefix). test_base_name is the name of an individual test without
  // parameter index. For the test SequenceA/FooTest.DoBar/1 FooTest is
  // test case base name and DoBar is test base name.
  void AddTestPattern(const char* test_case_name,
                      const char* test_base_name,
                      TestMetaFactoryBase<ParamType>* meta_factory) {
    tests_.push_back(linked_ptr<TestInfo>(new TestInfo(test_case_name,
                                                       test_base_name,
                                                       meta_factory)));
  }
  // INSTANTIATE_TEST_CASE_P macro uses AddGenerator() to record information
  // about a generator.
  int AddTestCaseInstantiation(const string& instantiation_name,
                               GeneratorCreationFunc* func,
                               const char* /* file */,
                               int /* line */) {
    instantiations_.push_back(::std::make_pair(instantiation_name, func));
    return 0;  // Return value used only to run this method in namespace scope.
  }
  // UnitTest class invokes this method to register tests in this test case
  // test cases right before running tests in RUN_ALL_TESTS macro.
  // This method should not be called more then once on any single
  // instance of a ParameterizedTestCaseInfoBase derived class.
  // UnitTest has a guard to prevent from calling this method more then once.
  virtual void RegisterTests() {
    for (typename TestInfoContainer::iterator test_it = tests_.begin();
         test_it != tests_.end(); ++test_it) {
      linked_ptr<TestInfo> test_info = *test_it;
      for (typename InstantiationContainer::iterator gen_it =
               instantiations_.begin(); gen_it != instantiations_.end();
               ++gen_it) {
        const string& instantiation_name = gen_it->first;
        ParamGenerator<ParamType> generator((*gen_it->second)());

        Message test_case_name_stream;
        if ( !instantiation_name.empty() )
          test_case_name_stream << instantiation_name << "/";
        test_case_name_stream << test_info->test_case_base_name;

        int i = 0;
        for (typename ParamGenerator<ParamType>::iterator param_it =
                 generator.begin();
             param_it != generator.end(); ++param_it, ++i) {
          Message test_name_stream;
          test_name_stream << test_info->test_base_name << "/" << i;
          MakeAndRegisterTestInfo(
              test_case_name_stream.GetString().c_str(),
              test_name_stream.GetString().c_str(),
              NULL,  // No type parameter.
              PrintToString(*param_it).c_str(),
              GetTestCaseTypeId(),
              TestCase::SetUpTestCase,
              TestCase::TearDownTestCase,
              test_info->test_meta_factory->CreateTestFactory(*param_it));
        }  // for param_it
      }  // for gen_it
    }  // for test_it
  }  // RegisterTests

 private:
  // LocalTestInfo structure keeps information about a single test registered
  // with TEST_P macro.
  struct TestInfo {
    TestInfo(const char* a_test_case_base_name,
             const char* a_test_base_name,
             TestMetaFactoryBase<ParamType>* a_test_meta_factory) :
        test_case_base_name(a_test_case_base_name),
        test_base_name(a_test_base_name),
        test_meta_factory(a_test_meta_factory) {}

    const string test_case_base_name;
    const string test_base_name;
    const scoped_ptr<TestMetaFactoryBase<ParamType> > test_meta_factory;
  };
  typedef ::std::vector<linked_ptr<TestInfo> > TestInfoContainer;
  // Keeps pairs of <Instantiation name, Sequence generator creation function>
  // received from INSTANTIATE_TEST_CASE_P macros.
  typedef ::std::vector<std::pair<string, GeneratorCreationFunc*> >
      InstantiationContainer;

  const string test_case_name_;
  TestInfoContainer tests_;
  InstantiationContainer instantiations_;

  GTEST_DISALLOW_COPY_AND_ASSIGN_(ParameterizedTestCaseInfo);
};  // class ParameterizedTestCaseInfo

// INTERNAL IMPLEMENTATION - DO NOT USE IN USER CODE.
//
// ParameterizedTestCaseRegistry contains a map of ParameterizedTestCaseInfoBase
// classes accessed by test case names. TEST_P and INSTANTIATE_TEST_CASE_P
// macros use it to locate their corresponding ParameterizedTestCaseInfo
// descriptors.
class ParameterizedTestCaseRegistry {
 public:
  ParameterizedTestCaseRegistry() {}
  ~ParameterizedTestCaseRegistry() {
    for (TestCaseInfoContainer::iterator it = test_case_infos_.begin();
         it != test_case_infos_.end(); ++it) {
      delete *it;
    }
  }

  // Looks up or creates and returns a structure containing information about
  // tests and instantiations of a particular test case.
  template <class TestCase>
  ParameterizedTestCaseInfo<TestCase>* GetTestCasePatternHolder(
      const char* test_case_name,
      const char* file,
      int line) {
    ParameterizedTestCaseInfo<TestCase>* typed_test_info = NULL;
    for (TestCaseInfoContainer::iterator it = test_case_infos_.begin();
         it != test_case_infos_.end(); ++it) {
      if ((*it)->GetTestCaseName() == test_case_name) {
        if ((*it)->GetTestCaseTypeId() != GetTypeId<TestCase>()) {
          // Complain about incorrect usage of Google Test facilities
          // and terminate the program since we cannot guaranty correct
          // test case setup and tear-down in this case.
          ReportInvalidTestCaseType(test_case_name,  file, line);
          posix::Abort();
        } else {
          // At this point we are sure that the object we found is of the same
          // type we are looking for, so we downcast it to that type
          // without further checks.
          typed_test_info = CheckedDowncastToActualType<
              ParameterizedTestCaseInfo<TestCase> >(*it);
        }
        break;
      }
    }
    if (typed_test_info == NULL) {
      typed_test_info = new ParameterizedTestCaseInfo<TestCase>(test_case_name);
      test_case_infos_.push_back(typed_test_info);
    }
    return typed_test_info;
  }
  void RegisterTests() {
    for (TestCaseInfoContainer::iterator it = test_case_infos_.begin();
         it != test_case_infos_.end(); ++it) {
      (*it)->RegisterTests();
    }
  }

 private:
  typedef ::std::vector<ParameterizedTestCaseInfoBase*> TestCaseInfoContainer;

  TestCaseInfoContainer test_case_infos_;

  GTEST_DISALLOW_COPY_AND_ASSIGN_(ParameterizedTestCaseRegistry);
};

}  // namespace internal
}  // namespace testing

#endif  //  GTEST_HAS_PARAM_TEST

#endif  // GTEST_INCLUDE_GTEST_INTERNAL_GTEST_PARAM_UTIL_H_
// This file was GENERATED by command:
//     pump.py gtest-param-util-generated.h.pump
// DO NOT EDIT BY HAND!!!

// Copyright 2008 Google Inc.
// All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Author: vladl@google.com (Vlad Losev)

// Type and function utilities for implementing parameterized tests.
// This file is generated by a SCRIPT.  DO NOT EDIT BY HAND!
//
// Currently Google Test supports at most 50 arguments in Values,
// and at most 10 arguments in Combine. Please contact
// googletestframework@googlegroups.com if you need more.
// Please note that the number of arguments to Combine is limited
// by the maximum arity of the implementation of tr1::tuple which is
// currently set at 10.

#ifndef GTEST_INCLUDE_GTEST_INTERNAL_GTEST_PARAM_UTIL_GENERATED_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_GTEST_PARAM_UTIL_GENERATED_H_

// scripts/fuse_gtest.py depends on gtest's own header being #included
// *unconditionally*.  Therefore these #includes cannot be moved
// inside #if GTEST_HAS_PARAM_TEST.

#if GTEST_HAS_PARAM_TEST

namespace testing {

// Forward declarations of ValuesIn(), which is implemented in
// include/gtest/gtest-param-test.h.
template <typename ForwardIterator>
internal::ParamGenerator<
  typename ::testing::internal::IteratorTraits<ForwardIterator>::value_type>
ValuesIn(ForwardIterator begin, ForwardIterator end);

template <typename T, size_t N>
internal::ParamGenerator<T> ValuesIn(const T (&array)[N]);

template <class Container>
internal::ParamGenerator<typename Container::value_type> ValuesIn(
    const Container& container);

namespace internal {

// Used in the Values() function to provide polymorphic capabilities.
template <typename T1>
class ValueArray1 {
 public:
  explicit ValueArray1(T1 v1) : v1_(v1) {}

  template <typename T>
  operator ParamGenerator<T>() const { return ValuesIn(&v1_, &v1_ + 1); }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray1& other);

  const T1 v1_;
};

template <typename T1, typename T2>
class ValueArray2 {
 public:
  ValueArray2(T1 v1, T2 v2) : v1_(v1), v2_(v2) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray2& other);

  const T1 v1_;
  const T2 v2_;
};

template <typename T1, typename T2, typename T3>
class ValueArray3 {
 public:
  ValueArray3(T1 v1, T2 v2, T3 v3) : v1_(v1), v2_(v2), v3_(v3) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray3& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
};

template <typename T1, typename T2, typename T3, typename T4>
class ValueArray4 {
 public:
  ValueArray4(T1 v1, T2 v2, T3 v3, T4 v4) : v1_(v1), v2_(v2), v3_(v3),
      v4_(v4) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray4& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5>
class ValueArray5 {
 public:
  ValueArray5(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5) : v1_(v1), v2_(v2), v3_(v3),
      v4_(v4), v5_(v5) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray5& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6>
class ValueArray6 {
 public:
  ValueArray6(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6) : v1_(v1), v2_(v2),
      v3_(v3), v4_(v4), v5_(v5), v6_(v6) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray6& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7>
class ValueArray7 {
 public:
  ValueArray7(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7) : v1_(v1),
      v2_(v2), v3_(v3), v4_(v4), v5_(v5), v6_(v6), v7_(v7) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray7& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8>
class ValueArray8 {
 public:
  ValueArray8(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7,
      T8 v8) : v1_(v1), v2_(v2), v3_(v3), v4_(v4), v5_(v5), v6_(v6), v7_(v7),
      v8_(v8) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray8& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9>
class ValueArray9 {
 public:
  ValueArray9(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8,
      T9 v9) : v1_(v1), v2_(v2), v3_(v3), v4_(v4), v5_(v5), v6_(v6), v7_(v7),
      v8_(v8), v9_(v9) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray9& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10>
class ValueArray10 {
 public:
  ValueArray10(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10) : v1_(v1), v2_(v2), v3_(v3), v4_(v4), v5_(v5), v6_(v6), v7_(v7),
      v8_(v8), v9_(v9), v10_(v10) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray10& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11>
class ValueArray11 {
 public:
  ValueArray11(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11) : v1_(v1), v2_(v2), v3_(v3), v4_(v4), v5_(v5), v6_(v6),
      v7_(v7), v8_(v8), v9_(v9), v10_(v10), v11_(v11) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray11& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12>
class ValueArray12 {
 public:
  ValueArray12(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12) : v1_(v1), v2_(v2), v3_(v3), v4_(v4), v5_(v5),
      v6_(v6), v7_(v7), v8_(v8), v9_(v9), v10_(v10), v11_(v11), v12_(v12) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray12& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13>
class ValueArray13 {
 public:
  ValueArray13(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13) : v1_(v1), v2_(v2), v3_(v3), v4_(v4),
      v5_(v5), v6_(v6), v7_(v7), v8_(v8), v9_(v9), v10_(v10), v11_(v11),
      v12_(v12), v13_(v13) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray13& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14>
class ValueArray14 {
 public:
  ValueArray14(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14) : v1_(v1), v2_(v2), v3_(v3),
      v4_(v4), v5_(v5), v6_(v6), v7_(v7), v8_(v8), v9_(v9), v10_(v10),
      v11_(v11), v12_(v12), v13_(v13), v14_(v14) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray14& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15>
class ValueArray15 {
 public:
  ValueArray15(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15) : v1_(v1), v2_(v2),
      v3_(v3), v4_(v4), v5_(v5), v6_(v6), v7_(v7), v8_(v8), v9_(v9), v10_(v10),
      v11_(v11), v12_(v12), v13_(v13), v14_(v14), v15_(v15) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray15& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16>
class ValueArray16 {
 public:
  ValueArray16(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16) : v1_(v1),
      v2_(v2), v3_(v3), v4_(v4), v5_(v5), v6_(v6), v7_(v7), v8_(v8), v9_(v9),
      v10_(v10), v11_(v11), v12_(v12), v13_(v13), v14_(v14), v15_(v15),
      v16_(v16) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray16& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17>
class ValueArray17 {
 public:
  ValueArray17(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16,
      T17 v17) : v1_(v1), v2_(v2), v3_(v3), v4_(v4), v5_(v5), v6_(v6), v7_(v7),
      v8_(v8), v9_(v9), v10_(v10), v11_(v11), v12_(v12), v13_(v13), v14_(v14),
      v15_(v15), v16_(v16), v17_(v17) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray17& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18>
class ValueArray18 {
 public:
  ValueArray18(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18) : v1_(v1), v2_(v2), v3_(v3), v4_(v4), v5_(v5), v6_(v6), v7_(v7),
      v8_(v8), v9_(v9), v10_(v10), v11_(v11), v12_(v12), v13_(v13), v14_(v14),
      v15_(v15), v16_(v16), v17_(v17), v18_(v18) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray18& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19>
class ValueArray19 {
 public:
  ValueArray19(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19) : v1_(v1), v2_(v2), v3_(v3), v4_(v4), v5_(v5), v6_(v6),
      v7_(v7), v8_(v8), v9_(v9), v10_(v10), v11_(v11), v12_(v12), v13_(v13),
      v14_(v14), v15_(v15), v16_(v16), v17_(v17), v18_(v18), v19_(v19) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray19& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20>
class ValueArray20 {
 public:
  ValueArray20(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20) : v1_(v1), v2_(v2), v3_(v3), v4_(v4), v5_(v5),
      v6_(v6), v7_(v7), v8_(v8), v9_(v9), v10_(v10), v11_(v11), v12_(v12),
      v13_(v13), v14_(v14), v15_(v15), v16_(v16), v17_(v17), v18_(v18),
      v19_(v19), v20_(v20) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray20& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21>
class ValueArray21 {
 public:
  ValueArray21(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21) : v1_(v1), v2_(v2), v3_(v3), v4_(v4),
      v5_(v5), v6_(v6), v7_(v7), v8_(v8), v9_(v9), v10_(v10), v11_(v11),
      v12_(v12), v13_(v13), v14_(v14), v15_(v15), v16_(v16), v17_(v17),
      v18_(v18), v19_(v19), v20_(v20), v21_(v21) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray21& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22>
class ValueArray22 {
 public:
  ValueArray22(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22) : v1_(v1), v2_(v2), v3_(v3),
      v4_(v4), v5_(v5), v6_(v6), v7_(v7), v8_(v8), v9_(v9), v10_(v10),
      v11_(v11), v12_(v12), v13_(v13), v14_(v14), v15_(v15), v16_(v16),
      v17_(v17), v18_(v18), v19_(v19), v20_(v20), v21_(v21), v22_(v22) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray22& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23>
class ValueArray23 {
 public:
  ValueArray23(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23) : v1_(v1), v2_(v2),
      v3_(v3), v4_(v4), v5_(v5), v6_(v6), v7_(v7), v8_(v8), v9_(v9), v10_(v10),
      v11_(v11), v12_(v12), v13_(v13), v14_(v14), v15_(v15), v16_(v16),
      v17_(v17), v18_(v18), v19_(v19), v20_(v20), v21_(v21), v22_(v22),
      v23_(v23) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_,
        v23_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray23& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
  const T23 v23_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24>
class ValueArray24 {
 public:
  ValueArray24(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24) : v1_(v1),
      v2_(v2), v3_(v3), v4_(v4), v5_(v5), v6_(v6), v7_(v7), v8_(v8), v9_(v9),
      v10_(v10), v11_(v11), v12_(v12), v13_(v13), v14_(v14), v15_(v15),
      v16_(v16), v17_(v17), v18_(v18), v19_(v19), v20_(v20), v21_(v21),
      v22_(v22), v23_(v23), v24_(v24) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_, v23_,
        v24_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray24& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
  const T23 v23_;
  const T24 v24_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25>
class ValueArray25 {
 public:
  ValueArray25(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24,
      T25 v25) : v1_(v1), v2_(v2), v3_(v3), v4_(v4), v5_(v5), v6_(v6), v7_(v7),
      v8_(v8), v9_(v9), v10_(v10), v11_(v11), v12_(v12), v13_(v13), v14_(v14),
      v15_(v15), v16_(v16), v17_(v17), v18_(v18), v19_(v19), v20_(v20),
      v21_(v21), v22_(v22), v23_(v23), v24_(v24), v25_(v25) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_, v23_,
        v24_, v25_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray25& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
  const T23 v23_;
  const T24 v24_;
  const T25 v25_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26>
class ValueArray26 {
 public:
  ValueArray26(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
      T26 v26) : v1_(v1), v2_(v2), v3_(v3), v4_(v4), v5_(v5), v6_(v6), v7_(v7),
      v8_(v8), v9_(v9), v10_(v10), v11_(v11), v12_(v12), v13_(v13), v14_(v14),
      v15_(v15), v16_(v16), v17_(v17), v18_(v18), v19_(v19), v20_(v20),
      v21_(v21), v22_(v22), v23_(v23), v24_(v24), v25_(v25), v26_(v26) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_, v23_,
        v24_, v25_, v26_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray26& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
  const T23 v23_;
  const T24 v24_;
  const T25 v25_;
  const T26 v26_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27>
class ValueArray27 {
 public:
  ValueArray27(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
      T26 v26, T27 v27) : v1_(v1), v2_(v2), v3_(v3), v4_(v4), v5_(v5), v6_(v6),
      v7_(v7), v8_(v8), v9_(v9), v10_(v10), v11_(v11), v12_(v12), v13_(v13),
      v14_(v14), v15_(v15), v16_(v16), v17_(v17), v18_(v18), v19_(v19),
      v20_(v20), v21_(v21), v22_(v22), v23_(v23), v24_(v24), v25_(v25),
      v26_(v26), v27_(v27) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_, v23_,
        v24_, v25_, v26_, v27_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray27& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
  const T23 v23_;
  const T24 v24_;
  const T25 v25_;
  const T26 v26_;
  const T27 v27_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28>
class ValueArray28 {
 public:
  ValueArray28(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
      T26 v26, T27 v27, T28 v28) : v1_(v1), v2_(v2), v3_(v3), v4_(v4), v5_(v5),
      v6_(v6), v7_(v7), v8_(v8), v9_(v9), v10_(v10), v11_(v11), v12_(v12),
      v13_(v13), v14_(v14), v15_(v15), v16_(v16), v17_(v17), v18_(v18),
      v19_(v19), v20_(v20), v21_(v21), v22_(v22), v23_(v23), v24_(v24),
      v25_(v25), v26_(v26), v27_(v27), v28_(v28) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_, v23_,
        v24_, v25_, v26_, v27_, v28_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray28& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
  const T23 v23_;
  const T24 v24_;
  const T25 v25_;
  const T26 v26_;
  const T27 v27_;
  const T28 v28_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29>
class ValueArray29 {
 public:
  ValueArray29(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
      T26 v26, T27 v27, T28 v28, T29 v29) : v1_(v1), v2_(v2), v3_(v3), v4_(v4),
      v5_(v5), v6_(v6), v7_(v7), v8_(v8), v9_(v9), v10_(v10), v11_(v11),
      v12_(v12), v13_(v13), v14_(v14), v15_(v15), v16_(v16), v17_(v17),
      v18_(v18), v19_(v19), v20_(v20), v21_(v21), v22_(v22), v23_(v23),
      v24_(v24), v25_(v25), v26_(v26), v27_(v27), v28_(v28), v29_(v29) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_, v23_,
        v24_, v25_, v26_, v27_, v28_, v29_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray29& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
  const T23 v23_;
  const T24 v24_;
  const T25 v25_;
  const T26 v26_;
  const T27 v27_;
  const T28 v28_;
  const T29 v29_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30>
class ValueArray30 {
 public:
  ValueArray30(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
      T26 v26, T27 v27, T28 v28, T29 v29, T30 v30) : v1_(v1), v2_(v2), v3_(v3),
      v4_(v4), v5_(v5), v6_(v6), v7_(v7), v8_(v8), v9_(v9), v10_(v10),
      v11_(v11), v12_(v12), v13_(v13), v14_(v14), v15_(v15), v16_(v16),
      v17_(v17), v18_(v18), v19_(v19), v20_(v20), v21_(v21), v22_(v22),
      v23_(v23), v24_(v24), v25_(v25), v26_(v26), v27_(v27), v28_(v28),
      v29_(v29), v30_(v30) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_, v23_,
        v24_, v25_, v26_, v27_, v28_, v29_, v30_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray30& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
  const T23 v23_;
  const T24 v24_;
  const T25 v25_;
  const T26 v26_;
  const T27 v27_;
  const T28 v28_;
  const T29 v29_;
  const T30 v30_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31>
class ValueArray31 {
 public:
  ValueArray31(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
      T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31) : v1_(v1), v2_(v2),
      v3_(v3), v4_(v4), v5_(v5), v6_(v6), v7_(v7), v8_(v8), v9_(v9), v10_(v10),
      v11_(v11), v12_(v12), v13_(v13), v14_(v14), v15_(v15), v16_(v16),
      v17_(v17), v18_(v18), v19_(v19), v20_(v20), v21_(v21), v22_(v22),
      v23_(v23), v24_(v24), v25_(v25), v26_(v26), v27_(v27), v28_(v28),
      v29_(v29), v30_(v30), v31_(v31) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_, v23_,
        v24_, v25_, v26_, v27_, v28_, v29_, v30_, v31_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray31& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
  const T23 v23_;
  const T24 v24_;
  const T25 v25_;
  const T26 v26_;
  const T27 v27_;
  const T28 v28_;
  const T29 v29_;
  const T30 v30_;
  const T31 v31_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32>
class ValueArray32 {
 public:
  ValueArray32(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
      T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31, T32 v32) : v1_(v1),
      v2_(v2), v3_(v3), v4_(v4), v5_(v5), v6_(v6), v7_(v7), v8_(v8), v9_(v9),
      v10_(v10), v11_(v11), v12_(v12), v13_(v13), v14_(v14), v15_(v15),
      v16_(v16), v17_(v17), v18_(v18), v19_(v19), v20_(v20), v21_(v21),
      v22_(v22), v23_(v23), v24_(v24), v25_(v25), v26_(v26), v27_(v27),
      v28_(v28), v29_(v29), v30_(v30), v31_(v31), v32_(v32) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_, v23_,
        v24_, v25_, v26_, v27_, v28_, v29_, v30_, v31_, v32_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray32& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
  const T23 v23_;
  const T24 v24_;
  const T25 v25_;
  const T26 v26_;
  const T27 v27_;
  const T28 v28_;
  const T29 v29_;
  const T30 v30_;
  const T31 v31_;
  const T32 v32_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33>
class ValueArray33 {
 public:
  ValueArray33(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
      T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31, T32 v32,
      T33 v33) : v1_(v1), v2_(v2), v3_(v3), v4_(v4), v5_(v5), v6_(v6), v7_(v7),
      v8_(v8), v9_(v9), v10_(v10), v11_(v11), v12_(v12), v13_(v13), v14_(v14),
      v15_(v15), v16_(v16), v17_(v17), v18_(v18), v19_(v19), v20_(v20),
      v21_(v21), v22_(v22), v23_(v23), v24_(v24), v25_(v25), v26_(v26),
      v27_(v27), v28_(v28), v29_(v29), v30_(v30), v31_(v31), v32_(v32),
      v33_(v33) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_, v23_,
        v24_, v25_, v26_, v27_, v28_, v29_, v30_, v31_, v32_, v33_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray33& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
  const T23 v23_;
  const T24 v24_;
  const T25 v25_;
  const T26 v26_;
  const T27 v27_;
  const T28 v28_;
  const T29 v29_;
  const T30 v30_;
  const T31 v31_;
  const T32 v32_;
  const T33 v33_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34>
class ValueArray34 {
 public:
  ValueArray34(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
      T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31, T32 v32, T33 v33,
      T34 v34) : v1_(v1), v2_(v2), v3_(v3), v4_(v4), v5_(v5), v6_(v6), v7_(v7),
      v8_(v8), v9_(v9), v10_(v10), v11_(v11), v12_(v12), v13_(v13), v14_(v14),
      v15_(v15), v16_(v16), v17_(v17), v18_(v18), v19_(v19), v20_(v20),
      v21_(v21), v22_(v22), v23_(v23), v24_(v24), v25_(v25), v26_(v26),
      v27_(v27), v28_(v28), v29_(v29), v30_(v30), v31_(v31), v32_(v32),
      v33_(v33), v34_(v34) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_, v23_,
        v24_, v25_, v26_, v27_, v28_, v29_, v30_, v31_, v32_, v33_, v34_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray34& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
  const T23 v23_;
  const T24 v24_;
  const T25 v25_;
  const T26 v26_;
  const T27 v27_;
  const T28 v28_;
  const T29 v29_;
  const T30 v30_;
  const T31 v31_;
  const T32 v32_;
  const T33 v33_;
  const T34 v34_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35>
class ValueArray35 {
 public:
  ValueArray35(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
      T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31, T32 v32, T33 v33,
      T34 v34, T35 v35) : v1_(v1), v2_(v2), v3_(v3), v4_(v4), v5_(v5), v6_(v6),
      v7_(v7), v8_(v8), v9_(v9), v10_(v10), v11_(v11), v12_(v12), v13_(v13),
      v14_(v14), v15_(v15), v16_(v16), v17_(v17), v18_(v18), v19_(v19),
      v20_(v20), v21_(v21), v22_(v22), v23_(v23), v24_(v24), v25_(v25),
      v26_(v26), v27_(v27), v28_(v28), v29_(v29), v30_(v30), v31_(v31),
      v32_(v32), v33_(v33), v34_(v34), v35_(v35) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_, v23_,
        v24_, v25_, v26_, v27_, v28_, v29_, v30_, v31_, v32_, v33_, v34_,
        v35_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray35& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
  const T23 v23_;
  const T24 v24_;
  const T25 v25_;
  const T26 v26_;
  const T27 v27_;
  const T28 v28_;
  const T29 v29_;
  const T30 v30_;
  const T31 v31_;
  const T32 v32_;
  const T33 v33_;
  const T34 v34_;
  const T35 v35_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36>
class ValueArray36 {
 public:
  ValueArray36(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
      T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31, T32 v32, T33 v33,
      T34 v34, T35 v35, T36 v36) : v1_(v1), v2_(v2), v3_(v3), v4_(v4), v5_(v5),
      v6_(v6), v7_(v7), v8_(v8), v9_(v9), v10_(v10), v11_(v11), v12_(v12),
      v13_(v13), v14_(v14), v15_(v15), v16_(v16), v17_(v17), v18_(v18),
      v19_(v19), v20_(v20), v21_(v21), v22_(v22), v23_(v23), v24_(v24),
      v25_(v25), v26_(v26), v27_(v27), v28_(v28), v29_(v29), v30_(v30),
      v31_(v31), v32_(v32), v33_(v33), v34_(v34), v35_(v35), v36_(v36) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_, v23_,
        v24_, v25_, v26_, v27_, v28_, v29_, v30_, v31_, v32_, v33_, v34_, v35_,
        v36_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray36& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
  const T23 v23_;
  const T24 v24_;
  const T25 v25_;
  const T26 v26_;
  const T27 v27_;
  const T28 v28_;
  const T29 v29_;
  const T30 v30_;
  const T31 v31_;
  const T32 v32_;
  const T33 v33_;
  const T34 v34_;
  const T35 v35_;
  const T36 v36_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37>
class ValueArray37 {
 public:
  ValueArray37(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
      T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31, T32 v32, T33 v33,
      T34 v34, T35 v35, T36 v36, T37 v37) : v1_(v1), v2_(v2), v3_(v3), v4_(v4),
      v5_(v5), v6_(v6), v7_(v7), v8_(v8), v9_(v9), v10_(v10), v11_(v11),
      v12_(v12), v13_(v13), v14_(v14), v15_(v15), v16_(v16), v17_(v17),
      v18_(v18), v19_(v19), v20_(v20), v21_(v21), v22_(v22), v23_(v23),
      v24_(v24), v25_(v25), v26_(v26), v27_(v27), v28_(v28), v29_(v29),
      v30_(v30), v31_(v31), v32_(v32), v33_(v33), v34_(v34), v35_(v35),
      v36_(v36), v37_(v37) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_, v23_,
        v24_, v25_, v26_, v27_, v28_, v29_, v30_, v31_, v32_, v33_, v34_, v35_,
        v36_, v37_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray37& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
  const T23 v23_;
  const T24 v24_;
  const T25 v25_;
  const T26 v26_;
  const T27 v27_;
  const T28 v28_;
  const T29 v29_;
  const T30 v30_;
  const T31 v31_;
  const T32 v32_;
  const T33 v33_;
  const T34 v34_;
  const T35 v35_;
  const T36 v36_;
  const T37 v37_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38>
class ValueArray38 {
 public:
  ValueArray38(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
      T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31, T32 v32, T33 v33,
      T34 v34, T35 v35, T36 v36, T37 v37, T38 v38) : v1_(v1), v2_(v2), v3_(v3),
      v4_(v4), v5_(v5), v6_(v6), v7_(v7), v8_(v8), v9_(v9), v10_(v10),
      v11_(v11), v12_(v12), v13_(v13), v14_(v14), v15_(v15), v16_(v16),
      v17_(v17), v18_(v18), v19_(v19), v20_(v20), v21_(v21), v22_(v22),
      v23_(v23), v24_(v24), v25_(v25), v26_(v26), v27_(v27), v28_(v28),
      v29_(v29), v30_(v30), v31_(v31), v32_(v32), v33_(v33), v34_(v34),
      v35_(v35), v36_(v36), v37_(v37), v38_(v38) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_, v23_,
        v24_, v25_, v26_, v27_, v28_, v29_, v30_, v31_, v32_, v33_, v34_, v35_,
        v36_, v37_, v38_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray38& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
  const T23 v23_;
  const T24 v24_;
  const T25 v25_;
  const T26 v26_;
  const T27 v27_;
  const T28 v28_;
  const T29 v29_;
  const T30 v30_;
  const T31 v31_;
  const T32 v32_;
  const T33 v33_;
  const T34 v34_;
  const T35 v35_;
  const T36 v36_;
  const T37 v37_;
  const T38 v38_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39>
class ValueArray39 {
 public:
  ValueArray39(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
      T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31, T32 v32, T33 v33,
      T34 v34, T35 v35, T36 v36, T37 v37, T38 v38, T39 v39) : v1_(v1), v2_(v2),
      v3_(v3), v4_(v4), v5_(v5), v6_(v6), v7_(v7), v8_(v8), v9_(v9), v10_(v10),
      v11_(v11), v12_(v12), v13_(v13), v14_(v14), v15_(v15), v16_(v16),
      v17_(v17), v18_(v18), v19_(v19), v20_(v20), v21_(v21), v22_(v22),
      v23_(v23), v24_(v24), v25_(v25), v26_(v26), v27_(v27), v28_(v28),
      v29_(v29), v30_(v30), v31_(v31), v32_(v32), v33_(v33), v34_(v34),
      v35_(v35), v36_(v36), v37_(v37), v38_(v38), v39_(v39) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_, v23_,
        v24_, v25_, v26_, v27_, v28_, v29_, v30_, v31_, v32_, v33_, v34_, v35_,
        v36_, v37_, v38_, v39_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray39& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
  const T23 v23_;
  const T24 v24_;
  const T25 v25_;
  const T26 v26_;
  const T27 v27_;
  const T28 v28_;
  const T29 v29_;
  const T30 v30_;
  const T31 v31_;
  const T32 v32_;
  const T33 v33_;
  const T34 v34_;
  const T35 v35_;
  const T36 v36_;
  const T37 v37_;
  const T38 v38_;
  const T39 v39_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40>
class ValueArray40 {
 public:
  ValueArray40(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
      T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31, T32 v32, T33 v33,
      T34 v34, T35 v35, T36 v36, T37 v37, T38 v38, T39 v39, T40 v40) : v1_(v1),
      v2_(v2), v3_(v3), v4_(v4), v5_(v5), v6_(v6), v7_(v7), v8_(v8), v9_(v9),
      v10_(v10), v11_(v11), v12_(v12), v13_(v13), v14_(v14), v15_(v15),
      v16_(v16), v17_(v17), v18_(v18), v19_(v19), v20_(v20), v21_(v21),
      v22_(v22), v23_(v23), v24_(v24), v25_(v25), v26_(v26), v27_(v27),
      v28_(v28), v29_(v29), v30_(v30), v31_(v31), v32_(v32), v33_(v33),
      v34_(v34), v35_(v35), v36_(v36), v37_(v37), v38_(v38), v39_(v39),
      v40_(v40) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_, v23_,
        v24_, v25_, v26_, v27_, v28_, v29_, v30_, v31_, v32_, v33_, v34_, v35_,
        v36_, v37_, v38_, v39_, v40_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray40& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
  const T23 v23_;
  const T24 v24_;
  const T25 v25_;
  const T26 v26_;
  const T27 v27_;
  const T28 v28_;
  const T29 v29_;
  const T30 v30_;
  const T31 v31_;
  const T32 v32_;
  const T33 v33_;
  const T34 v34_;
  const T35 v35_;
  const T36 v36_;
  const T37 v37_;
  const T38 v38_;
  const T39 v39_;
  const T40 v40_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41>
class ValueArray41 {
 public:
  ValueArray41(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
      T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31, T32 v32, T33 v33,
      T34 v34, T35 v35, T36 v36, T37 v37, T38 v38, T39 v39, T40 v40,
      T41 v41) : v1_(v1), v2_(v2), v3_(v3), v4_(v4), v5_(v5), v6_(v6), v7_(v7),
      v8_(v8), v9_(v9), v10_(v10), v11_(v11), v12_(v12), v13_(v13), v14_(v14),
      v15_(v15), v16_(v16), v17_(v17), v18_(v18), v19_(v19), v20_(v20),
      v21_(v21), v22_(v22), v23_(v23), v24_(v24), v25_(v25), v26_(v26),
      v27_(v27), v28_(v28), v29_(v29), v30_(v30), v31_(v31), v32_(v32),
      v33_(v33), v34_(v34), v35_(v35), v36_(v36), v37_(v37), v38_(v38),
      v39_(v39), v40_(v40), v41_(v41) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_, v23_,
        v24_, v25_, v26_, v27_, v28_, v29_, v30_, v31_, v32_, v33_, v34_, v35_,
        v36_, v37_, v38_, v39_, v40_, v41_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray41& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
  const T23 v23_;
  const T24 v24_;
  const T25 v25_;
  const T26 v26_;
  const T27 v27_;
  const T28 v28_;
  const T29 v29_;
  const T30 v30_;
  const T31 v31_;
  const T32 v32_;
  const T33 v33_;
  const T34 v34_;
  const T35 v35_;
  const T36 v36_;
  const T37 v37_;
  const T38 v38_;
  const T39 v39_;
  const T40 v40_;
  const T41 v41_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42>
class ValueArray42 {
 public:
  ValueArray42(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
      T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31, T32 v32, T33 v33,
      T34 v34, T35 v35, T36 v36, T37 v37, T38 v38, T39 v39, T40 v40, T41 v41,
      T42 v42) : v1_(v1), v2_(v2), v3_(v3), v4_(v4), v5_(v5), v6_(v6), v7_(v7),
      v8_(v8), v9_(v9), v10_(v10), v11_(v11), v12_(v12), v13_(v13), v14_(v14),
      v15_(v15), v16_(v16), v17_(v17), v18_(v18), v19_(v19), v20_(v20),
      v21_(v21), v22_(v22), v23_(v23), v24_(v24), v25_(v25), v26_(v26),
      v27_(v27), v28_(v28), v29_(v29), v30_(v30), v31_(v31), v32_(v32),
      v33_(v33), v34_(v34), v35_(v35), v36_(v36), v37_(v37), v38_(v38),
      v39_(v39), v40_(v40), v41_(v41), v42_(v42) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_, v23_,
        v24_, v25_, v26_, v27_, v28_, v29_, v30_, v31_, v32_, v33_, v34_, v35_,
        v36_, v37_, v38_, v39_, v40_, v41_, v42_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray42& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
  const T23 v23_;
  const T24 v24_;
  const T25 v25_;
  const T26 v26_;
  const T27 v27_;
  const T28 v28_;
  const T29 v29_;
  const T30 v30_;
  const T31 v31_;
  const T32 v32_;
  const T33 v33_;
  const T34 v34_;
  const T35 v35_;
  const T36 v36_;
  const T37 v37_;
  const T38 v38_;
  const T39 v39_;
  const T40 v40_;
  const T41 v41_;
  const T42 v42_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43>
class ValueArray43 {
 public:
  ValueArray43(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
      T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31, T32 v32, T33 v33,
      T34 v34, T35 v35, T36 v36, T37 v37, T38 v38, T39 v39, T40 v40, T41 v41,
      T42 v42, T43 v43) : v1_(v1), v2_(v2), v3_(v3), v4_(v4), v5_(v5), v6_(v6),
      v7_(v7), v8_(v8), v9_(v9), v10_(v10), v11_(v11), v12_(v12), v13_(v13),
      v14_(v14), v15_(v15), v16_(v16), v17_(v17), v18_(v18), v19_(v19),
      v20_(v20), v21_(v21), v22_(v22), v23_(v23), v24_(v24), v25_(v25),
      v26_(v26), v27_(v27), v28_(v28), v29_(v29), v30_(v30), v31_(v31),
      v32_(v32), v33_(v33), v34_(v34), v35_(v35), v36_(v36), v37_(v37),
      v38_(v38), v39_(v39), v40_(v40), v41_(v41), v42_(v42), v43_(v43) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_, v23_,
        v24_, v25_, v26_, v27_, v28_, v29_, v30_, v31_, v32_, v33_, v34_, v35_,
        v36_, v37_, v38_, v39_, v40_, v41_, v42_, v43_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray43& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
  const T23 v23_;
  const T24 v24_;
  const T25 v25_;
  const T26 v26_;
  const T27 v27_;
  const T28 v28_;
  const T29 v29_;
  const T30 v30_;
  const T31 v31_;
  const T32 v32_;
  const T33 v33_;
  const T34 v34_;
  const T35 v35_;
  const T36 v36_;
  const T37 v37_;
  const T38 v38_;
  const T39 v39_;
  const T40 v40_;
  const T41 v41_;
  const T42 v42_;
  const T43 v43_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43, typename T44>
class ValueArray44 {
 public:
  ValueArray44(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
      T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31, T32 v32, T33 v33,
      T34 v34, T35 v35, T36 v36, T37 v37, T38 v38, T39 v39, T40 v40, T41 v41,
      T42 v42, T43 v43, T44 v44) : v1_(v1), v2_(v2), v3_(v3), v4_(v4), v5_(v5),
      v6_(v6), v7_(v7), v8_(v8), v9_(v9), v10_(v10), v11_(v11), v12_(v12),
      v13_(v13), v14_(v14), v15_(v15), v16_(v16), v17_(v17), v18_(v18),
      v19_(v19), v20_(v20), v21_(v21), v22_(v22), v23_(v23), v24_(v24),
      v25_(v25), v26_(v26), v27_(v27), v28_(v28), v29_(v29), v30_(v30),
      v31_(v31), v32_(v32), v33_(v33), v34_(v34), v35_(v35), v36_(v36),
      v37_(v37), v38_(v38), v39_(v39), v40_(v40), v41_(v41), v42_(v42),
      v43_(v43), v44_(v44) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_, v23_,
        v24_, v25_, v26_, v27_, v28_, v29_, v30_, v31_, v32_, v33_, v34_, v35_,
        v36_, v37_, v38_, v39_, v40_, v41_, v42_, v43_, v44_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray44& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
  const T23 v23_;
  const T24 v24_;
  const T25 v25_;
  const T26 v26_;
  const T27 v27_;
  const T28 v28_;
  const T29 v29_;
  const T30 v30_;
  const T31 v31_;
  const T32 v32_;
  const T33 v33_;
  const T34 v34_;
  const T35 v35_;
  const T36 v36_;
  const T37 v37_;
  const T38 v38_;
  const T39 v39_;
  const T40 v40_;
  const T41 v41_;
  const T42 v42_;
  const T43 v43_;
  const T44 v44_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43, typename T44, typename T45>
class ValueArray45 {
 public:
  ValueArray45(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
      T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31, T32 v32, T33 v33,
      T34 v34, T35 v35, T36 v36, T37 v37, T38 v38, T39 v39, T40 v40, T41 v41,
      T42 v42, T43 v43, T44 v44, T45 v45) : v1_(v1), v2_(v2), v3_(v3), v4_(v4),
      v5_(v5), v6_(v6), v7_(v7), v8_(v8), v9_(v9), v10_(v10), v11_(v11),
      v12_(v12), v13_(v13), v14_(v14), v15_(v15), v16_(v16), v17_(v17),
      v18_(v18), v19_(v19), v20_(v20), v21_(v21), v22_(v22), v23_(v23),
      v24_(v24), v25_(v25), v26_(v26), v27_(v27), v28_(v28), v29_(v29),
      v30_(v30), v31_(v31), v32_(v32), v33_(v33), v34_(v34), v35_(v35),
      v36_(v36), v37_(v37), v38_(v38), v39_(v39), v40_(v40), v41_(v41),
      v42_(v42), v43_(v43), v44_(v44), v45_(v45) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_, v23_,
        v24_, v25_, v26_, v27_, v28_, v29_, v30_, v31_, v32_, v33_, v34_, v35_,
        v36_, v37_, v38_, v39_, v40_, v41_, v42_, v43_, v44_, v45_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray45& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
  const T23 v23_;
  const T24 v24_;
  const T25 v25_;
  const T26 v26_;
  const T27 v27_;
  const T28 v28_;
  const T29 v29_;
  const T30 v30_;
  const T31 v31_;
  const T32 v32_;
  const T33 v33_;
  const T34 v34_;
  const T35 v35_;
  const T36 v36_;
  const T37 v37_;
  const T38 v38_;
  const T39 v39_;
  const T40 v40_;
  const T41 v41_;
  const T42 v42_;
  const T43 v43_;
  const T44 v44_;
  const T45 v45_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43, typename T44, typename T45,
    typename T46>
class ValueArray46 {
 public:
  ValueArray46(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
      T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31, T32 v32, T33 v33,
      T34 v34, T35 v35, T36 v36, T37 v37, T38 v38, T39 v39, T40 v40, T41 v41,
      T42 v42, T43 v43, T44 v44, T45 v45, T46 v46) : v1_(v1), v2_(v2), v3_(v3),
      v4_(v4), v5_(v5), v6_(v6), v7_(v7), v8_(v8), v9_(v9), v10_(v10),
      v11_(v11), v12_(v12), v13_(v13), v14_(v14), v15_(v15), v16_(v16),
      v17_(v17), v18_(v18), v19_(v19), v20_(v20), v21_(v21), v22_(v22),
      v23_(v23), v24_(v24), v25_(v25), v26_(v26), v27_(v27), v28_(v28),
      v29_(v29), v30_(v30), v31_(v31), v32_(v32), v33_(v33), v34_(v34),
      v35_(v35), v36_(v36), v37_(v37), v38_(v38), v39_(v39), v40_(v40),
      v41_(v41), v42_(v42), v43_(v43), v44_(v44), v45_(v45), v46_(v46) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_, v23_,
        v24_, v25_, v26_, v27_, v28_, v29_, v30_, v31_, v32_, v33_, v34_, v35_,
        v36_, v37_, v38_, v39_, v40_, v41_, v42_, v43_, v44_, v45_, v46_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray46& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
  const T23 v23_;
  const T24 v24_;
  const T25 v25_;
  const T26 v26_;
  const T27 v27_;
  const T28 v28_;
  const T29 v29_;
  const T30 v30_;
  const T31 v31_;
  const T32 v32_;
  const T33 v33_;
  const T34 v34_;
  const T35 v35_;
  const T36 v36_;
  const T37 v37_;
  const T38 v38_;
  const T39 v39_;
  const T40 v40_;
  const T41 v41_;
  const T42 v42_;
  const T43 v43_;
  const T44 v44_;
  const T45 v45_;
  const T46 v46_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43, typename T44, typename T45,
    typename T46, typename T47>
class ValueArray47 {
 public:
  ValueArray47(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
      T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31, T32 v32, T33 v33,
      T34 v34, T35 v35, T36 v36, T37 v37, T38 v38, T39 v39, T40 v40, T41 v41,
      T42 v42, T43 v43, T44 v44, T45 v45, T46 v46, T47 v47) : v1_(v1), v2_(v2),
      v3_(v3), v4_(v4), v5_(v5), v6_(v6), v7_(v7), v8_(v8), v9_(v9), v10_(v10),
      v11_(v11), v12_(v12), v13_(v13), v14_(v14), v15_(v15), v16_(v16),
      v17_(v17), v18_(v18), v19_(v19), v20_(v20), v21_(v21), v22_(v22),
      v23_(v23), v24_(v24), v25_(v25), v26_(v26), v27_(v27), v28_(v28),
      v29_(v29), v30_(v30), v31_(v31), v32_(v32), v33_(v33), v34_(v34),
      v35_(v35), v36_(v36), v37_(v37), v38_(v38), v39_(v39), v40_(v40),
      v41_(v41), v42_(v42), v43_(v43), v44_(v44), v45_(v45), v46_(v46),
      v47_(v47) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_, v23_,
        v24_, v25_, v26_, v27_, v28_, v29_, v30_, v31_, v32_, v33_, v34_, v35_,
        v36_, v37_, v38_, v39_, v40_, v41_, v42_, v43_, v44_, v45_, v46_,
        v47_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray47& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
  const T23 v23_;
  const T24 v24_;
  const T25 v25_;
  const T26 v26_;
  const T27 v27_;
  const T28 v28_;
  const T29 v29_;
  const T30 v30_;
  const T31 v31_;
  const T32 v32_;
  const T33 v33_;
  const T34 v34_;
  const T35 v35_;
  const T36 v36_;
  const T37 v37_;
  const T38 v38_;
  const T39 v39_;
  const T40 v40_;
  const T41 v41_;
  const T42 v42_;
  const T43 v43_;
  const T44 v44_;
  const T45 v45_;
  const T46 v46_;
  const T47 v47_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43, typename T44, typename T45,
    typename T46, typename T47, typename T48>
class ValueArray48 {
 public:
  ValueArray48(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
      T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31, T32 v32, T33 v33,
      T34 v34, T35 v35, T36 v36, T37 v37, T38 v38, T39 v39, T40 v40, T41 v41,
      T42 v42, T43 v43, T44 v44, T45 v45, T46 v46, T47 v47, T48 v48) : v1_(v1),
      v2_(v2), v3_(v3), v4_(v4), v5_(v5), v6_(v6), v7_(v7), v8_(v8), v9_(v9),
      v10_(v10), v11_(v11), v12_(v12), v13_(v13), v14_(v14), v15_(v15),
      v16_(v16), v17_(v17), v18_(v18), v19_(v19), v20_(v20), v21_(v21),
      v22_(v22), v23_(v23), v24_(v24), v25_(v25), v26_(v26), v27_(v27),
      v28_(v28), v29_(v29), v30_(v30), v31_(v31), v32_(v32), v33_(v33),
      v34_(v34), v35_(v35), v36_(v36), v37_(v37), v38_(v38), v39_(v39),
      v40_(v40), v41_(v41), v42_(v42), v43_(v43), v44_(v44), v45_(v45),
      v46_(v46), v47_(v47), v48_(v48) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_, v23_,
        v24_, v25_, v26_, v27_, v28_, v29_, v30_, v31_, v32_, v33_, v34_, v35_,
        v36_, v37_, v38_, v39_, v40_, v41_, v42_, v43_, v44_, v45_, v46_, v47_,
        v48_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray48& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
  const T23 v23_;
  const T24 v24_;
  const T25 v25_;
  const T26 v26_;
  const T27 v27_;
  const T28 v28_;
  const T29 v29_;
  const T30 v30_;
  const T31 v31_;
  const T32 v32_;
  const T33 v33_;
  const T34 v34_;
  const T35 v35_;
  const T36 v36_;
  const T37 v37_;
  const T38 v38_;
  const T39 v39_;
  const T40 v40_;
  const T41 v41_;
  const T42 v42_;
  const T43 v43_;
  const T44 v44_;
  const T45 v45_;
  const T46 v46_;
  const T47 v47_;
  const T48 v48_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43, typename T44, typename T45,
    typename T46, typename T47, typename T48, typename T49>
class ValueArray49 {
 public:
  ValueArray49(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
      T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31, T32 v32, T33 v33,
      T34 v34, T35 v35, T36 v36, T37 v37, T38 v38, T39 v39, T40 v40, T41 v41,
      T42 v42, T43 v43, T44 v44, T45 v45, T46 v46, T47 v47, T48 v48,
      T49 v49) : v1_(v1), v2_(v2), v3_(v3), v4_(v4), v5_(v5), v6_(v6), v7_(v7),
      v8_(v8), v9_(v9), v10_(v10), v11_(v11), v12_(v12), v13_(v13), v14_(v14),
      v15_(v15), v16_(v16), v17_(v17), v18_(v18), v19_(v19), v20_(v20),
      v21_(v21), v22_(v22), v23_(v23), v24_(v24), v25_(v25), v26_(v26),
      v27_(v27), v28_(v28), v29_(v29), v30_(v30), v31_(v31), v32_(v32),
      v33_(v33), v34_(v34), v35_(v35), v36_(v36), v37_(v37), v38_(v38),
      v39_(v39), v40_(v40), v41_(v41), v42_(v42), v43_(v43), v44_(v44),
      v45_(v45), v46_(v46), v47_(v47), v48_(v48), v49_(v49) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_, v23_,
        v24_, v25_, v26_, v27_, v28_, v29_, v30_, v31_, v32_, v33_, v34_, v35_,
        v36_, v37_, v38_, v39_, v40_, v41_, v42_, v43_, v44_, v45_, v46_, v47_,
        v48_, v49_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray49& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
  const T23 v23_;
  const T24 v24_;
  const T25 v25_;
  const T26 v26_;
  const T27 v27_;
  const T28 v28_;
  const T29 v29_;
  const T30 v30_;
  const T31 v31_;
  const T32 v32_;
  const T33 v33_;
  const T34 v34_;
  const T35 v35_;
  const T36 v36_;
  const T37 v37_;
  const T38 v38_;
  const T39 v39_;
  const T40 v40_;
  const T41 v41_;
  const T42 v42_;
  const T43 v43_;
  const T44 v44_;
  const T45 v45_;
  const T46 v46_;
  const T47 v47_;
  const T48 v48_;
  const T49 v49_;
};

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43, typename T44, typename T45,
    typename T46, typename T47, typename T48, typename T49, typename T50>
class ValueArray50 {
 public:
  ValueArray50(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
      T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
      T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
      T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31, T32 v32, T33 v33,
      T34 v34, T35 v35, T36 v36, T37 v37, T38 v38, T39 v39, T40 v40, T41 v41,
      T42 v42, T43 v43, T44 v44, T45 v45, T46 v46, T47 v47, T48 v48, T49 v49,
      T50 v50) : v1_(v1), v2_(v2), v3_(v3), v4_(v4), v5_(v5), v6_(v6), v7_(v7),
      v8_(v8), v9_(v9), v10_(v10), v11_(v11), v12_(v12), v13_(v13), v14_(v14),
      v15_(v15), v16_(v16), v17_(v17), v18_(v18), v19_(v19), v20_(v20),
      v21_(v21), v22_(v22), v23_(v23), v24_(v24), v25_(v25), v26_(v26),
      v27_(v27), v28_(v28), v29_(v29), v30_(v30), v31_(v31), v32_(v32),
      v33_(v33), v34_(v34), v35_(v35), v36_(v36), v37_(v37), v38_(v38),
      v39_(v39), v40_(v40), v41_(v41), v42_(v42), v43_(v43), v44_(v44),
      v45_(v45), v46_(v46), v47_(v47), v48_(v48), v49_(v49), v50_(v50) {}

  template <typename T>
  operator ParamGenerator<T>() const {
    const T array[] = {v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_, v11_,
        v12_, v13_, v14_, v15_, v16_, v17_, v18_, v19_, v20_, v21_, v22_, v23_,
        v24_, v25_, v26_, v27_, v28_, v29_, v30_, v31_, v32_, v33_, v34_, v35_,
        v36_, v37_, v38_, v39_, v40_, v41_, v42_, v43_, v44_, v45_, v46_, v47_,
        v48_, v49_, v50_};
    return ValuesIn(array);
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const ValueArray50& other);

  const T1 v1_;
  const T2 v2_;
  const T3 v3_;
  const T4 v4_;
  const T5 v5_;
  const T6 v6_;
  const T7 v7_;
  const T8 v8_;
  const T9 v9_;
  const T10 v10_;
  const T11 v11_;
  const T12 v12_;
  const T13 v13_;
  const T14 v14_;
  const T15 v15_;
  const T16 v16_;
  const T17 v17_;
  const T18 v18_;
  const T19 v19_;
  const T20 v20_;
  const T21 v21_;
  const T22 v22_;
  const T23 v23_;
  const T24 v24_;
  const T25 v25_;
  const T26 v26_;
  const T27 v27_;
  const T28 v28_;
  const T29 v29_;
  const T30 v30_;
  const T31 v31_;
  const T32 v32_;
  const T33 v33_;
  const T34 v34_;
  const T35 v35_;
  const T36 v36_;
  const T37 v37_;
  const T38 v38_;
  const T39 v39_;
  const T40 v40_;
  const T41 v41_;
  const T42 v42_;
  const T43 v43_;
  const T44 v44_;
  const T45 v45_;
  const T46 v46_;
  const T47 v47_;
  const T48 v48_;
  const T49 v49_;
  const T50 v50_;
};

# if GTEST_HAS_COMBINE
// INTERNAL IMPLEMENTATION - DO NOT USE IN USER CODE.
//
// Generates values from the Cartesian product of values produced
// by the argument generators.
//
template <typename T1, typename T2>
class CartesianProductGenerator2
    : public ParamGeneratorInterface< ::std::tr1::tuple<T1, T2> > {
 public:
  typedef ::std::tr1::tuple<T1, T2> ParamType;

  CartesianProductGenerator2(const ParamGenerator<T1>& g1,
      const ParamGenerator<T2>& g2)
      : g1_(g1), g2_(g2) {}
  virtual ~CartesianProductGenerator2() {}

  virtual ParamIteratorInterface<ParamType>* Begin() const {
    return new Iterator(this, g1_, g1_.begin(), g2_, g2_.begin());
  }
  virtual ParamIteratorInterface<ParamType>* End() const {
    return new Iterator(this, g1_, g1_.end(), g2_, g2_.end());
  }

 private:
  class Iterator : public ParamIteratorInterface<ParamType> {
   public:
    Iterator(const ParamGeneratorInterface<ParamType>* base,
      const ParamGenerator<T1>& g1,
      const typename ParamGenerator<T1>::iterator& current1,
      const ParamGenerator<T2>& g2,
      const typename ParamGenerator<T2>::iterator& current2)
        : base_(base),
          begin1_(g1.begin()), end1_(g1.end()), current1_(current1),
          begin2_(g2.begin()), end2_(g2.end()), current2_(current2)    {
      ComputeCurrentValue();
    }
    virtual ~Iterator() {}

    virtual const ParamGeneratorInterface<ParamType>* BaseGenerator() const {
      return base_;
    }
    // Advance should not be called on beyond-of-range iterators
    // so no component iterators must be beyond end of range, either.
    virtual void Advance() {
      assert(!AtEnd());
      ++current2_;
      if (current2_ == end2_) {
        current2_ = begin2_;
        ++current1_;
      }
      ComputeCurrentValue();
    }
    virtual ParamIteratorInterface<ParamType>* Clone() const {
      return new Iterator(*this);
    }
    virtual const ParamType* Current() const { return &current_value_; }
    virtual bool Equals(const ParamIteratorInterface<ParamType>& other) const {
      // Having the same base generator guarantees that the other
      // iterator is of the same type and we can downcast.
      GTEST_CHECK_(BaseGenerator() == other.BaseGenerator())
          << "The program attempted to compare iterators "
          << "from different generators." << std::endl;
      const Iterator* typed_other =
          CheckedDowncastToActualType<const Iterator>(&other);
      // We must report iterators equal if they both point beyond their
      // respective ranges. That can happen in a variety of fashions,
      // so we have to consult AtEnd().
      return (AtEnd() && typed_other->AtEnd()) ||
         (
          current1_ == typed_other->current1_ &&
          current2_ == typed_other->current2_);
    }

   private:
    Iterator(const Iterator& other)
        : base_(other.base_),
        begin1_(other.begin1_),
        end1_(other.end1_),
        current1_(other.current1_),
        begin2_(other.begin2_),
        end2_(other.end2_),
        current2_(other.current2_) {
      ComputeCurrentValue();
    }

    void ComputeCurrentValue() {
      if (!AtEnd())
        current_value_ = ParamType(*current1_, *current2_);
    }
    bool AtEnd() const {
      // We must report iterator past the end of the range when either of the
      // component iterators has reached the end of its range.
      return
          current1_ == end1_ ||
          current2_ == end2_;
    }

    // No implementation - assignment is unsupported.
    void operator=(const Iterator& other);

    const ParamGeneratorInterface<ParamType>* const base_;
    // begin[i]_ and end[i]_ define the i-th range that Iterator traverses.
    // current[i]_ is the actual traversing iterator.
    const typename ParamGenerator<T1>::iterator begin1_;
    const typename ParamGenerator<T1>::iterator end1_;
    typename ParamGenerator<T1>::iterator current1_;
    const typename ParamGenerator<T2>::iterator begin2_;
    const typename ParamGenerator<T2>::iterator end2_;
    typename ParamGenerator<T2>::iterator current2_;
    ParamType current_value_;
  };  // class CartesianProductGenerator2::Iterator

  // No implementation - assignment is unsupported.
  void operator=(const CartesianProductGenerator2& other);

  const ParamGenerator<T1> g1_;
  const ParamGenerator<T2> g2_;
};  // class CartesianProductGenerator2


template <typename T1, typename T2, typename T3>
class CartesianProductGenerator3
    : public ParamGeneratorInterface< ::std::tr1::tuple<T1, T2, T3> > {
 public:
  typedef ::std::tr1::tuple<T1, T2, T3> ParamType;

  CartesianProductGenerator3(const ParamGenerator<T1>& g1,
      const ParamGenerator<T2>& g2, const ParamGenerator<T3>& g3)
      : g1_(g1), g2_(g2), g3_(g3) {}
  virtual ~CartesianProductGenerator3() {}

  virtual ParamIteratorInterface<ParamType>* Begin() const {
    return new Iterator(this, g1_, g1_.begin(), g2_, g2_.begin(), g3_,
        g3_.begin());
  }
  virtual ParamIteratorInterface<ParamType>* End() const {
    return new Iterator(this, g1_, g1_.end(), g2_, g2_.end(), g3_, g3_.end());
  }

 private:
  class Iterator : public ParamIteratorInterface<ParamType> {
   public:
    Iterator(const ParamGeneratorInterface<ParamType>* base,
      const ParamGenerator<T1>& g1,
      const typename ParamGenerator<T1>::iterator& current1,
      const ParamGenerator<T2>& g2,
      const typename ParamGenerator<T2>::iterator& current2,
      const ParamGenerator<T3>& g3,
      const typename ParamGenerator<T3>::iterator& current3)
        : base_(base),
          begin1_(g1.begin()), end1_(g1.end()), current1_(current1),
          begin2_(g2.begin()), end2_(g2.end()), current2_(current2),
          begin3_(g3.begin()), end3_(g3.end()), current3_(current3)    {
      ComputeCurrentValue();
    }
    virtual ~Iterator() {}

    virtual const ParamGeneratorInterface<ParamType>* BaseGenerator() const {
      return base_;
    }
    // Advance should not be called on beyond-of-range iterators
    // so no component iterators must be beyond end of range, either.
    virtual void Advance() {
      assert(!AtEnd());
      ++current3_;
      if (current3_ == end3_) {
        current3_ = begin3_;
        ++current2_;
      }
      if (current2_ == end2_) {
        current2_ = begin2_;
        ++current1_;
      }
      ComputeCurrentValue();
    }
    virtual ParamIteratorInterface<ParamType>* Clone() const {
      return new Iterator(*this);
    }
    virtual const ParamType* Current() const { return &current_value_; }
    virtual bool Equals(const ParamIteratorInterface<ParamType>& other) const {
      // Having the same base generator guarantees that the other
      // iterator is of the same type and we can downcast.
      GTEST_CHECK_(BaseGenerator() == other.BaseGenerator())
          << "The program attempted to compare iterators "
          << "from different generators." << std::endl;
      const Iterator* typed_other =
          CheckedDowncastToActualType<const Iterator>(&other);
      // We must report iterators equal if they both point beyond their
      // respective ranges. That can happen in a variety of fashions,
      // so we have to consult AtEnd().
      return (AtEnd() && typed_other->AtEnd()) ||
         (
          current1_ == typed_other->current1_ &&
          current2_ == typed_other->current2_ &&
          current3_ == typed_other->current3_);
    }

   private:
    Iterator(const Iterator& other)
        : base_(other.base_),
        begin1_(other.begin1_),
        end1_(other.end1_),
        current1_(other.current1_),
        begin2_(other.begin2_),
        end2_(other.end2_),
        current2_(other.current2_),
        begin3_(other.begin3_),
        end3_(other.end3_),
        current3_(other.current3_) {
      ComputeCurrentValue();
    }

    void ComputeCurrentValue() {
      if (!AtEnd())
        current_value_ = ParamType(*current1_, *current2_, *current3_);
    }
    bool AtEnd() const {
      // We must report iterator past the end of the range when either of the
      // component iterators has reached the end of its range.
      return
          current1_ == end1_ ||
          current2_ == end2_ ||
          current3_ == end3_;
    }

    // No implementation - assignment is unsupported.
    void operator=(const Iterator& other);

    const ParamGeneratorInterface<ParamType>* const base_;
    // begin[i]_ and end[i]_ define the i-th range that Iterator traverses.
    // current[i]_ is the actual traversing iterator.
    const typename ParamGenerator<T1>::iterator begin1_;
    const typename ParamGenerator<T1>::iterator end1_;
    typename ParamGenerator<T1>::iterator current1_;
    const typename ParamGenerator<T2>::iterator begin2_;
    const typename ParamGenerator<T2>::iterator end2_;
    typename ParamGenerator<T2>::iterator current2_;
    const typename ParamGenerator<T3>::iterator begin3_;
    const typename ParamGenerator<T3>::iterator end3_;
    typename ParamGenerator<T3>::iterator current3_;
    ParamType current_value_;
  };  // class CartesianProductGenerator3::Iterator

  // No implementation - assignment is unsupported.
  void operator=(const CartesianProductGenerator3& other);

  const ParamGenerator<T1> g1_;
  const ParamGenerator<T2> g2_;
  const ParamGenerator<T3> g3_;
};  // class CartesianProductGenerator3


template <typename T1, typename T2, typename T3, typename T4>
class CartesianProductGenerator4
    : public ParamGeneratorInterface< ::std::tr1::tuple<T1, T2, T3, T4> > {
 public:
  typedef ::std::tr1::tuple<T1, T2, T3, T4> ParamType;

  CartesianProductGenerator4(const ParamGenerator<T1>& g1,
      const ParamGenerator<T2>& g2, const ParamGenerator<T3>& g3,
      const ParamGenerator<T4>& g4)
      : g1_(g1), g2_(g2), g3_(g3), g4_(g4) {}
  virtual ~CartesianProductGenerator4() {}

  virtual ParamIteratorInterface<ParamType>* Begin() const {
    return new Iterator(this, g1_, g1_.begin(), g2_, g2_.begin(), g3_,
        g3_.begin(), g4_, g4_.begin());
  }
  virtual ParamIteratorInterface<ParamType>* End() const {
    return new Iterator(this, g1_, g1_.end(), g2_, g2_.end(), g3_, g3_.end(),
        g4_, g4_.end());
  }

 private:
  class Iterator : public ParamIteratorInterface<ParamType> {
   public:
    Iterator(const ParamGeneratorInterface<ParamType>* base,
      const ParamGenerator<T1>& g1,
      const typename ParamGenerator<T1>::iterator& current1,
      const ParamGenerator<T2>& g2,
      const typename ParamGenerator<T2>::iterator& current2,
      const ParamGenerator<T3>& g3,
      const typename ParamGenerator<T3>::iterator& current3,
      const ParamGenerator<T4>& g4,
      const typename ParamGenerator<T4>::iterator& current4)
        : base_(base),
          begin1_(g1.begin()), end1_(g1.end()), current1_(current1),
          begin2_(g2.begin()), end2_(g2.end()), current2_(current2),
          begin3_(g3.begin()), end3_(g3.end()), current3_(current3),
          begin4_(g4.begin()), end4_(g4.end()), current4_(current4)    {
      ComputeCurrentValue();
    }
    virtual ~Iterator() {}

    virtual const ParamGeneratorInterface<ParamType>* BaseGenerator() const {
      return base_;
    }
    // Advance should not be called on beyond-of-range iterators
    // so no component iterators must be beyond end of range, either.
    virtual void Advance() {
      assert(!AtEnd());
      ++current4_;
      if (current4_ == end4_) {
        current4_ = begin4_;
        ++current3_;
      }
      if (current3_ == end3_) {
        current3_ = begin3_;
        ++current2_;
      }
      if (current2_ == end2_) {
        current2_ = begin2_;
        ++current1_;
      }
      ComputeCurrentValue();
    }
    virtual ParamIteratorInterface<ParamType>* Clone() const {
      return new Iterator(*this);
    }
    virtual const ParamType* Current() const { return &current_value_; }
    virtual bool Equals(const ParamIteratorInterface<ParamType>& other) const {
      // Having the same base generator guarantees that the other
      // iterator is of the same type and we can downcast.
      GTEST_CHECK_(BaseGenerator() == other.BaseGenerator())
          << "The program attempted to compare iterators "
          << "from different generators." << std::endl;
      const Iterator* typed_other =
          CheckedDowncastToActualType<const Iterator>(&other);
      // We must report iterators equal if they both point beyond their
      // respective ranges. That can happen in a variety of fashions,
      // so we have to consult AtEnd().
      return (AtEnd() && typed_other->AtEnd()) ||
         (
          current1_ == typed_other->current1_ &&
          current2_ == typed_other->current2_ &&
          current3_ == typed_other->current3_ &&
          current4_ == typed_other->current4_);
    }

   private:
    Iterator(const Iterator& other)
        : base_(other.base_),
        begin1_(other.begin1_),
        end1_(other.end1_),
        current1_(other.current1_),
        begin2_(other.begin2_),
        end2_(other.end2_),
        current2_(other.current2_),
        begin3_(other.begin3_),
        end3_(other.end3_),
        current3_(other.current3_),
        begin4_(other.begin4_),
        end4_(other.end4_),
        current4_(other.current4_) {
      ComputeCurrentValue();
    }

    void ComputeCurrentValue() {
      if (!AtEnd())
        current_value_ = ParamType(*current1_, *current2_, *current3_,
            *current4_);
    }
    bool AtEnd() const {
      // We must report iterator past the end of the range when either of the
      // component iterators has reached the end of its range.
      return
          current1_ == end1_ ||
          current2_ == end2_ ||
          current3_ == end3_ ||
          current4_ == end4_;
    }

    // No implementation - assignment is unsupported.
    void operator=(const Iterator& other);

    const ParamGeneratorInterface<ParamType>* const base_;
    // begin[i]_ and end[i]_ define the i-th range that Iterator traverses.
    // current[i]_ is the actual traversing iterator.
    const typename ParamGenerator<T1>::iterator begin1_;
    const typename ParamGenerator<T1>::iterator end1_;
    typename ParamGenerator<T1>::iterator current1_;
    const typename ParamGenerator<T2>::iterator begin2_;
    const typename ParamGenerator<T2>::iterator end2_;
    typename ParamGenerator<T2>::iterator current2_;
    const typename ParamGenerator<T3>::iterator begin3_;
    const typename ParamGenerator<T3>::iterator end3_;
    typename ParamGenerator<T3>::iterator current3_;
    const typename ParamGenerator<T4>::iterator begin4_;
    const typename ParamGenerator<T4>::iterator end4_;
    typename ParamGenerator<T4>::iterator current4_;
    ParamType current_value_;
  };  // class CartesianProductGenerator4::Iterator

  // No implementation - assignment is unsupported.
  void operator=(const CartesianProductGenerator4& other);

  const ParamGenerator<T1> g1_;
  const ParamGenerator<T2> g2_;
  const ParamGenerator<T3> g3_;
  const ParamGenerator<T4> g4_;
};  // class CartesianProductGenerator4


template <typename T1, typename T2, typename T3, typename T4, typename T5>
class CartesianProductGenerator5
    : public ParamGeneratorInterface< ::std::tr1::tuple<T1, T2, T3, T4, T5> > {
 public:
  typedef ::std::tr1::tuple<T1, T2, T3, T4, T5> ParamType;

  CartesianProductGenerator5(const ParamGenerator<T1>& g1,
      const ParamGenerator<T2>& g2, const ParamGenerator<T3>& g3,
      const ParamGenerator<T4>& g4, const ParamGenerator<T5>& g5)
      : g1_(g1), g2_(g2), g3_(g3), g4_(g4), g5_(g5) {}
  virtual ~CartesianProductGenerator5() {}

  virtual ParamIteratorInterface<ParamType>* Begin() const {
    return new Iterator(this, g1_, g1_.begin(), g2_, g2_.begin(), g3_,
        g3_.begin(), g4_, g4_.begin(), g5_, g5_.begin());
  }
  virtual ParamIteratorInterface<ParamType>* End() const {
    return new Iterator(this, g1_, g1_.end(), g2_, g2_.end(), g3_, g3_.end(),
        g4_, g4_.end(), g5_, g5_.end());
  }

 private:
  class Iterator : public ParamIteratorInterface<ParamType> {
   public:
    Iterator(const ParamGeneratorInterface<ParamType>* base,
      const ParamGenerator<T1>& g1,
      const typename ParamGenerator<T1>::iterator& current1,
      const ParamGenerator<T2>& g2,
      const typename ParamGenerator<T2>::iterator& current2,
      const ParamGenerator<T3>& g3,
      const typename ParamGenerator<T3>::iterator& current3,
      const ParamGenerator<T4>& g4,
      const typename ParamGenerator<T4>::iterator& current4,
      const ParamGenerator<T5>& g5,
      const typename ParamGenerator<T5>::iterator& current5)
        : base_(base),
          begin1_(g1.begin()), end1_(g1.end()), current1_(current1),
          begin2_(g2.begin()), end2_(g2.end()), current2_(current2),
          begin3_(g3.begin()), end3_(g3.end()), current3_(current3),
          begin4_(g4.begin()), end4_(g4.end()), current4_(current4),
          begin5_(g5.begin()), end5_(g5.end()), current5_(current5)    {
      ComputeCurrentValue();
    }
    virtual ~Iterator() {}

    virtual const ParamGeneratorInterface<ParamType>* BaseGenerator() const {
      return base_;
    }
    // Advance should not be called on beyond-of-range iterators
    // so no component iterators must be beyond end of range, either.
    virtual void Advance() {
      assert(!AtEnd());
      ++current5_;
      if (current5_ == end5_) {
        current5_ = begin5_;
        ++current4_;
      }
      if (current4_ == end4_) {
        current4_ = begin4_;
        ++current3_;
      }
      if (current3_ == end3_) {
        current3_ = begin3_;
        ++current2_;
      }
      if (current2_ == end2_) {
        current2_ = begin2_;
        ++current1_;
      }
      ComputeCurrentValue();
    }
    virtual ParamIteratorInterface<ParamType>* Clone() const {
      return new Iterator(*this);
    }
    virtual const ParamType* Current() const { return &current_value_; }
    virtual bool Equals(const ParamIteratorInterface<ParamType>& other) const {
      // Having the same base generator guarantees that the other
      // iterator is of the same type and we can downcast.
      GTEST_CHECK_(BaseGenerator() == other.BaseGenerator())
          << "The program attempted to compare iterators "
          << "from different generators." << std::endl;
      const Iterator* typed_other =
          CheckedDowncastToActualType<const Iterator>(&other);
      // We must report iterators equal if they both point beyond their
      // respective ranges. That can happen in a variety of fashions,
      // so we have to consult AtEnd().
      return (AtEnd() && typed_other->AtEnd()) ||
         (
          current1_ == typed_other->current1_ &&
          current2_ == typed_other->current2_ &&
          current3_ == typed_other->current3_ &&
          current4_ == typed_other->current4_ &&
          current5_ == typed_other->current5_);
    }

   private:
    Iterator(const Iterator& other)
        : base_(other.base_),
        begin1_(other.begin1_),
        end1_(other.end1_),
        current1_(other.current1_),
        begin2_(other.begin2_),
        end2_(other.end2_),
        current2_(other.current2_),
        begin3_(other.begin3_),
        end3_(other.end3_),
        current3_(other.current3_),
        begin4_(other.begin4_),
        end4_(other.end4_),
        current4_(other.current4_),
        begin5_(other.begin5_),
        end5_(other.end5_),
        current5_(other.current5_) {
      ComputeCurrentValue();
    }

    void ComputeCurrentValue() {
      if (!AtEnd())
        current_value_ = ParamType(*current1_, *current2_, *current3_,
            *current4_, *current5_);
    }
    bool AtEnd() const {
      // We must report iterator past the end of the range when either of the
      // component iterators has reached the end of its range.
      return
          current1_ == end1_ ||
          current2_ == end2_ ||
          current3_ == end3_ ||
          current4_ == end4_ ||
          current5_ == end5_;
    }

    // No implementation - assignment is unsupported.
    void operator=(const Iterator& other);

    const ParamGeneratorInterface<ParamType>* const base_;
    // begin[i]_ and end[i]_ define the i-th range that Iterator traverses.
    // current[i]_ is the actual traversing iterator.
    const typename ParamGenerator<T1>::iterator begin1_;
    const typename ParamGenerator<T1>::iterator end1_;
    typename ParamGenerator<T1>::iterator current1_;
    const typename ParamGenerator<T2>::iterator begin2_;
    const typename ParamGenerator<T2>::iterator end2_;
    typename ParamGenerator<T2>::iterator current2_;
    const typename ParamGenerator<T3>::iterator begin3_;
    const typename ParamGenerator<T3>::iterator end3_;
    typename ParamGenerator<T3>::iterator current3_;
    const typename ParamGenerator<T4>::iterator begin4_;
    const typename ParamGenerator<T4>::iterator end4_;
    typename ParamGenerator<T4>::iterator current4_;
    const typename ParamGenerator<T5>::iterator begin5_;
    const typename ParamGenerator<T5>::iterator end5_;
    typename ParamGenerator<T5>::iterator current5_;
    ParamType current_value_;
  };  // class CartesianProductGenerator5::Iterator

  // No implementation - assignment is unsupported.
  void operator=(const CartesianProductGenerator5& other);

  const ParamGenerator<T1> g1_;
  const ParamGenerator<T2> g2_;
  const ParamGenerator<T3> g3_;
  const ParamGenerator<T4> g4_;
  const ParamGenerator<T5> g5_;
};  // class CartesianProductGenerator5


template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6>
class CartesianProductGenerator6
    : public ParamGeneratorInterface< ::std::tr1::tuple<T1, T2, T3, T4, T5,
        T6> > {
 public:
  typedef ::std::tr1::tuple<T1, T2, T3, T4, T5, T6> ParamType;

  CartesianProductGenerator6(const ParamGenerator<T1>& g1,
      const ParamGenerator<T2>& g2, const ParamGenerator<T3>& g3,
      const ParamGenerator<T4>& g4, const ParamGenerator<T5>& g5,
      const ParamGenerator<T6>& g6)
      : g1_(g1), g2_(g2), g3_(g3), g4_(g4), g5_(g5), g6_(g6) {}
  virtual ~CartesianProductGenerator6() {}

  virtual ParamIteratorInterface<ParamType>* Begin() const {
    return new Iterator(this, g1_, g1_.begin(), g2_, g2_.begin(), g3_,
        g3_.begin(), g4_, g4_.begin(), g5_, g5_.begin(), g6_, g6_.begin());
  }
  virtual ParamIteratorInterface<ParamType>* End() const {
    return new Iterator(this, g1_, g1_.end(), g2_, g2_.end(), g3_, g3_.end(),
        g4_, g4_.end(), g5_, g5_.end(), g6_, g6_.end());
  }

 private:
  class Iterator : public ParamIteratorInterface<ParamType> {
   public:
    Iterator(const ParamGeneratorInterface<ParamType>* base,
      const ParamGenerator<T1>& g1,
      const typename ParamGenerator<T1>::iterator& current1,
      const ParamGenerator<T2>& g2,
      const typename ParamGenerator<T2>::iterator& current2,
      const ParamGenerator<T3>& g3,
      const typename ParamGenerator<T3>::iterator& current3,
      const ParamGenerator<T4>& g4,
      const typename ParamGenerator<T4>::iterator& current4,
      const ParamGenerator<T5>& g5,
      const typename ParamGenerator<T5>::iterator& current5,
      const ParamGenerator<T6>& g6,
      const typename ParamGenerator<T6>::iterator& current6)
        : base_(base),
          begin1_(g1.begin()), end1_(g1.end()), current1_(current1),
          begin2_(g2.begin()), end2_(g2.end()), current2_(current2),
          begin3_(g3.begin()), end3_(g3.end()), current3_(current3),
          begin4_(g4.begin()), end4_(g4.end()), current4_(current4),
          begin5_(g5.begin()), end5_(g5.end()), current5_(current5),
          begin6_(g6.begin()), end6_(g6.end()), current6_(current6)    {
      ComputeCurrentValue();
    }
    virtual ~Iterator() {}

    virtual const ParamGeneratorInterface<ParamType>* BaseGenerator() const {
      return base_;
    }
    // Advance should not be called on beyond-of-range iterators
    // so no component iterators must be beyond end of range, either.
    virtual void Advance() {
      assert(!AtEnd());
      ++current6_;
      if (current6_ == end6_) {
        current6_ = begin6_;
        ++current5_;
      }
      if (current5_ == end5_) {
        current5_ = begin5_;
        ++current4_;
      }
      if (current4_ == end4_) {
        current4_ = begin4_;
        ++current3_;
      }
      if (current3_ == end3_) {
        current3_ = begin3_;
        ++current2_;
      }
      if (current2_ == end2_) {
        current2_ = begin2_;
        ++current1_;
      }
      ComputeCurrentValue();
    }
    virtual ParamIteratorInterface<ParamType>* Clone() const {
      return new Iterator(*this);
    }
    virtual const ParamType* Current() const { return &current_value_; }
    virtual bool Equals(const ParamIteratorInterface<ParamType>& other) const {
      // Having the same base generator guarantees that the other
      // iterator is of the same type and we can downcast.
      GTEST_CHECK_(BaseGenerator() == other.BaseGenerator())
          << "The program attempted to compare iterators "
          << "from different generators." << std::endl;
      const Iterator* typed_other =
          CheckedDowncastToActualType<const Iterator>(&other);
      // We must report iterators equal if they both point beyond their
      // respective ranges. That can happen in a variety of fashions,
      // so we have to consult AtEnd().
      return (AtEnd() && typed_other->AtEnd()) ||
         (
          current1_ == typed_other->current1_ &&
          current2_ == typed_other->current2_ &&
          current3_ == typed_other->current3_ &&
          current4_ == typed_other->current4_ &&
          current5_ == typed_other->current5_ &&
          current6_ == typed_other->current6_);
    }

   private:
    Iterator(const Iterator& other)
        : base_(other.base_),
        begin1_(other.begin1_),
        end1_(other.end1_),
        current1_(other.current1_),
        begin2_(other.begin2_),
        end2_(other.end2_),
        current2_(other.current2_),
        begin3_(other.begin3_),
        end3_(other.end3_),
        current3_(other.current3_),
        begin4_(other.begin4_),
        end4_(other.end4_),
        current4_(other.current4_),
        begin5_(other.begin5_),
        end5_(other.end5_),
        current5_(other.current5_),
        begin6_(other.begin6_),
        end6_(other.end6_),
        current6_(other.current6_) {
      ComputeCurrentValue();
    }

    void ComputeCurrentValue() {
      if (!AtEnd())
        current_value_ = ParamType(*current1_, *current2_, *current3_,
            *current4_, *current5_, *current6_);
    }
    bool AtEnd() const {
      // We must report iterator past the end of the range when either of the
      // component iterators has reached the end of its range.
      return
          current1_ == end1_ ||
          current2_ == end2_ ||
          current3_ == end3_ ||
          current4_ == end4_ ||
          current5_ == end5_ ||
          current6_ == end6_;
    }

    // No implementation - assignment is unsupported.
    void operator=(const Iterator& other);

    const ParamGeneratorInterface<ParamType>* const base_;
    // begin[i]_ and end[i]_ define the i-th range that Iterator traverses.
    // current[i]_ is the actual traversing iterator.
    const typename ParamGenerator<T1>::iterator begin1_;
    const typename ParamGenerator<T1>::iterator end1_;
    typename ParamGenerator<T1>::iterator current1_;
    const typename ParamGenerator<T2>::iterator begin2_;
    const typename ParamGenerator<T2>::iterator end2_;
    typename ParamGenerator<T2>::iterator current2_;
    const typename ParamGenerator<T3>::iterator begin3_;
    const typename ParamGenerator<T3>::iterator end3_;
    typename ParamGenerator<T3>::iterator current3_;
    const typename ParamGenerator<T4>::iterator begin4_;
    const typename ParamGenerator<T4>::iterator end4_;
    typename ParamGenerator<T4>::iterator current4_;
    const typename ParamGenerator<T5>::iterator begin5_;
    const typename ParamGenerator<T5>::iterator end5_;
    typename ParamGenerator<T5>::iterator current5_;
    const typename ParamGenerator<T6>::iterator begin6_;
    const typename ParamGenerator<T6>::iterator end6_;
    typename ParamGenerator<T6>::iterator current6_;
    ParamType current_value_;
  };  // class CartesianProductGenerator6::Iterator

  // No implementation - assignment is unsupported.
  void operator=(const CartesianProductGenerator6& other);

  const ParamGenerator<T1> g1_;
  const ParamGenerator<T2> g2_;
  const ParamGenerator<T3> g3_;
  const ParamGenerator<T4> g4_;
  const ParamGenerator<T5> g5_;
  const ParamGenerator<T6> g6_;
};  // class CartesianProductGenerator6


template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7>
class CartesianProductGenerator7
    : public ParamGeneratorInterface< ::std::tr1::tuple<T1, T2, T3, T4, T5, T6,
        T7> > {
 public:
  typedef ::std::tr1::tuple<T1, T2, T3, T4, T5, T6, T7> ParamType;

  CartesianProductGenerator7(const ParamGenerator<T1>& g1,
      const ParamGenerator<T2>& g2, const ParamGenerator<T3>& g3,
      const ParamGenerator<T4>& g4, const ParamGenerator<T5>& g5,
      const ParamGenerator<T6>& g6, const ParamGenerator<T7>& g7)
      : g1_(g1), g2_(g2), g3_(g3), g4_(g4), g5_(g5), g6_(g6), g7_(g7) {}
  virtual ~CartesianProductGenerator7() {}

  virtual ParamIteratorInterface<ParamType>* Begin() const {
    return new Iterator(this, g1_, g1_.begin(), g2_, g2_.begin(), g3_,
        g3_.begin(), g4_, g4_.begin(), g5_, g5_.begin(), g6_, g6_.begin(), g7_,
        g7_.begin());
  }
  virtual ParamIteratorInterface<ParamType>* End() const {
    return new Iterator(this, g1_, g1_.end(), g2_, g2_.end(), g3_, g3_.end(),
        g4_, g4_.end(), g5_, g5_.end(), g6_, g6_.end(), g7_, g7_.end());
  }

 private:
  class Iterator : public ParamIteratorInterface<ParamType> {
   public:
    Iterator(const ParamGeneratorInterface<ParamType>* base,
      const ParamGenerator<T1>& g1,
      const typename ParamGenerator<T1>::iterator& current1,
      const ParamGenerator<T2>& g2,
      const typename ParamGenerator<T2>::iterator& current2,
      const ParamGenerator<T3>& g3,
      const typename ParamGenerator<T3>::iterator& current3,
      const ParamGenerator<T4>& g4,
      const typename ParamGenerator<T4>::iterator& current4,
      const ParamGenerator<T5>& g5,
      const typename ParamGenerator<T5>::iterator& current5,
      const ParamGenerator<T6>& g6,
      const typename ParamGenerator<T6>::iterator& current6,
      const ParamGenerator<T7>& g7,
      const typename ParamGenerator<T7>::iterator& current7)
        : base_(base),
          begin1_(g1.begin()), end1_(g1.end()), current1_(current1),
          begin2_(g2.begin()), end2_(g2.end()), current2_(current2),
          begin3_(g3.begin()), end3_(g3.end()), current3_(current3),
          begin4_(g4.begin()), end4_(g4.end()), current4_(current4),
          begin5_(g5.begin()), end5_(g5.end()), current5_(current5),
          begin6_(g6.begin()), end6_(g6.end()), current6_(current6),
          begin7_(g7.begin()), end7_(g7.end()), current7_(current7)    {
      ComputeCurrentValue();
    }
    virtual ~Iterator() {}

    virtual const ParamGeneratorInterface<ParamType>* BaseGenerator() const {
      return base_;
    }
    // Advance should not be called on beyond-of-range iterators
    // so no component iterators must be beyond end of range, either.
    virtual void Advance() {
      assert(!AtEnd());
      ++current7_;
      if (current7_ == end7_) {
        current7_ = begin7_;
        ++current6_;
      }
      if (current6_ == end6_) {
        current6_ = begin6_;
        ++current5_;
      }
      if (current5_ == end5_) {
        current5_ = begin5_;
        ++current4_;
      }
      if (current4_ == end4_) {
        current4_ = begin4_;
        ++current3_;
      }
      if (current3_ == end3_) {
        current3_ = begin3_;
        ++current2_;
      }
      if (current2_ == end2_) {
        current2_ = begin2_;
        ++current1_;
      }
      ComputeCurrentValue();
    }
    virtual ParamIteratorInterface<ParamType>* Clone() const {
      return new Iterator(*this);
    }
    virtual const ParamType* Current() const { return &current_value_; }
    virtual bool Equals(const ParamIteratorInterface<ParamType>& other) const {
      // Having the same base generator guarantees that the other
      // iterator is of the same type and we can downcast.
      GTEST_CHECK_(BaseGenerator() == other.BaseGenerator())
          << "The program attempted to compare iterators "
          << "from different generators." << std::endl;
      const Iterator* typed_other =
          CheckedDowncastToActualType<const Iterator>(&other);
      // We must report iterators equal if they both point beyond their
      // respective ranges. That can happen in a variety of fashions,
      // so we have to consult AtEnd().
      return (AtEnd() && typed_other->AtEnd()) ||
         (
          current1_ == typed_other->current1_ &&
          current2_ == typed_other->current2_ &&
          current3_ == typed_other->current3_ &&
          current4_ == typed_other->current4_ &&
          current5_ == typed_other->current5_ &&
          current6_ == typed_other->current6_ &&
          current7_ == typed_other->current7_);
    }

   private:
    Iterator(const Iterator& other)
        : base_(other.base_),
        begin1_(other.begin1_),
        end1_(other.end1_),
        current1_(other.current1_),
        begin2_(other.begin2_),
        end2_(other.end2_),
        current2_(other.current2_),
        begin3_(other.begin3_),
        end3_(other.end3_),
        current3_(other.current3_),
        begin4_(other.begin4_),
        end4_(other.end4_),
        current4_(other.current4_),
        begin5_(other.begin5_),
        end5_(other.end5_),
        current5_(other.current5_),
        begin6_(other.begin6_),
        end6_(other.end6_),
        current6_(other.current6_),
        begin7_(other.begin7_),
        end7_(other.end7_),
        current7_(other.current7_) {
      ComputeCurrentValue();
    }

    void ComputeCurrentValue() {
      if (!AtEnd())
        current_value_ = ParamType(*current1_, *current2_, *current3_,
            *current4_, *current5_, *current6_, *current7_);
    }
    bool AtEnd() const {
      // We must report iterator past the end of the range when either of the
      // component iterators has reached the end of its range.
      return
          current1_ == end1_ ||
          current2_ == end2_ ||
          current3_ == end3_ ||
          current4_ == end4_ ||
          current5_ == end5_ ||
          current6_ == end6_ ||
          current7_ == end7_;
    }

    // No implementation - assignment is unsupported.
    void operator=(const Iterator& other);

    const ParamGeneratorInterface<ParamType>* const base_;
    // begin[i]_ and end[i]_ define the i-th range that Iterator traverses.
    // current[i]_ is the actual traversing iterator.
    const typename ParamGenerator<T1>::iterator begin1_;
    const typename ParamGenerator<T1>::iterator end1_;
    typename ParamGenerator<T1>::iterator current1_;
    const typename ParamGenerator<T2>::iterator begin2_;
    const typename ParamGenerator<T2>::iterator end2_;
    typename ParamGenerator<T2>::iterator current2_;
    const typename ParamGenerator<T3>::iterator begin3_;
    const typename ParamGenerator<T3>::iterator end3_;
    typename ParamGenerator<T3>::iterator current3_;
    const typename ParamGenerator<T4>::iterator begin4_;
    const typename ParamGenerator<T4>::iterator end4_;
    typename ParamGenerator<T4>::iterator current4_;
    const typename ParamGenerator<T5>::iterator begin5_;
    const typename ParamGenerator<T5>::iterator end5_;
    typename ParamGenerator<T5>::iterator current5_;
    const typename ParamGenerator<T6>::iterator begin6_;
    const typename ParamGenerator<T6>::iterator end6_;
    typename ParamGenerator<T6>::iterator current6_;
    const typename ParamGenerator<T7>::iterator begin7_;
    const typename ParamGenerator<T7>::iterator end7_;
    typename ParamGenerator<T7>::iterator current7_;
    ParamType current_value_;
  };  // class CartesianProductGenerator7::Iterator

  // No implementation - assignment is unsupported.
  void operator=(const CartesianProductGenerator7& other);

  const ParamGenerator<T1> g1_;
  const ParamGenerator<T2> g2_;
  const ParamGenerator<T3> g3_;
  const ParamGenerator<T4> g4_;
  const ParamGenerator<T5> g5_;
  const ParamGenerator<T6> g6_;
  const ParamGenerator<T7> g7_;
};  // class CartesianProductGenerator7


template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8>
class CartesianProductGenerator8
    : public ParamGeneratorInterface< ::std::tr1::tuple<T1, T2, T3, T4, T5, T6,
        T7, T8> > {
 public:
  typedef ::std::tr1::tuple<T1, T2, T3, T4, T5, T6, T7, T8> ParamType;

  CartesianProductGenerator8(const ParamGenerator<T1>& g1,
      const ParamGenerator<T2>& g2, const ParamGenerator<T3>& g3,
      const ParamGenerator<T4>& g4, const ParamGenerator<T5>& g5,
      const ParamGenerator<T6>& g6, const ParamGenerator<T7>& g7,
      const ParamGenerator<T8>& g8)
      : g1_(g1), g2_(g2), g3_(g3), g4_(g4), g5_(g5), g6_(g6), g7_(g7),
          g8_(g8) {}
  virtual ~CartesianProductGenerator8() {}

  virtual ParamIteratorInterface<ParamType>* Begin() const {
    return new Iterator(this, g1_, g1_.begin(), g2_, g2_.begin(), g3_,
        g3_.begin(), g4_, g4_.begin(), g5_, g5_.begin(), g6_, g6_.begin(), g7_,
        g7_.begin(), g8_, g8_.begin());
  }
  virtual ParamIteratorInterface<ParamType>* End() const {
    return new Iterator(this, g1_, g1_.end(), g2_, g2_.end(), g3_, g3_.end(),
        g4_, g4_.end(), g5_, g5_.end(), g6_, g6_.end(), g7_, g7_.end(), g8_,
        g8_.end());
  }

 private:
  class Iterator : public ParamIteratorInterface<ParamType> {
   public:
    Iterator(const ParamGeneratorInterface<ParamType>* base,
      const ParamGenerator<T1>& g1,
      const typename ParamGenerator<T1>::iterator& current1,
      const ParamGenerator<T2>& g2,
      const typename ParamGenerator<T2>::iterator& current2,
      const ParamGenerator<T3>& g3,
      const typename ParamGenerator<T3>::iterator& current3,
      const ParamGenerator<T4>& g4,
      const typename ParamGenerator<T4>::iterator& current4,
      const ParamGenerator<T5>& g5,
      const typename ParamGenerator<T5>::iterator& current5,
      const ParamGenerator<T6>& g6,
      const typename ParamGenerator<T6>::iterator& current6,
      const ParamGenerator<T7>& g7,
      const typename ParamGenerator<T7>::iterator& current7,
      const ParamGenerator<T8>& g8,
      const typename ParamGenerator<T8>::iterator& current8)
        : base_(base),
          begin1_(g1.begin()), end1_(g1.end()), current1_(current1),
          begin2_(g2.begin()), end2_(g2.end()), current2_(current2),
          begin3_(g3.begin()), end3_(g3.end()), current3_(current3),
          begin4_(g4.begin()), end4_(g4.end()), current4_(current4),
          begin5_(g5.begin()), end5_(g5.end()), current5_(current5),
          begin6_(g6.begin()), end6_(g6.end()), current6_(current6),
          begin7_(g7.begin()), end7_(g7.end()), current7_(current7),
          begin8_(g8.begin()), end8_(g8.end()), current8_(current8)    {
      ComputeCurrentValue();
    }
    virtual ~Iterator() {}

    virtual const ParamGeneratorInterface<ParamType>* BaseGenerator() const {
      return base_;
    }
    // Advance should not be called on beyond-of-range iterators
    // so no component iterators must be beyond end of range, either.
    virtual void Advance() {
      assert(!AtEnd());
      ++current8_;
      if (current8_ == end8_) {
        current8_ = begin8_;
        ++current7_;
      }
      if (current7_ == end7_) {
        current7_ = begin7_;
        ++current6_;
      }
      if (current6_ == end6_) {
        current6_ = begin6_;
        ++current5_;
      }
      if (current5_ == end5_) {
        current5_ = begin5_;
        ++current4_;
      }
      if (current4_ == end4_) {
        current4_ = begin4_;
        ++current3_;
      }
      if (current3_ == end3_) {
        current3_ = begin3_;
        ++current2_;
      }
      if (current2_ == end2_) {
        current2_ = begin2_;
        ++current1_;
      }
      ComputeCurrentValue();
    }
    virtual ParamIteratorInterface<ParamType>* Clone() const {
      return new Iterator(*this);
    }
    virtual const ParamType* Current() const { return &current_value_; }
    virtual bool Equals(const ParamIteratorInterface<ParamType>& other) const {
      // Having the same base generator guarantees that the other
      // iterator is of the same type and we can downcast.
      GTEST_CHECK_(BaseGenerator() == other.BaseGenerator())
          << "The program attempted to compare iterators "
          << "from different generators." << std::endl;
      const Iterator* typed_other =
          CheckedDowncastToActualType<const Iterator>(&other);
      // We must report iterators equal if they both point beyond their
      // respective ranges. That can happen in a variety of fashions,
      // so we have to consult AtEnd().
      return (AtEnd() && typed_other->AtEnd()) ||
         (
          current1_ == typed_other->current1_ &&
          current2_ == typed_other->current2_ &&
          current3_ == typed_other->current3_ &&
          current4_ == typed_other->current4_ &&
          current5_ == typed_other->current5_ &&
          current6_ == typed_other->current6_ &&
          current7_ == typed_other->current7_ &&
          current8_ == typed_other->current8_);
    }

   private:
    Iterator(const Iterator& other)
        : base_(other.base_),
        begin1_(other.begin1_),
        end1_(other.end1_),
        current1_(other.current1_),
        begin2_(other.begin2_),
        end2_(other.end2_),
        current2_(other.current2_),
        begin3_(other.begin3_),
        end3_(other.end3_),
        current3_(other.current3_),
        begin4_(other.begin4_),
        end4_(other.end4_),
        current4_(other.current4_),
        begin5_(other.begin5_),
        end5_(other.end5_),
        current5_(other.current5_),
        begin6_(other.begin6_),
        end6_(other.end6_),
        current6_(other.current6_),
        begin7_(other.begin7_),
        end7_(other.end7_),
        current7_(other.current7_),
        begin8_(other.begin8_),
        end8_(other.end8_),
        current8_(other.current8_) {
      ComputeCurrentValue();
    }

    void ComputeCurrentValue() {
      if (!AtEnd())
        current_value_ = ParamType(*current1_, *current2_, *current3_,
            *current4_, *current5_, *current6_, *current7_, *current8_);
    }
    bool AtEnd() const {
      // We must report iterator past the end of the range when either of the
      // component iterators has reached the end of its range.
      return
          current1_ == end1_ ||
          current2_ == end2_ ||
          current3_ == end3_ ||
          current4_ == end4_ ||
          current5_ == end5_ ||
          current6_ == end6_ ||
          current7_ == end7_ ||
          current8_ == end8_;
    }

    // No implementation - assignment is unsupported.
    void operator=(const Iterator& other);

    const ParamGeneratorInterface<ParamType>* const base_;
    // begin[i]_ and end[i]_ define the i-th range that Iterator traverses.
    // current[i]_ is the actual traversing iterator.
    const typename ParamGenerator<T1>::iterator begin1_;
    const typename ParamGenerator<T1>::iterator end1_;
    typename ParamGenerator<T1>::iterator current1_;
    const typename ParamGenerator<T2>::iterator begin2_;
    const typename ParamGenerator<T2>::iterator end2_;
    typename ParamGenerator<T2>::iterator current2_;
    const typename ParamGenerator<T3>::iterator begin3_;
    const typename ParamGenerator<T3>::iterator end3_;
    typename ParamGenerator<T3>::iterator current3_;
    const typename ParamGenerator<T4>::iterator begin4_;
    const typename ParamGenerator<T4>::iterator end4_;
    typename ParamGenerator<T4>::iterator current4_;
    const typename ParamGenerator<T5>::iterator begin5_;
    const typename ParamGenerator<T5>::iterator end5_;
    typename ParamGenerator<T5>::iterator current5_;
    const typename ParamGenerator<T6>::iterator begin6_;
    const typename ParamGenerator<T6>::iterator end6_;
    typename ParamGenerator<T6>::iterator current6_;
    const typename ParamGenerator<T7>::iterator begin7_;
    const typename ParamGenerator<T7>::iterator end7_;
    typename ParamGenerator<T7>::iterator current7_;
    const typename ParamGenerator<T8>::iterator begin8_;
    const typename ParamGenerator<T8>::iterator end8_;
    typename ParamGenerator<T8>::iterator current8_;
    ParamType current_value_;
  };  // class CartesianProductGenerator8::Iterator

  // No implementation - assignment is unsupported.
  void operator=(const CartesianProductGenerator8& other);

  const ParamGenerator<T1> g1_;
  const ParamGenerator<T2> g2_;
  const ParamGenerator<T3> g3_;
  const ParamGenerator<T4> g4_;
  const ParamGenerator<T5> g5_;
  const ParamGenerator<T6> g6_;
  const ParamGenerator<T7> g7_;
  const ParamGenerator<T8> g8_;
};  // class CartesianProductGenerator8


template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9>
class CartesianProductGenerator9
    : public ParamGeneratorInterface< ::std::tr1::tuple<T1, T2, T3, T4, T5, T6,
        T7, T8, T9> > {
 public:
  typedef ::std::tr1::tuple<T1, T2, T3, T4, T5, T6, T7, T8, T9> ParamType;

  CartesianProductGenerator9(const ParamGenerator<T1>& g1,
      const ParamGenerator<T2>& g2, const ParamGenerator<T3>& g3,
      const ParamGenerator<T4>& g4, const ParamGenerator<T5>& g5,
      const ParamGenerator<T6>& g6, const ParamGenerator<T7>& g7,
      const ParamGenerator<T8>& g8, const ParamGenerator<T9>& g9)
      : g1_(g1), g2_(g2), g3_(g3), g4_(g4), g5_(g5), g6_(g6), g7_(g7), g8_(g8),
          g9_(g9) {}
  virtual ~CartesianProductGenerator9() {}

  virtual ParamIteratorInterface<ParamType>* Begin() const {
    return new Iterator(this, g1_, g1_.begin(), g2_, g2_.begin(), g3_,
        g3_.begin(), g4_, g4_.begin(), g5_, g5_.begin(), g6_, g6_.begin(), g7_,
        g7_.begin(), g8_, g8_.begin(), g9_, g9_.begin());
  }
  virtual ParamIteratorInterface<ParamType>* End() const {
    return new Iterator(this, g1_, g1_.end(), g2_, g2_.end(), g3_, g3_.end(),
        g4_, g4_.end(), g5_, g5_.end(), g6_, g6_.end(), g7_, g7_.end(), g8_,
        g8_.end(), g9_, g9_.end());
  }

 private:
  class Iterator : public ParamIteratorInterface<ParamType> {
   public:
    Iterator(const ParamGeneratorInterface<ParamType>* base,
      const ParamGenerator<T1>& g1,
      const typename ParamGenerator<T1>::iterator& current1,
      const ParamGenerator<T2>& g2,
      const typename ParamGenerator<T2>::iterator& current2,
      const ParamGenerator<T3>& g3,
      const typename ParamGenerator<T3>::iterator& current3,
      const ParamGenerator<T4>& g4,
      const typename ParamGenerator<T4>::iterator& current4,
      const ParamGenerator<T5>& g5,
      const typename ParamGenerator<T5>::iterator& current5,
      const ParamGenerator<T6>& g6,
      const typename ParamGenerator<T6>::iterator& current6,
      const ParamGenerator<T7>& g7,
      const typename ParamGenerator<T7>::iterator& current7,
      const ParamGenerator<T8>& g8,
      const typename ParamGenerator<T8>::iterator& current8,
      const ParamGenerator<T9>& g9,
      const typename ParamGenerator<T9>::iterator& current9)
        : base_(base),
          begin1_(g1.begin()), end1_(g1.end()), current1_(current1),
          begin2_(g2.begin()), end2_(g2.end()), current2_(current2),
          begin3_(g3.begin()), end3_(g3.end()), current3_(current3),
          begin4_(g4.begin()), end4_(g4.end()), current4_(current4),
          begin5_(g5.begin()), end5_(g5.end()), current5_(current5),
          begin6_(g6.begin()), end6_(g6.end()), current6_(current6),
          begin7_(g7.begin()), end7_(g7.end()), current7_(current7),
          begin8_(g8.begin()), end8_(g8.end()), current8_(current8),
          begin9_(g9.begin()), end9_(g9.end()), current9_(current9)    {
      ComputeCurrentValue();
    }
    virtual ~Iterator() {}

    virtual const ParamGeneratorInterface<ParamType>* BaseGenerator() const {
      return base_;
    }
    // Advance should not be called on beyond-of-range iterators
    // so no component iterators must be beyond end of range, either.
    virtual void Advance() {
      assert(!AtEnd());
      ++current9_;
      if (current9_ == end9_) {
        current9_ = begin9_;
        ++current8_;
      }
      if (current8_ == end8_) {
        current8_ = begin8_;
        ++current7_;
      }
      if (current7_ == end7_) {
        current7_ = begin7_;
        ++current6_;
      }
      if (current6_ == end6_) {
        current6_ = begin6_;
        ++current5_;
      }
      if (current5_ == end5_) {
        current5_ = begin5_;
        ++current4_;
      }
      if (current4_ == end4_) {
        current4_ = begin4_;
        ++current3_;
      }
      if (current3_ == end3_) {
        current3_ = begin3_;
        ++current2_;
      }
      if (current2_ == end2_) {
        current2_ = begin2_;
        ++current1_;
      }
      ComputeCurrentValue();
    }
    virtual ParamIteratorInterface<ParamType>* Clone() const {
      return new Iterator(*this);
    }
    virtual const ParamType* Current() const { return &current_value_; }
    virtual bool Equals(const ParamIteratorInterface<ParamType>& other) const {
      // Having the same base generator guarantees that the other
      // iterator is of the same type and we can downcast.
      GTEST_CHECK_(BaseGenerator() == other.BaseGenerator())
          << "The program attempted to compare iterators "
          << "from different generators." << std::endl;
      const Iterator* typed_other =
          CheckedDowncastToActualType<const Iterator>(&other);
      // We must report iterators equal if they both point beyond their
      // respective ranges. That can happen in a variety of fashions,
      // so we have to consult AtEnd().
      return (AtEnd() && typed_other->AtEnd()) ||
         (
          current1_ == typed_other->current1_ &&
          current2_ == typed_other->current2_ &&
          current3_ == typed_other->current3_ &&
          current4_ == typed_other->current4_ &&
          current5_ == typed_other->current5_ &&
          current6_ == typed_other->current6_ &&
          current7_ == typed_other->current7_ &&
          current8_ == typed_other->current8_ &&
          current9_ == typed_other->current9_);
    }

   private:
    Iterator(const Iterator& other)
        : base_(other.base_),
        begin1_(other.begin1_),
        end1_(other.end1_),
        current1_(other.current1_),
        begin2_(other.begin2_),
        end2_(other.end2_),
        current2_(other.current2_),
        begin3_(other.begin3_),
        end3_(other.end3_),
        current3_(other.current3_),
        begin4_(other.begin4_),
        end4_(other.end4_),
        current4_(other.current4_),
        begin5_(other.begin5_),
        end5_(other.end5_),
        current5_(other.current5_),
        begin6_(other.begin6_),
        end6_(other.end6_),
        current6_(other.current6_),
        begin7_(other.begin7_),
        end7_(other.end7_),
        current7_(other.current7_),
        begin8_(other.begin8_),
        end8_(other.end8_),
        current8_(other.current8_),
        begin9_(other.begin9_),
        end9_(other.end9_),
        current9_(other.current9_) {
      ComputeCurrentValue();
    }

    void ComputeCurrentValue() {
      if (!AtEnd())
        current_value_ = ParamType(*current1_, *current2_, *current3_,
            *current4_, *current5_, *current6_, *current7_, *current8_,
            *current9_);
    }
    bool AtEnd() const {
      // We must report iterator past the end of the range when either of the
      // component iterators has reached the end of its range.
      return
          current1_ == end1_ ||
          current2_ == end2_ ||
          current3_ == end3_ ||
          current4_ == end4_ ||
          current5_ == end5_ ||
          current6_ == end6_ ||
          current7_ == end7_ ||
          current8_ == end8_ ||
          current9_ == end9_;
    }

    // No implementation - assignment is unsupported.
    void operator=(const Iterator& other);

    const ParamGeneratorInterface<ParamType>* const base_;
    // begin[i]_ and end[i]_ define the i-th range that Iterator traverses.
    // current[i]_ is the actual traversing iterator.
    const typename ParamGenerator<T1>::iterator begin1_;
    const typename ParamGenerator<T1>::iterator end1_;
    typename ParamGenerator<T1>::iterator current1_;
    const typename ParamGenerator<T2>::iterator begin2_;
    const typename ParamGenerator<T2>::iterator end2_;
    typename ParamGenerator<T2>::iterator current2_;
    const typename ParamGenerator<T3>::iterator begin3_;
    const typename ParamGenerator<T3>::iterator end3_;
    typename ParamGenerator<T3>::iterator current3_;
    const typename ParamGenerator<T4>::iterator begin4_;
    const typename ParamGenerator<T4>::iterator end4_;
    typename ParamGenerator<T4>::iterator current4_;
    const typename ParamGenerator<T5>::iterator begin5_;
    const typename ParamGenerator<T5>::iterator end5_;
    typename ParamGenerator<T5>::iterator current5_;
    const typename ParamGenerator<T6>::iterator begin6_;
    const typename ParamGenerator<T6>::iterator end6_;
    typename ParamGenerator<T6>::iterator current6_;
    const typename ParamGenerator<T7>::iterator begin7_;
    const typename ParamGenerator<T7>::iterator end7_;
    typename ParamGenerator<T7>::iterator current7_;
    const typename ParamGenerator<T8>::iterator begin8_;
    const typename ParamGenerator<T8>::iterator end8_;
    typename ParamGenerator<T8>::iterator current8_;
    const typename ParamGenerator<T9>::iterator begin9_;
    const typename ParamGenerator<T9>::iterator end9_;
    typename ParamGenerator<T9>::iterator current9_;
    ParamType current_value_;
  };  // class CartesianProductGenerator9::Iterator

  // No implementation - assignment is unsupported.
  void operator=(const CartesianProductGenerator9& other);

  const ParamGenerator<T1> g1_;
  const ParamGenerator<T2> g2_;
  const ParamGenerator<T3> g3_;
  const ParamGenerator<T4> g4_;
  const ParamGenerator<T5> g5_;
  const ParamGenerator<T6> g6_;
  const ParamGenerator<T7> g7_;
  const ParamGenerator<T8> g8_;
  const ParamGenerator<T9> g9_;
};  // class CartesianProductGenerator9


template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10>
class CartesianProductGenerator10
    : public ParamGeneratorInterface< ::std::tr1::tuple<T1, T2, T3, T4, T5, T6,
        T7, T8, T9, T10> > {
 public:
  typedef ::std::tr1::tuple<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> ParamType;

  CartesianProductGenerator10(const ParamGenerator<T1>& g1,
      const ParamGenerator<T2>& g2, const ParamGenerator<T3>& g3,
      const ParamGenerator<T4>& g4, const ParamGenerator<T5>& g5,
      const ParamGenerator<T6>& g6, const ParamGenerator<T7>& g7,
      const ParamGenerator<T8>& g8, const ParamGenerator<T9>& g9,
      const ParamGenerator<T10>& g10)
      : g1_(g1), g2_(g2), g3_(g3), g4_(g4), g5_(g5), g6_(g6), g7_(g7), g8_(g8),
          g9_(g9), g10_(g10) {}
  virtual ~CartesianProductGenerator10() {}

  virtual ParamIteratorInterface<ParamType>* Begin() const {
    return new Iterator(this, g1_, g1_.begin(), g2_, g2_.begin(), g3_,
        g3_.begin(), g4_, g4_.begin(), g5_, g5_.begin(), g6_, g6_.begin(), g7_,
        g7_.begin(), g8_, g8_.begin(), g9_, g9_.begin(), g10_, g10_.begin());
  }
  virtual ParamIteratorInterface<ParamType>* End() const {
    return new Iterator(this, g1_, g1_.end(), g2_, g2_.end(), g3_, g3_.end(),
        g4_, g4_.end(), g5_, g5_.end(), g6_, g6_.end(), g7_, g7_.end(), g8_,
        g8_.end(), g9_, g9_.end(), g10_, g10_.end());
  }

 private:
  class Iterator : public ParamIteratorInterface<ParamType> {
   public:
    Iterator(const ParamGeneratorInterface<ParamType>* base,
      const ParamGenerator<T1>& g1,
      const typename ParamGenerator<T1>::iterator& current1,
      const ParamGenerator<T2>& g2,
      const typename ParamGenerator<T2>::iterator& current2,
      const ParamGenerator<T3>& g3,
      const typename ParamGenerator<T3>::iterator& current3,
      const ParamGenerator<T4>& g4,
      const typename ParamGenerator<T4>::iterator& current4,
      const ParamGenerator<T5>& g5,
      const typename ParamGenerator<T5>::iterator& current5,
      const ParamGenerator<T6>& g6,
      const typename ParamGenerator<T6>::iterator& current6,
      const ParamGenerator<T7>& g7,
      const typename ParamGenerator<T7>::iterator& current7,
      const ParamGenerator<T8>& g8,
      const typename ParamGenerator<T8>::iterator& current8,
      const ParamGenerator<T9>& g9,
      const typename ParamGenerator<T9>::iterator& current9,
      const ParamGenerator<T10>& g10,
      const typename ParamGenerator<T10>::iterator& current10)
        : base_(base),
          begin1_(g1.begin()), end1_(g1.end()), current1_(current1),
          begin2_(g2.begin()), end2_(g2.end()), current2_(current2),
          begin3_(g3.begin()), end3_(g3.end()), current3_(current3),
          begin4_(g4.begin()), end4_(g4.end()), current4_(current4),
          begin5_(g5.begin()), end5_(g5.end()), current5_(current5),
          begin6_(g6.begin()), end6_(g6.end()), current6_(current6),
          begin7_(g7.begin()), end7_(g7.end()), current7_(current7),
          begin8_(g8.begin()), end8_(g8.end()), current8_(current8),
          begin9_(g9.begin()), end9_(g9.end()), current9_(current9),
          begin10_(g10.begin()), end10_(g10.end()), current10_(current10)    {
      ComputeCurrentValue();
    }
    virtual ~Iterator() {}

    virtual const ParamGeneratorInterface<ParamType>* BaseGenerator() const {
      return base_;
    }
    // Advance should not be called on beyond-of-range iterators
    // so no component iterators must be beyond end of range, either.
    virtual void Advance() {
      assert(!AtEnd());
      ++current10_;
      if (current10_ == end10_) {
        current10_ = begin10_;
        ++current9_;
      }
      if (current9_ == end9_) {
        current9_ = begin9_;
        ++current8_;
      }
      if (current8_ == end8_) {
        current8_ = begin8_;
        ++current7_;
      }
      if (current7_ == end7_) {
        current7_ = begin7_;
        ++current6_;
      }
      if (current6_ == end6_) {
        current6_ = begin6_;
        ++current5_;
      }
      if (current5_ == end5_) {
        current5_ = begin5_;
        ++current4_;
      }
      if (current4_ == end4_) {
        current4_ = begin4_;
        ++current3_;
      }
      if (current3_ == end3_) {
        current3_ = begin3_;
        ++current2_;
      }
      if (current2_ == end2_) {
        current2_ = begin2_;
        ++current1_;
      }
      ComputeCurrentValue();
    }
    virtual ParamIteratorInterface<ParamType>* Clone() const {
      return new Iterator(*this);
    }
    virtual const ParamType* Current() const { return &current_value_; }
    virtual bool Equals(const ParamIteratorInterface<ParamType>& other) const {
      // Having the same base generator guarantees that the other
      // iterator is of the same type and we can downcast.
      GTEST_CHECK_(BaseGenerator() == other.BaseGenerator())
          << "The program attempted to compare iterators "
          << "from different generators." << std::endl;
      const Iterator* typed_other =
          CheckedDowncastToActualType<const Iterator>(&other);
      // We must report iterators equal if they both point beyond their
      // respective ranges. That can happen in a variety of fashions,
      // so we have to consult AtEnd().
      return (AtEnd() && typed_other->AtEnd()) ||
         (
          current1_ == typed_other->current1_ &&
          current2_ == typed_other->current2_ &&
          current3_ == typed_other->current3_ &&
          current4_ == typed_other->current4_ &&
          current5_ == typed_other->current5_ &&
          current6_ == typed_other->current6_ &&
          current7_ == typed_other->current7_ &&
          current8_ == typed_other->current8_ &&
          current9_ == typed_other->current9_ &&
          current10_ == typed_other->current10_);
    }

   private:
    Iterator(const Iterator& other)
        : base_(other.base_),
        begin1_(other.begin1_),
        end1_(other.end1_),
        current1_(other.current1_),
        begin2_(other.begin2_),
        end2_(other.end2_),
        current2_(other.current2_),
        begin3_(other.begin3_),
        end3_(other.end3_),
        current3_(other.current3_),
        begin4_(other.begin4_),
        end4_(other.end4_),
        current4_(other.current4_),
        begin5_(other.begin5_),
        end5_(other.end5_),
        current5_(other.current5_),
        begin6_(other.begin6_),
        end6_(other.end6_),
        current6_(other.current6_),
        begin7_(other.begin7_),
        end7_(other.end7_),
        current7_(other.current7_),
        begin8_(other.begin8_),
        end8_(other.end8_),
        current8_(other.current8_),
        begin9_(other.begin9_),
        end9_(other.end9_),
        current9_(other.current9_),
        begin10_(other.begin10_),
        end10_(other.end10_),
        current10_(other.current10_) {
      ComputeCurrentValue();
    }

    void ComputeCurrentValue() {
      if (!AtEnd())
        current_value_ = ParamType(*current1_, *current2_, *current3_,
            *current4_, *current5_, *current6_, *current7_, *current8_,
            *current9_, *current10_);
    }
    bool AtEnd() const {
      // We must report iterator past the end of the range when either of the
      // component iterators has reached the end of its range.
      return
          current1_ == end1_ ||
          current2_ == end2_ ||
          current3_ == end3_ ||
          current4_ == end4_ ||
          current5_ == end5_ ||
          current6_ == end6_ ||
          current7_ == end7_ ||
          current8_ == end8_ ||
          current9_ == end9_ ||
          current10_ == end10_;
    }

    // No implementation - assignment is unsupported.
    void operator=(const Iterator& other);

    const ParamGeneratorInterface<ParamType>* const base_;
    // begin[i]_ and end[i]_ define the i-th range that Iterator traverses.
    // current[i]_ is the actual traversing iterator.
    const typename ParamGenerator<T1>::iterator begin1_;
    const typename ParamGenerator<T1>::iterator end1_;
    typename ParamGenerator<T1>::iterator current1_;
    const typename ParamGenerator<T2>::iterator begin2_;
    const typename ParamGenerator<T2>::iterator end2_;
    typename ParamGenerator<T2>::iterator current2_;
    const typename ParamGenerator<T3>::iterator begin3_;
    const typename ParamGenerator<T3>::iterator end3_;
    typename ParamGenerator<T3>::iterator current3_;
    const typename ParamGenerator<T4>::iterator begin4_;
    const typename ParamGenerator<T4>::iterator end4_;
    typename ParamGenerator<T4>::iterator current4_;
    const typename ParamGenerator<T5>::iterator begin5_;
    const typename ParamGenerator<T5>::iterator end5_;
    typename ParamGenerator<T5>::iterator current5_;
    const typename ParamGenerator<T6>::iterator begin6_;
    const typename ParamGenerator<T6>::iterator end6_;
    typename ParamGenerator<T6>::iterator current6_;
    const typename ParamGenerator<T7>::iterator begin7_;
    const typename ParamGenerator<T7>::iterator end7_;
    typename ParamGenerator<T7>::iterator current7_;
    const typename ParamGenerator<T8>::iterator begin8_;
    const typename ParamGenerator<T8>::iterator end8_;
    typename ParamGenerator<T8>::iterator current8_;
    const typename ParamGenerator<T9>::iterator begin9_;
    const typename ParamGenerator<T9>::iterator end9_;
    typename ParamGenerator<T9>::iterator current9_;
    const typename ParamGenerator<T10>::iterator begin10_;
    const typename ParamGenerator<T10>::iterator end10_;
    typename ParamGenerator<T10>::iterator current10_;
    ParamType current_value_;
  };  // class CartesianProductGenerator10::Iterator

  // No implementation - assignment is unsupported.
  void operator=(const CartesianProductGenerator10& other);

  const ParamGenerator<T1> g1_;
  const ParamGenerator<T2> g2_;
  const ParamGenerator<T3> g3_;
  const ParamGenerator<T4> g4_;
  const ParamGenerator<T5> g5_;
  const ParamGenerator<T6> g6_;
  const ParamGenerator<T7> g7_;
  const ParamGenerator<T8> g8_;
  const ParamGenerator<T9> g9_;
  const ParamGenerator<T10> g10_;
};  // class CartesianProductGenerator10


// INTERNAL IMPLEMENTATION - DO NOT USE IN USER CODE.
//
// Helper classes providing Combine() with polymorphic features. They allow
// casting CartesianProductGeneratorN<T> to ParamGenerator<U> if T is
// convertible to U.
//
template <class Generator1, class Generator2>
class CartesianProductHolder2 {
 public:
CartesianProductHolder2(const Generator1& g1, const Generator2& g2)
      : g1_(g1), g2_(g2) {}
  template <typename T1, typename T2>
  operator ParamGenerator< ::std::tr1::tuple<T1, T2> >() const {
    return ParamGenerator< ::std::tr1::tuple<T1, T2> >(
        new CartesianProductGenerator2<T1, T2>(
        static_cast<ParamGenerator<T1> >(g1_),
        static_cast<ParamGenerator<T2> >(g2_)));
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const CartesianProductHolder2& other);

  const Generator1 g1_;
  const Generator2 g2_;
};  // class CartesianProductHolder2

template <class Generator1, class Generator2, class Generator3>
class CartesianProductHolder3 {
 public:
CartesianProductHolder3(const Generator1& g1, const Generator2& g2,
    const Generator3& g3)
      : g1_(g1), g2_(g2), g3_(g3) {}
  template <typename T1, typename T2, typename T3>
  operator ParamGenerator< ::std::tr1::tuple<T1, T2, T3> >() const {
    return ParamGenerator< ::std::tr1::tuple<T1, T2, T3> >(
        new CartesianProductGenerator3<T1, T2, T3>(
        static_cast<ParamGenerator<T1> >(g1_),
        static_cast<ParamGenerator<T2> >(g2_),
        static_cast<ParamGenerator<T3> >(g3_)));
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const CartesianProductHolder3& other);

  const Generator1 g1_;
  const Generator2 g2_;
  const Generator3 g3_;
};  // class CartesianProductHolder3

template <class Generator1, class Generator2, class Generator3,
    class Generator4>
class CartesianProductHolder4 {
 public:
CartesianProductHolder4(const Generator1& g1, const Generator2& g2,
    const Generator3& g3, const Generator4& g4)
      : g1_(g1), g2_(g2), g3_(g3), g4_(g4) {}
  template <typename T1, typename T2, typename T3, typename T4>
  operator ParamGenerator< ::std::tr1::tuple<T1, T2, T3, T4> >() const {
    return ParamGenerator< ::std::tr1::tuple<T1, T2, T3, T4> >(
        new CartesianProductGenerator4<T1, T2, T3, T4>(
        static_cast<ParamGenerator<T1> >(g1_),
        static_cast<ParamGenerator<T2> >(g2_),
        static_cast<ParamGenerator<T3> >(g3_),
        static_cast<ParamGenerator<T4> >(g4_)));
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const CartesianProductHolder4& other);

  const Generator1 g1_;
  const Generator2 g2_;
  const Generator3 g3_;
  const Generator4 g4_;
};  // class CartesianProductHolder4

template <class Generator1, class Generator2, class Generator3,
    class Generator4, class Generator5>
class CartesianProductHolder5 {
 public:
CartesianProductHolder5(const Generator1& g1, const Generator2& g2,
    const Generator3& g3, const Generator4& g4, const Generator5& g5)
      : g1_(g1), g2_(g2), g3_(g3), g4_(g4), g5_(g5) {}
  template <typename T1, typename T2, typename T3, typename T4, typename T5>
  operator ParamGenerator< ::std::tr1::tuple<T1, T2, T3, T4, T5> >() const {
    return ParamGenerator< ::std::tr1::tuple<T1, T2, T3, T4, T5> >(
        new CartesianProductGenerator5<T1, T2, T3, T4, T5>(
        static_cast<ParamGenerator<T1> >(g1_),
        static_cast<ParamGenerator<T2> >(g2_),
        static_cast<ParamGenerator<T3> >(g3_),
        static_cast<ParamGenerator<T4> >(g4_),
        static_cast<ParamGenerator<T5> >(g5_)));
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const CartesianProductHolder5& other);

  const Generator1 g1_;
  const Generator2 g2_;
  const Generator3 g3_;
  const Generator4 g4_;
  const Generator5 g5_;
};  // class CartesianProductHolder5

template <class Generator1, class Generator2, class Generator3,
    class Generator4, class Generator5, class Generator6>
class CartesianProductHolder6 {
 public:
CartesianProductHolder6(const Generator1& g1, const Generator2& g2,
    const Generator3& g3, const Generator4& g4, const Generator5& g5,
    const Generator6& g6)
      : g1_(g1), g2_(g2), g3_(g3), g4_(g4), g5_(g5), g6_(g6) {}
  template <typename T1, typename T2, typename T3, typename T4, typename T5,
      typename T6>
  operator ParamGenerator< ::std::tr1::tuple<T1, T2, T3, T4, T5, T6> >() const {
    return ParamGenerator< ::std::tr1::tuple<T1, T2, T3, T4, T5, T6> >(
        new CartesianProductGenerator6<T1, T2, T3, T4, T5, T6>(
        static_cast<ParamGenerator<T1> >(g1_),
        static_cast<ParamGenerator<T2> >(g2_),
        static_cast<ParamGenerator<T3> >(g3_),
        static_cast<ParamGenerator<T4> >(g4_),
        static_cast<ParamGenerator<T5> >(g5_),
        static_cast<ParamGenerator<T6> >(g6_)));
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const CartesianProductHolder6& other);

  const Generator1 g1_;
  const Generator2 g2_;
  const Generator3 g3_;
  const Generator4 g4_;
  const Generator5 g5_;
  const Generator6 g6_;
};  // class CartesianProductHolder6

template <class Generator1, class Generator2, class Generator3,
    class Generator4, class Generator5, class Generator6, class Generator7>
class CartesianProductHolder7 {
 public:
CartesianProductHolder7(const Generator1& g1, const Generator2& g2,
    const Generator3& g3, const Generator4& g4, const Generator5& g5,
    const Generator6& g6, const Generator7& g7)
      : g1_(g1), g2_(g2), g3_(g3), g4_(g4), g5_(g5), g6_(g6), g7_(g7) {}
  template <typename T1, typename T2, typename T3, typename T4, typename T5,
      typename T6, typename T7>
  operator ParamGenerator< ::std::tr1::tuple<T1, T2, T3, T4, T5, T6,
      T7> >() const {
    return ParamGenerator< ::std::tr1::tuple<T1, T2, T3, T4, T5, T6, T7> >(
        new CartesianProductGenerator7<T1, T2, T3, T4, T5, T6, T7>(
        static_cast<ParamGenerator<T1> >(g1_),
        static_cast<ParamGenerator<T2> >(g2_),
        static_cast<ParamGenerator<T3> >(g3_),
        static_cast<ParamGenerator<T4> >(g4_),
        static_cast<ParamGenerator<T5> >(g5_),
        static_cast<ParamGenerator<T6> >(g6_),
        static_cast<ParamGenerator<T7> >(g7_)));
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const CartesianProductHolder7& other);

  const Generator1 g1_;
  const Generator2 g2_;
  const Generator3 g3_;
  const Generator4 g4_;
  const Generator5 g5_;
  const Generator6 g6_;
  const Generator7 g7_;
};  // class CartesianProductHolder7

template <class Generator1, class Generator2, class Generator3,
    class Generator4, class Generator5, class Generator6, class Generator7,
    class Generator8>
class CartesianProductHolder8 {
 public:
CartesianProductHolder8(const Generator1& g1, const Generator2& g2,
    const Generator3& g3, const Generator4& g4, const Generator5& g5,
    const Generator6& g6, const Generator7& g7, const Generator8& g8)
      : g1_(g1), g2_(g2), g3_(g3), g4_(g4), g5_(g5), g6_(g6), g7_(g7),
          g8_(g8) {}
  template <typename T1, typename T2, typename T3, typename T4, typename T5,
      typename T6, typename T7, typename T8>
  operator ParamGenerator< ::std::tr1::tuple<T1, T2, T3, T4, T5, T6, T7,
      T8> >() const {
    return ParamGenerator< ::std::tr1::tuple<T1, T2, T3, T4, T5, T6, T7, T8> >(
        new CartesianProductGenerator8<T1, T2, T3, T4, T5, T6, T7, T8>(
        static_cast<ParamGenerator<T1> >(g1_),
        static_cast<ParamGenerator<T2> >(g2_),
        static_cast<ParamGenerator<T3> >(g3_),
        static_cast<ParamGenerator<T4> >(g4_),
        static_cast<ParamGenerator<T5> >(g5_),
        static_cast<ParamGenerator<T6> >(g6_),
        static_cast<ParamGenerator<T7> >(g7_),
        static_cast<ParamGenerator<T8> >(g8_)));
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const CartesianProductHolder8& other);

  const Generator1 g1_;
  const Generator2 g2_;
  const Generator3 g3_;
  const Generator4 g4_;
  const Generator5 g5_;
  const Generator6 g6_;
  const Generator7 g7_;
  const Generator8 g8_;
};  // class CartesianProductHolder8

template <class Generator1, class Generator2, class Generator3,
    class Generator4, class Generator5, class Generator6, class Generator7,
    class Generator8, class Generator9>
class CartesianProductHolder9 {
 public:
CartesianProductHolder9(const Generator1& g1, const Generator2& g2,
    const Generator3& g3, const Generator4& g4, const Generator5& g5,
    const Generator6& g6, const Generator7& g7, const Generator8& g8,
    const Generator9& g9)
      : g1_(g1), g2_(g2), g3_(g3), g4_(g4), g5_(g5), g6_(g6), g7_(g7), g8_(g8),
          g9_(g9) {}
  template <typename T1, typename T2, typename T3, typename T4, typename T5,
      typename T6, typename T7, typename T8, typename T9>
  operator ParamGenerator< ::std::tr1::tuple<T1, T2, T3, T4, T5, T6, T7, T8,
      T9> >() const {
    return ParamGenerator< ::std::tr1::tuple<T1, T2, T3, T4, T5, T6, T7, T8,
        T9> >(
        new CartesianProductGenerator9<T1, T2, T3, T4, T5, T6, T7, T8, T9>(
        static_cast<ParamGenerator<T1> >(g1_),
        static_cast<ParamGenerator<T2> >(g2_),
        static_cast<ParamGenerator<T3> >(g3_),
        static_cast<ParamGenerator<T4> >(g4_),
        static_cast<ParamGenerator<T5> >(g5_),
        static_cast<ParamGenerator<T6> >(g6_),
        static_cast<ParamGenerator<T7> >(g7_),
        static_cast<ParamGenerator<T8> >(g8_),
        static_cast<ParamGenerator<T9> >(g9_)));
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const CartesianProductHolder9& other);

  const Generator1 g1_;
  const Generator2 g2_;
  const Generator3 g3_;
  const Generator4 g4_;
  const Generator5 g5_;
  const Generator6 g6_;
  const Generator7 g7_;
  const Generator8 g8_;
  const Generator9 g9_;
};  // class CartesianProductHolder9

template <class Generator1, class Generator2, class Generator3,
    class Generator4, class Generator5, class Generator6, class Generator7,
    class Generator8, class Generator9, class Generator10>
class CartesianProductHolder10 {
 public:
CartesianProductHolder10(const Generator1& g1, const Generator2& g2,
    const Generator3& g3, const Generator4& g4, const Generator5& g5,
    const Generator6& g6, const Generator7& g7, const Generator8& g8,
    const Generator9& g9, const Generator10& g10)
      : g1_(g1), g2_(g2), g3_(g3), g4_(g4), g5_(g5), g6_(g6), g7_(g7), g8_(g8),
          g9_(g9), g10_(g10) {}
  template <typename T1, typename T2, typename T3, typename T4, typename T5,
      typename T6, typename T7, typename T8, typename T9, typename T10>
  operator ParamGenerator< ::std::tr1::tuple<T1, T2, T3, T4, T5, T6, T7, T8,
      T9, T10> >() const {
    return ParamGenerator< ::std::tr1::tuple<T1, T2, T3, T4, T5, T6, T7, T8,
        T9, T10> >(
        new CartesianProductGenerator10<T1, T2, T3, T4, T5, T6, T7, T8, T9,
            T10>(
        static_cast<ParamGenerator<T1> >(g1_),
        static_cast<ParamGenerator<T2> >(g2_),
        static_cast<ParamGenerator<T3> >(g3_),
        static_cast<ParamGenerator<T4> >(g4_),
        static_cast<ParamGenerator<T5> >(g5_),
        static_cast<ParamGenerator<T6> >(g6_),
        static_cast<ParamGenerator<T7> >(g7_),
        static_cast<ParamGenerator<T8> >(g8_),
        static_cast<ParamGenerator<T9> >(g9_),
        static_cast<ParamGenerator<T10> >(g10_)));
  }

 private:
  // No implementation - assignment is unsupported.
  void operator=(const CartesianProductHolder10& other);

  const Generator1 g1_;
  const Generator2 g2_;
  const Generator3 g3_;
  const Generator4 g4_;
  const Generator5 g5_;
  const Generator6 g6_;
  const Generator7 g7_;
  const Generator8 g8_;
  const Generator9 g9_;
  const Generator10 g10_;
};  // class CartesianProductHolder10

# endif  // GTEST_HAS_COMBINE

}  // namespace internal
}  // namespace testing

#endif  //  GTEST_HAS_PARAM_TEST

#endif  // GTEST_INCLUDE_GTEST_INTERNAL_GTEST_PARAM_UTIL_GENERATED_H_

#if GTEST_HAS_PARAM_TEST

namespace testing {

// Functions producing parameter generators.
//
// Google Test uses these generators to produce parameters for value-
// parameterized tests. When a parameterized test case is instantiated
// with a particular generator, Google Test creates and runs tests
// for each element in the sequence produced by the generator.
//
// In the following sample, tests from test case FooTest are instantiated
// each three times with parameter values 3, 5, and 8:
//
// class FooTest : public TestWithParam<int> { ... };
//
// TEST_P(FooTest, TestThis) {
// }
// TEST_P(FooTest, TestThat) {
// }
// INSTANTIATE_TEST_CASE_P(TestSequence, FooTest, Values(3, 5, 8));
//

// Range() returns generators providing sequences of values in a range.
//
// Synopsis:
// Range(start, end)
//   - returns a generator producing a sequence of values {start, start+1,
//     start+2, ..., }.
// Range(start, end, step)
//   - returns a generator producing a sequence of values {start, start+step,
//     start+step+step, ..., }.
// Notes:
//   * The generated sequences never include end. For example, Range(1, 5)
//     returns a generator producing a sequence {1, 2, 3, 4}. Range(1, 9, 2)
//     returns a generator producing {1, 3, 5, 7}.
//   * start and end must have the same type. That type may be any integral or
//     floating-point type or a user defined type satisfying these conditions:
//     * It must be assignable (have operator=() defined).
//     * It must have operator+() (operator+(int-compatible type) for
//       two-operand version).
//     * It must have operator<() defined.
//     Elements in the resulting sequences will also have that type.
//   * Condition start < end must be satisfied in order for resulting sequences
//     to contain any elements.
//
template <typename T, typename IncrementT>
internal::ParamGenerator<T> Range(T start, T end, IncrementT step) {
  return internal::ParamGenerator<T>(
      new internal::RangeGenerator<T, IncrementT>(start, end, step));
}

template <typename T>
internal::ParamGenerator<T> Range(T start, T end) {
  return Range(start, end, 1);
}

// ValuesIn() function allows generation of tests with parameters coming from
// a container.
//
// Synopsis:
// ValuesIn(const T (&array)[N])
//   - returns a generator producing sequences with elements from
//     a C-style array.
// ValuesIn(const Container& container)
//   - returns a generator producing sequences with elements from
//     an STL-style container.
// ValuesIn(Iterator begin, Iterator end)
//   - returns a generator producing sequences with elements from
//     a range [begin, end) defined by a pair of STL-style iterators. These
//     iterators can also be plain C pointers.
//
// Please note that ValuesIn copies the values from the containers
// passed in and keeps them to generate tests in RUN_ALL_TESTS().
//
// Examples:
//
// This instantiates tests from test case StringTest
// each with C-string values of "foo", "bar", and "baz":
//
// const char* strings[] = {"foo", "bar", "baz"};
// INSTANTIATE_TEST_CASE_P(StringSequence, SrtingTest, ValuesIn(strings));
//
// This instantiates tests from test case StlStringTest
// each with STL strings with values "a" and "b":
//
// ::std::vector< ::std::string> GetParameterStrings() {
//   ::std::vector< ::std::string> v;
//   v.push_back("a");
//   v.push_back("b");
//   return v;
// }
//
// INSTANTIATE_TEST_CASE_P(CharSequence,
//                         StlStringTest,
//                         ValuesIn(GetParameterStrings()));
//
//
// This will also instantiate tests from CharTest
// each with parameter values 'a' and 'b':
//
// ::std::list<char> GetParameterChars() {
//   ::std::list<char> list;
//   list.push_back('a');
//   list.push_back('b');
//   return list;
// }
// ::std::list<char> l = GetParameterChars();
// INSTANTIATE_TEST_CASE_P(CharSequence2,
//                         CharTest,
//                         ValuesIn(l.begin(), l.end()));
//
template <typename ForwardIterator>
internal::ParamGenerator<
  typename ::testing::internal::IteratorTraits<ForwardIterator>::value_type>
ValuesIn(ForwardIterator begin, ForwardIterator end) {
  typedef typename ::testing::internal::IteratorTraits<ForwardIterator>
      ::value_type ParamType;
  return internal::ParamGenerator<ParamType>(
      new internal::ValuesInIteratorRangeGenerator<ParamType>(begin, end));
}

template <typename T, size_t N>
internal::ParamGenerator<T> ValuesIn(const T (&array)[N]) {
  return ValuesIn(array, array + N);
}

template <class Container>
internal::ParamGenerator<typename Container::value_type> ValuesIn(
    const Container& container) {
  return ValuesIn(container.begin(), container.end());
}

// Values() allows generating tests from explicitly specified list of
// parameters.
//
// Synopsis:
// Values(T v1, T v2, ..., T vN)
//   - returns a generator producing sequences with elements v1, v2, ..., vN.
//
// For example, this instantiates tests from test case BarTest each
// with values "one", "two", and "three":
//
// INSTANTIATE_TEST_CASE_P(NumSequence, BarTest, Values("one", "two", "three"));
//
// This instantiates tests from test case BazTest each with values 1, 2, 3.5.
// The exact type of values will depend on the type of parameter in BazTest.
//
// INSTANTIATE_TEST_CASE_P(FloatingNumbers, BazTest, Values(1, 2, 3.5));
//
// Currently, Values() supports from 1 to 50 parameters.
//
template <typename T1>
internal::ValueArray1<T1> Values(T1 v1) {
  return internal::ValueArray1<T1>(v1);
}

template <typename T1, typename T2>
internal::ValueArray2<T1, T2> Values(T1 v1, T2 v2) {
  return internal::ValueArray2<T1, T2>(v1, v2);
}

template <typename T1, typename T2, typename T3>
internal::ValueArray3<T1, T2, T3> Values(T1 v1, T2 v2, T3 v3) {
  return internal::ValueArray3<T1, T2, T3>(v1, v2, v3);
}

template <typename T1, typename T2, typename T3, typename T4>
internal::ValueArray4<T1, T2, T3, T4> Values(T1 v1, T2 v2, T3 v3, T4 v4) {
  return internal::ValueArray4<T1, T2, T3, T4>(v1, v2, v3, v4);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
internal::ValueArray5<T1, T2, T3, T4, T5> Values(T1 v1, T2 v2, T3 v3, T4 v4,
    T5 v5) {
  return internal::ValueArray5<T1, T2, T3, T4, T5>(v1, v2, v3, v4, v5);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6>
internal::ValueArray6<T1, T2, T3, T4, T5, T6> Values(T1 v1, T2 v2, T3 v3,
    T4 v4, T5 v5, T6 v6) {
  return internal::ValueArray6<T1, T2, T3, T4, T5, T6>(v1, v2, v3, v4, v5, v6);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7>
internal::ValueArray7<T1, T2, T3, T4, T5, T6, T7> Values(T1 v1, T2 v2, T3 v3,
    T4 v4, T5 v5, T6 v6, T7 v7) {
  return internal::ValueArray7<T1, T2, T3, T4, T5, T6, T7>(v1, v2, v3, v4, v5,
      v6, v7);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8>
internal::ValueArray8<T1, T2, T3, T4, T5, T6, T7, T8> Values(T1 v1, T2 v2,
    T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8) {
  return internal::ValueArray8<T1, T2, T3, T4, T5, T6, T7, T8>(v1, v2, v3, v4,
      v5, v6, v7, v8);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9>
internal::ValueArray9<T1, T2, T3, T4, T5, T6, T7, T8, T9> Values(T1 v1, T2 v2,
    T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9) {
  return internal::ValueArray9<T1, T2, T3, T4, T5, T6, T7, T8, T9>(v1, v2, v3,
      v4, v5, v6, v7, v8, v9);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10>
internal::ValueArray10<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> Values(T1 v1,
    T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9, T10 v10) {
  return internal::ValueArray10<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>(v1,
      v2, v3, v4, v5, v6, v7, v8, v9, v10);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11>
internal::ValueArray11<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10,
    T11> Values(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
    T10 v10, T11 v11) {
  return internal::ValueArray11<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10,
      T11>(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12>
internal::ValueArray12<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
    T12> Values(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
    T10 v10, T11 v11, T12 v12) {
  return internal::ValueArray12<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12>(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13>
internal::ValueArray13<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
    T13> Values(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
    T10 v10, T11 v11, T12 v12, T13 v13) {
  return internal::ValueArray13<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13>(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14>
internal::ValueArray14<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14> Values(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
    T10 v10, T11 v11, T12 v12, T13 v13, T14 v14) {
  return internal::ValueArray14<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14>(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13,
      v14);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15>
internal::ValueArray15<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15> Values(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8,
    T9 v9, T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15) {
  return internal::ValueArray15<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15>(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12,
      v13, v14, v15);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16>
internal::ValueArray16<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16> Values(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7,
    T8 v8, T9 v9, T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15,
    T16 v16) {
  return internal::ValueArray16<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16>(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11,
      v12, v13, v14, v15, v16);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17>
internal::ValueArray17<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17> Values(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7,
    T8 v8, T9 v9, T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15,
    T16 v16, T17 v17) {
  return internal::ValueArray17<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17>(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
      v11, v12, v13, v14, v15, v16, v17);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18>
internal::ValueArray18<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18> Values(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6,
    T7 v7, T8 v8, T9 v9, T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15,
    T16 v16, T17 v17, T18 v18) {
  return internal::ValueArray18<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18>(v1, v2, v3, v4, v5, v6, v7, v8, v9,
      v10, v11, v12, v13, v14, v15, v16, v17, v18);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19>
internal::ValueArray19<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19> Values(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5,
    T6 v6, T7 v7, T8 v8, T9 v9, T10 v10, T11 v11, T12 v12, T13 v13, T14 v14,
    T15 v15, T16 v16, T17 v17, T18 v18, T19 v19) {
  return internal::ValueArray19<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19>(v1, v2, v3, v4, v5, v6, v7, v8,
      v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20>
internal::ValueArray20<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20> Values(T1 v1, T2 v2, T3 v3, T4 v4,
    T5 v5, T6 v6, T7 v7, T8 v8, T9 v9, T10 v10, T11 v11, T12 v12, T13 v13,
    T14 v14, T15 v15, T16 v16, T17 v17, T18 v18, T19 v19, T20 v20) {
  return internal::ValueArray20<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20>(v1, v2, v3, v4, v5, v6, v7,
      v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21>
internal::ValueArray21<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21> Values(T1 v1, T2 v2, T3 v3, T4 v4,
    T5 v5, T6 v6, T7 v7, T8 v8, T9 v9, T10 v10, T11 v11, T12 v12, T13 v13,
    T14 v14, T15 v15, T16 v16, T17 v17, T18 v18, T19 v19, T20 v20, T21 v21) {
  return internal::ValueArray21<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21>(v1, v2, v3, v4, v5, v6,
      v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22>
internal::ValueArray22<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22> Values(T1 v1, T2 v2, T3 v3,
    T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9, T10 v10, T11 v11, T12 v12,
    T13 v13, T14 v14, T15 v15, T16 v16, T17 v17, T18 v18, T19 v19, T20 v20,
    T21 v21, T22 v22) {
  return internal::ValueArray22<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22>(v1, v2, v3, v4,
      v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
      v20, v21, v22);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23>
internal::ValueArray23<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23> Values(T1 v1, T2 v2,
    T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9, T10 v10, T11 v11, T12 v12,
    T13 v13, T14 v14, T15 v15, T16 v16, T17 v17, T18 v18, T19 v19, T20 v20,
    T21 v21, T22 v22, T23 v23) {
  return internal::ValueArray23<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23>(v1, v2, v3,
      v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
      v20, v21, v22, v23);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24>
internal::ValueArray24<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24> Values(T1 v1, T2 v2,
    T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9, T10 v10, T11 v11, T12 v12,
    T13 v13, T14 v14, T15 v15, T16 v16, T17 v17, T18 v18, T19 v19, T20 v20,
    T21 v21, T22 v22, T23 v23, T24 v24) {
  return internal::ValueArray24<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24>(v1, v2,
      v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
      v19, v20, v21, v22, v23, v24);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25>
internal::ValueArray25<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25> Values(T1 v1,
    T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9, T10 v10, T11 v11,
    T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17, T18 v18, T19 v19,
    T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25) {
  return internal::ValueArray25<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25>(v1,
      v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17,
      v18, v19, v20, v21, v22, v23, v24, v25);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26>
internal::ValueArray26<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25,
    T26> Values(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
    T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
    T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
    T26 v26) {
  return internal::ValueArray26<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25,
      T26>(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
      v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27>
internal::ValueArray27<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
    T27> Values(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
    T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
    T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
    T26 v26, T27 v27) {
  return internal::ValueArray27<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25,
      T26, T27>(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14,
      v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28>
internal::ValueArray28<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27,
    T28> Values(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
    T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
    T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
    T26 v26, T27 v27, T28 v28) {
  return internal::ValueArray28<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25,
      T26, T27, T28>(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13,
      v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27,
      v28);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29>
internal::ValueArray29<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
    T29> Values(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
    T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
    T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
    T26 v26, T27 v27, T28 v28, T29 v29) {
  return internal::ValueArray29<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25,
      T26, T27, T28, T29>(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12,
      v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26,
      v27, v28, v29);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30>
internal::ValueArray30<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
    T29, T30> Values(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8,
    T9 v9, T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16,
    T17 v17, T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24,
    T25 v25, T26 v26, T27 v27, T28 v28, T29 v29, T30 v30) {
  return internal::ValueArray30<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25,
      T26, T27, T28, T29, T30>(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11,
      v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25,
      v26, v27, v28, v29, v30);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31>
internal::ValueArray31<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
    T29, T30, T31> Values(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7,
    T8 v8, T9 v9, T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15,
    T16 v16, T17 v17, T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23,
    T24 v24, T25 v25, T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31) {
  return internal::ValueArray31<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25,
      T26, T27, T28, T29, T30, T31>(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
      v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24,
      v25, v26, v27, v28, v29, v30, v31);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32>
internal::ValueArray32<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
    T29, T30, T31, T32> Values(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7,
    T8 v8, T9 v9, T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15,
    T16 v16, T17 v17, T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23,
    T24 v24, T25 v25, T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31,
    T32 v32) {
  return internal::ValueArray32<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25,
      T26, T27, T28, T29, T30, T31, T32>(v1, v2, v3, v4, v5, v6, v7, v8, v9,
      v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23,
      v24, v25, v26, v27, v28, v29, v30, v31, v32);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33>
internal::ValueArray33<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
    T29, T30, T31, T32, T33> Values(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6,
    T7 v7, T8 v8, T9 v9, T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15,
    T16 v16, T17 v17, T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23,
    T24 v24, T25 v25, T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31,
    T32 v32, T33 v33) {
  return internal::ValueArray33<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25,
      T26, T27, T28, T29, T30, T31, T32, T33>(v1, v2, v3, v4, v5, v6, v7, v8,
      v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23,
      v24, v25, v26, v27, v28, v29, v30, v31, v32, v33);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34>
internal::ValueArray34<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
    T29, T30, T31, T32, T33, T34> Values(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5,
    T6 v6, T7 v7, T8 v8, T9 v9, T10 v10, T11 v11, T12 v12, T13 v13, T14 v14,
    T15 v15, T16 v16, T17 v17, T18 v18, T19 v19, T20 v20, T21 v21, T22 v22,
    T23 v23, T24 v24, T25 v25, T26 v26, T27 v27, T28 v28, T29 v29, T30 v30,
    T31 v31, T32 v32, T33 v33, T34 v34) {
  return internal::ValueArray34<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25,
      T26, T27, T28, T29, T30, T31, T32, T33, T34>(v1, v2, v3, v4, v5, v6, v7,
      v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22,
      v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35>
internal::ValueArray35<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
    T29, T30, T31, T32, T33, T34, T35> Values(T1 v1, T2 v2, T3 v3, T4 v4,
    T5 v5, T6 v6, T7 v7, T8 v8, T9 v9, T10 v10, T11 v11, T12 v12, T13 v13,
    T14 v14, T15 v15, T16 v16, T17 v17, T18 v18, T19 v19, T20 v20, T21 v21,
    T22 v22, T23 v23, T24 v24, T25 v25, T26 v26, T27 v27, T28 v28, T29 v29,
    T30 v30, T31 v31, T32 v32, T33 v33, T34 v34, T35 v35) {
  return internal::ValueArray35<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25,
      T26, T27, T28, T29, T30, T31, T32, T33, T34, T35>(v1, v2, v3, v4, v5, v6,
      v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21,
      v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36>
internal::ValueArray36<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
    T29, T30, T31, T32, T33, T34, T35, T36> Values(T1 v1, T2 v2, T3 v3, T4 v4,
    T5 v5, T6 v6, T7 v7, T8 v8, T9 v9, T10 v10, T11 v11, T12 v12, T13 v13,
    T14 v14, T15 v15, T16 v16, T17 v17, T18 v18, T19 v19, T20 v20, T21 v21,
    T22 v22, T23 v23, T24 v24, T25 v25, T26 v26, T27 v27, T28 v28, T29 v29,
    T30 v30, T31 v31, T32 v32, T33 v33, T34 v34, T35 v35, T36 v36) {
  return internal::ValueArray36<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25,
      T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36>(v1, v2, v3, v4,
      v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
      v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33,
      v34, v35, v36);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37>
internal::ValueArray37<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
    T29, T30, T31, T32, T33, T34, T35, T36, T37> Values(T1 v1, T2 v2, T3 v3,
    T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9, T10 v10, T11 v11, T12 v12,
    T13 v13, T14 v14, T15 v15, T16 v16, T17 v17, T18 v18, T19 v19, T20 v20,
    T21 v21, T22 v22, T23 v23, T24 v24, T25 v25, T26 v26, T27 v27, T28 v28,
    T29 v29, T30 v30, T31 v31, T32 v32, T33 v33, T34 v34, T35 v35, T36 v36,
    T37 v37) {
  return internal::ValueArray37<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25,
      T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37>(v1, v2, v3,
      v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
      v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33,
      v34, v35, v36, v37);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38>
internal::ValueArray38<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
    T29, T30, T31, T32, T33, T34, T35, T36, T37, T38> Values(T1 v1, T2 v2,
    T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9, T10 v10, T11 v11, T12 v12,
    T13 v13, T14 v14, T15 v15, T16 v16, T17 v17, T18 v18, T19 v19, T20 v20,
    T21 v21, T22 v22, T23 v23, T24 v24, T25 v25, T26 v26, T27 v27, T28 v28,
    T29 v29, T30 v30, T31 v31, T32 v32, T33 v33, T34 v34, T35 v35, T36 v36,
    T37 v37, T38 v38) {
  return internal::ValueArray38<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25,
      T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38>(v1, v2,
      v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
      v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32,
      v33, v34, v35, v36, v37, v38);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39>
internal::ValueArray39<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
    T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39> Values(T1 v1, T2 v2,
    T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9, T10 v10, T11 v11, T12 v12,
    T13 v13, T14 v14, T15 v15, T16 v16, T17 v17, T18 v18, T19 v19, T20 v20,
    T21 v21, T22 v22, T23 v23, T24 v24, T25 v25, T26 v26, T27 v27, T28 v28,
    T29 v29, T30 v30, T31 v31, T32 v32, T33 v33, T34 v34, T35 v35, T36 v36,
    T37 v37, T38 v38, T39 v39) {
  return internal::ValueArray39<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25,
      T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39>(v1,
      v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17,
      v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31,
      v32, v33, v34, v35, v36, v37, v38, v39);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40>
internal::ValueArray40<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
    T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40> Values(T1 v1,
    T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9, T10 v10, T11 v11,
    T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17, T18 v18, T19 v19,
    T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25, T26 v26, T27 v27,
    T28 v28, T29 v29, T30 v30, T31 v31, T32 v32, T33 v33, T34 v34, T35 v35,
    T36 v36, T37 v37, T38 v38, T39 v39, T40 v40) {
  return internal::ValueArray40<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25,
      T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39,
      T40>(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15,
      v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29,
      v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41>
internal::ValueArray41<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
    T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40,
    T41> Values(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
    T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
    T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
    T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31, T32 v32, T33 v33,
    T34 v34, T35 v35, T36 v36, T37 v37, T38 v38, T39 v39, T40 v40, T41 v41) {
  return internal::ValueArray41<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25,
      T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39,
      T40, T41>(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14,
      v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28,
      v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42>
internal::ValueArray42<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
    T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41,
    T42> Values(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
    T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
    T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
    T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31, T32 v32, T33 v33,
    T34 v34, T35 v35, T36 v36, T37 v37, T38 v38, T39 v39, T40 v40, T41 v41,
    T42 v42) {
  return internal::ValueArray42<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25,
      T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39,
      T40, T41, T42>(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13,
      v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27,
      v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41,
      v42);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43>
internal::ValueArray43<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
    T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42,
    T43> Values(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
    T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
    T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
    T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31, T32 v32, T33 v33,
    T34 v34, T35 v35, T36 v36, T37 v37, T38 v38, T39 v39, T40 v40, T41 v41,
    T42 v42, T43 v43) {
  return internal::ValueArray43<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25,
      T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39,
      T40, T41, T42, T43>(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12,
      v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26,
      v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40,
      v41, v42, v43);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43, typename T44>
internal::ValueArray44<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
    T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43,
    T44> Values(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9,
    T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16, T17 v17,
    T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24, T25 v25,
    T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31, T32 v32, T33 v33,
    T34 v34, T35 v35, T36 v36, T37 v37, T38 v38, T39 v39, T40 v40, T41 v41,
    T42 v42, T43 v43, T44 v44) {
  return internal::ValueArray44<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25,
      T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39,
      T40, T41, T42, T43, T44>(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11,
      v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25,
      v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39,
      v40, v41, v42, v43, v44);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43, typename T44, typename T45>
internal::ValueArray45<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
    T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43,
    T44, T45> Values(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8,
    T9 v9, T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15, T16 v16,
    T17 v17, T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23, T24 v24,
    T25 v25, T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31, T32 v32,
    T33 v33, T34 v34, T35 v35, T36 v36, T37 v37, T38 v38, T39 v39, T40 v40,
    T41 v41, T42 v42, T43 v43, T44 v44, T45 v45) {
  return internal::ValueArray45<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25,
      T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39,
      T40, T41, T42, T43, T44, T45>(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
      v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24,
      v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38,
      v39, v40, v41, v42, v43, v44, v45);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43, typename T44, typename T45,
    typename T46>
internal::ValueArray46<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
    T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43,
    T44, T45, T46> Values(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7,
    T8 v8, T9 v9, T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15,
    T16 v16, T17 v17, T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23,
    T24 v24, T25 v25, T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31,
    T32 v32, T33 v33, T34 v34, T35 v35, T36 v36, T37 v37, T38 v38, T39 v39,
    T40 v40, T41 v41, T42 v42, T43 v43, T44 v44, T45 v45, T46 v46) {
  return internal::ValueArray46<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25,
      T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39,
      T40, T41, T42, T43, T44, T45, T46>(v1, v2, v3, v4, v5, v6, v7, v8, v9,
      v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23,
      v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37,
      v38, v39, v40, v41, v42, v43, v44, v45, v46);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43, typename T44, typename T45,
    typename T46, typename T47>
internal::ValueArray47<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
    T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43,
    T44, T45, T46, T47> Values(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7,
    T8 v8, T9 v9, T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15,
    T16 v16, T17 v17, T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23,
    T24 v24, T25 v25, T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31,
    T32 v32, T33 v33, T34 v34, T35 v35, T36 v36, T37 v37, T38 v38, T39 v39,
    T40 v40, T41 v41, T42 v42, T43 v43, T44 v44, T45 v45, T46 v46, T47 v47) {
  return internal::ValueArray47<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25,
      T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39,
      T40, T41, T42, T43, T44, T45, T46, T47>(v1, v2, v3, v4, v5, v6, v7, v8,
      v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23,
      v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37,
      v38, v39, v40, v41, v42, v43, v44, v45, v46, v47);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43, typename T44, typename T45,
    typename T46, typename T47, typename T48>
internal::ValueArray48<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
    T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43,
    T44, T45, T46, T47, T48> Values(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6,
    T7 v7, T8 v8, T9 v9, T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15,
    T16 v16, T17 v17, T18 v18, T19 v19, T20 v20, T21 v21, T22 v22, T23 v23,
    T24 v24, T25 v25, T26 v26, T27 v27, T28 v28, T29 v29, T30 v30, T31 v31,
    T32 v32, T33 v33, T34 v34, T35 v35, T36 v36, T37 v37, T38 v38, T39 v39,
    T40 v40, T41 v41, T42 v42, T43 v43, T44 v44, T45 v45, T46 v46, T47 v47,
    T48 v48) {
  return internal::ValueArray48<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25,
      T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39,
      T40, T41, T42, T43, T44, T45, T46, T47, T48>(v1, v2, v3, v4, v5, v6, v7,
      v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22,
      v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36,
      v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43, typename T44, typename T45,
    typename T46, typename T47, typename T48, typename T49>
internal::ValueArray49<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
    T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43,
    T44, T45, T46, T47, T48, T49> Values(T1 v1, T2 v2, T3 v3, T4 v4, T5 v5,
    T6 v6, T7 v7, T8 v8, T9 v9, T10 v10, T11 v11, T12 v12, T13 v13, T14 v14,
    T15 v15, T16 v16, T17 v17, T18 v18, T19 v19, T20 v20, T21 v21, T22 v22,
    T23 v23, T24 v24, T25 v25, T26 v26, T27 v27, T28 v28, T29 v29, T30 v30,
    T31 v31, T32 v32, T33 v33, T34 v34, T35 v35, T36 v36, T37 v37, T38 v38,
    T39 v39, T40 v40, T41 v41, T42 v42, T43 v43, T44 v44, T45 v45, T46 v46,
    T47 v47, T48 v48, T49 v49) {
  return internal::ValueArray49<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25,
      T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39,
      T40, T41, T42, T43, T44, T45, T46, T47, T48, T49>(v1, v2, v3, v4, v5, v6,
      v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21,
      v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35,
      v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49);
}

template <typename T1, typename T2, typename T3, typename T4, typename T5,
    typename T6, typename T7, typename T8, typename T9, typename T10,
    typename T11, typename T12, typename T13, typename T14, typename T15,
    typename T16, typename T17, typename T18, typename T19, typename T20,
    typename T21, typename T22, typename T23, typename T24, typename T25,
    typename T26, typename T27, typename T28, typename T29, typename T30,
    typename T31, typename T32, typename T33, typename T34, typename T35,
    typename T36, typename T37, typename T38, typename T39, typename T40,
    typename T41, typename T42, typename T43, typename T44, typename T45,
    typename T46, typename T47, typename T48, typename T49, typename T50>
internal::ValueArray50<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
    T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28,
    T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43,
    T44, T45, T46, T47, T48, T49, T50> Values(T1 v1, T2 v2, T3 v3, T4 v4,
    T5 v5, T6 v6, T7 v7, T8 v8, T9 v9, T10 v10, T11 v11, T12 v12, T13 v13,
    T14 v14, T15 v15, T16 v16, T17 v17, T18 v18, T19 v19, T20 v20, T21 v21,
    T22 v22, T23 v23, T24 v24, T25 v25, T26 v26, T27 v27, T28 v28, T29 v29,
    T30 v30, T31 v31, T32 v32, T33 v33, T34 v34, T35 v35, T36 v36, T37 v37,
    T38 v38, T39 v39, T40 v40, T41 v41, T42 v42, T43 v43, T44 v44, T45 v45,
    T46 v46, T47 v47, T48 v48, T49 v49, T50 v50) {
  return internal::ValueArray50<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
      T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25,
      T26, T27, T28, T29, T30, T31, T32, T33, T34, T35, T36, T37, T38, T39,
      T40, T41, T42, T43, T44, T45, T46, T47, T48, T49, T50>(v1, v2, v3, v4,
      v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
      v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33,
      v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47,
      v48, v49, v50);
}

// Bool() allows generating tests with parameters in a set of (false, true).
//
// Synopsis:
// Bool()
//   - returns a generator producing sequences with elements {false, true}.
//
// It is useful when testing code that depends on Boolean flags. Combinations
// of multiple flags can be tested when several Bool()'s are combined using
// Combine() function.
//
// In the following example all tests in the test case FlagDependentTest
// will be instantiated twice with parameters false and true.
//
// class FlagDependentTest : public testing::TestWithParam<bool> {
//   virtual void SetUp() {
//     external_flag = GetParam();
//   }
// }
// INSTANTIATE_TEST_CASE_P(BoolSequence, FlagDependentTest, Bool());
//
inline internal::ParamGenerator<bool> Bool() {
  return Values(false, true);
}

# if GTEST_HAS_COMBINE
// Combine() allows the user to combine two or more sequences to produce
// values of a Cartesian product of those sequences' elements.
//
// Synopsis:
// Combine(gen1, gen2, ..., genN)
//   - returns a generator producing sequences with elements coming from
//     the Cartesian product of elements from the sequences generated by
//     gen1, gen2, ..., genN. The sequence elements will have a type of
//     tuple<T1, T2, ..., TN> where T1, T2, ..., TN are the types
//     of elements from sequences produces by gen1, gen2, ..., genN.
//
// Combine can have up to 10 arguments. This number is currently limited
// by the maximum number of elements in the tuple implementation used by Google
// Test.
//
// Example:
//
// This will instantiate tests in test case AnimalTest each one with
// the parameter values tuple("cat", BLACK), tuple("cat", WHITE),
// tuple("dog", BLACK), and tuple("dog", WHITE):
//
// enum Color { BLACK, GRAY, WHITE };
// class AnimalTest
//     : public testing::TestWithParam<tuple<const char*, Color> > {...};
//
// TEST_P(AnimalTest, AnimalLooksNice) {...}
//
// INSTANTIATE_TEST_CASE_P(AnimalVariations, AnimalTest,
//                         Combine(Values("cat", "dog"),
//                                 Values(BLACK, WHITE)));
//
// This will instantiate tests in FlagDependentTest with all variations of two
// Boolean flags:
//
// class FlagDependentTest
//     : public testing::TestWithParam<tuple(bool, bool)> > {
//   virtual void SetUp() {
//     // Assigns external_flag_1 and external_flag_2 values from the tuple.
//     tie(external_flag_1, external_flag_2) = GetParam();
//   }
// };
//
// TEST_P(FlagDependentTest, TestFeature1) {
//   // Test your code using external_flag_1 and external_flag_2 here.
// }
// INSTANTIATE_TEST_CASE_P(TwoBoolSequence, FlagDependentTest,
//                         Combine(Bool(), Bool()));
//
template <typename Generator1, typename Generator2>
internal::CartesianProductHolder2<Generator1, Generator2> Combine(
    const Generator1& g1, const Generator2& g2) {
  return internal::CartesianProductHolder2<Generator1, Generator2>(
      g1, g2);
}

template <typename Generator1, typename Generator2, typename Generator3>
internal::CartesianProductHolder3<Generator1, Generator2, Generator3> Combine(
    const Generator1& g1, const Generator2& g2, const Generator3& g3) {
  return internal::CartesianProductHolder3<Generator1, Generator2, Generator3>(
      g1, g2, g3);
}

template <typename Generator1, typename Generator2, typename Generator3,
    typename Generator4>
internal::CartesianProductHolder4<Generator1, Generator2, Generator3,
    Generator4> Combine(
    const Generator1& g1, const Generator2& g2, const Generator3& g3,
        const Generator4& g4) {
  return internal::CartesianProductHolder4<Generator1, Generator2, Generator3,
      Generator4>(
      g1, g2, g3, g4);
}

template <typename Generator1, typename Generator2, typename Generator3,
    typename Generator4, typename Generator5>
internal::CartesianProductHolder5<Generator1, Generator2, Generator3,
    Generator4, Generator5> Combine(
    const Generator1& g1, const Generator2& g2, const Generator3& g3,
        const Generator4& g4, const Generator5& g5) {
  return internal::CartesianProductHolder5<Generator1, Generator2, Generator3,
      Generator4, Generator5>(
      g1, g2, g3, g4, g5);
}

template <typename Generator1, typename Generator2, typename Generator3,
    typename Generator4, typename Generator5, typename Generator6>
internal::CartesianProductHolder6<Generator1, Generator2, Generator3,
    Generator4, Generator5, Generator6> Combine(
    const Generator1& g1, const Generator2& g2, const Generator3& g3,
        const Generator4& g4, const Generator5& g5, const Generator6& g6) {
  return internal::CartesianProductHolder6<Generator1, Generator2, Generator3,
      Generator4, Generator5, Generator6>(
      g1, g2, g3, g4, g5, g6);
}

template <typename Generator1, typename Generator2, typename Generator3,
    typename Generator4, typename Generator5, typename Generator6,
    typename Generator7>
internal::CartesianProductHolder7<Generator1, Generator2, Generator3,
    Generator4, Generator5, Generator6, Generator7> Combine(
    const Generator1& g1, const Generator2& g2, const Generator3& g3,
        const Generator4& g4, const Generator5& g5, const Generator6& g6,
        const Generator7& g7) {
  return internal::CartesianProductHolder7<Generator1, Generator2, Generator3,
      Generator4, Generator5, Generator6, Generator7>(
      g1, g2, g3, g4, g5, g6, g7);
}

template <typename Generator1, typename Generator2, typename Generator3,
    typename Generator4, typename Generator5, typename Generator6,
    typename Generator7, typename Generator8>
internal::CartesianProductHolder8<Generator1, Generator2, Generator3,
    Generator4, Generator5, Generator6, Generator7, Generator8> Combine(
    const Generator1& g1, const Generator2& g2, const Generator3& g3,
        const Generator4& g4, const Generator5& g5, const Generator6& g6,
        const Generator7& g7, const Generator8& g8) {
  return internal::CartesianProductHolder8<Generator1, Generator2, Generator3,
      Generator4, Generator5, Generator6, Generator7, Generator8>(
      g1, g2, g3, g4, g5, g6, g7, g8);
}

template <typename Generator1, typename Generator2, typename Generator3,
    typename Generator4, typename Generator5, typename Generator6,
    typename Generator7, typename Generator8, typename Generator9>
internal::CartesianProductHolder9<Generator1, Generator2, Generator3,
    Generator4, Generator5, Generator6, Generator7, Generator8,
    Generator9> Combine(
    const Generator1& g1, const Generator2& g2, const Generator3& g3,
        const Generator4& g4, const Generator5& g5, const Generator6& g6,
        const Generator7& g7, const Generator8& g8, const Generator9& g9) {
  return internal::CartesianProductHolder9<Generator1, Generator2, Generator3,
      Generator4, Generator5, Generator6, Generator7, Generator8, Generator9>(
      g1, g2, g3, g4, g5, g6, g7, g8, g9);
}

template <typename Generator1, typename Generator2, typename Generator3,
    typename Generator4, typename Generator5, typename Generator6,
    typename Generator7, typename Generator8, typename Generator9,
    typename Generator10>
internal::CartesianProductHolder10<Generator1, Generator2, Generator3,
    Generator4, Generator5, Generator6, Generator7, Generator8, Generator9,
    Generator10> Combine(
    const Generator1& g1, const Generator2& g2, const Generator3& g3,
        const Generator4& g4, const Generator5& g5, const Generator6& g6,
        const Generator7& g7, const Generator8& g8, const Generator9& g9,
        const Generator10& g10) {
  return internal::CartesianProductHolder10<Generator1, Generator2, Generator3,
      Generator4, Generator5, Generator6, Generator7, Generator8, Generator9,
      Generator10>(
      g1, g2, g3, g4, g5, g6, g7, g8, g9, g10);
}
# endif  // GTEST_HAS_COMBINE



# define TEST_P(test_case_name, test_name) \
  class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) \
      : public test_case_name { \
   public: \
    GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() {} \
    virtual void TestBody(); \
   private: \
    static int AddToRegistry() { \
      ::testing::UnitTest::GetInstance()->parameterized_test_registry(). \
          GetTestCasePatternHolder<test_case_name>(\
              #test_case_name, __FILE__, __LINE__)->AddTestPattern(\
                  #test_case_name, \
                  #test_name, \
                  new ::testing::internal::TestMetaFactory< \
                      GTEST_TEST_CLASS_NAME_(test_case_name, test_name)>()); \
      return 0; \
    } \
    static int gtest_registering_dummy_; \
    GTEST_DISALLOW_COPY_AND_ASSIGN_(\
        GTEST_TEST_CLASS_NAME_(test_case_name, test_name)); \
  }; \
  int GTEST_TEST_CLASS_NAME_(test_case_name, \
                             test_name)::gtest_registering_dummy_ = \
      GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::AddToRegistry(); \
  void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::TestBody()

# define INSTANTIATE_TEST_CASE_P(prefix, test_case_name, generator) \
  ::testing::internal::ParamGenerator<test_case_name::ParamType> \
      gtest_##prefix##test_case_name##_EvalGenerator_() { return generator; } \
  int gtest_##prefix##test_case_name##_dummy_ = \
      ::testing::UnitTest::GetInstance()->parameterized_test_registry(). \
          GetTestCasePatternHolder<test_case_name>(\
              #test_case_name, __FILE__, __LINE__)->AddTestCaseInstantiation(\
                  #prefix, \
                  &gtest_##prefix##test_case_name##_EvalGenerator_, \
                  __FILE__, __LINE__)

}  // namespace testing

#endif  // GTEST_HAS_PARAM_TEST

#endif  // GTEST_INCLUDE_GTEST_GTEST_PARAM_TEST_H_
// Copyright 2006, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Author: wan@google.com (Zhanyong Wan)
//
// Google C++ Testing Framework definitions useful in production code.

#ifndef GTEST_INCLUDE_GTEST_GTEST_PROD_H_
#define GTEST_INCLUDE_GTEST_GTEST_PROD_H_

// When you need to test the private or protected members of a class,
// use the FRIEND_TEST macro to declare your tests as friends of the
// class.  For example:
//
// class MyClass {
//  private:
//   void MyMethod();
//   FRIEND_TEST(MyClassTest, MyMethod);
// };
//
// class MyClassTest : public testing::Test {
//   // ...
// };
//
// TEST_F(MyClassTest, MyMethod) {
//   // Can call MyClass::MyMethod() here.
// }

#define FRIEND_TEST(test_case_name, test_name)\
friend class test_case_name##_##test_name##_Test

#endif  // GTEST_INCLUDE_GTEST_GTEST_PROD_H_
// Copyright 2008, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Author: mheule@google.com (Markus Heule)
//

#ifndef GTEST_INCLUDE_GTEST_GTEST_TEST_PART_H_
#define GTEST_INCLUDE_GTEST_GTEST_TEST_PART_H_

#include <iosfwd>
#include <vector>

namespace testing {

// A copyable object representing the result of a test part (i.e. an
// assertion or an explicit FAIL(), ADD_FAILURE(), or SUCCESS()).
//
// Don't inherit from TestPartResult as its destructor is not virtual.
class GTEST_API_ TestPartResult {
 public:
  // The possible outcomes of a test part (i.e. an assertion or an
  // explicit SUCCEED(), FAIL(), or ADD_FAILURE()).
  enum Type {
    kSuccess,          // Succeeded.
    kNonFatalFailure,  // Failed but the test can continue.
    kFatalFailure      // Failed and the test should be terminated.
  };

  // C'tor.  TestPartResult does NOT have a default constructor.
  // Always use this constructor (with parameters) to create a
  // TestPartResult object.
  TestPartResult(Type a_type,
                 const char* a_file_name,
                 int a_line_number,
                 const char* a_message)
      : type_(a_type),
        file_name_(a_file_name),
        line_number_(a_line_number),
        summary_(ExtractSummary(a_message)),
        message_(a_message) {
  }

  // Gets the outcome of the test part.
  Type type() const { return type_; }

  // Gets the name of the source file where the test part took place, or
  // NULL if it's unknown.
  const char* file_name() const { return file_name_.c_str(); }

  // Gets the line in the source file where the test part took place,
  // or -1 if it's unknown.
  int line_number() const { return line_number_; }

  // Gets the summary of the failure message.
  const char* summary() const { return summary_.c_str(); }

  // Gets the message associated with the test part.
  const char* message() const { return message_.c_str(); }

  // Returns true iff the test part passed.
  bool passed() const { return type_ == kSuccess; }

  // Returns true iff the test part failed.
  bool failed() const { return type_ != kSuccess; }

  // Returns true iff the test part non-fatally failed.
  bool nonfatally_failed() const { return type_ == kNonFatalFailure; }

  // Returns true iff the test part fatally failed.
  bool fatally_failed() const { return type_ == kFatalFailure; }
 private:
  Type type_;

  // Gets the summary of the failure message by omitting the stack
  // trace in it.
  static internal::String ExtractSummary(const char* message);

  // The name of the source file where the test part took place, or
  // NULL if the source file is unknown.
  internal::String file_name_;
  // The line in the source file where the test part took place, or -1
  // if the line number is unknown.
  int line_number_;
  internal::String summary_;  // The test failure summary.
  internal::String message_;  // The test failure message.
};

// Prints a TestPartResult object.
std::ostream& operator<<(std::ostream& os, const TestPartResult& result);

// An array of TestPartResult objects.
//
// Don't inherit from TestPartResultArray as its destructor is not
// virtual.
class GTEST_API_ TestPartResultArray {
 public:
  TestPartResultArray() {}

  // Appends the given TestPartResult to the array.
  void Append(const TestPartResult& result);

  // Returns the TestPartResult at the given index (0-based).
  const TestPartResult& GetTestPartResult(int index) const;

  // Returns the number of TestPartResult objects in the array.
  int size() const;

 private:
  std::vector<TestPartResult> array_;

  GTEST_DISALLOW_COPY_AND_ASSIGN_(TestPartResultArray);
};

// This interface knows how to report a test part result.
class TestPartResultReporterInterface {
 public:
  virtual ~TestPartResultReporterInterface() {}

  virtual void ReportTestPartResult(const TestPartResult& result) = 0;
};

namespace internal {

// This helper class is used by {ASSERT|EXPECT}_NO_FATAL_FAILURE to check if a
// statement generates new fatal failures. To do so it registers itself as the
// current test part result reporter. Besides checking if fatal failures were
// reported, it only delegates the reporting to the former result reporter.
// The original result reporter is restored in the destructor.
// INTERNAL IMPLEMENTATION - DO NOT USE IN A USER PROGRAM.
class GTEST_API_ HasNewFatalFailureHelper
    : public TestPartResultReporterInterface {
 public:
  HasNewFatalFailureHelper();
  virtual ~HasNewFatalFailureHelper();
  virtual void ReportTestPartResult(const TestPartResult& result);
  bool has_new_fatal_failure() const { return has_new_fatal_failure_; }
 private:
  bool has_new_fatal_failure_;
  TestPartResultReporterInterface* original_reporter_;

  GTEST_DISALLOW_COPY_AND_ASSIGN_(HasNewFatalFailureHelper);
};

}  // namespace internal

}  // namespace testing

#endif  // GTEST_INCLUDE_GTEST_GTEST_TEST_PART_H_
// Copyright 2008 Google Inc.
// All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Author: wan@google.com (Zhanyong Wan)

#ifndef GTEST_INCLUDE_GTEST_GTEST_TYPED_TEST_H_
#define GTEST_INCLUDE_GTEST_GTEST_TYPED_TEST_H_

// This header implements typed tests and type-parameterized tests.

// Typed (aka type-driven) tests repeat the same test for types in a
// list.  You must know which types you want to test with when writing
// typed tests. Here's how you do it:

#if 0

// First, define a fixture class template.  It should be parameterized
// by a type.  Remember to derive it from testing::Test.
template <typename T>
class FooTest : public testing::Test {
 public:
  ...
  typedef std::list<T> List;
  static T shared_;
  T value_;
};

// Next, associate a list of types with the test case, which will be
// repeated for each type in the list.  The typedef is necessary for
// the macro to parse correctly.
typedef testing::Types<char, int, unsigned int> MyTypes;
TYPED_TEST_CASE(FooTest, MyTypes);

// If the type list contains only one type, you can write that type
// directly without Types<...>:
//   TYPED_TEST_CASE(FooTest, int);

// Then, use TYPED_TEST() instead of TEST_F() to define as many typed
// tests for this test case as you want.
TYPED_TEST(FooTest, DoesBlah) {
  // Inside a test, refer to TypeParam to get the type parameter.
  // Since we are inside a derived class template, C++ requires use to
  // visit the members of FooTest via 'this'.
  TypeParam n = this->value_;

  // To visit static members of the fixture, add the TestFixture::
  // prefix.
  n += TestFixture::shared_;

  // To refer to typedefs in the fixture, add the "typename
  // TestFixture::" prefix.
  typename TestFixture::List values;
  values.push_back(n);
  ...
}

TYPED_TEST(FooTest, HasPropertyA) { ... }

#endif  // 0

// Type-parameterized tests are abstract test patterns parameterized
// by a type.  Compared with typed tests, type-parameterized tests
// allow you to define the test pattern without knowing what the type
// parameters are.  The defined pattern can be instantiated with
// different types any number of times, in any number of translation
// units.
//
// If you are designing an interface or concept, you can define a
// suite of type-parameterized tests to verify properties that any
// valid implementation of the interface/concept should have.  Then,
// each implementation can easily instantiate the test suite to verify
// that it conforms to the requirements, without having to write
// similar tests repeatedly.  Here's an example:

#if 0

// First, define a fixture class template.  It should be parameterized
// by a type.  Remember to derive it from testing::Test.
template <typename T>
class FooTest : public testing::Test {
  ...
};

// Next, declare that you will define a type-parameterized test case
// (the _P suffix is for "parameterized" or "pattern", whichever you
// prefer):
TYPED_TEST_CASE_P(FooTest);

// Then, use TYPED_TEST_P() to define as many type-parameterized tests
// for this type-parameterized test case as you want.
TYPED_TEST_P(FooTest, DoesBlah) {
  // Inside a test, refer to TypeParam to get the type parameter.
  TypeParam n = 0;
  ...
}

TYPED_TEST_P(FooTest, HasPropertyA) { ... }

// Now the tricky part: you need to register all test patterns before
// you can instantiate them.  The first argument of the macro is the
// test case name; the rest are the names of the tests in this test
// case.
REGISTER_TYPED_TEST_CASE_P(FooTest,
                           DoesBlah, HasPropertyA);

// Finally, you are free to instantiate the pattern with the types you
// want.  If you put the above code in a header file, you can #include
// it in multiple C++ source files and instantiate it multiple times.
//
// To distinguish different instances of the pattern, the first
// argument to the INSTANTIATE_* macro is a prefix that will be added
// to the actual test case name.  Remember to pick unique prefixes for
// different instances.
typedef testing::Types<char, int, unsigned int> MyTypes;
INSTANTIATE_TYPED_TEST_CASE_P(My, FooTest, MyTypes);

// If the type list contains only one type, you can write that type
// directly without Types<...>:
//   INSTANTIATE_TYPED_TEST_CASE_P(My, FooTest, int);

#endif  // 0


// Implements typed tests.

#if GTEST_HAS_TYPED_TEST

// INTERNAL IMPLEMENTATION - DO NOT USE IN USER CODE.
//
// Expands to the name of the typedef for the type parameters of the
// given test case.
# define GTEST_TYPE_PARAMS_(TestCaseName) gtest_type_params_##TestCaseName##_

// The 'Types' template argument below must have spaces around it
// since some compilers may choke on '>>' when passing a template
// instance (e.g. Types<int>)
# define TYPED_TEST_CASE(CaseName, Types) \
  typedef ::testing::internal::TypeList< Types >::type \
      GTEST_TYPE_PARAMS_(CaseName)

# define TYPED_TEST(CaseName, TestName) \
  template <typename gtest_TypeParam_> \
  class GTEST_TEST_CLASS_NAME_(CaseName, TestName) \
      : public CaseName<gtest_TypeParam_> { \
   private: \
    typedef CaseName<gtest_TypeParam_> TestFixture; \
    typedef gtest_TypeParam_ TypeParam; \
    virtual void TestBody(); \
  }; \
  bool gtest_##CaseName##_##TestName##_registered_ GTEST_ATTRIBUTE_UNUSED_ = \
      ::testing::internal::TypeParameterizedTest< \
          CaseName, \
          ::testing::internal::TemplateSel< \
              GTEST_TEST_CLASS_NAME_(CaseName, TestName)>, \
          GTEST_TYPE_PARAMS_(CaseName)>::Register(\
              "", #CaseName, #TestName, 0); \
  template <typename gtest_TypeParam_> \
  void GTEST_TEST_CLASS_NAME_(CaseName, TestName)<gtest_TypeParam_>::TestBody()

#endif  // GTEST_HAS_TYPED_TEST

// Implements type-parameterized tests.

#if GTEST_HAS_TYPED_TEST_P

// INTERNAL IMPLEMENTATION - DO NOT USE IN USER CODE.
//
// Expands to the namespace name that the type-parameterized tests for
// the given type-parameterized test case are defined in.  The exact
// name of the namespace is subject to change without notice.
# define GTEST_CASE_NAMESPACE_(TestCaseName) \
  gtest_case_##TestCaseName##_

// INTERNAL IMPLEMENTATION - DO NOT USE IN USER CODE.
//
// Expands to the name of the variable used to remember the names of
// the defined tests in the given test case.
# define GTEST_TYPED_TEST_CASE_P_STATE_(TestCaseName) \
  gtest_typed_test_case_p_state_##TestCaseName##_

// INTERNAL IMPLEMENTATION - DO NOT USE IN USER CODE DIRECTLY.
//
// Expands to the name of the variable used to remember the names of
// the registered tests in the given test case.
# define GTEST_REGISTERED_TEST_NAMES_(TestCaseName) \
  gtest_registered_test_names_##TestCaseName##_

// The variables defined in the type-parameterized test macros are
// static as typically these macros are used in a .h file that can be
// #included in multiple translation units linked together.
# define TYPED_TEST_CASE_P(CaseName) \
  static ::testing::internal::TypedTestCasePState \
      GTEST_TYPED_TEST_CASE_P_STATE_(CaseName)

# define TYPED_TEST_P(CaseName, TestName) \
  namespace GTEST_CASE_NAMESPACE_(CaseName) { \
  template <typename gtest_TypeParam_> \
  class TestName : public CaseName<gtest_TypeParam_> { \
   private: \
    typedef CaseName<gtest_TypeParam_> TestFixture; \
    typedef gtest_TypeParam_ TypeParam; \
    virtual void TestBody(); \
  }; \
  static bool gtest_##TestName##_defined_ GTEST_ATTRIBUTE_UNUSED_ = \
      GTEST_TYPED_TEST_CASE_P_STATE_(CaseName).AddTestName(\
          __FILE__, __LINE__, #CaseName, #TestName); \
  } \
  template <typename gtest_TypeParam_> \
  void GTEST_CASE_NAMESPACE_(CaseName)::TestName<gtest_TypeParam_>::TestBody()

# define REGISTER_TYPED_TEST_CASE_P(CaseName, ...) \
  namespace GTEST_CASE_NAMESPACE_(CaseName) { \
  typedef ::testing::internal::Templates<__VA_ARGS__>::type gtest_AllTests_; \
  } \
  static const char* const GTEST_REGISTERED_TEST_NAMES_(CaseName) = \
      GTEST_TYPED_TEST_CASE_P_STATE_(CaseName).VerifyRegisteredTestNames(\
          __FILE__, __LINE__, #__VA_ARGS__)

// The 'Types' template argument below must have spaces around it
// since some compilers may choke on '>>' when passing a template
// instance (e.g. Types<int>)
# define INSTANTIATE_TYPED_TEST_CASE_P(Prefix, CaseName, Types) \
  bool gtest_##Prefix##_##CaseName GTEST_ATTRIBUTE_UNUSED_ = \
      ::testing::internal::TypeParameterizedTestCase<CaseName, \
          GTEST_CASE_NAMESPACE_(CaseName)::gtest_AllTests_, \
          ::testing::internal::TypeList< Types >::type>::Register(\
              #Prefix, #CaseName, GTEST_REGISTERED_TEST_NAMES_(CaseName))

#endif  // GTEST_HAS_TYPED_TEST_P

#endif  // GTEST_INCLUDE_GTEST_GTEST_TYPED_TEST_H_

// Depending on the platform, different string classes are available.
// On Linux, in addition to ::std::string, Google also makes use of
// class ::string, which has the same interface as ::std::string, but
// has a different implementation.
//
// The user can define GTEST_HAS_GLOBAL_STRING to 1 to indicate that
// ::string is available AND is a distinct type to ::std::string, or
// define it to 0 to indicate otherwise.
//
// If the user's ::std::string and ::string are the same class due to
// aliasing, he should define GTEST_HAS_GLOBAL_STRING to 0.
//
// If the user doesn't define GTEST_HAS_GLOBAL_STRING, it is defined
// heuristically.

namespace testing {

// Declares the flags.

// This flag temporary enables the disabled tests.
GTEST_DECLARE_bool_(also_run_disabled_tests);

// This flag brings the debugger on an assertion failure.
GTEST_DECLARE_bool_(break_on_failure);

// This flag controls whether Google Test catches all test-thrown exceptions
// and logs them as failures.
GTEST_DECLARE_bool_(catch_exceptions);

// This flag enables using colors in terminal output. Available values are
// "yes" to enable colors, "no" (disable colors), or "auto" (the default)
// to let Google Test decide.
GTEST_DECLARE_string_(color);

// This flag sets up the filter to select by name using a glob pattern
// the tests to run. If the filter is not given all tests are executed.
GTEST_DECLARE_string_(filter);

// This flag causes the Google Test to list tests. None of the tests listed
// are actually run if the flag is provided.
GTEST_DECLARE_bool_(list_tests);

// This flag controls whether Google Test emits a detailed XML report to a file
// in addition to its normal textual output.
GTEST_DECLARE_string_(output);

// This flags control whether Google Test prints the elapsed time for each
// test.
GTEST_DECLARE_bool_(print_time);

// This flag specifies the random number seed.
GTEST_DECLARE_int32_(random_seed);

// This flag sets how many times the tests are repeated. The default value
// is 1. If the value is -1 the tests are repeating forever.
GTEST_DECLARE_int32_(repeat);

// This flag controls whether Google Test includes Google Test internal
// stack frames in failure stack traces.
GTEST_DECLARE_bool_(show_internal_stack_frames);

// When this flag is specified, tests' order is randomized on every iteration.
GTEST_DECLARE_bool_(shuffle);

// This flag specifies the maximum number of stack frames to be
// printed in a failure message.
GTEST_DECLARE_int32_(stack_trace_depth);

// When this flag is specified, a failed assertion will throw an
// exception if exceptions are enabled, or exit the program with a
// non-zero code otherwise.
GTEST_DECLARE_bool_(throw_on_failure);

// When this flag is set with a "host:port" string, on supported
// platforms test results are streamed to the specified port on
// the specified host machine.
GTEST_DECLARE_string_(stream_result_to);

// The upper limit for valid stack trace depths.
const int kMaxStackTraceDepth = 100;

namespace internal {

class AssertHelper;
class DefaultGlobalTestPartResultReporter;
class ExecDeathTest;
class NoExecDeathTest;
class FinalSuccessChecker;
class GTestFlagSaver;
class TestResultAccessor;
class TestEventListenersAccessor;
class TestEventRepeater;
class WindowsDeathTest;
class UnitTestImpl* GetUnitTestImpl();
void ReportFailureInUnknownLocation(TestPartResult::Type result_type,
                                    const String& message);

// Converts a streamable value to a String.  A NULL pointer is
// converted to "(null)".  When the input value is a ::string,
// ::std::string, ::wstring, or ::std::wstring object, each NUL
// character in it is replaced with "\\0".
// Declared in gtest-internal.h but defined here, so that it has access
// to the definition of the Message class, required by the ARM
// compiler.
template <typename T>
String StreamableToString(const T& streamable) {
  return (Message() << streamable).GetString();
}

}  // namespace internal

// The friend relationship of some of these classes is cyclic.
// If we don't forward declare them the compiler might confuse the classes
// in friendship clauses with same named classes on the scope.
class Test;
class TestCase;
class TestInfo;
class UnitTest;

// A class for indicating whether an assertion was successful.  When
// the assertion wasn't successful, the AssertionResult object
// remembers a non-empty message that describes how it failed.
//
// To create an instance of this class, use one of the factory functions
// (AssertionSuccess() and AssertionFailure()).
//
// This class is useful for two purposes:
//   1. Defining predicate functions to be used with Boolean test assertions
//      EXPECT_TRUE/EXPECT_FALSE and their ASSERT_ counterparts
//   2. Defining predicate-format functions to be
//      used with predicate assertions (ASSERT_PRED_FORMAT*, etc).
//
// For example, if you define IsEven predicate:
//
//   testing::AssertionResult IsEven(int n) {
//     if ((n % 2) == 0)
//       return testing::AssertionSuccess();
//     else
//       return testing::AssertionFailure() << n << " is odd";
//   }
//
// Then the failed expectation EXPECT_TRUE(IsEven(Fib(5)))
// will print the message
//
//   Value of: IsEven(Fib(5))
//     Actual: false (5 is odd)
//   Expected: true
//
// instead of a more opaque
//
//   Value of: IsEven(Fib(5))
//     Actual: false
//   Expected: true
//
// in case IsEven is a simple Boolean predicate.
//
// If you expect your predicate to be reused and want to support informative
// messages in EXPECT_FALSE and ASSERT_FALSE (negative assertions show up
// about half as often as positive ones in our tests), supply messages for
// both success and failure cases:
//
//   testing::AssertionResult IsEven(int n) {
//     if ((n % 2) == 0)
//       return testing::AssertionSuccess() << n << " is even";
//     else
//       return testing::AssertionFailure() << n << " is odd";
//   }
//
// Then a statement EXPECT_FALSE(IsEven(Fib(6))) will print
//
//   Value of: IsEven(Fib(6))
//     Actual: true (8 is even)
//   Expected: false
//
// NB: Predicates that support negative Boolean assertions have reduced
// performance in positive ones so be careful not to use them in tests
// that have lots (tens of thousands) of positive Boolean assertions.
//
// To use this class with EXPECT_PRED_FORMAT assertions such as:
//
//   // Verifies that Foo() returns an even number.
//   EXPECT_PRED_FORMAT1(IsEven, Foo());
//
// you need to define:
//
//   testing::AssertionResult IsEven(const char* expr, int n) {
//     if ((n % 2) == 0)
//       return testing::AssertionSuccess();
//     else
//       return testing::AssertionFailure()
//         << "Expected: " << expr << " is even\n  Actual: it's " << n;
//   }
//
// If Foo() returns 5, you will see the following message:
//
//   Expected: Foo() is even
//     Actual: it's 5
//
class GTEST_API_ AssertionResult {
 public:
  // Copy constructor.
  // Used in EXPECT_TRUE/FALSE(assertion_result).
  AssertionResult(const AssertionResult& other);
  // Used in the EXPECT_TRUE/FALSE(bool_expression).
  explicit AssertionResult(bool success) : success_(success) {}

  // Returns true iff the assertion succeeded.
  operator bool() const { return success_; }  // NOLINT

  // Returns the assertion's negation. Used with EXPECT/ASSERT_FALSE.
  AssertionResult operator!() const;

  // Returns the text streamed into this AssertionResult. Test assertions
  // use it when they fail (i.e., the predicate's outcome doesn't match the
  // assertion's expectation). When nothing has been streamed into the
  // object, returns an empty string.
  const char* message() const {
    return message_.get() != NULL ?  message_->c_str() : "";
  }
  // TODO(vladl@google.com): Remove this after making sure no clients use it.
  // Deprecated; please use message() instead.
  const char* failure_message() const { return message(); }

  // Streams a custom failure message into this object.
  template <typename T> AssertionResult& operator<<(const T& value) {
    AppendMessage(Message() << value);
    return *this;
  }

  // Allows streaming basic output manipulators such as endl or flush into
  // this object.
  AssertionResult& operator<<(
      ::std::ostream& (*basic_manipulator)(::std::ostream& stream)) {
    AppendMessage(Message() << basic_manipulator);
    return *this;
  }

 private:
  // Appends the contents of message to message_.
  void AppendMessage(const Message& a_message) {
    if (message_.get() == NULL)
      message_.reset(new ::std::string);
    message_->append(a_message.GetString().c_str());
  }

  // Stores result of the assertion predicate.
  bool success_;
  // Stores the message describing the condition in case the expectation
  // construct is not satisfied with the predicate's outcome.
  // Referenced via a pointer to avoid taking too much stack frame space
  // with test assertions.
  internal::scoped_ptr< ::std::string> message_;

  GTEST_DISALLOW_ASSIGN_(AssertionResult);
};

// Makes a successful assertion result.
GTEST_API_ AssertionResult AssertionSuccess();

// Makes a failed assertion result.
GTEST_API_ AssertionResult AssertionFailure();

// Makes a failed assertion result with the given failure message.
// Deprecated; use AssertionFailure() << msg.
GTEST_API_ AssertionResult AssertionFailure(const Message& msg);

// The abstract class that all tests inherit from.
//
// In Google Test, a unit test program contains one or many TestCases, and
// each TestCase contains one or many Tests.
//
// When you define a test using the TEST macro, you don't need to
// explicitly derive from Test - the TEST macro automatically does
// this for you.
//
// The only time you derive from Test is when defining a test fixture
// to be used a TEST_F.  For example:
//
//   class FooTest : public testing::Test {
//    protected:
//     virtual void SetUp() { ... }
//     virtual void TearDown() { ... }
//     ...
//   };
//
//   TEST_F(FooTest, Bar) { ... }
//   TEST_F(FooTest, Baz) { ... }
//
// Test is not copyable.
class GTEST_API_ Test {
 public:
  friend class TestInfo;

  // Defines types for pointers to functions that set up and tear down
  // a test case.
  typedef internal::SetUpTestCaseFunc SetUpTestCaseFunc;
  typedef internal::TearDownTestCaseFunc TearDownTestCaseFunc;

  // The d'tor is virtual as we intend to inherit from Test.
  virtual ~Test();

  // Sets up the stuff shared by all tests in this test case.
  //
  // Google Test will call Foo::SetUpTestCase() before running the first
  // test in test case Foo.  Hence a sub-class can define its own
  // SetUpTestCase() method to shadow the one defined in the super
  // class.
  static void SetUpTestCase() {}

  // Tears down the stuff shared by all tests in this test case.
  //
  // Google Test will call Foo::TearDownTestCase() after running the last
  // test in test case Foo.  Hence a sub-class can define its own
  // TearDownTestCase() method to shadow the one defined in the super
  // class.
  static void TearDownTestCase() {}

  // Returns true iff the current test has a fatal failure.
  static bool HasFatalFailure();

  // Returns true iff the current test has a non-fatal failure.
  static bool HasNonfatalFailure();

  // Returns true iff the current test has a (either fatal or
  // non-fatal) failure.
  static bool HasFailure() { return HasFatalFailure() || HasNonfatalFailure(); }

  // Logs a property for the current test.  Only the last value for a given
  // key is remembered.
  // These are public static so they can be called from utility functions
  // that are not members of the test fixture.
  // The arguments are const char* instead strings, as Google Test is used
  // on platforms where string doesn't compile.
  //
  // Note that a driving consideration for these RecordProperty methods
  // was to produce xml output suited to the Greenspan charting utility,
  // which at present will only chart values that fit in a 32-bit int. It
  // is the user's responsibility to restrict their values to 32-bit ints
  // if they intend them to be used with Greenspan.
  static void RecordProperty(const char* key, const char* value);
  static void RecordProperty(const char* key, int value);

 protected:
  // Creates a Test object.
  Test();

  // Sets up the test fixture.
  virtual void SetUp();

  // Tears down the test fixture.
  virtual void TearDown();

 private:
  // Returns true iff the current test has the same fixture class as
  // the first test in the current test case.
  static bool HasSameFixtureClass();

  // Runs the test after the test fixture has been set up.
  //
  // A sub-class must implement this to define the test logic.
  //
  // DO NOT OVERRIDE THIS FUNCTION DIRECTLY IN A USER PROGRAM.
  // Instead, use the TEST or TEST_F macro.
  virtual void TestBody() = 0;

  // Sets up, executes, and tears down the test.
  void Run();

  // Deletes self.  We deliberately pick an unusual name for this
  // internal method to avoid clashing with names used in user TESTs.
  void DeleteSelf_() { delete this; }

  // Uses a GTestFlagSaver to save and restore all Google Test flags.
  const internal::GTestFlagSaver* const gtest_flag_saver_;

  // Often a user mis-spells SetUp() as Setup() and spends a long time
  // wondering why it is never called by Google Test.  The declaration of
  // the following method is solely for catching such an error at
  // compile time:
  //
  //   - The return type is deliberately chosen to be not void, so it
  //   will be a conflict if a user declares void Setup() in his test
  //   fixture.
  //
  //   - This method is private, so it will be another compiler error
  //   if a user calls it from his test fixture.
  //
  // DO NOT OVERRIDE THIS FUNCTION.
  //
  // If you see an error about overriding the following function or
  // about it being private, you have mis-spelled SetUp() as Setup().
  struct Setup_should_be_spelled_SetUp {};
  virtual Setup_should_be_spelled_SetUp* Setup() { return NULL; }

  // We disallow copying Tests.
  GTEST_DISALLOW_COPY_AND_ASSIGN_(Test);
};

typedef internal::TimeInMillis TimeInMillis;

// A copyable object representing a user specified test property which can be
// output as a key/value string pair.
//
// Don't inherit from TestProperty as its destructor is not virtual.
class TestProperty {
 public:
  // C'tor.  TestProperty does NOT have a default constructor.
  // Always use this constructor (with parameters) to create a
  // TestProperty object.
  TestProperty(const char* a_key, const char* a_value) :
    key_(a_key), value_(a_value) {
  }

  // Gets the user supplied key.
  const char* key() const {
    return key_.c_str();
  }

  // Gets the user supplied value.
  const char* value() const {
    return value_.c_str();
  }

  // Sets a new value, overriding the one supplied in the constructor.
  void SetValue(const char* new_value) {
    value_ = new_value;
  }

 private:
  // The key supplied by the user.
  internal::String key_;
  // The value supplied by the user.
  internal::String value_;
};

// The result of a single Test.  This includes a list of
// TestPartResults, a list of TestProperties, a count of how many
// death tests there are in the Test, and how much time it took to run
// the Test.
//
// TestResult is not copyable.
class GTEST_API_ TestResult {
 public:
  // Creates an empty TestResult.
  TestResult();

  // D'tor.  Do not inherit from TestResult.
  ~TestResult();

  // Gets the number of all test parts.  This is the sum of the number
  // of successful test parts and the number of failed test parts.
  int total_part_count() const;

  // Returns the number of the test properties.
  int test_property_count() const;

  // Returns true iff the test passed (i.e. no test part failed).
  bool Passed() const { return !Failed(); }

  // Returns true iff the test failed.
  bool Failed() const;

  // Returns true iff the test fatally failed.
  bool HasFatalFailure() const;

  // Returns true iff the test has a non-fatal failure.
  bool HasNonfatalFailure() const;

  // Returns the elapsed time, in milliseconds.
  TimeInMillis elapsed_time() const { return elapsed_time_; }

  // Returns the i-th test part result among all the results. i can range
  // from 0 to test_property_count() - 1. If i is not in that range, aborts
  // the program.
  const TestPartResult& GetTestPartResult(int i) const;

  // Returns the i-th test property. i can range from 0 to
  // test_property_count() - 1. If i is not in that range, aborts the
  // program.
  const TestProperty& GetTestProperty(int i) const;

 private:
  friend class TestInfo;
  friend class UnitTest;
  friend class internal::DefaultGlobalTestPartResultReporter;
  friend class internal::ExecDeathTest;
  friend class internal::TestResultAccessor;
  friend class internal::UnitTestImpl;
  friend class internal::WindowsDeathTest;

  // Gets the vector of TestPartResults.
  const std::vector<TestPartResult>& test_part_results() const {
    return test_part_results_;
  }

  // Gets the vector of TestProperties.
  const std::vector<TestProperty>& test_properties() const {
    return test_properties_;
  }

  // Sets the elapsed time.
  void set_elapsed_time(TimeInMillis elapsed) { elapsed_time_ = elapsed; }

  // Adds a test property to the list. The property is validated and may add
  // a non-fatal failure if invalid (e.g., if it conflicts with reserved
  // key names). If a property is already recorded for the same key, the
  // value will be updated, rather than storing multiple values for the same
  // key.
  void RecordProperty(const TestProperty& test_property);

  // Adds a failure if the key is a reserved attribute of Google Test
  // testcase tags.  Returns true if the property is valid.
  // TODO(russr): Validate attribute names are legal and human readable.
  static bool ValidateTestProperty(const TestProperty& test_property);

  // Adds a test part result to the list.
  void AddTestPartResult(const TestPartResult& test_part_result);

  // Returns the death test count.
  int death_test_count() const { return death_test_count_; }

  // Increments the death test count, returning the new count.
  int increment_death_test_count() { return ++death_test_count_; }

  // Clears the test part results.
  void ClearTestPartResults();

  // Clears the object.
  void Clear();

  // Protects mutable state of the property vector and of owned
  // properties, whose values may be updated.
  internal::Mutex test_properites_mutex_;

  // The vector of TestPartResults
  std::vector<TestPartResult> test_part_results_;
  // The vector of TestProperties
  std::vector<TestProperty> test_properties_;
  // Running count of death tests.
  int death_test_count_;
  // The elapsed time, in milliseconds.
  TimeInMillis elapsed_time_;

  // We disallow copying TestResult.
  GTEST_DISALLOW_COPY_AND_ASSIGN_(TestResult);
};  // class TestResult

// A TestInfo object stores the following information about a test:
//
//   Test case name
//   Test name
//   Whether the test should be run
//   A function pointer that creates the test object when invoked
//   Test result
//
// The constructor of TestInfo registers itself with the UnitTest
// singleton such that the RUN_ALL_TESTS() macro knows which tests to
// run.
class GTEST_API_ TestInfo {
 public:
  // Destructs a TestInfo object.  This function is not virtual, so
  // don't inherit from TestInfo.
  ~TestInfo();

  // Returns the test case name.
  const char* test_case_name() const { return test_case_name_.c_str(); }

  // Returns the test name.
  const char* name() const { return name_.c_str(); }

  // Returns the name of the parameter type, or NULL if this is not a typed
  // or a type-parameterized test.
  const char* type_param() const {
    if (type_param_.get() != NULL)
      return type_param_->c_str();
    return NULL;
  }

  // Returns the text representation of the value parameter, or NULL if this
  // is not a value-parameterized test.
  const char* value_param() const {
    if (value_param_.get() != NULL)
      return value_param_->c_str();
    return NULL;
  }

  // Returns true if this test should run, that is if the test is not disabled
  // (or it is disabled but the also_run_disabled_tests flag has been specified)
  // and its full name matches the user-specified filter.
  //
  // Google Test allows the user to filter the tests by their full names.
  // The full name of a test Bar in test case Foo is defined as
  // "Foo.Bar".  Only the tests that match the filter will run.
  //
  // A filter is a colon-separated list of glob (not regex) patterns,
  // optionally followed by a '-' and a colon-separated list of
  // negative patterns (tests to exclude).  A test is run if it
  // matches one of the positive patterns and does not match any of
  // the negative patterns.
  //
  // For example, *A*:Foo.* is a filter that matches any string that
  // contains the character 'A' or starts with "Foo.".
  bool should_run() const { return should_run_; }

  // Returns the result of the test.
  const TestResult* result() const { return &result_; }

 private:

#if GTEST_HAS_DEATH_TEST
  friend class internal::DefaultDeathTestFactory;
#endif  // GTEST_HAS_DEATH_TEST
  friend class Test;
  friend class TestCase;
  friend class internal::UnitTestImpl;
  friend TestInfo* internal::MakeAndRegisterTestInfo(
      const char* test_case_name, const char* name,
      const char* type_param,
      const char* value_param,
      internal::TypeId fixture_class_id,
      Test::SetUpTestCaseFunc set_up_tc,
      Test::TearDownTestCaseFunc tear_down_tc,
      internal::TestFactoryBase* factory);

  // Constructs a TestInfo object. The newly constructed instance assumes
  // ownership of the factory object.
  TestInfo(const char* test_case_name, const char* name,
           const char* a_type_param,
           const char* a_value_param,
           internal::TypeId fixture_class_id,
           internal::TestFactoryBase* factory);

  // Increments the number of death tests encountered in this test so
  // far.
  int increment_death_test_count() {
    return result_.increment_death_test_count();
  }

  // Creates the test object, runs it, records its result, and then
  // deletes it.
  void Run();

  static void ClearTestResult(TestInfo* test_info) {
    test_info->result_.Clear();
  }

  // These fields are immutable properties of the test.
  const std::string test_case_name_;     // Test case name
  const std::string name_;               // Test name
  // Name of the parameter type, or NULL if this is not a typed or a
  // type-parameterized test.
  const internal::scoped_ptr<const ::std::string> type_param_;
  // Text representation of the value parameter, or NULL if this is not a
  // value-parameterized test.
  const internal::scoped_ptr<const ::std::string> value_param_;
  const internal::TypeId fixture_class_id_;   // ID of the test fixture class
  bool should_run_;                 // True iff this test should run
  bool is_disabled_;                // True iff this test is disabled
  bool matches_filter_;             // True if this test matches the
                                    // user-specified filter.
  internal::TestFactoryBase* const factory_;  // The factory that creates
                                              // the test object

  // This field is mutable and needs to be reset before running the
  // test for the second time.
  TestResult result_;

  GTEST_DISALLOW_COPY_AND_ASSIGN_(TestInfo);
};

// A test case, which consists of a vector of TestInfos.
//
// TestCase is not copyable.
class GTEST_API_ TestCase {
 public:
  // Creates a TestCase with the given name.
  //
  // TestCase does NOT have a default constructor.  Always use this
  // constructor to create a TestCase object.
  //
  // Arguments:
  //
  //   name:         name of the test case
  //   a_type_param: the name of the test's type parameter, or NULL if
  //                 this is not a type-parameterized test.
  //   set_up_tc:    pointer to the function that sets up the test case
  //   tear_down_tc: pointer to the function that tears down the test case
  TestCase(const char* name, const char* a_type_param,
           Test::SetUpTestCaseFunc set_up_tc,
           Test::TearDownTestCaseFunc tear_down_tc);

  // Destructor of TestCase.
  virtual ~TestCase();

  // Gets the name of the TestCase.
  const char* name() const { return name_.c_str(); }

  // Returns the name of the parameter type, or NULL if this is not a
  // type-parameterized test case.
  const char* type_param() const {
    if (type_param_.get() != NULL)
      return type_param_->c_str();
    return NULL;
  }

  // Returns true if any test in this test case should run.
  bool should_run() const { return should_run_; }

  // Gets the number of successful tests in this test case.
  int successful_test_count() const;

  // Gets the number of failed tests in this test case.
  int failed_test_count() const;

  // Gets the number of disabled tests in this test case.
  int disabled_test_count() const;

  // Get the number of tests in this test case that should run.
  int test_to_run_count() const;

  // Gets the number of all tests in this test case.
  int total_test_count() const;

  // Returns true iff the test case passed.
  bool Passed() const { return !Failed(); }

  // Returns true iff the test case failed.
  bool Failed() const { return failed_test_count() > 0; }

  // Returns the elapsed time, in milliseconds.
  TimeInMillis elapsed_time() const { return elapsed_time_; }

  // Returns the i-th test among all the tests. i can range from 0 to
  // total_test_count() - 1. If i is not in that range, returns NULL.
  const TestInfo* GetTestInfo(int i) const;

 private:
  friend class Test;
  friend class internal::UnitTestImpl;

  // Gets the (mutable) vector of TestInfos in this TestCase.
  std::vector<TestInfo*>& test_info_list() { return test_info_list_; }

  // Gets the (immutable) vector of TestInfos in this TestCase.
  const std::vector<TestInfo*>& test_info_list() const {
    return test_info_list_;
  }

  // Returns the i-th test among all the tests. i can range from 0 to
  // total_test_count() - 1. If i is not in that range, returns NULL.
  TestInfo* GetMutableTestInfo(int i);

  // Sets the should_run member.
  void set_should_run(bool should) { should_run_ = should; }

  // Adds a TestInfo to this test case.  Will delete the TestInfo upon
  // destruction of the TestCase object.
  void AddTestInfo(TestInfo * test_info);

  // Clears the results of all tests in this test case.
  void ClearResult();

  // Clears the results of all tests in the given test case.
  static void ClearTestCaseResult(TestCase* test_case) {
    test_case->ClearResult();
  }

  // Runs every test in this TestCase.
  void Run();

  // Runs SetUpTestCase() for this TestCase.  This wrapper is needed
  // for catching exceptions thrown from SetUpTestCase().
  void RunSetUpTestCase() { (*set_up_tc_)(); }

  // Runs TearDownTestCase() for this TestCase.  This wrapper is
  // needed for catching exceptions thrown from TearDownTestCase().
  void RunTearDownTestCase() { (*tear_down_tc_)(); }

  // Returns true iff test passed.
  static bool TestPassed(const TestInfo* test_info) {
    return test_info->should_run() && test_info->result()->Passed();
  }

  // Returns true iff test failed.
  static bool TestFailed(const TestInfo* test_info) {
    return test_info->should_run() && test_info->result()->Failed();
  }

  // Returns true iff test is disabled.
  static bool TestDisabled(const TestInfo* test_info) {
    return test_info->is_disabled_;
  }

  // Returns true if the given test should run.
  static bool ShouldRunTest(const TestInfo* test_info) {
    return test_info->should_run();
  }

  // Shuffles the tests in this test case.
  void ShuffleTests(internal::Random* random);

  // Restores the test order to before the first shuffle.
  void UnshuffleTests();

  // Name of the test case.
  internal::String name_;
  // Name of the parameter type, or NULL if this is not a typed or a
  // type-parameterized test.
  const internal::scoped_ptr<const ::std::string> type_param_;
  // The vector of TestInfos in their original order.  It owns the
  // elements in the vector.
  std::vector<TestInfo*> test_info_list_;
  // Provides a level of indirection for the test list to allow easy
  // shuffling and restoring the test order.  The i-th element in this
  // vector is the index of the i-th test in the shuffled test list.
  std::vector<int> test_indices_;
  // Pointer to the function that sets up the test case.
  Test::SetUpTestCaseFunc set_up_tc_;
  // Pointer to the function that tears down the test case.
  Test::TearDownTestCaseFunc tear_down_tc_;
  // True iff any test in this test case should run.
  bool should_run_;
  // Elapsed time, in milliseconds.
  TimeInMillis elapsed_time_;

  // We disallow copying TestCases.
  GTEST_DISALLOW_COPY_AND_ASSIGN_(TestCase);
};

// An Environment object is capable of setting up and tearing down an
// environment.  The user should subclass this to define his own
// environment(s).
//
// An Environment object does the set-up and tear-down in virtual
// methods SetUp() and TearDown() instead of the constructor and the
// destructor, as:
//
//   1. You cannot safely throw from a destructor.  This is a problem
//      as in some cases Google Test is used where exceptions are enabled, and
//      we may want to implement ASSERT_* using exceptions where they are
//      available.
//   2. You cannot use ASSERT_* directly in a constructor or
//      destructor.
class Environment {
 public:
  // The d'tor is virtual as we need to subclass Environment.
  virtual ~Environment() {}

  // Override this to define how to set up the environment.
  virtual void SetUp() {}

  // Override this to define how to tear down the environment.
  virtual void TearDown() {}
 private:
  // If you see an error about overriding the following function or
  // about it being private, you have mis-spelled SetUp() as Setup().
  struct Setup_should_be_spelled_SetUp {};
  virtual Setup_should_be_spelled_SetUp* Setup() { return NULL; }
};

// The interface for tracing execution of tests. The methods are organized in
// the order the corresponding events are fired.
class TestEventListener {
 public:
  virtual ~TestEventListener() {}

  // Fired before any test activity starts.
  virtual void OnTestProgramStart(const UnitTest& unit_test) = 0;

  // Fired before each iteration of tests starts.  There may be more than
  // one iteration if GTEST_FLAG(repeat) is set. iteration is the iteration
  // index, starting from 0.
  virtual void OnTestIterationStart(const UnitTest& unit_test,
                                    int iteration) = 0;

  // Fired before environment set-up for each iteration of tests starts.
  virtual void OnEnvironmentsSetUpStart(const UnitTest& unit_test) = 0;

  // Fired after environment set-up for each iteration of tests ends.
  virtual void OnEnvironmentsSetUpEnd(const UnitTest& unit_test) = 0;

  // Fired before the test case starts.
  virtual void OnTestCaseStart(const TestCase& test_case) = 0;

  // Fired before the test starts.
  virtual void OnTestStart(const TestInfo& test_info) = 0;

  // Fired after a failed assertion or a SUCCEED() invocation.
  virtual void OnTestPartResult(const TestPartResult& test_part_result) = 0;

  // Fired after the test ends.
  virtual void OnTestEnd(const TestInfo& test_info) = 0;

  // Fired after the test case ends.
  virtual void OnTestCaseEnd(const TestCase& test_case) = 0;

  // Fired before environment tear-down for each iteration of tests starts.
  virtual void OnEnvironmentsTearDownStart(const UnitTest& unit_test) = 0;

  // Fired after environment tear-down for each iteration of tests ends.
  virtual void OnEnvironmentsTearDownEnd(const UnitTest& unit_test) = 0;

  // Fired after each iteration of tests finishes.
  virtual void OnTestIterationEnd(const UnitTest& unit_test,
                                  int iteration) = 0;

  // Fired after all test activities have ended.
  virtual void OnTestProgramEnd(const UnitTest& unit_test) = 0;
};

// The convenience class for users who need to override just one or two
// methods and are not concerned that a possible change to a signature of
// the methods they override will not be caught during the build.  For
// comments about each method please see the definition of TestEventListener
// above.
class EmptyTestEventListener : public TestEventListener {
 public:
  virtual void OnTestProgramStart(const UnitTest& /*unit_test*/) {}
  virtual void OnTestIterationStart(const UnitTest& /*unit_test*/,
                                    int /*iteration*/) {}
  virtual void OnEnvironmentsSetUpStart(const UnitTest& /*unit_test*/) {}
  virtual void OnEnvironmentsSetUpEnd(const UnitTest& /*unit_test*/) {}
  virtual void OnTestCaseStart(const TestCase& /*test_case*/) {}
  virtual void OnTestStart(const TestInfo& /*test_info*/) {}
  virtual void OnTestPartResult(const TestPartResult& /*test_part_result*/) {}
  virtual void OnTestEnd(const TestInfo& /*test_info*/) {}
  virtual void OnTestCaseEnd(const TestCase& /*test_case*/) {}
  virtual void OnEnvironmentsTearDownStart(const UnitTest& /*unit_test*/) {}
  virtual void OnEnvironmentsTearDownEnd(const UnitTest& /*unit_test*/) {}
  virtual void OnTestIterationEnd(const UnitTest& /*unit_test*/,
                                  int /*iteration*/) {}
  virtual void OnTestProgramEnd(const UnitTest& /*unit_test*/) {}
};

// TestEventListeners lets users add listeners to track events in Google Test.
class GTEST_API_ TestEventListeners {
 public:
  TestEventListeners();
  ~TestEventListeners();

  // Appends an event listener to the end of the list. Google Test assumes
  // the ownership of the listener (i.e. it will delete the listener when
  // the test program finishes).
  void Append(TestEventListener* listener);

  // Removes the given event listener from the list and returns it.  It then
  // becomes the caller's responsibility to delete the listener. Returns
  // NULL if the listener is not found in the list.
  TestEventListener* Release(TestEventListener* listener);

  // Returns the standard listener responsible for the default console
  // output.  Can be removed from the listeners list to shut down default
  // console output.  Note that removing this object from the listener list
  // with Release transfers its ownership to the caller and makes this
  // function return NULL the next time.
  TestEventListener* default_result_printer() const {
    return default_result_printer_;
  }

  // Returns the standard listener responsible for the default XML output
  // controlled by the --gtest_output=xml flag.  Can be removed from the
  // listeners list by users who want to shut down the default XML output
  // controlled by this flag and substitute it with custom one.  Note that
  // removing this object from the listener list with Release transfers its
  // ownership to the caller and makes this function return NULL the next
  // time.
  TestEventListener* default_xml_generator() const {
    return default_xml_generator_;
  }

 private:
  friend class TestCase;
  friend class TestInfo;
  friend class internal::DefaultGlobalTestPartResultReporter;
  friend class internal::NoExecDeathTest;
  friend class internal::TestEventListenersAccessor;
  friend class internal::UnitTestImpl;

  // Returns repeater that broadcasts the TestEventListener events to all
  // subscribers.
  TestEventListener* repeater();

  // Sets the default_result_printer attribute to the provided listener.
  // The listener is also added to the listener list and previous
  // default_result_printer is removed from it and deleted. The listener can
  // also be NULL in which case it will not be added to the list. Does
  // nothing if the previous and the current listener objects are the same.
  void SetDefaultResultPrinter(TestEventListener* listener);

  // Sets the default_xml_generator attribute to the provided listener.  The
  // listener is also added to the listener list and previous
  // default_xml_generator is removed from it and deleted. The listener can
  // also be NULL in which case it will not be added to the list. Does
  // nothing if the previous and the current listener objects are the same.
  void SetDefaultXmlGenerator(TestEventListener* listener);

  // Controls whether events will be forwarded by the repeater to the
  // listeners in the list.
  bool EventForwardingEnabled() const;
  void SuppressEventForwarding();

  // The actual list of listeners.
  internal::TestEventRepeater* repeater_;
  // Listener responsible for the standard result output.
  TestEventListener* default_result_printer_;
  // Listener responsible for the creation of the XML output file.
  TestEventListener* default_xml_generator_;

  // We disallow copying TestEventListeners.
  GTEST_DISALLOW_COPY_AND_ASSIGN_(TestEventListeners);
};

// A UnitTest consists of a vector of TestCases.
//
// This is a singleton class.  The only instance of UnitTest is
// created when UnitTest::GetInstance() is first called.  This
// instance is never deleted.
//
// UnitTest is not copyable.
//
// This class is thread-safe as long as the methods are called
// according to their specification.
class GTEST_API_ UnitTest {
 public:
  // Gets the singleton UnitTest object.  The first time this method
  // is called, a UnitTest object is constructed and returned.
  // Consecutive calls will return the same object.
  static UnitTest* GetInstance();

  // Runs all tests in this UnitTest object and prints the result.
  // Returns 0 if successful, or 1 otherwise.
  //
  // This method can only be called from the main thread.
  //
  // INTERNAL IMPLEMENTATION - DO NOT USE IN A USER PROGRAM.
  int Run() GTEST_MUST_USE_RESULT_;

  // Returns the working directory when the first TEST() or TEST_F()
  // was executed.  The UnitTest object owns the string.
  const char* original_working_dir() const;

  // Returns the TestCase object for the test that's currently running,
  // or NULL if no test is running.
  const TestCase* current_test_case() const;

  // Returns the TestInfo object for the test that's currently running,
  // or NULL if no test is running.
  const TestInfo* current_test_info() const;

  // Returns the random seed used at the start of the current test run.
  int random_seed() const;

#if GTEST_HAS_PARAM_TEST
  // Returns the ParameterizedTestCaseRegistry object used to keep track of
  // value-parameterized tests and instantiate and register them.
  //
  // INTERNAL IMPLEMENTATION - DO NOT USE IN A USER PROGRAM.
  internal::ParameterizedTestCaseRegistry& parameterized_test_registry();
#endif  // GTEST_HAS_PARAM_TEST

  // Gets the number of successful test cases.
  int successful_test_case_count() const;

  // Gets the number of failed test cases.
  int failed_test_case_count() const;

  // Gets the number of all test cases.
  int total_test_case_count() const;

  // Gets the number of all test cases that contain at least one test
  // that should run.
  int test_case_to_run_count() const;

  // Gets the number of successful tests.
  int successful_test_count() const;

  // Gets the number of failed tests.
  int failed_test_count() const;

  // Gets the number of disabled tests.
  int disabled_test_count() const;

  // Gets the number of all tests.
  int total_test_count() const;

  // Gets the number of tests that should run.
  int test_to_run_count() const;

  // Gets the elapsed time, in milliseconds.
  TimeInMillis elapsed_time() const;

  // Returns true iff the unit test passed (i.e. all test cases passed).
  bool Passed() const;

  // Returns true iff the unit test failed (i.e. some test case failed
  // or something outside of all tests failed).
  bool Failed() const;

  // Gets the i-th test case among all the test cases. i can range from 0 to
  // total_test_case_count() - 1. If i is not in that range, returns NULL.
  const TestCase* GetTestCase(int i) const;

  // Returns the list of event listeners that can be used to track events
  // inside Google Test.
  TestEventListeners& listeners();

 private:
  // Registers and returns a global test environment.  When a test
  // program is run, all global test environments will be set-up in
  // the order they were registered.  After all tests in the program
  // have finished, all global test environments will be torn-down in
  // the *reverse* order they were registered.
  //
  // The UnitTest object takes ownership of the given environment.
  //
  // This method can only be called from the main thread.
  Environment* AddEnvironment(Environment* env);

  // Adds a TestPartResult to the current TestResult object.  All
  // Google Test assertion macros (e.g. ASSERT_TRUE, EXPECT_EQ, etc)
  // eventually call this to report their results.  The user code
  // should use the assertion macros instead of calling this directly.
  void AddTestPartResult(TestPartResult::Type result_type,
                         const char* file_name,
                         int line_number,
                         const internal::String& message,
                         const internal::String& os_stack_trace);

  // Adds a TestProperty to the current TestResult object. If the result already
  // contains a property with the same key, the value will be updated.
  void RecordPropertyForCurrentTest(const char* key, const char* value);

  // Gets the i-th test case among all the test cases. i can range from 0 to
  // total_test_case_count() - 1. If i is not in that range, returns NULL.
  TestCase* GetMutableTestCase(int i);

  // Accessors for the implementation object.
  internal::UnitTestImpl* impl() { return impl_; }
  const internal::UnitTestImpl* impl() const { return impl_; }

  // These classes and funcions are friends as they need to access private
  // members of UnitTest.
  friend class Test;
  friend class internal::AssertHelper;
  friend class internal::ScopedTrace;
  friend Environment* AddGlobalTestEnvironment(Environment* env);
  friend internal::UnitTestImpl* internal::GetUnitTestImpl();
  friend void internal::ReportFailureInUnknownLocation(
      TestPartResult::Type result_type,
      const internal::String& message);

  // Creates an empty UnitTest.
  UnitTest();

  // D'tor
  virtual ~UnitTest();

  // Pushes a trace defined by SCOPED_TRACE() on to the per-thread
  // Google Test trace stack.
  void PushGTestTrace(const internal::TraceInfo& trace);

  // Pops a trace from the per-thread Google Test trace stack.
  void PopGTestTrace();

  // Protects mutable state in *impl_.  This is mutable as some const
  // methods need to lock it too.
  mutable internal::Mutex mutex_;

  // Opaque implementation object.  This field is never changed once
  // the object is constructed.  We don't mark it as const here, as
  // doing so will cause a warning in the constructor of UnitTest.
  // Mutable state in *impl_ is protected by mutex_.
  internal::UnitTestImpl* impl_;

  // We disallow copying UnitTest.
  GTEST_DISALLOW_COPY_AND_ASSIGN_(UnitTest);
};

// A convenient wrapper for adding an environment for the test
// program.
//
// You should call this before RUN_ALL_TESTS() is called, probably in
// main().  If you use gtest_main, you need to call this before main()
// starts for it to take effect.  For example, you can define a global
// variable like this:
//
//   testing::Environment* const foo_env =
//       testing::AddGlobalTestEnvironment(new FooEnvironment);
//
// However, we strongly recommend you to write your own main() and
// call AddGlobalTestEnvironment() there, as relying on initialization
// of global variables makes the code harder to read and may cause
// problems when you register multiple environments from different
// translation units and the environments have dependencies among them
// (remember that the compiler doesn't guarantee the order in which
// global variables from different translation units are initialized).
inline Environment* AddGlobalTestEnvironment(Environment* env) {
  return UnitTest::GetInstance()->AddEnvironment(env);
}

// Initializes Google Test.  This must be called before calling
// RUN_ALL_TESTS().  In particular, it parses a command line for the
// flags that Google Test recognizes.  Whenever a Google Test flag is
// seen, it is removed from argv, and *argc is decremented.
//
// No value is returned.  Instead, the Google Test flag variables are
// updated.
//
// Calling the function for the second time has no user-visible effect.
GTEST_API_ void InitGoogleTest(int* argc, char** argv);

// This overloaded version can be used in Windows programs compiled in
// UNICODE mode.
GTEST_API_ void InitGoogleTest(int* argc, wchar_t** argv);

namespace internal {

// Formats a comparison assertion (e.g. ASSERT_EQ, EXPECT_LT, and etc)
// operand to be used in a failure message.  The type (but not value)
// of the other operand may affect the format.  This allows us to
// print a char* as a raw pointer when it is compared against another
// char*, and print it as a C string when it is compared against an
// std::string object, for example.
//
// The default implementation ignores the type of the other operand.
// Some specialized versions are used to handle formatting wide or
// narrow C strings.
//
// INTERNAL IMPLEMENTATION - DO NOT USE IN A USER PROGRAM.
template <typename T1, typename T2>
String FormatForComparisonFailureMessage(const T1& value,
                                         const T2& /* other_operand */) {
  // C++Builder compiles this incorrectly if the namespace isn't explicitly
  // given.
  return ::testing::PrintToString(value);
}

// The helper function for {ASSERT|EXPECT}_EQ.
template <typename T1, typename T2>
AssertionResult CmpHelperEQ(const char* expected_expression,
                            const char* actual_expression,
                            const T1& expected,
                            const T2& actual) {
#ifdef _MSC_VER
# pragma warning(push)          // Saves the current warning state.
# pragma warning(disable:4389)  // Temporarily disables warning on
                               // signed/unsigned mismatch.
#endif

  if (expected == actual) {
    return AssertionSuccess();
  }

#ifdef _MSC_VER
# pragma warning(pop)          // Restores the warning state.
#endif

  return EqFailure(expected_expression,
                   actual_expression,
                   FormatForComparisonFailureMessage(expected, actual),
                   FormatForComparisonFailureMessage(actual, expected),
                   false);
}

// With this overloaded version, we allow anonymous enums to be used
// in {ASSERT|EXPECT}_EQ when compiled with gcc 4, as anonymous enums
// can be implicitly cast to BiggestInt.
GTEST_API_ AssertionResult CmpHelperEQ(const char* expected_expression,
                                       const char* actual_expression,
                                       BiggestInt expected,
                                       BiggestInt actual);

// The helper class for {ASSERT|EXPECT}_EQ.  The template argument
// lhs_is_null_literal is true iff the first argument to ASSERT_EQ()
// is a null pointer literal.  The following default implementation is
// for lhs_is_null_literal being false.
template <bool lhs_is_null_literal>
class EqHelper {
 public:
  // This templatized version is for the general case.
  template <typename T1, typename T2>
  static AssertionResult Compare(const char* expected_expression,
                                 const char* actual_expression,
                                 const T1& expected,
                                 const T2& actual) {
    return CmpHelperEQ(expected_expression, actual_expression, expected,
                       actual);
  }

  // With this overloaded version, we allow anonymous enums to be used
  // in {ASSERT|EXPECT}_EQ when compiled with gcc 4, as anonymous
  // enums can be implicitly cast to BiggestInt.
  //
  // Even though its body looks the same as the above version, we
  // cannot merge the two, as it will make anonymous enums unhappy.
  static AssertionResult Compare(const char* expected_expression,
                                 const char* actual_expression,
                                 BiggestInt expected,
                                 BiggestInt actual) {
    return CmpHelperEQ(expected_expression, actual_expression, expected,
                       actual);
  }
};

// This specialization is used when the first argument to ASSERT_EQ()
// is a null pointer literal, like NULL, false, or 0.
template <>
class EqHelper<true> {
 public:
  // We define two overloaded versions of Compare().  The first
  // version will be picked when the second argument to ASSERT_EQ() is
  // NOT a pointer, e.g. ASSERT_EQ(0, AnIntFunction()) or
  // EXPECT_EQ(false, a_bool).
  template <typename T1, typename T2>
  static AssertionResult Compare(
      const char* expected_expression,
      const char* actual_expression,
      const T1& expected,
      const T2& actual,
      // The following line prevents this overload from being considered if T2
      // is not a pointer type.  We need this because ASSERT_EQ(NULL, my_ptr)
      // expands to Compare("", "", NULL, my_ptr), which requires a conversion
      // to match the Secret* in the other overload, which would otherwise make
      // this template match better.
      typename EnableIf<!is_pointer<T2>::value>::type* = 0) {
    return CmpHelperEQ(expected_expression, actual_expression, expected,
                       actual);
  }

  // This version will be picked when the second argument to ASSERT_EQ() is a
  // pointer, e.g. ASSERT_EQ(NULL, a_pointer).
  template <typename T>
  static AssertionResult Compare(
      const char* expected_expression,
      const char* actual_expression,
      // We used to have a second template parameter instead of Secret*.  That
      // template parameter would deduce to 'long', making this a better match
      // than the first overload even without the first overload's EnableIf.
      // Unfortunately, gcc with -Wconversion-null warns when "passing NULL to
      // non-pointer argument" (even a deduced integral argument), so the old
      // implementation caused warnings in user code.
      Secret* /* expected (NULL) */,
      T* actual) {
    // We already know that 'expected' is a null pointer.
    return CmpHelperEQ(expected_expression, actual_expression,
                       static_cast<T*>(NULL), actual);
  }
};

// A macro for implementing the helper functions needed to implement
// ASSERT_?? and EXPECT_??.  It is here just to avoid copy-and-paste
// of similar code.
//
// For each templatized helper function, we also define an overloaded
// version for BiggestInt in order to reduce code bloat and allow
// anonymous enums to be used with {ASSERT|EXPECT}_?? when compiled
// with gcc 4.
//
// INTERNAL IMPLEMENTATION - DO NOT USE IN A USER PROGRAM.
#define GTEST_IMPL_CMP_HELPER_(op_name, op)\
template <typename T1, typename T2>\
AssertionResult CmpHelper##op_name(const char* expr1, const char* expr2, \
                                   const T1& val1, const T2& val2) {\
  if (val1 op val2) {\
    return AssertionSuccess();\
  } else {\
    return AssertionFailure() \
        << "Expected: (" << expr1 << ") " #op " (" << expr2\
        << "), actual: " << FormatForComparisonFailureMessage(val1, val2)\
        << " vs " << FormatForComparisonFailureMessage(val2, val1);\
  }\
}\
GTEST_API_ AssertionResult CmpHelper##op_name(\
    const char* expr1, const char* expr2, BiggestInt val1, BiggestInt val2)

// INTERNAL IMPLEMENTATION - DO NOT USE IN A USER PROGRAM.

// Implements the helper function for {ASSERT|EXPECT}_NE
GTEST_IMPL_CMP_HELPER_(NE, !=);
// Implements the helper function for {ASSERT|EXPECT}_LE
GTEST_IMPL_CMP_HELPER_(LE, <=);
// Implements the helper function for {ASSERT|EXPECT}_LT
GTEST_IMPL_CMP_HELPER_(LT, < );
// Implements the helper function for {ASSERT|EXPECT}_GE
GTEST_IMPL_CMP_HELPER_(GE, >=);
// Implements the helper function for {ASSERT|EXPECT}_GT
GTEST_IMPL_CMP_HELPER_(GT, > );

#undef GTEST_IMPL_CMP_HELPER_

// The helper function for {ASSERT|EXPECT}_STREQ.
//
// INTERNAL IMPLEMENTATION - DO NOT USE IN A USER PROGRAM.
GTEST_API_ AssertionResult CmpHelperSTREQ(const char* expected_expression,
                                          const char* actual_expression,
                                          const char* expected,
                                          const char* actual);

// The helper function for {ASSERT|EXPECT}_STRCASEEQ.
//
// INTERNAL IMPLEMENTATION - DO NOT USE IN A USER PROGRAM.
GTEST_API_ AssertionResult CmpHelperSTRCASEEQ(const char* expected_expression,
                                              const char* actual_expression,
                                              const char* expected,
                                              const char* actual);

// The helper function for {ASSERT|EXPECT}_STRNE.
//
// INTERNAL IMPLEMENTATION - DO NOT USE IN A USER PROGRAM.
GTEST_API_ AssertionResult CmpHelperSTRNE(const char* s1_expression,
                                          const char* s2_expression,
                                          const char* s1,
                                          const char* s2);

// The helper function for {ASSERT|EXPECT}_STRCASENE.
//
// INTERNAL IMPLEMENTATION - DO NOT USE IN A USER PROGRAM.
GTEST_API_ AssertionResult CmpHelperSTRCASENE(const char* s1_expression,
                                              const char* s2_expression,
                                              const char* s1,
                                              const char* s2);


// Helper function for *_STREQ on wide strings.
//
// INTERNAL IMPLEMENTATION - DO NOT USE IN A USER PROGRAM.
GTEST_API_ AssertionResult CmpHelperSTREQ(const char* expected_expression,
                                          const char* actual_expression,
                                          const wchar_t* expected,
                                          const wchar_t* actual);

// Helper function for *_STRNE on wide strings.
//
// INTERNAL IMPLEMENTATION - DO NOT USE IN A USER PROGRAM.
GTEST_API_ AssertionResult CmpHelperSTRNE(const char* s1_expression,
                                          const char* s2_expression,
                                          const wchar_t* s1,
                                          const wchar_t* s2);

}  // namespace internal

// IsSubstring() and IsNotSubstring() are intended to be used as the
// first argument to {EXPECT,ASSERT}_PRED_FORMAT2(), not by
// themselves.  They check whether needle is a substring of haystack
// (NULL is considered a substring of itself only), and return an
// appropriate error message when they fail.
//
// The {needle,haystack}_expr arguments are the stringified
// expressions that generated the two real arguments.
GTEST_API_ AssertionResult IsSubstring(
    const char* needle_expr, const char* haystack_expr,
    const char* needle, const char* haystack);
GTEST_API_ AssertionResult IsSubstring(
    const char* needle_expr, const char* haystack_expr,
    const wchar_t* needle, const wchar_t* haystack);
GTEST_API_ AssertionResult IsNotSubstring(
    const char* needle_expr, const char* haystack_expr,
    const char* needle, const char* haystack);
GTEST_API_ AssertionResult IsNotSubstring(
    const char* needle_expr, const char* haystack_expr,
    const wchar_t* needle, const wchar_t* haystack);
GTEST_API_ AssertionResult IsSubstring(
    const char* needle_expr, const char* haystack_expr,
    const ::std::string& needle, const ::std::string& haystack);
GTEST_API_ AssertionResult IsNotSubstring(
    const char* needle_expr, const char* haystack_expr,
    const ::std::string& needle, const ::std::string& haystack);

#if GTEST_HAS_STD_WSTRING
GTEST_API_ AssertionResult IsSubstring(
    const char* needle_expr, const char* haystack_expr,
    const ::std::wstring& needle, const ::std::wstring& haystack);
GTEST_API_ AssertionResult IsNotSubstring(
    const char* needle_expr, const char* haystack_expr,
    const ::std::wstring& needle, const ::std::wstring& haystack);
#endif  // GTEST_HAS_STD_WSTRING

namespace internal {

// Helper template function for comparing floating-points.
//
// Template parameter:
//
//   RawType: the raw floating-point type (either float or double)
//
// INTERNAL IMPLEMENTATION - DO NOT USE IN A USER PROGRAM.
template <typename RawType>
AssertionResult CmpHelperFloatingPointEQ(const char* expected_expression,
                                         const char* actual_expression,
                                         RawType expected,
                                         RawType actual) {
  const FloatingPoint<RawType> lhs(expected), rhs(actual);

  if (lhs.AlmostEquals(rhs)) {
    return AssertionSuccess();
  }

  ::std::stringstream expected_ss;
  expected_ss << std::setprecision(std::numeric_limits<RawType>::digits10 + 2)
              << expected;

  ::std::stringstream actual_ss;
  actual_ss << std::setprecision(std::numeric_limits<RawType>::digits10 + 2)
            << actual;

  return EqFailure(expected_expression,
                   actual_expression,
                   StringStreamToString(&expected_ss),
                   StringStreamToString(&actual_ss),
                   false);
}

// Helper function for implementing ASSERT_NEAR.
//
// INTERNAL IMPLEMENTATION - DO NOT USE IN A USER PROGRAM.
GTEST_API_ AssertionResult DoubleNearPredFormat(const char* expr1,
                                                const char* expr2,
                                                const char* abs_error_expr,
                                                double val1,
                                                double val2,
                                                double abs_error);

// INTERNAL IMPLEMENTATION - DO NOT USE IN USER CODE.
// A class that enables one to stream messages to assertion macros
class GTEST_API_ AssertHelper {
 public:
  // Constructor.
  AssertHelper(TestPartResult::Type type,
               const char* file,
               int line,
               const char* message);
  ~AssertHelper();

  // Message assignment is a semantic trick to enable assertion
  // streaming; see the GTEST_MESSAGE_ macro below.
  void operator=(const Message& message) const;

 private:
  // We put our data in a struct so that the size of the AssertHelper class can
  // be as small as possible.  This is important because gcc is incapable of
  // re-using stack space even for temporary variables, so every EXPECT_EQ
  // reserves stack space for another AssertHelper.
  struct AssertHelperData {
    AssertHelperData(TestPartResult::Type t,
                     const char* srcfile,
                     int line_num,
                     const char* msg)
        : type(t), file(srcfile), line(line_num), message(msg) { }

    TestPartResult::Type const type;
    const char*        const file;
    int                const line;
    String             const message;

   private:
    GTEST_DISALLOW_COPY_AND_ASSIGN_(AssertHelperData);
  };

  AssertHelperData* const data_;

  GTEST_DISALLOW_COPY_AND_ASSIGN_(AssertHelper);
};

}  // namespace internal

#if GTEST_HAS_PARAM_TEST
// The pure interface class that all value-parameterized tests inherit from.
// A value-parameterized class must inherit from both ::testing::Test and
// ::testing::WithParamInterface. In most cases that just means inheriting
// from ::testing::TestWithParam, but more complicated test hierarchies
// may need to inherit from Test and WithParamInterface at different levels.
//
// This interface has support for accessing the test parameter value via
// the GetParam() method.
//
// Use it with one of the parameter generator defining functions, like Range(),
// Values(), ValuesIn(), Bool(), and Combine().
//
// class FooTest : public ::testing::TestWithParam<int> {
//  protected:
//   FooTest() {
//     // Can use GetParam() here.
//   }
//   virtual ~FooTest() {
//     // Can use GetParam() here.
//   }
//   virtual void SetUp() {
//     // Can use GetParam() here.
//   }
//   virtual void TearDown {
//     // Can use GetParam() here.
//   }
// };
// TEST_P(FooTest, DoesBar) {
//   // Can use GetParam() method here.
//   Foo foo;
//   ASSERT_TRUE(foo.DoesBar(GetParam()));
// }
// INSTANTIATE_TEST_CASE_P(OneToTenRange, FooTest, ::testing::Range(1, 10));

template <typename T>
class WithParamInterface {
 public:
  typedef T ParamType;
  virtual ~WithParamInterface() {}

  // The current parameter value. Is also available in the test fixture's
  // constructor. This member function is non-static, even though it only
  // references static data, to reduce the opportunity for incorrect uses
  // like writing 'WithParamInterface<bool>::GetParam()' for a test that
  // uses a fixture whose parameter type is int.
  const ParamType& GetParam() const { return *parameter_; }

 private:
  // Sets parameter value. The caller is responsible for making sure the value
  // remains alive and unchanged throughout the current test.
  static void SetParam(const ParamType* parameter) {
    parameter_ = parameter;
  }

  // Static value used for accessing parameter during a test lifetime.
  static const ParamType* parameter_;

  // TestClass must be a subclass of WithParamInterface<T> and Test.
  template <class TestClass> friend class internal::ParameterizedTestFactory;
};

template <typename T>
const T* WithParamInterface<T>::parameter_ = NULL;

// Most value-parameterized classes can ignore the existence of
// WithParamInterface, and can just inherit from ::testing::TestWithParam.

template <typename T>
class TestWithParam : public Test, public WithParamInterface<T> {
};

#endif  // GTEST_HAS_PARAM_TEST

// Macros for indicating success/failure in test code.

// ADD_FAILURE unconditionally adds a failure to the current test.
// SUCCEED generates a success - it doesn't automatically make the
// current test successful, as a test is only successful when it has
// no failure.
//
// EXPECT_* verifies that a certain condition is satisfied.  If not,
// it behaves like ADD_FAILURE.  In particular:
//
//   EXPECT_TRUE  verifies that a Boolean condition is true.
//   EXPECT_FALSE verifies that a Boolean condition is false.
//
// FAIL and ASSERT_* are similar to ADD_FAILURE and EXPECT_*, except
// that they will also abort the current function on failure.  People
// usually want the fail-fast behavior of FAIL and ASSERT_*, but those
// writing data-driven tests often find themselves using ADD_FAILURE
// and EXPECT_* more.
//
// Examples:
//
//   EXPECT_TRUE(server.StatusIsOK());
//   ASSERT_FALSE(server.HasPendingRequest(port))
//       << "There are still pending requests " << "on port " << port;

// Generates a nonfatal failure with a generic message.
#define ADD_FAILURE() GTEST_NONFATAL_FAILURE_("Failed")

// Generates a nonfatal failure at the given source file location with
// a generic message.
#define ADD_FAILURE_AT(file, line) \
  GTEST_MESSAGE_AT_(file, line, "Failed", \
                    ::testing::TestPartResult::kNonFatalFailure)

// Generates a fatal failure with a generic message.
#define GTEST_FAIL() GTEST_FATAL_FAILURE_("Failed")

// Define this macro to 1 to omit the definition of FAIL(), which is a
// generic name and clashes with some other libraries.
#if !GTEST_DONT_DEFINE_FAIL
# define FAIL() GTEST_FAIL()
#endif

// Generates a success with a generic message.
#define GTEST_SUCCEED() GTEST_SUCCESS_("Succeeded")

// Define this macro to 1 to omit the definition of SUCCEED(), which
// is a generic name and clashes with some other libraries.
#if !GTEST_DONT_DEFINE_SUCCEED
# define SUCCEED() GTEST_SUCCEED()
#endif

// Macros for testing exceptions.
//
//    * {ASSERT|EXPECT}_THROW(statement, expected_exception):
//         Tests that the statement throws the expected exception.
//    * {ASSERT|EXPECT}_NO_THROW(statement):
//         Tests that the statement doesn't throw any exception.
//    * {ASSERT|EXPECT}_ANY_THROW(statement):
//         Tests that the statement throws an exception.

#define EXPECT_THROW(statement, expected_exception) \
  GTEST_TEST_THROW_(statement, expected_exception, GTEST_NONFATAL_FAILURE_)
#define EXPECT_NO_THROW(statement) \
  GTEST_TEST_NO_THROW_(statement, GTEST_NONFATAL_FAILURE_)
#define EXPECT_ANY_THROW(statement) \
  GTEST_TEST_ANY_THROW_(statement, GTEST_NONFATAL_FAILURE_)
#define ASSERT_THROW(statement, expected_exception) \
  GTEST_TEST_THROW_(statement, expected_exception, GTEST_FATAL_FAILURE_)
#define ASSERT_NO_THROW(statement) \
  GTEST_TEST_NO_THROW_(statement, GTEST_FATAL_FAILURE_)
#define ASSERT_ANY_THROW(statement) \
  GTEST_TEST_ANY_THROW_(statement, GTEST_FATAL_FAILURE_)

// Boolean assertions. Condition can be either a Boolean expression or an
// AssertionResult. For more information on how to use AssertionResult with
// these macros see comments on that class.
#define EXPECT_TRUE(condition) \
  GTEST_TEST_BOOLEAN_(condition, #condition, false, true, \
                      GTEST_NONFATAL_FAILURE_)
#define EXPECT_FALSE(condition) \
  GTEST_TEST_BOOLEAN_(!(condition), #condition, true, false, \
                      GTEST_NONFATAL_FAILURE_)
#define ASSERT_TRUE(condition) \
  GTEST_TEST_BOOLEAN_(condition, #condition, false, true, \
                      GTEST_FATAL_FAILURE_)
#define ASSERT_FALSE(condition) \
  GTEST_TEST_BOOLEAN_(!(condition), #condition, true, false, \
                      GTEST_FATAL_FAILURE_)

// Includes the auto-generated header that implements a family of
// generic predicate assertion macros.
// Copyright 2006, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// This file is AUTOMATICALLY GENERATED on 09/24/2010 by command
// 'gen_gtest_pred_impl.py 5'.  DO NOT EDIT BY HAND!
//
// Implements a family of generic predicate assertion macros.

#ifndef GTEST_INCLUDE_GTEST_GTEST_PRED_IMPL_H_
#define GTEST_INCLUDE_GTEST_GTEST_PRED_IMPL_H_

// Makes sure this header is not included before gtest.h.
#ifndef GTEST_INCLUDE_GTEST_GTEST_H_
# error Do not include gtest_pred_impl.h directly.  Include gtest.h instead.
#endif  // GTEST_INCLUDE_GTEST_GTEST_H_

// This header implements a family of generic predicate assertion
// macros:
//
//   ASSERT_PRED_FORMAT1(pred_format, v1)
//   ASSERT_PRED_FORMAT2(pred_format, v1, v2)
//   ...
//
// where pred_format is a function or functor that takes n (in the
// case of ASSERT_PRED_FORMATn) values and their source expression
// text, and returns a testing::AssertionResult.  See the definition
// of ASSERT_EQ in gtest.h for an example.
//
// If you don't care about formatting, you can use the more
// restrictive version:
//
//   ASSERT_PRED1(pred, v1)
//   ASSERT_PRED2(pred, v1, v2)
//   ...
//
// where pred is an n-ary function or functor that returns bool,
// and the values v1, v2, ..., must support the << operator for
// streaming to std::ostream.
//
// We also define the EXPECT_* variations.
//
// For now we only support predicates whose arity is at most 5.
// Please email googletestframework@googlegroups.com if you need
// support for higher arities.

// GTEST_ASSERT_ is the basic statement to which all of the assertions
// in this file reduce.  Don't use this in your code.

#define GTEST_ASSERT_(expression, on_failure) \
  GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
  if (const ::testing::AssertionResult gtest_ar = (expression)) \
    ; \
  else \
    on_failure(gtest_ar.failure_message())


// Helper function for implementing {EXPECT|ASSERT}_PRED1.  Don't use
// this in your code.
template <typename Pred,
          typename T1>
AssertionResult AssertPred1Helper(const char* pred_text,
                                  const char* e1,
                                  Pred pred,
                                  const T1& v1) {
  if (pred(v1)) return AssertionSuccess();

  return AssertionFailure() << pred_text << "("
                            << e1 << ") evaluates to false, where"
                            << "\n" << e1 << " evaluates to " << v1;
}

// Internal macro for implementing {EXPECT|ASSERT}_PRED_FORMAT1.
// Don't use this in your code.
#define GTEST_PRED_FORMAT1_(pred_format, v1, on_failure)\
  GTEST_ASSERT_(pred_format(#v1, v1),\
                on_failure)

// Internal macro for implementing {EXPECT|ASSERT}_PRED1.  Don't use
// this in your code.
#define GTEST_PRED1_(pred, v1, on_failure)\
  GTEST_ASSERT_(::testing::AssertPred1Helper(#pred, \
                                             #v1, \
                                             pred, \
                                             v1), on_failure)

// Unary predicate assertion macros.
#define EXPECT_PRED_FORMAT1(pred_format, v1) \
  GTEST_PRED_FORMAT1_(pred_format, v1, GTEST_NONFATAL_FAILURE_)
#define EXPECT_PRED1(pred, v1) \
  GTEST_PRED1_(pred, v1, GTEST_NONFATAL_FAILURE_)
#define ASSERT_PRED_FORMAT1(pred_format, v1) \
  GTEST_PRED_FORMAT1_(pred_format, v1, GTEST_FATAL_FAILURE_)
#define ASSERT_PRED1(pred, v1) \
  GTEST_PRED1_(pred, v1, GTEST_FATAL_FAILURE_)



// Helper function for implementing {EXPECT|ASSERT}_PRED2.  Don't use
// this in your code.
template <typename Pred,
          typename T1,
          typename T2>
AssertionResult AssertPred2Helper(const char* pred_text,
                                  const char* e1,
                                  const char* e2,
                                  Pred pred,
                                  const T1& v1,
                                  const T2& v2) {
  if (pred(v1, v2)) return AssertionSuccess();

  return AssertionFailure() << pred_text << "("
                            << e1 << ", "
                            << e2 << ") evaluates to false, where"
                            << "\n" << e1 << " evaluates to " << v1
                            << "\n" << e2 << " evaluates to " << v2;
}

// Internal macro for implementing {EXPECT|ASSERT}_PRED_FORMAT2.
// Don't use this in your code.
#define GTEST_PRED_FORMAT2_(pred_format, v1, v2, on_failure)\
  GTEST_ASSERT_(pred_format(#v1, #v2, v1, v2),\
                on_failure)

// Internal macro for implementing {EXPECT|ASSERT}_PRED2.  Don't use
// this in your code.
#define GTEST_PRED2_(pred, v1, v2, on_failure)\
  GTEST_ASSERT_(::testing::AssertPred2Helper(#pred, \
                                             #v1, \
                                             #v2, \
                                             pred, \
                                             v1, \
                                             v2), on_failure)

// Binary predicate assertion macros.
#define EXPECT_PRED_FORMAT2(pred_format, v1, v2) \
  GTEST_PRED_FORMAT2_(pred_format, v1, v2, GTEST_NONFATAL_FAILURE_)
#define EXPECT_PRED2(pred, v1, v2) \
  GTEST_PRED2_(pred, v1, v2, GTEST_NONFATAL_FAILURE_)
#define ASSERT_PRED_FORMAT2(pred_format, v1, v2) \
  GTEST_PRED_FORMAT2_(pred_format, v1, v2, GTEST_FATAL_FAILURE_)
#define ASSERT_PRED2(pred, v1, v2) \
  GTEST_PRED2_(pred, v1, v2, GTEST_FATAL_FAILURE_)



// Helper function for implementing {EXPECT|ASSERT}_PRED3.  Don't use
// this in your code.
template <typename Pred,
          typename T1,
          typename T2,
          typename T3>
AssertionResult AssertPred3Helper(const char* pred_text,
                                  const char* e1,
                                  const char* e2,
                                  const char* e3,
                                  Pred pred,
                                  const T1& v1,
                                  const T2& v2,
                                  const T3& v3) {
  if (pred(v1, v2, v3)) return AssertionSuccess();

  return AssertionFailure() << pred_text << "("
                            << e1 << ", "
                            << e2 << ", "
                            << e3 << ") evaluates to false, where"
                            << "\n" << e1 << " evaluates to " << v1
                            << "\n" << e2 << " evaluates to " << v2
                            << "\n" << e3 << " evaluates to " << v3;
}

// Internal macro for implementing {EXPECT|ASSERT}_PRED_FORMAT3.
// Don't use this in your code.
#define GTEST_PRED_FORMAT3_(pred_format, v1, v2, v3, on_failure)\
  GTEST_ASSERT_(pred_format(#v1, #v2, #v3, v1, v2, v3),\
                on_failure)

// Internal macro for implementing {EXPECT|ASSERT}_PRED3.  Don't use
// this in your code.
#define GTEST_PRED3_(pred, v1, v2, v3, on_failure)\
  GTEST_ASSERT_(::testing::AssertPred3Helper(#pred, \
                                             #v1, \
                                             #v2, \
                                             #v3, \
                                             pred, \
                                             v1, \
                                             v2, \
                                             v3), on_failure)

// Ternary predicate assertion macros.
#define EXPECT_PRED_FORMAT3(pred_format, v1, v2, v3) \
  GTEST_PRED_FORMAT3_(pred_format, v1, v2, v3, GTEST_NONFATAL_FAILURE_)
#define EXPECT_PRED3(pred, v1, v2, v3) \
  GTEST_PRED3_(pred, v1, v2, v3, GTEST_NONFATAL_FAILURE_)
#define ASSERT_PRED_FORMAT3(pred_format, v1, v2, v3) \
  GTEST_PRED_FORMAT3_(pred_format, v1, v2, v3, GTEST_FATAL_FAILURE_)
#define ASSERT_PRED3(pred, v1, v2, v3) \
  GTEST_PRED3_(pred, v1, v2, v3, GTEST_FATAL_FAILURE_)



// Helper function for implementing {EXPECT|ASSERT}_PRED4.  Don't use
// this in your code.
template <typename Pred,
          typename T1,
          typename T2,
          typename T3,
          typename T4>
AssertionResult AssertPred4Helper(const char* pred_text,
                                  const char* e1,
                                  const char* e2,
                                  const char* e3,
                                  const char* e4,
                                  Pred pred,
                                  const T1& v1,
                                  const T2& v2,
                                  const T3& v3,
                                  const T4& v4) {
  if (pred(v1, v2, v3, v4)) return AssertionSuccess();

  return AssertionFailure() << pred_text << "("
                            << e1 << ", "
                            << e2 << ", "
                            << e3 << ", "
                            << e4 << ") evaluates to false, where"
                            << "\n" << e1 << " evaluates to " << v1
                            << "\n" << e2 << " evaluates to " << v2
                            << "\n" << e3 << " evaluates to " << v3
                            << "\n" << e4 << " evaluates to " << v4;
}

// Internal macro for implementing {EXPECT|ASSERT}_PRED_FORMAT4.
// Don't use this in your code.
#define GTEST_PRED_FORMAT4_(pred_format, v1, v2, v3, v4, on_failure)\
  GTEST_ASSERT_(pred_format(#v1, #v2, #v3, #v4, v1, v2, v3, v4),\
                on_failure)

// Internal macro for implementing {EXPECT|ASSERT}_PRED4.  Don't use
// this in your code.
#define GTEST_PRED4_(pred, v1, v2, v3, v4, on_failure)\
  GTEST_ASSERT_(::testing::AssertPred4Helper(#pred, \
                                             #v1, \
                                             #v2, \
                                             #v3, \
                                             #v4, \
                                             pred, \
                                             v1, \
                                             v2, \
                                             v3, \
                                             v4), on_failure)

// 4-ary predicate assertion macros.
#define EXPECT_PRED_FORMAT4(pred_format, v1, v2, v3, v4) \
  GTEST_PRED_FORMAT4_(pred_format, v1, v2, v3, v4, GTEST_NONFATAL_FAILURE_)
#define EXPECT_PRED4(pred, v1, v2, v3, v4) \
  GTEST_PRED4_(pred, v1, v2, v3, v4, GTEST_NONFATAL_FAILURE_)
#define ASSERT_PRED_FORMAT4(pred_format, v1, v2, v3, v4) \
  GTEST_PRED_FORMAT4_(pred_format, v1, v2, v3, v4, GTEST_FATAL_FAILURE_)
#define ASSERT_PRED4(pred, v1, v2, v3, v4) \
  GTEST_PRED4_(pred, v1, v2, v3, v4, GTEST_FATAL_FAILURE_)



// Helper function for implementing {EXPECT|ASSERT}_PRED5.  Don't use
// this in your code.
template <typename Pred,
          typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5>
AssertionResult AssertPred5Helper(const char* pred_text,
                                  const char* e1,
                                  const char* e2,
                                  const char* e3,
                                  const char* e4,
                                  const char* e5,
                                  Pred pred,
                                  const T1& v1,
                                  const T2& v2,
                                  const T3& v3,
                                  const T4& v4,
                                  const T5& v5) {
  if (pred(v1, v2, v3, v4, v5)) return AssertionSuccess();

  return AssertionFailure() << pred_text << "("
                            << e1 << ", "
                            << e2 << ", "
                            << e3 << ", "
                            << e4 << ", "
                            << e5 << ") evaluates to false, where"
                            << "\n" << e1 << " evaluates to " << v1
                            << "\n" << e2 << " evaluates to " << v2
                            << "\n" << e3 << " evaluates to " << v3
                            << "\n" << e4 << " evaluates to " << v4
                            << "\n" << e5 << " evaluates to " << v5;
}

// Internal macro for implementing {EXPECT|ASSERT}_PRED_FORMAT5.
// Don't use this in your code.
#define GTEST_PRED_FORMAT5_(pred_format, v1, v2, v3, v4, v5, on_failure)\
  GTEST_ASSERT_(pred_format(#v1, #v2, #v3, #v4, #v5, v1, v2, v3, v4, v5),\
                on_failure)

// Internal macro for implementing {EXPECT|ASSERT}_PRED5.  Don't use
// this in your code.
#define GTEST_PRED5_(pred, v1, v2, v3, v4, v5, on_failure)\
  GTEST_ASSERT_(::testing::AssertPred5Helper(#pred, \
                                             #v1, \
                                             #v2, \
                                             #v3, \
                                             #v4, \
                                             #v5, \
                                             pred, \
                                             v1, \
                                             v2, \
                                             v3, \
                                             v4, \
                                             v5), on_failure)

// 5-ary predicate assertion macros.
#define EXPECT_PRED_FORMAT5(pred_format, v1, v2, v3, v4, v5) \
  GTEST_PRED_FORMAT5_(pred_format, v1, v2, v3, v4, v5, GTEST_NONFATAL_FAILURE_)
#define EXPECT_PRED5(pred, v1, v2, v3, v4, v5) \
  GTEST_PRED5_(pred, v1, v2, v3, v4, v5, GTEST_NONFATAL_FAILURE_)
#define ASSERT_PRED_FORMAT5(pred_format, v1, v2, v3, v4, v5) \
  GTEST_PRED_FORMAT5_(pred_format, v1, v2, v3, v4, v5, GTEST_FATAL_FAILURE_)
#define ASSERT_PRED5(pred, v1, v2, v3, v4, v5) \
  GTEST_PRED5_(pred, v1, v2, v3, v4, v5, GTEST_FATAL_FAILURE_)



#endif  // GTEST_INCLUDE_GTEST_GTEST_PRED_IMPL_H_

// Macros for testing equalities and inequalities.
//
//    * {ASSERT|EXPECT}_EQ(expected, actual): Tests that expected == actual
//    * {ASSERT|EXPECT}_NE(v1, v2):           Tests that v1 != v2
//    * {ASSERT|EXPECT}_LT(v1, v2):           Tests that v1 < v2
//    * {ASSERT|EXPECT}_LE(v1, v2):           Tests that v1 <= v2
//    * {ASSERT|EXPECT}_GT(v1, v2):           Tests that v1 > v2
//    * {ASSERT|EXPECT}_GE(v1, v2):           Tests that v1 >= v2
//
// When they are not, Google Test prints both the tested expressions and
// their actual values.  The values must be compatible built-in types,
// or you will get a compiler error.  By "compatible" we mean that the
// values can be compared by the respective operator.
//
// Note:
//
//   1. It is possible to make a user-defined type work with
//   {ASSERT|EXPECT}_??(), but that requires overloading the
//   comparison operators and is thus discouraged by the Google C++
//   Usage Guide.  Therefore, you are advised to use the
//   {ASSERT|EXPECT}_TRUE() macro to assert that two objects are
//   equal.
//
//   2. The {ASSERT|EXPECT}_??() macros do pointer comparisons on
//   pointers (in particular, C strings).  Therefore, if you use it
//   with two C strings, you are testing how their locations in memory
//   are related, not how their content is related.  To compare two C
//   strings by content, use {ASSERT|EXPECT}_STR*().
//
//   3. {ASSERT|EXPECT}_EQ(expected, actual) is preferred to
//   {ASSERT|EXPECT}_TRUE(expected == actual), as the former tells you
//   what the actual value is when it fails, and similarly for the
//   other comparisons.
//
//   4. Do not depend on the order in which {ASSERT|EXPECT}_??()
//   evaluate their arguments, which is undefined.
//
//   5. These macros evaluate their arguments exactly once.
//
// Examples:
//
//   EXPECT_NE(5, Foo());
//   EXPECT_EQ(NULL, a_pointer);
//   ASSERT_LT(i, array_size);
//   ASSERT_GT(records.size(), 0) << "There is no record left.";

#define EXPECT_EQ(expected, actual) \
  EXPECT_PRED_FORMAT2(::testing::internal:: \
                      EqHelper<GTEST_IS_NULL_LITERAL_(expected)>::Compare, \
                      expected, actual)
#define EXPECT_NE(expected, actual) \
  EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperNE, expected, actual)
#define EXPECT_LE(val1, val2) \
  EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperLE, val1, val2)
#define EXPECT_LT(val1, val2) \
  EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperLT, val1, val2)
#define EXPECT_GE(val1, val2) \
  EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperGE, val1, val2)
#define EXPECT_GT(val1, val2) \
  EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperGT, val1, val2)

#define GTEST_ASSERT_EQ(expected, actual) \
  ASSERT_PRED_FORMAT2(::testing::internal:: \
                      EqHelper<GTEST_IS_NULL_LITERAL_(expected)>::Compare, \
                      expected, actual)
#define GTEST_ASSERT_NE(val1, val2) \
  ASSERT_PRED_FORMAT2(::testing::internal::CmpHelperNE, val1, val2)
#define GTEST_ASSERT_LE(val1, val2) \
  ASSERT_PRED_FORMAT2(::testing::internal::CmpHelperLE, val1, val2)
#define GTEST_ASSERT_LT(val1, val2) \
  ASSERT_PRED_FORMAT2(::testing::internal::CmpHelperLT, val1, val2)
#define GTEST_ASSERT_GE(val1, val2) \
  ASSERT_PRED_FORMAT2(::testing::internal::CmpHelperGE, val1, val2)
#define GTEST_ASSERT_GT(val1, val2) \
  ASSERT_PRED_FORMAT2(::testing::internal::CmpHelperGT, val1, val2)

// Define macro GTEST_DONT_DEFINE_ASSERT_XY to 1 to omit the definition of
// ASSERT_XY(), which clashes with some users' own code.

#if !GTEST_DONT_DEFINE_ASSERT_EQ
# define ASSERT_EQ(val1, val2) GTEST_ASSERT_EQ(val1, val2)
#endif

#if !GTEST_DONT_DEFINE_ASSERT_NE
# define ASSERT_NE(val1, val2) GTEST_ASSERT_NE(val1, val2)
#endif

#if !GTEST_DONT_DEFINE_ASSERT_LE
# define ASSERT_LE(val1, val2) GTEST_ASSERT_LE(val1, val2)
#endif

#if !GTEST_DONT_DEFINE_ASSERT_LT
# define ASSERT_LT(val1, val2) GTEST_ASSERT_LT(val1, val2)
#endif

#if !GTEST_DONT_DEFINE_ASSERT_GE
# define ASSERT_GE(val1, val2) GTEST_ASSERT_GE(val1, val2)
#endif

#if !GTEST_DONT_DEFINE_ASSERT_GT
# define ASSERT_GT(val1, val2) GTEST_ASSERT_GT(val1, val2)
#endif

// C String Comparisons.  All tests treat NULL and any non-NULL string
// as different.  Two NULLs are equal.
//
//    * {ASSERT|EXPECT}_STREQ(s1, s2):     Tests that s1 == s2
//    * {ASSERT|EXPECT}_STRNE(s1, s2):     Tests that s1 != s2
//    * {ASSERT|EXPECT}_STRCASEEQ(s1, s2): Tests that s1 == s2, ignoring case
//    * {ASSERT|EXPECT}_STRCASENE(s1, s2): Tests that s1 != s2, ignoring case
//
// For wide or narrow string objects, you can use the
// {ASSERT|EXPECT}_??() macros.
//
// Don't depend on the order in which the arguments are evaluated,
// which is undefined.
//
// These macros evaluate their arguments exactly once.

#define EXPECT_STREQ(expected, actual) \
  EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperSTREQ, expected, actual)
#define EXPECT_STRNE(s1, s2) \
  EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperSTRNE, s1, s2)
#define EXPECT_STRCASEEQ(expected, actual) \
  EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperSTRCASEEQ, expected, actual)
#define EXPECT_STRCASENE(s1, s2)\
  EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperSTRCASENE, s1, s2)

#define ASSERT_STREQ(expected, actual) \
  ASSERT_PRED_FORMAT2(::testing::internal::CmpHelperSTREQ, expected, actual)
#define ASSERT_STRNE(s1, s2) \
  ASSERT_PRED_FORMAT2(::testing::internal::CmpHelperSTRNE, s1, s2)
#define ASSERT_STRCASEEQ(expected, actual) \
  ASSERT_PRED_FORMAT2(::testing::internal::CmpHelperSTRCASEEQ, expected, actual)
#define ASSERT_STRCASENE(s1, s2)\
  ASSERT_PRED_FORMAT2(::testing::internal::CmpHelperSTRCASENE, s1, s2)

// Macros for comparing floating-point numbers.
//
//    * {ASSERT|EXPECT}_FLOAT_EQ(expected, actual):
//         Tests that two float values are almost equal.
//    * {ASSERT|EXPECT}_DOUBLE_EQ(expected, actual):
//         Tests that two double values are almost equal.
//    * {ASSERT|EXPECT}_NEAR(v1, v2, abs_error):
//         Tests that v1 and v2 are within the given distance to each other.
//
// Google Test uses ULP-based comparison to automatically pick a default
// error bound that is appropriate for the operands.  See the
// FloatingPoint template class in gtest-internal.h if you are
// interested in the implementation details.

#define EXPECT_FLOAT_EQ(expected, actual)\
  EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperFloatingPointEQ<float>, \
                      expected, actual)

#define EXPECT_DOUBLE_EQ(expected, actual)\
  EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperFloatingPointEQ<double>, \
                      expected, actual)

#define ASSERT_FLOAT_EQ(expected, actual)\
  ASSERT_PRED_FORMAT2(::testing::internal::CmpHelperFloatingPointEQ<float>, \
                      expected, actual)

#define ASSERT_DOUBLE_EQ(expected, actual)\
  ASSERT_PRED_FORMAT2(::testing::internal::CmpHelperFloatingPointEQ<double>, \
                      expected, actual)

#define EXPECT_NEAR(val1, val2, abs_error)\
  EXPECT_PRED_FORMAT3(::testing::internal::DoubleNearPredFormat, \
                      val1, val2, abs_error)

#define ASSERT_NEAR(val1, val2, abs_error)\
  ASSERT_PRED_FORMAT3(::testing::internal::DoubleNearPredFormat, \
                      val1, val2, abs_error)

// These predicate format functions work on floating-point values, and
// can be used in {ASSERT|EXPECT}_PRED_FORMAT2*(), e.g.
//
//   EXPECT_PRED_FORMAT2(testing::DoubleLE, Foo(), 5.0);

// Asserts that val1 is less than, or almost equal to, val2.  Fails
// otherwise.  In particular, it fails if either val1 or val2 is NaN.
GTEST_API_ AssertionResult FloatLE(const char* expr1, const char* expr2,
                                   float val1, float val2);
GTEST_API_ AssertionResult DoubleLE(const char* expr1, const char* expr2,
                                    double val1, double val2);


#if GTEST_OS_WINDOWS

// Macros that test for HRESULT failure and success, these are only useful
// on Windows, and rely on Windows SDK macros and APIs to compile.
//
//    * {ASSERT|EXPECT}_HRESULT_{SUCCEEDED|FAILED}(expr)
//
// When expr unexpectedly fails or succeeds, Google Test prints the
// expected result and the actual result with both a human-readable
// string representation of the error, if available, as well as the
// hex result code.
# define EXPECT_HRESULT_SUCCEEDED(expr) \
    EXPECT_PRED_FORMAT1(::testing::internal::IsHRESULTSuccess, (expr))

# define ASSERT_HRESULT_SUCCEEDED(expr) \
    ASSERT_PRED_FORMAT1(::testing::internal::IsHRESULTSuccess, (expr))

# define EXPECT_HRESULT_FAILED(expr) \
    EXPECT_PRED_FORMAT1(::testing::internal::IsHRESULTFailure, (expr))

# define ASSERT_HRESULT_FAILED(expr) \
    ASSERT_PRED_FORMAT1(::testing::internal::IsHRESULTFailure, (expr))

#endif  // GTEST_OS_WINDOWS

// Macros that execute statement and check that it doesn't generate new fatal
// failures in the current thread.
//
//   * {ASSERT|EXPECT}_NO_FATAL_FAILURE(statement);
//
// Examples:
//
//   EXPECT_NO_FATAL_FAILURE(Process());
//   ASSERT_NO_FATAL_FAILURE(Process()) << "Process() failed";
//
#define ASSERT_NO_FATAL_FAILURE(statement) \
    GTEST_TEST_NO_FATAL_FAILURE_(statement, GTEST_FATAL_FAILURE_)
#define EXPECT_NO_FATAL_FAILURE(statement) \
    GTEST_TEST_NO_FATAL_FAILURE_(statement, GTEST_NONFATAL_FAILURE_)

// Causes a trace (including the source file path, the current line
// number, and the given message) to be included in every test failure
// message generated by code in the current scope.  The effect is
// undone when the control leaves the current scope.
//
// The message argument can be anything streamable to std::ostream.
//
// In the implementation, we include the current line number as part
// of the dummy variable name, thus allowing multiple SCOPED_TRACE()s
// to appear in the same block - as long as they are on different
// lines.
#define SCOPED_TRACE(message) \
  ::testing::internal::ScopedTrace GTEST_CONCAT_TOKEN_(gtest_trace_, __LINE__)(\
    __FILE__, __LINE__, ::testing::Message() << (message))

// Compile-time assertion for type equality.
// StaticAssertTypeEq<type1, type2>() compiles iff type1 and type2 are
// the same type.  The value it returns is not interesting.
//
// Instead of making StaticAssertTypeEq a class template, we make it a
// function template that invokes a helper class template.  This
// prevents a user from misusing StaticAssertTypeEq<T1, T2> by
// defining objects of that type.
//
// CAVEAT:
//
// When used inside a method of a class template,
// StaticAssertTypeEq<T1, T2>() is effective ONLY IF the method is
// instantiated.  For example, given:
//
//   template <typename T> class Foo {
//    public:
//     void Bar() { testing::StaticAssertTypeEq<int, T>(); }
//   };
//
// the code:
//
//   void Test1() { Foo<bool> foo; }
//
// will NOT generate a compiler error, as Foo<bool>::Bar() is never
// actually instantiated.  Instead, you need:
//
//   void Test2() { Foo<bool> foo; foo.Bar(); }
//
// to cause a compiler error.
template <typename T1, typename T2>
bool StaticAssertTypeEq() {
  (void)internal::StaticAssertTypeEqHelper<T1, T2>();
  return true;
}

// Defines a test.
//
// The first parameter is the name of the test case, and the second
// parameter is the name of the test within the test case.
//
// The convention is to end the test case name with "Test".  For
// example, a test case for the Foo class can be named FooTest.
//
// The user should put his test code between braces after using this
// macro.  Example:
//
//   TEST(FooTest, InitializesCorrectly) {
//     Foo foo;
//     EXPECT_TRUE(foo.StatusIsOK());
//   }

// Note that we call GetTestTypeId() instead of GetTypeId<
// ::testing::Test>() here to get the type ID of testing::Test.  This
// is to work around a suspected linker bug when using Google Test as
// a framework on Mac OS X.  The bug causes GetTypeId<
// ::testing::Test>() to return different values depending on whether
// the call is from the Google Test framework itself or from user test
// code.  GetTestTypeId() is guaranteed to always return the same
// value, as it always calls GetTypeId<>() from the Google Test
// framework.
#define GTEST_TEST(test_case_name, test_name)\
  GTEST_TEST_(test_case_name, test_name, \
              ::testing::Test, ::testing::internal::GetTestTypeId())

// Define this macro to 1 to omit the definition of TEST(), which
// is a generic name and clashes with some other libraries.
#if !GTEST_DONT_DEFINE_TEST
# define TEST(test_case_name, test_name) GTEST_TEST(test_case_name, test_name)
#endif

// Defines a test that uses a test fixture.
//
// The first parameter is the name of the test fixture class, which
// also doubles as the test case name.  The second parameter is the
// name of the test within the test case.
//
// A test fixture class must be declared earlier.  The user should put
// his test code between braces after using this macro.  Example:
//
//   class FooTest : public testing::Test {
//    protected:
//     virtual void SetUp() { b_.AddElement(3); }
//
//     Foo a_;
//     Foo b_;
//   };
//
//   TEST_F(FooTest, InitializesCorrectly) {
//     EXPECT_TRUE(a_.StatusIsOK());
//   }
//
//   TEST_F(FooTest, ReturnsElementCountCorrectly) {
//     EXPECT_EQ(0, a_.size());
//     EXPECT_EQ(1, b_.size());
//   }

#define TEST_F(test_fixture, test_name)\
  GTEST_TEST_(test_fixture, test_name, test_fixture, \
              ::testing::internal::GetTypeId<test_fixture>())

// Use this macro in main() to run all tests.  It returns 0 if all
// tests are successful, or 1 otherwise.
//
// RUN_ALL_TESTS() should be invoked after the command line has been
// parsed by InitGoogleTest().

#define RUN_ALL_TESTS()\
  (::testing::UnitTest::GetInstance()->Run())

}  // namespace testing

#endif  // GTEST_INCLUDE_GTEST_GTEST_H_
