// Copyright 2017 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// This header file defines macros for declaring attributes for functions,
// types, and variables.
//
// These macros are used within Abseil and allow the compiler to optimize, where
// applicable, certain function calls.
//
// This file is used for both C and C++!
//
// Most macros here are exposing GCC or Clang features, and are stubbed out for
// other compilers.
//
// GCC attributes documentation:
//   https://gcc.gnu.org/onlinedocs/gcc-4.7.0/gcc/Function-Attributes.html
//   https://gcc.gnu.org/onlinedocs/gcc-4.7.0/gcc/Variable-Attributes.html
//   https://gcc.gnu.org/onlinedocs/gcc-4.7.0/gcc/Type-Attributes.html
//
// Most attributes in this file are already supported by GCC 4.7. However, some
// of them are not supported in older version of Clang. Thus, we check
// `__has_attribute()` first. If the check fails, we check if we are on GCC and
// assume the attribute exists on GCC (which is verified on GCC 4.7).
//
// -----------------------------------------------------------------------------
// Sanitizer Attributes
// -----------------------------------------------------------------------------
//
// Sanitizer-related attributes are not "defined" in this file (and indeed
// are not defined as such in any file). To utilize the following
// sanitizer-related attributes within your builds, define the following macros
// within your build using a `-D` flag, along with the given value for
// `-fsanitize`:
//
//   * `ADDRESS_SANITIZER` + `-fsanitize=address` (Clang, GCC 4.8)
//   * `MEMORY_SANITIZER` + `-fsanitize=memory` (Clang-only)
//   * `THREAD_SANITIZER + `-fsanitize=thread` (Clang, GCC 4.8+)
//   * `UNDEFINED_BEHAVIOR_SANITIZER` + `-fsanitize=undefined` (Clang, GCC 4.9+)
//   * `CONTROL_FLOW_INTEGRITY` + -fsanitize=cfi (Clang-only)
//
// Example:
//
//   // Enable branches in the Abseil code that are tagged for ASan:
//   $ bazel build --copt=-DADDRESS_SANITIZER --copt=-fsanitize=address
//     --linkopt=-fsanitize=address *target*
//
// Since these macro names are only supported by GCC and Clang, we only check
// for `__GNUC__` (GCC or Clang) and the above macros.
#ifndef OTABSL_BASE_ATTRIBUTES_H_
#define OTABSL_BASE_ATTRIBUTES_H_

// OTABSL_HAVE_ATTRIBUTE
//
// A function-like feature checking macro that is a wrapper around
// `__has_attribute`, which is defined by GCC 5+ and Clang and evaluates to a
// nonzero constant integer if the attribute is supported or 0 if not.
//
// It evaluates to zero if `__has_attribute` is not defined by the compiler.
//
// GCC: https://gcc.gnu.org/gcc-5/changes.html
// Clang: https://clang.llvm.org/docs/LanguageExtensions.html
#ifdef __has_attribute
#define OTABSL_HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define OTABSL_HAVE_ATTRIBUTE(x) 0
#endif

// OTABSL_HAVE_CPP_ATTRIBUTE
//
// A function-like feature checking macro that accepts C++11 style attributes.
// It's a wrapper around `__has_cpp_attribute`, defined by ISO C++ SD-6
// (https://en.cppreference.com/w/cpp/experimental/feature_test). If we don't
// find `__has_cpp_attribute`, will evaluate to 0.
#if defined(__cplusplus) && defined(__has_cpp_attribute)
// NOTE: requiring __cplusplus above should not be necessary, but
// works around https://bugs.llvm.org/show_bug.cgi?id=23435.
#define OTABSL_HAVE_CPP_ATTRIBUTE(x) __has_cpp_attribute(x)
#else
#define OTABSL_HAVE_CPP_ATTRIBUTE(x) 0
#endif

// -----------------------------------------------------------------------------
// Function Attributes
// -----------------------------------------------------------------------------
//
// GCC: https://gcc.gnu.org/onlinedocs/gcc/Function-Attributes.html
// Clang: https://clang.llvm.org/docs/AttributeReference.html

// OTABSL_PRINTF_ATTRIBUTE
// OTABSL_SCANF_ATTRIBUTE
//
// Tells the compiler to perform `printf` format string checking if the
// compiler supports it; see the 'format' attribute in
// <https://gcc.gnu.org/onlinedocs/gcc-4.7.0/gcc/Function-Attributes.html>.
//
// Note: As the GCC manual states, "[s]ince non-static C++ methods
// have an implicit 'this' argument, the arguments of such methods
// should be counted from two, not one."
#if OTABSL_HAVE_ATTRIBUTE(format) || (defined(__GNUC__) && !defined(__clang__))
#define OTABSL_PRINTF_ATTRIBUTE(string_index, first_to_check) \
  __attribute__((__format__(__printf__, string_index, first_to_check)))
#define OTABSL_SCANF_ATTRIBUTE(string_index, first_to_check) \
  __attribute__((__format__(__scanf__, string_index, first_to_check)))
#else
#define OTABSL_PRINTF_ATTRIBUTE(string_index, first_to_check)
#define OTABSL_SCANF_ATTRIBUTE(string_index, first_to_check)
#endif

// OTABSL_ATTRIBUTE_ALWAYS_INLINE
// OTABSL_ATTRIBUTE_NOINLINE
//
// Forces functions to either inline or not inline. Introduced in gcc 3.1.
#if OTABSL_HAVE_ATTRIBUTE(always_inline) || \
    (defined(__GNUC__) && !defined(__clang__))
#define OTABSL_ATTRIBUTE_ALWAYS_INLINE __attribute__((always_inline))
#define OTABSL_HAVE_ATTRIBUTE_ALWAYS_INLINE 1
#else
#define OTABSL_ATTRIBUTE_ALWAYS_INLINE
#endif

#if OTABSL_HAVE_ATTRIBUTE(noinline) || (defined(__GNUC__) && !defined(__clang__))
#define OTABSL_ATTRIBUTE_NOINLINE __attribute__((noinline))
#define OTABSL_HAVE_ATTRIBUTE_NOINLINE 1
#else
#define OTABSL_ATTRIBUTE_NOINLINE
#endif

// OTABSL_ATTRIBUTE_NO_TAIL_CALL
//
// Prevents the compiler from optimizing away stack frames for functions which
// end in a call to another function.
#if OTABSL_HAVE_ATTRIBUTE(disable_tail_calls)
#define OTABSL_HAVE_ATTRIBUTE_NO_TAIL_CALL 1
#define OTABSL_ATTRIBUTE_NO_TAIL_CALL __attribute__((disable_tail_calls))
#elif defined(__GNUC__) && !defined(__clang__)
#define OTABSL_HAVE_ATTRIBUTE_NO_TAIL_CALL 1
#define OTABSL_ATTRIBUTE_NO_TAIL_CALL \
  __attribute__((optimize("no-optimize-sibling-calls")))
#else
#define OTABSL_ATTRIBUTE_NO_TAIL_CALL
#define OTABSL_HAVE_ATTRIBUTE_NO_TAIL_CALL 0
#endif

// OTABSL_ATTRIBUTE_WEAK
//
// Tags a function as weak for the purposes of compilation and linking.
// Weak attributes currently do not work properly in LLVM's Windows backend,
// so disable them there. See https://bugs.llvm.org/show_bug.cgi?id=37598
// for further information.
// The MinGW compiler doesn't complain about the weak attribute until the link
// step, presumably because Windows doesn't use ELF binaries.
#if (OTABSL_HAVE_ATTRIBUTE(weak) ||                   \
     (defined(__GNUC__) && !defined(__clang__))) && \
    !(defined(__llvm__) && defined(_WIN32)) && !defined(__MINGW32__)
#undef OTABSL_ATTRIBUTE_WEAK
#define OTABSL_ATTRIBUTE_WEAK __attribute__((weak))
#define OTABSL_HAVE_ATTRIBUTE_WEAK 1
#else
#define OTABSL_ATTRIBUTE_WEAK
#define OTABSL_HAVE_ATTRIBUTE_WEAK 0
#endif

// OTABSL_ATTRIBUTE_NONNULL
//
// Tells the compiler either (a) that a particular function parameter
// should be a non-null pointer, or (b) that all pointer arguments should
// be non-null.
//
// Note: As the GCC manual states, "[s]ince non-static C++ methods
// have an implicit 'this' argument, the arguments of such methods
// should be counted from two, not one."
//
// Args are indexed starting at 1.
//
// For non-static class member functions, the implicit `this` argument
// is arg 1, and the first explicit argument is arg 2. For static class member
// functions, there is no implicit `this`, and the first explicit argument is
// arg 1.
//
// Example:
//
//   /* arg_a cannot be null, but arg_b can */
//   void Function(void* arg_a, void* arg_b) OTABSL_ATTRIBUTE_NONNULL(1);
//
//   class C {
//     /* arg_a cannot be null, but arg_b can */
//     void Method(void* arg_a, void* arg_b) OTABSL_ATTRIBUTE_NONNULL(2);
//
//     /* arg_a cannot be null, but arg_b can */
//     static void StaticMethod(void* arg_a, void* arg_b)
//     OTABSL_ATTRIBUTE_NONNULL(1);
//   };
//
// If no arguments are provided, then all pointer arguments should be non-null.
//
//  /* No pointer arguments may be null. */
//  void Function(void* arg_a, void* arg_b, int arg_c) OTABSL_ATTRIBUTE_NONNULL();
//
// NOTE: The GCC nonnull attribute actually accepts a list of arguments, but
// OTABSL_ATTRIBUTE_NONNULL does not.
#if OTABSL_HAVE_ATTRIBUTE(nonnull) || (defined(__GNUC__) && !defined(__clang__))
#define OTABSL_ATTRIBUTE_NONNULL(arg_index) __attribute__((nonnull(arg_index)))
#else
#define OTABSL_ATTRIBUTE_NONNULL(...)
#endif

// OTABSL_ATTRIBUTE_NORETURN
//
// Tells the compiler that a given function never returns.
#if OTABSL_HAVE_ATTRIBUTE(noreturn) || (defined(__GNUC__) && !defined(__clang__))
#define OTABSL_ATTRIBUTE_NORETURN __attribute__((noreturn))
#elif defined(_MSC_VER)
#define OTABSL_ATTRIBUTE_NORETURN __declspec(noreturn)
#else
#define OTABSL_ATTRIBUTE_NORETURN
#endif

// OTABSL_ATTRIBUTE_NO_SANITIZE_ADDRESS
//
// Tells the AddressSanitizer (or other memory testing tools) to ignore a given
// function. Useful for cases when a function reads random locations on stack,
// calls _exit from a cloned subprocess, deliberately accesses buffer
// out of bounds or does other scary things with memory.
// NOTE: GCC supports AddressSanitizer(asan) since 4.8.
// https://gcc.gnu.org/gcc-4.8/changes.html
#if defined(__GNUC__)
#define OTABSL_ATTRIBUTE_NO_SANITIZE_ADDRESS __attribute__((no_sanitize_address))
#else
#define OTABSL_ATTRIBUTE_NO_SANITIZE_ADDRESS
#endif

// OTABSL_ATTRIBUTE_NO_SANITIZE_MEMORY
//
// Tells the  MemorySanitizer to relax the handling of a given function. All
// "Use of uninitialized value" warnings from such functions will be suppressed,
// and all values loaded from memory will be considered fully initialized.
// This attribute is similar to the ADDRESS_SANITIZER attribute above, but deals
// with initialized-ness rather than addressability issues.
// NOTE: MemorySanitizer(msan) is supported by Clang but not GCC.
#if defined(__clang__)
#define OTABSL_ATTRIBUTE_NO_SANITIZE_MEMORY __attribute__((no_sanitize_memory))
#else
#define OTABSL_ATTRIBUTE_NO_SANITIZE_MEMORY
#endif

// OTABSL_ATTRIBUTE_NO_SANITIZE_THREAD
//
// Tells the ThreadSanitizer to not instrument a given function.
// NOTE: GCC supports ThreadSanitizer(tsan) since 4.8.
// https://gcc.gnu.org/gcc-4.8/changes.html
#if defined(__GNUC__)
#define OTABSL_ATTRIBUTE_NO_SANITIZE_THREAD __attribute__((no_sanitize_thread))
#else
#define OTABSL_ATTRIBUTE_NO_SANITIZE_THREAD
#endif

// OTABSL_ATTRIBUTE_NO_SANITIZE_UNDEFINED
//
// Tells the UndefinedSanitizer to ignore a given function. Useful for cases
// where certain behavior (eg. division by zero) is being used intentionally.
// NOTE: GCC supports UndefinedBehaviorSanitizer(ubsan) since 4.9.
// https://gcc.gnu.org/gcc-4.9/changes.html
#if defined(__GNUC__) && \
    (defined(UNDEFINED_BEHAVIOR_SANITIZER) || defined(ADDRESS_SANITIZER))
#define OTABSL_ATTRIBUTE_NO_SANITIZE_UNDEFINED \
  __attribute__((no_sanitize("undefined")))
#else
#define OTABSL_ATTRIBUTE_NO_SANITIZE_UNDEFINED
#endif

// OTABSL_ATTRIBUTE_NO_SANITIZE_CFI
//
// Tells the ControlFlowIntegrity sanitizer to not instrument a given function.
// See https://clang.llvm.org/docs/ControlFlowIntegrity.html for details.
#if defined(__GNUC__) && defined(CONTROL_FLOW_INTEGRITY)
#define OTABSL_ATTRIBUTE_NO_SANITIZE_CFI __attribute__((no_sanitize("cfi")))
#else
#define OTABSL_ATTRIBUTE_NO_SANITIZE_CFI
#endif

// OTABSL_ATTRIBUTE_NO_SANITIZE_SAFESTACK
//
// Tells the SafeStack to not instrument a given function.
// See https://clang.llvm.org/docs/SafeStack.html for details.
#if defined(__GNUC__) && defined(SAFESTACK_SANITIZER)
#define OTABSL_ATTRIBUTE_NO_SANITIZE_SAFESTACK \
  __attribute__((no_sanitize("safe-stack")))
#else
#define OTABSL_ATTRIBUTE_NO_SANITIZE_SAFESTACK
#endif

// OTABSL_ATTRIBUTE_RETURNS_NONNULL
//
// Tells the compiler that a particular function never returns a null pointer.
#if OTABSL_HAVE_ATTRIBUTE(returns_nonnull) || \
    (defined(__GNUC__) && \
     (__GNUC__ > 5 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 9)) && \
     !defined(__clang__))
#define OTABSL_ATTRIBUTE_RETURNS_NONNULL __attribute__((returns_nonnull))
#else
#define OTABSL_ATTRIBUTE_RETURNS_NONNULL
#endif

// OTABSL_HAVE_ATTRIBUTE_SECTION
//
// Indicates whether labeled sections are supported. Weak symbol support is
// a prerequisite. Labeled sections are not supported on Darwin/iOS.
#ifdef OTABSL_HAVE_ATTRIBUTE_SECTION
#error OTABSL_HAVE_ATTRIBUTE_SECTION cannot be directly set
#elif (OTABSL_HAVE_ATTRIBUTE(section) ||                \
       (defined(__GNUC__) && !defined(__clang__))) && \
    !defined(__APPLE__) && OTABSL_HAVE_ATTRIBUTE_WEAK
#define OTABSL_HAVE_ATTRIBUTE_SECTION 1

// OTABSL_ATTRIBUTE_SECTION
//
// Tells the compiler/linker to put a given function into a section and define
// `__start_ ## name` and `__stop_ ## name` symbols to bracket the section.
// This functionality is supported by GNU linker.  Any function annotated with
// `OTABSL_ATTRIBUTE_SECTION` must not be inlined, or it will be placed into
// whatever section its caller is placed into.
//
#ifndef OTABSL_ATTRIBUTE_SECTION
#define OTABSL_ATTRIBUTE_SECTION(name) \
  __attribute__((section(#name))) __attribute__((noinline))
#endif


// OTABSL_ATTRIBUTE_SECTION_VARIABLE
//
// Tells the compiler/linker to put a given variable into a section and define
// `__start_ ## name` and `__stop_ ## name` symbols to bracket the section.
// This functionality is supported by GNU linker.
#ifndef OTABSL_ATTRIBUTE_SECTION_VARIABLE
#define OTABSL_ATTRIBUTE_SECTION_VARIABLE(name) __attribute__((section(#name)))
#endif

// OTABSL_DECLARE_ATTRIBUTE_SECTION_VARS
//
// A weak section declaration to be used as a global declaration
// for OTABSL_ATTRIBUTE_SECTION_START|STOP(name) to compile and link
// even without functions with OTABSL_ATTRIBUTE_SECTION(name).
// OTABSL_DEFINE_ATTRIBUTE_SECTION should be in the exactly one file; it's
// a no-op on ELF but not on Mach-O.
//
#ifndef OTABSL_DECLARE_ATTRIBUTE_SECTION_VARS
#define OTABSL_DECLARE_ATTRIBUTE_SECTION_VARS(name) \
  extern char __start_##name[] OTABSL_ATTRIBUTE_WEAK;    \
  extern char __stop_##name[] OTABSL_ATTRIBUTE_WEAK
#endif
#ifndef OTABSL_DEFINE_ATTRIBUTE_SECTION_VARS
#define OTABSL_INIT_ATTRIBUTE_SECTION_VARS(name)
#define OTABSL_DEFINE_ATTRIBUTE_SECTION_VARS(name)
#endif

// OTABSL_ATTRIBUTE_SECTION_START
//
// Returns `void*` pointers to start/end of a section of code with
// functions having OTABSL_ATTRIBUTE_SECTION(name).
// Returns 0 if no such functions exist.
// One must OTABSL_DECLARE_ATTRIBUTE_SECTION_VARS(name) for this to compile and
// link.
//
#define OTABSL_ATTRIBUTE_SECTION_START(name) \
  (reinterpret_cast<void *>(__start_##name))
#define OTABSL_ATTRIBUTE_SECTION_STOP(name) \
  (reinterpret_cast<void *>(__stop_##name))

#else  // !OTABSL_HAVE_ATTRIBUTE_SECTION

#define OTABSL_HAVE_ATTRIBUTE_SECTION 0

// provide dummy definitions
#define OTABSL_ATTRIBUTE_SECTION(name)
#define OTABSL_ATTRIBUTE_SECTION_VARIABLE(name)
#define OTABSL_INIT_ATTRIBUTE_SECTION_VARS(name)
#define OTABSL_DEFINE_ATTRIBUTE_SECTION_VARS(name)
#define OTABSL_DECLARE_ATTRIBUTE_SECTION_VARS(name)
#define OTABSL_ATTRIBUTE_SECTION_START(name) (reinterpret_cast<void *>(0))
#define OTABSL_ATTRIBUTE_SECTION_STOP(name) (reinterpret_cast<void *>(0))

#endif  // OTABSL_ATTRIBUTE_SECTION

// OTABSL_ATTRIBUTE_STACK_ALIGN_FOR_OLD_LIBC
//
// Support for aligning the stack on 32-bit x86.
#if OTABSL_HAVE_ATTRIBUTE(force_align_arg_pointer) || \
    (defined(__GNUC__) && !defined(__clang__))
#if defined(__i386__)
#define OTABSL_ATTRIBUTE_STACK_ALIGN_FOR_OLD_LIBC \
  __attribute__((force_align_arg_pointer))
#define OTABSL_REQUIRE_STACK_ALIGN_TRAMPOLINE (0)
#elif defined(__x86_64__)
#define OTABSL_REQUIRE_STACK_ALIGN_TRAMPOLINE (1)
#define OTABSL_ATTRIBUTE_STACK_ALIGN_FOR_OLD_LIBC
#else  // !__i386__ && !__x86_64
#define OTABSL_REQUIRE_STACK_ALIGN_TRAMPOLINE (0)
#define OTABSL_ATTRIBUTE_STACK_ALIGN_FOR_OLD_LIBC
#endif  // __i386__
#else
#define OTABSL_ATTRIBUTE_STACK_ALIGN_FOR_OLD_LIBC
#define OTABSL_REQUIRE_STACK_ALIGN_TRAMPOLINE (0)
#endif

// OTABSL_MUST_USE_RESULT
//
// Tells the compiler to warn about unused results.
//
// When annotating a function, it must appear as the first part of the
// declaration or definition. The compiler will warn if the return value from
// such a function is unused:
//
//   OTABSL_MUST_USE_RESULT Sprocket* AllocateSprocket();
//   AllocateSprocket();  // Triggers a warning.
//
// When annotating a class, it is equivalent to annotating every function which
// returns an instance.
//
//   class OTABSL_MUST_USE_RESULT Sprocket {};
//   Sprocket();  // Triggers a warning.
//
//   Sprocket MakeSprocket();
//   MakeSprocket();  // Triggers a warning.
//
// Note that references and pointers are not instances:
//
//   Sprocket* SprocketPointer();
//   SprocketPointer();  // Does *not* trigger a warning.
//
// OTABSL_MUST_USE_RESULT allows using cast-to-void to suppress the unused result
// warning. For that, warn_unused_result is used only for clang but not for gcc.
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=66425
//
// Note: past advice was to place the macro after the argument list.
#if OTABSL_HAVE_ATTRIBUTE(nodiscard)
#define OTABSL_MUST_USE_RESULT [[nodiscard]]
#elif defined(__clang__) && OTABSL_HAVE_ATTRIBUTE(warn_unused_result)
#define OTABSL_MUST_USE_RESULT __attribute__((warn_unused_result))
#else
#define OTABSL_MUST_USE_RESULT
#endif

// OTABSL_ATTRIBUTE_HOT, OTABSL_ATTRIBUTE_COLD
//
// Tells GCC that a function is hot or cold. GCC can use this information to
// improve static analysis, i.e. a conditional branch to a cold function
// is likely to be not-taken.
// This annotation is used for function declarations.
//
// Example:
//
//   int foo() OTABSL_ATTRIBUTE_HOT;
#if OTABSL_HAVE_ATTRIBUTE(hot) || (defined(__GNUC__) && !defined(__clang__))
#define OTABSL_ATTRIBUTE_HOT __attribute__((hot))
#else
#define OTABSL_ATTRIBUTE_HOT
#endif

#if OTABSL_HAVE_ATTRIBUTE(cold) || (defined(__GNUC__) && !defined(__clang__))
#define OTABSL_ATTRIBUTE_COLD __attribute__((cold))
#else
#define OTABSL_ATTRIBUTE_COLD
#endif

// OTABSL_XRAY_ALWAYS_INSTRUMENT, OTABSL_XRAY_NEVER_INSTRUMENT, OTABSL_XRAY_LOG_ARGS
//
// We define the OTABSL_XRAY_ALWAYS_INSTRUMENT and OTABSL_XRAY_NEVER_INSTRUMENT
// macro used as an attribute to mark functions that must always or never be
// instrumented by XRay. Currently, this is only supported in Clang/LLVM.
//
// For reference on the LLVM XRay instrumentation, see
// http://llvm.org/docs/XRay.html.
//
// A function with the XRAY_ALWAYS_INSTRUMENT macro attribute in its declaration
// will always get the XRay instrumentation sleds. These sleds may introduce
// some binary size and runtime overhead and must be used sparingly.
//
// These attributes only take effect when the following conditions are met:
//
//   * The file/target is built in at least C++11 mode, with a Clang compiler
//     that supports XRay attributes.
//   * The file/target is built with the -fxray-instrument flag set for the
//     Clang/LLVM compiler.
//   * The function is defined in the translation unit (the compiler honors the
//     attribute in either the definition or the declaration, and must match).
//
// There are cases when, even when building with XRay instrumentation, users
// might want to control specifically which functions are instrumented for a
// particular build using special-case lists provided to the compiler. These
// special case lists are provided to Clang via the
// -fxray-always-instrument=... and -fxray-never-instrument=... flags. The
// attributes in source take precedence over these special-case lists.
//
// To disable the XRay attributes at build-time, users may define
// OTABSL_NO_XRAY_ATTRIBUTES. Do NOT define OTABSL_NO_XRAY_ATTRIBUTES on specific
// packages/targets, as this may lead to conflicting definitions of functions at
// link-time.
//
#if OTABSL_HAVE_CPP_ATTRIBUTE(clang::xray_always_instrument) && \
    !defined(OTABSL_NO_XRAY_ATTRIBUTES)
#define OTABSL_XRAY_ALWAYS_INSTRUMENT [[clang::xray_always_instrument]]
#define OTABSL_XRAY_NEVER_INSTRUMENT [[clang::xray_never_instrument]]
#if OTABSL_HAVE_CPP_ATTRIBUTE(clang::xray_log_args)
#define OTABSL_XRAY_LOG_ARGS(N) \
    [[clang::xray_always_instrument, clang::xray_log_args(N)]]
#else
#define OTABSL_XRAY_LOG_ARGS(N) [[clang::xray_always_instrument]]
#endif
#else
#define OTABSL_XRAY_ALWAYS_INSTRUMENT
#define OTABSL_XRAY_NEVER_INSTRUMENT
#define OTABSL_XRAY_LOG_ARGS(N)
#endif

// OTABSL_ATTRIBUTE_REINITIALIZES
//
// Indicates that a member function reinitializes the entire object to a known
// state, independent of the previous state of the object.
//
// The clang-tidy check bugprone-use-after-move allows member functions marked
// with this attribute to be called on objects that have been moved from;
// without the attribute, this would result in a use-after-move warning.
#if OTABSL_HAVE_CPP_ATTRIBUTE(clang::reinitializes)
#define OTABSL_ATTRIBUTE_REINITIALIZES [[clang::reinitializes]]
#else
#define OTABSL_ATTRIBUTE_REINITIALIZES
#endif

// -----------------------------------------------------------------------------
// Variable Attributes
// -----------------------------------------------------------------------------

// OTABSL_ATTRIBUTE_UNUSED
//
// Prevents the compiler from complaining about variables that appear unused.
#if OTABSL_HAVE_ATTRIBUTE(unused) || (defined(__GNUC__) && !defined(__clang__))
#undef OTABSL_ATTRIBUTE_UNUSED
#define OTABSL_ATTRIBUTE_UNUSED __attribute__((__unused__))
#else
#define OTABSL_ATTRIBUTE_UNUSED
#endif

// OTABSL_ATTRIBUTE_INITIAL_EXEC
//
// Tells the compiler to use "initial-exec" mode for a thread-local variable.
// See http://people.redhat.com/drepper/tls.pdf for the gory details.
#if OTABSL_HAVE_ATTRIBUTE(tls_model) || (defined(__GNUC__) && !defined(__clang__))
#define OTABSL_ATTRIBUTE_INITIAL_EXEC __attribute__((tls_model("initial-exec")))
#else
#define OTABSL_ATTRIBUTE_INITIAL_EXEC
#endif

// OTABSL_ATTRIBUTE_PACKED
//
// Instructs the compiler not to use natural alignment for a tagged data
// structure, but instead to reduce its alignment to 1. This attribute can
// either be applied to members of a structure or to a structure in its
// entirety. Applying this attribute (judiciously) to a structure in its
// entirety to optimize the memory footprint of very commonly-used structs is
// fine. Do not apply this attribute to a structure in its entirety if the
// purpose is to control the offsets of the members in the structure. Instead,
// apply this attribute only to structure members that need it.
//
// When applying OTABSL_ATTRIBUTE_PACKED only to specific structure members the
// natural alignment of structure members not annotated is preserved. Aligned
// member accesses are faster than non-aligned member accesses even if the
// targeted microprocessor supports non-aligned accesses.
#if OTABSL_HAVE_ATTRIBUTE(packed) || (defined(__GNUC__) && !defined(__clang__))
#define OTABSL_ATTRIBUTE_PACKED __attribute__((__packed__))
#else
#define OTABSL_ATTRIBUTE_PACKED
#endif

// OTABSL_ATTRIBUTE_FUNC_ALIGN
//
// Tells the compiler to align the function start at least to certain
// alignment boundary
#if OTABSL_HAVE_ATTRIBUTE(aligned) || (defined(__GNUC__) && !defined(__clang__))
#define OTABSL_ATTRIBUTE_FUNC_ALIGN(bytes) __attribute__((aligned(bytes)))
#else
#define OTABSL_ATTRIBUTE_FUNC_ALIGN(bytes)
#endif

// OTABSL_CONST_INIT
//
// A variable declaration annotated with the `OTABSL_CONST_INIT` attribute will
// not compile (on supported platforms) unless the variable has a constant
// initializer. This is useful for variables with static and thread storage
// duration, because it guarantees that they will not suffer from the so-called
// "static init order fiasco".  Prefer to put this attribute on the most visible
// declaration of the variable, if there's more than one, because code that
// accesses the variable can then use the attribute for optimization.
//
// Example:
//
//   class MyClass {
//    public:
//     OTABSL_CONST_INIT static MyType my_var;
//   };
//
//   MyType MyClass::my_var = MakeMyType(...);
//
// Note that this attribute is redundant if the variable is declared constexpr.
#if OTABSL_HAVE_CPP_ATTRIBUTE(clang::require_constant_initialization)
#define OTABSL_CONST_INIT [[clang::require_constant_initialization]]
#else
#define OTABSL_CONST_INIT
#endif  // OTABSL_HAVE_CPP_ATTRIBUTE(clang::require_constant_initialization)

#endif  // OTABSL_BASE_ATTRIBUTES_H_
