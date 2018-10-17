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

#endif // C10_MACROS_MACROS_H_
