#pragma once

#include <sstream>
#include <string>

// You can use the definition AT_CORE_STATIC_WINDOWS to control whether
// or not we apply __declspec.  You will want to set this as
// -DAT_CORE_STATIC_WINDOWS=1 when compiling code which links
// against ATen/core on Windows, when ATen/core is built as a
// static library (in which case, saying the symbol is coming
// from a DLL would be incorrect).

#define AT_CORE_EXPORT
#define AT_CORE_IMPORT

#ifdef _WIN32
  #ifndef AT_CORE_STATIC_WINDOWS
    #undef AT_CORE_EXPORT
    #undef AT_CORE_IMPORT
    #define AT_CORE_EXPORT __declspec(dllexport)
    #define AT_CORE_IMPORT __declspec(dllimport)
  #endif // !defined(AT_CORE_STATIC_WINDOWS)
#else  // _WIN32
  #if defined(__GNUC__) || defined(__llvm__)
    #undef AT_CORE_EXPORT
    #undef AT_CORE_IMPORT
    #define AT_CORE_EXPORT __attribute__((__visibility__("default")))
    #define AT_CORE_IMPORT AT_CORE_EXPORT
  #endif // defined(__GNUC__) || defined(__llvm__)
#endif  // _WIN32

#if defined(CAFFE2_BUILD_MAIN_LIBS) || defined(ATen_cpu_EXPORTS) || defined(caffe2_EXPORTS)
  #define AT_CORE_API AT_CORE_EXPORT
#else // defined(CAFFE2_BUILD_MAIN_LIBS) || defined(ATen_cpu_EXPORTS) || defined(caffe2_EXPORTS)
  #define AT_CORE_API AT_CORE_IMPORT
#endif // defined(CAFFE2_BUILD_MAIN_LIBS) || defined(ATen_cpu_EXPORTS) || defined(caffe2_EXPORTS)

// Disable the copy and assignment operator for a class. Note that this will
// disable the usage of the class in std containers.
#define AT_DISABLE_COPY_AND_ASSIGN(classname) \
  classname(const classname&) = delete;       \
  classname& operator=(const classname&) = delete

namespace at {
inline int stoi(const std::string& str) {
#if defined(__ANDROID__)
  std::stringstream ss;
  int n = 0;
  ss << str;
  ss >> n;
  return n;
#else
  return std::stoi(str);
#endif // defined(__ANDROID__)
}
} // namespace at
