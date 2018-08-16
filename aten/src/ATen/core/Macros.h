#pragma once

// You can use the definition AT_CORE_STATIC_WINDOWS to control whether
// or not we apply __declspec.  You will want to set this as
// -DAT_CORE_STATIC_WINDOWS=1 when compiling code which links
// against ATen/core on Windows, when ATen/core is built as a
// static library (in which case, saying the symbol is coming
// from a DLL would be incorrect).

#ifdef _WIN32
#if !defined(AT_CORE_STATIC_WINDOWS)
// TODO: unfiy the controlling macros.
#if defined(CAFFE2_BUILD_MAIN_LIBS) || defined(ATen_cpu_EXPORTS) || defined(caffe2_EXPORTS)
#define AT_CORE_API __declspec(dllexport)
#else // defined(CAFFE2_BUILD_MAIN_LIBS) || defined(ATen_cpu_EXPORTS) || defined(caffe2_EXPORTS)
#define AT_CORE_API __declspec(dllimport)
#endif // defined(CAFFE2_BUILD_MAIN_LIBS) || defined(ATen_cpu_EXPORTS) || defined(caffe2_EXPORTS)
#else // !defined(AT_CORE_STATIC_WINDOWS)
#define AT_CORE_API
#endif // !defined(AT_CORE_STATIC_WINDOWS)
#else  // _WIN32
#if defined(__GNUC__)
#define AT_CORE_API __attribute__((__visibility__("default")))
#endif // defined(__GNUC__)
#endif  // _WIN32

// Disable the copy and assignment operator for a class. Note that this will
// disable the usage of the class in std containers.
#define AT_DISABLE_COPY_AND_ASSIGN(classname) \
  classname(const classname&) = delete;       \
  classname& operator=(const classname&) = delete
