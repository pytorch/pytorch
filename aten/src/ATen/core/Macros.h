#pragma once

// You can use the definition AT_CORE_STATIC_WINDOWS to control whether
// or not we apply __declspec.  You will want to set this as
// -DAT_CORE_STATIC_WINDOWS=1 when compiling code which links
// against ATen/core on Windows, when ATen/core is built as a
// static library (in which case, saying the symbol is coming
// from a DLL would be incorrect).

#ifdef _WIN32
#if !defined(AT_CORE_STATIC_WINDOWS)
#if defined(ATen_cpu_EXPORTS) || defined(caffe2_EXPORTS)
#define AT_CORE_API __declspec(dllexport)
#else
#define AT_CORE_API __declspec(dllimport)
#endif
#else
#define AT_CORE_API
#endif
#else
#define AT_CORE_API
#endif

// Disable the copy and assignment operator for a class. Note that this will
// disable the usage of the class in std containers.
#define AT_DISABLE_COPY_AND_ASSIGN(classname) \
  classname(const classname&) = delete;       \
  classname& operator=(const classname&) = delete
