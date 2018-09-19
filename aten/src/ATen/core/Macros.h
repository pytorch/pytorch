#pragma once

#include <sstream>
#include <string>

// You can use the definition AT_CORE_STATIC_WINDOWS to control whether
// or not we apply __declspec.  You will want to set this as
// -DAT_CORE_STATIC_WINDOWS=1 when compiling code which links
// against ATen/core on Windows, when ATen/core is built as a
// static library (in which case, saying the symbol is coming
// from a DLL would be incorrect).

#ifdef _WIN32
#if !defined(AT_CORE_STATIC_WINDOWS)
#define AT_CORE_EXPORT __declspec(dllexport)
#define AT_CORE_IMPORT __declspec(dllimport)
#else // !defined(AT_CORE_STATIC_WINDOWS)
#define AT_CORE_EXPORT
#define AT_CORE_IMPORT
#endif // !defined(AT_CORE_STATIC_WINDOWS)
#else  // _WIN32
#if defined(__GNUC__)
#define AT_CORE_EXPORT __attribute__((__visibility__("default")))
#else // defined(__GNUC__)
#define AT_CORE_EXPORT
#endif // defined(__GNUC__)
#define AT_CORE_IMPORT AT_CORE_EXPORT
#endif  // _WIN32

// AT_CORE_API is a macro that, depends on whether you are building the
// main library or not, resolves to either AT_CORE_EXPORT or
// AT_CORE_IMPORT.
//

// TODO: unify the controlling macros.
#if defined(CAFFE2_BUILD_MAIN_LIBS) || defined(ATen_cpu_EXPORTS) || defined(caffe2_EXPORTS)
#define AT_CORE_API AT_CORE_EXPORT
#else // defined(CAFFE2_BUILD_MAIN_LIBS) || defined(ATen_cpu_EXPORTS) || defined(caffe2_EXPORTS)
#define AT_CORE_API AT_CORE_IMPORT
#endif // defined(CAFFE2_BUILD_MAIN_LIBS) || defined(ATen_cpu_EXPORTS) || defined(caffe2_EXPORTS)

#ifdef __CUDACC__
// Designates functions callable from the host (CPU) and the device (GPU)
#define AT_HOST_DEVICE __host__ __device__
#define AT_DEVICE __device__
#define AT_HOST __host__
#else
#define AT_HOST_DEVICE
#define AT_HOST
#define AT_DEVICE
#endif

// Disable the copy and assignment operator for a class. Note that this will
// disable the usage of the class in std containers.
#define AT_DISABLE_COPY_AND_ASSIGN(classname) \
  classname(const classname&) = delete;       \
  classname& operator=(const classname&) = delete


#if defined(__ANDROID__)
#define AT_ANDROID 1
#define AT_MOBILE 1
#elif (defined(__APPLE__) &&                                            \
       (TARGET_IPHONE_SIMULATOR || TARGET_OS_SIMULATOR || TARGET_OS_IPHONE))
#define AT_IOS 1
#define AT_MOBILE 1
#elif (defined(__APPLE__) && TARGET_OS_MAC)
#define AT_IOS 1
#define AT_MOBILE 0
#else
#define AT_MOBILE 0
#endif // ANDROID / IOS / MACOS

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
