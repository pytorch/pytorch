#pragma once

#ifndef C10_USING_CUSTOM_GENERATED_MACROS
#include <c10/xpu/impl/xpu_cmake_macros.h>
#endif

// See c10/macros/Export.h for a detailed explanation of what the function
// of these macros are.  We need one set of macros for every separate library
// we build.

#ifdef _WIN32
#if defined(C10_XPU_BUILD_SHARED_LIBS)
#define C10_XPU_EXPORT __declspec(dllexport)
#define C10_XPU_IMPORT __declspec(dllimport)
#else
#define C10_XPU_EXPORT
#define C10_XPU_IMPORT
#endif
#else // _WIN32
#if defined(__GNUC__)
#define C10_XPU_EXPORT __attribute__((__visibility__("default")))
#else // defined(__GNUC__)
#define C10_XPU_EXPORT
#endif // defined(__GNUC__)
#define C10_XPU_IMPORT C10_XPU_EXPORT
#endif // _WIN32

// This one is being used by libc10_xpu.so
#ifdef C10_XPU_BUILD_MAIN_LIB
#define C10_XPU_API C10_XPU_EXPORT
#else
#define C10_XPU_API C10_XPU_IMPORT
#endif
