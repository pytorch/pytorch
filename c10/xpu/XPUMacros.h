#pragma once

// See c10/macros/Export.h for a detailed explanation of what the function
// of these macros are.  We need one set of macros for every separate library
// we build.

#if defined(__GNUC__)
#define C10_XPU_EXPORT __attribute__((__visibility__("default")))
#else // defined(__GNUC__)
#define C10_XPU_EXPORT
#endif // defined(__GNUC__)
#define C10_XPU_IMPORT C10_XPU_EXPORT

// This one is being used by libc10_xpu.so
#ifdef C10_XPU_BUILD_MAIN_LIB
#define C10_XPU_API C10_XPU_EXPORT
#else
#define C10_XPU_API C10_XPU_IMPORT
#endif
