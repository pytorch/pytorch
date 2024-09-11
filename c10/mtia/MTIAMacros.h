#pragma once

// See c10/macros/Export.h for a detailed explanation of what the function
// of these macros are.  We need one set of macros for every separate library
// we build.

#if defined(__GNUC__)
#define C10_MTIA_EXPORT __attribute__((__visibility__("default")))
#else // defined(__GNUC__)
#define C10_MTIA_EXPORT
#endif // defined(__GNUC__)
#define C10_MTIA_IMPORT C10_MTIA_EXPORT

// This one is being used by libc10_mtia.so
#ifdef C10_MTIA_BUILD_MAIN_LIB
#define C10_MTIA_API C10_MTIA_EXPORT
#else
#define C10_MTIA_API C10_MTIA_IMPORT
#endif
