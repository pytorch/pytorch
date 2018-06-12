#pragma once

/**
 * Macro for marking functions as having public visibility.
 * Ported from folly/CPortability.h
 */
#ifndef __GNUC_PREREQ
#if defined __GNUC__ && defined __GNUC_MINOR__
#define __GNUC_PREREQ(maj, min) \
  ((__GNUC__ << 16) + __GNUC_MINOR__ >= ((maj) << 16) + (min))
#else
#define __GNUC_PREREQ(maj, min) 0
#endif
#endif

// Defines AT_EXPORT and AT_IMPORT. On Windows, this corresponds to
// different declarations (dllexport and dllimport). On Linux/Mac, it just
// resolves to the same "default visibility" setting.
#if defined(_MSC_VER)
#if defined(CAFFE2_BUILD_SHARED_LIBS) || defined(ATen_cpu_EXPORTS)
#define AT_EXPORT __declspec(dllexport)
#define AT_IMPORT __declspec(dllimport)
#else
#define AT_EXPORT
#define AT_IMPORT
#endif
#else
#if defined(__GNUC__)
#if __GNUC_PREREQ(4, 9)
#define AT_EXPORT [[gnu::visibility("default")]]
#else
#define AT_EXPORT __attribute__((__visibility__("default")))
#endif
#else
#define AT_EXPORT
#endif
#define AT_IMPORT AT_EXPORT
#endif

// AT_API is a macro that, depends on whether you are building the
// main library or not, resolves to either AT_EXPORT or
// AT_IMPORT.
//
// it is defined as AT_EXPORT to fix a Windows global-variable-in-dll
// issue, and for anyone dependent on ATen it will be defined as
// AT_IMPORT.

#if defined(CAFFE2_BUILD_MAIN_LIB) || defined(ATen_cpu_EXPORTS) || defined(caffe2_EXPORTS)
#define AT_API AT_EXPORT
#else
#define AT_API AT_IMPORT
#endif
