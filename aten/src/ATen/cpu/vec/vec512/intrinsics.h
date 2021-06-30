#pragma once
#if (defined(__clang__) || defined(__GNUC__)) && (defined(__x86_64__) || defined(__i386__))
/* GCC or Clang, targeting x86/x86-64 */
#include <x86intrin.h>
#elif defined(_MSC_VER)
/* Microsoft C/C++-compatible compiler */
#include <intrin.h>
#endif
