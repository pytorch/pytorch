#ifndef SLANG_CPP_HOST_PRELUDE_H
#define SLANG_CPP_HOST_PRELUDE_H

#include <cmath>
#include <cstdio>
#include <cstring>

#define SLANG_COM_PTR_ENABLE_REF_OPERATOR 1

#include "../source/slang-rt/slang-rt.h"
#include "slang-com-ptr.h"
#include "slang-cpp-types.h"

#ifdef SLANG_LLVM
#include "slang-llvm.h"
#else // SLANG_LLVM
#if SLANG_GCC_FAMILY && __GNUC__ < 6
#include <cmath>
#define SLANG_PRELUDE_STD std::
#else
#include <math.h>
#define SLANG_PRELUDE_STD
#endif

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#endif // SLANG_LLVM

#if defined(_MSC_VER)
#define SLANG_PRELUDE_SHARED_LIB_EXPORT __declspec(dllexport)
#else
#define SLANG_PRELUDE_SHARED_LIB_EXPORT __attribute__((__visibility__("default")))
// #   define SLANG_PRELUDE_SHARED_LIB_EXPORT __attribute__ ((dllexport))
// __attribute__((__visibility__("default")))
#endif

#ifdef __cplusplus
#define SLANG_PRELUDE_EXTERN_C extern "C"
#define SLANG_PRELUDE_EXTERN_C_START \
    extern "C"                       \
    {
#define SLANG_PRELUDE_EXTERN_C_END }
#else
#define SLANG_PRELUDE_EXTERN_C
#define SLANG_PRELUDE_EXTERN_C_START
#define SLANG_PRELUDE_EXTERN_C_END
#endif

#include "slang-cpp-scalar-intrinsics.h"

using namespace Slang;

template<typename TResult, typename... Args>
using Slang_FuncType = TResult(SLANG_MCALL*)(Args...);

#endif
