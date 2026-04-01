// slang-rt.h
#ifndef SLANG_RT_H
#define SLANG_RT_H

#include "../core/slang-com-object.h"
#include "../core/slang-smart-pointer.h"
#include "../core/slang-string.h"

#ifdef SLANG_RT_DYNAMIC_EXPORT
#define SLANG_RT_API SLANG_DLL_EXPORT
#else
#define SLANG_RT_API
#endif

#if defined(_MSC_VER)
#define SLANG_PRELUDE_SHARED_LIB_EXPORT __declspec(dllexport)
#else
#define SLANG_PRELUDE_SHARED_LIB_EXPORT __attribute__((__visibility__("default")))
// #   define SLANG_PRELUDE_SHARED_LIB_EXPORT __attribute__ ((dllexport))
// __attribute__((__visibility__("default")))
#endif

#define SLANG_PRELUDE_EXPORT extern "C" SLANG_PRELUDE_SHARED_LIB_EXPORT

extern "C"
{
    SLANG_RT_API void SLANG_MCALL _slang_rt_abort(Slang::String errorMessage);
    SLANG_RT_API void* SLANG_MCALL _slang_rt_load_dll(Slang::String modulePath);
    SLANG_RT_API void* SLANG_MCALL
    _slang_rt_load_dll_func(void* moduleHandle, Slang::String modulePath, uint32_t argSize);
}

#endif
