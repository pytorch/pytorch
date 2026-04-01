#ifndef SLANG_PLATFORM_API_H
#define SLANG_PLATFORM_API_H

#if defined(SLANG_PLATFORM_DYNAMIC)
#if defined(_MSC_VER)
#ifdef SLANG_PLATFORM_DYNAMIC_EXPORT
#define SLANG_PLATFORM_API SLANG_DLL_EXPORT
#else
#define SLANG_PLATFORM_API __declspec(dllimport)
#endif
#else
// TODO: need to consider compiler capabilities
// #     ifdef SLANG_DYNAMIC_EXPORT
#define SLANG_PLATFORM_API SLANG_DLL_EXPORT
// #     endif
#endif
#endif

#ifndef SLANG_PLATFORM_API
#define SLANG_PLATFORM_API
#endif

#endif
