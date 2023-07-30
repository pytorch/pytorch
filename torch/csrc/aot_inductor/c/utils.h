#pragma once

#include <assert.h>

#ifdef __GNUC__
#define AOT_INDUCTOR_EXPORT \
  __attribute__((__visibility__("default"))) __attribute__((used))
#else // !__GNUC__
#ifdef _WIN32
#define AOT_INDUCTOR_EXPORT __declspec(dllexport)
#else // !_WIN32
#define AOT_INDUCTOR_EXPORT
#endif // _WIN32
#endif // __GNUC__

#define AOT_INDUCTOR_CHECK(cond, msg) \
  { assert((cond) && (msg)); }

typedef void* AOTInductorTensorHandle;
typedef void* AtenTensorHandle;
