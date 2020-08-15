#ifndef TH_SIMD_INC
#define TH_SIMD_INC

#include <stdint.h>
#include <stdlib.h>
#include <ATen/native/DispatchStub.h>

// Helper macros for initialization
#define FUNCTION_IMPL(NAME, EXT) \
    { (void *)NAME,    \
      EXT      \
    }

#define INIT_DISPATCH_PTR(OP)    \
  do {                           \
    size_t i;                       \
    for (i = 0; i < sizeof(THVector_(OP ## _DISPATCHTABLE)) / sizeof(FunctionDescription); ++i) { \
      THVector_(OP ## _DISPATCHPTR) = reinterpret_cast<decltype(THVector_(OP ## _DISPATCHPTR))>(THVector_(OP ## _DISPATCHTABLE)[i].function);                     \
      if (THVector_(OP ## _DISPATCHTABLE)[i].supportedSimdExt & hostSimdExts) {                       \
        break;                                                                                     \
      }                                                                                            \
    }                                                                                              \
  } while(0)


typedef struct FunctionDescription
{
  void *function;
  uint32_t supportedSimdExt;
} FunctionDescription;


enum SIMDExtensions
{
#if defined(__NEON__)
  SIMDExtension_NEON    = 0x1,
#elif defined(__PPC64__)
  SIMDExtension_VSX     = 0x1,
#else
  SIMDExtension_AVX2    = 0x1,
  SIMDExtension_AVX     = 0x2,
  SIMDExtension_SSE     = 0x4,
#endif
  SIMDExtension_DEFAULT = 0x0
};


#if defined(__arm__) || defined(__aarch64__) // incl. armel, armhf, arm64

 #if defined(__NEON__)

static inline uint32_t detectHostSIMDExtensions()
{
  return SIMDExtension_NEON;
}

 #else //ARM without NEON

static inline uint32_t detectHostSIMDExtensions()
{
  return SIMDExtension_DEFAULT;
}

 #endif

#elif defined(__s390x__)

static inline uint32_t detectHostSIMDExtensions()
{
  return SIMDExtension_DEFAULT;
}

#elif defined(__PPC64__)

 #if defined(__VSX__)

static inline uint32_t detectHostSIMDExtensions()
{
  uint32_t hostSimdExts = SIMDExtension_DEFAULT;
  char *evar;

  evar = getenv("TH_NO_VSX");
  if (evar == NULL || strncmp(evar, "1", 1) != 0)
    hostSimdExts = SIMDExtension_VSX;
  return hostSimdExts;
}

 #else //PPC64 without VSX

static inline uint32_t detectHostSIMDExtensions()
{
  return SIMDExtension_DEFAULT;
}

 #endif
 
#elif defined(__EMSCRIPTEN__)

static inline uint32_t detectHostSIMDExtensions()
{
  return SIMDExtension_DEFAULT;
}

#else   // x86

static inline uint32_t detectHostSIMDExtensions()
{
  using at::native::CPUCapability;
  switch (at::native::get_cpu_capability()) {
  case CPUCapability::AVX2:
    return SIMDExtension_AVX2 | SIMDExtension_AVX | SIMDExtension_SSE;
  case CPUCapability::AVX:
    return SIMDExtension_AVX | SIMDExtension_SSE;
  default:
    return SIMDExtension_SSE;
  }
}

#endif // end SIMD extension detection code

#endif
