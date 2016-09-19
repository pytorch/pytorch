#ifndef TH_SIMD_INC
#define TH_SIMD_INC

#include <stdint.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif

// Can be found on Intel ISA Reference for CPUID
#define CPUID_AVX2_BIT 0x10       // Bit 5 of EBX for EAX=0x7
#define CPUID_AVX_BIT  0x10000000 // Bit 28 of ECX for EAX=0x1
#define CPUID_SSE_BIT  0x2000000  // bit 25 of EDX for EAX=0x1

// Helper macros for initialization
#define FUNCTION_IMPL(NAME, EXT) \
    { .function=(void *)NAME,    \
      .supportedSimdExt=EXT      \
    }

#define INIT_DISPATCH_PTR(OP)    \
  do {                           \
    int i;                       \
    for (i = 0; i < sizeof(THVector_(OP ## _DISPATCHTABLE)) / sizeof(FunctionDescription); ++i) { \
      THVector_(OP ## _DISPATCHPTR) = THVector_(OP ## _DISPATCHTABLE)[i].function;                     \
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
#else
  SIMDExtension_AVX2    = 0x1,
  SIMDExtension_AVX     = 0x2,
  SIMDExtension_SSE     = 0x4,
#endif
  SIMDExtension_DEFAULT = 0x0
};

#if defined(__NEON__)

static inline uint32_t detectHostSIMDExtensions()
{
  return SIMDExtension_NEON;
}

#else // x86

static inline void cpuid(uint32_t *eax, uint32_t *ebx, uint32_t *ecx, uint32_t *edx)
{
#ifndef _MSC_VER
  uint32_t a = *eax, b, c, d;
  asm volatile ( "cpuid\n\t"
                 : "+a"(a), "=b"(b), "=c"(c), "=d"(d) );
  *eax = a;
  *ebx = b;
  *ecx = c;
  *edx = d;
#else
  uint32_t cpuInfo[4];
  __cpuid(cpuInfo, *eax);
  *eax = cpuInfo[0];
  *ebx = cpuInfo[1];
  *ecx = cpuInfo[2];
  *edx = cpuInfo[3];
#endif
}

static inline uint32_t detectHostSIMDExtensions()
{
  uint32_t eax, ebx, ecx, edx;
  uint32_t hostSimdExts = 0x0;

  // Check for AVX2. Requires separate CPUID
  eax = 0x7;
  cpuid(&eax, &ebx, &ecx, &edx);
  if (ebx & CPUID_AVX2_BIT)
    hostSimdExts |= SIMDExtension_AVX2;

  eax = 0x1;
  cpuid(&eax, &ebx, &ecx, &edx);
  if (ecx & CPUID_AVX_BIT)
    hostSimdExts |= SIMDExtension_AVX;
  if (edx & CPUID_SSE_BIT)
    hostSimdExts |= SIMDExtension_SSE;

  return hostSimdExts;
}

#endif // end x86 SIMD extension detection code

#endif
