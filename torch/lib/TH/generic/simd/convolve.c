#if defined(__AVX__)

#ifdef _MSC_VER
#include <intrin.h>

static __inline int __get_cpuid (unsigned int __level, unsigned int *__eax,
                                 unsigned int *__ebx, unsigned int *__ecx,
                                 unsigned int *__edx) {
  unsigned int cpui[4];
  __cpuid(cpui, __level);
  *__eax = cpui[0]; *__ebx = cpui[1]; *__ecx = cpui[2]; *__edx = cpui[3];
  return 1;
}

static void xgetbv(unsigned int op, unsigned int* eax, unsigned int* edx) {
  *eax = 0; *edx = 0;
  if (op == 0)
      *eax = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
}

#else

#if __i386__
#define __cpuid(__level, __eax, __ebx, __ecx, __edx) \
__asm("  pushl  %%ebx\n" \
"  cpuid\n" \
"  mov    %%ebx,%1\n" \
"  popl   %%ebx" \
: "=a"(__eax), "=r" (__ebx), "=c"(__ecx), "=d"(__edx) \
: "0"(__level))
#else
#define __cpuid(__level, __eax, __ebx, __ecx, __edx) \
__asm("cpuid" : "=a"(__eax), "=b" (__ebx), "=c"(__ecx), "=d"(__edx) \
: "0"(__level))
#endif

static __inline int __get_cpuid (unsigned int __level, unsigned int *__eax,
                                 unsigned int *__ebx, unsigned int *__ecx,
                                 unsigned int *__edx) {
  __cpuid(__level, *__eax, *__ebx, *__ecx, *__edx);
  return 1;
}

static void xgetbv(unsigned int op, unsigned int* eax, unsigned int* edx) {
  __asm__ __volatile__
  (".byte 0x0f, 0x01, 0xd0": "=a" (*eax), "=d" (*edx) : "c" (op) : "cc");
}

#endif

enum ECPUFeature
{
  kCPUFeature_SSE = 0x01,
  kCPUFeature_SSE2 = 0x02,
  kCPUFeature_SSE3 = 0x04,
  kCPUFeature_SSE3_S = 0x08,
  kCPUFeature_SSE4_1 = 0x10,
  kCPUFeature_SSE4_2 = 0x20,
  kCPUFeature_AVX = 0x40
};

static unsigned int checkCPUFeatures() {
  unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
  unsigned int features = 0;
  __get_cpuid(1, &eax, &ebx, &ecx, &edx);
  if( (edx & (1 << 25)) != 0 ) {
    features |= kCPUFeature_SSE;
  }
  if( (edx & (1 << 26)) != 0 ) {
    features |= kCPUFeature_SSE2;
  }
  if( (ecx & (1 << 0)) != 0 ) {
    features |= kCPUFeature_SSE3;
  }
  if( (ecx & (1 << 9)) != 0 ) {
    features |= kCPUFeature_SSE3_S;
  }
  if( (ecx & (1 << 19)) != 0 ) {
    features |= kCPUFeature_SSE4_1;
  }
  if( (ecx & (1 << 20)) != 0 ) {
    features |= kCPUFeature_SSE4_2;
  }
  if( (ecx & (1 << 28)) != 0 && (ecx & (1 << 27)) != 0 && (ecx & (1 << 26)) != 0 ) {
    xgetbv(0, &eax, &edx);
    if( (eax & 6) == 6 ) {
      features |= kCPUFeature_AVX;
    }
  }
  return features;
}

#include <stdio.h>

static int haveCPUFeature(unsigned int feature) {
  static unsigned int sCPUFeatures = 0;
  static int sDetectedCPUFeatures = 0;
  if (!sDetectedCPUFeatures) {
    sDetectedCPUFeatures = 1;
    sCPUFeatures = checkCPUFeatures();
    if ((sCPUFeatures & kCPUFeature_AVX) != 0) {
      printf("torch running avx\n");
    } else {
      printf("torch running sse \n");
    }
  }
  return (sCPUFeatures & feature) != 0;
}

#endif

#include <stdint.h>

void convolve_5x5_sse(float* output, float* input, float* kernel, int64_t outRows, int64_t outCols, int64_t outStride, int64_t inCols);
void convolve_5x5_avx(float* output, float* input, float* kernel, int64_t outRows, int64_t outCols, int64_t outStride, int64_t inCols);

void convolve_5x5(float* output, float* input, float* kernel, int64_t outRows, int64_t outCols, int64_t inCols) {
#if defined(__AVX__)
  int avx = haveCPUFeature(kCPUFeature_AVX);
  if (avx)
  {
    convolve_5x5_avx(output, input, kernel, outRows, outCols, outCols, inCols);
  }
  else
#endif
  {
    convolve_5x5_sse(output, input, kernel, outRows, outCols, outCols, inCols);
  }
}
