#pragma once

#include <cstdint>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include "caffe2/core/common.h"

namespace caffe2 {

class CpuId;

CAFFE2_API const CpuId& GetCpuId();

///////////////////////////////////////////////////////////////////////////////
// Implementation of CpuId that is borrowed from folly.
///////////////////////////////////////////////////////////////////////////////

/**
 * Identification of an Intel CPU.
 * Supports CPUID feature flags (EAX=1) and extended features (EAX=7, ECX=0).
 * Values from
 * http://www.intel.com/content/www/us/en/processors/processor-identification-cpuid-instruction-note.html
 */
class CpuId {
 public:
  CpuId();

#define X(name, r, bit)              \
  inline bool name() const {         \
    return ((r) & (1U << bit)) != 0; \
  }

// cpuid(1): Processor Info and Feature Bits.
#define C(name, bit) X(name, f1c_, bit)
  C(sse3, 0)
  C(pclmuldq, 1)
  C(dtes64, 2)
  C(monitor, 3)
  C(dscpl, 4)
  C(vmx, 5)
  C(smx, 6)
  C(eist, 7)
  C(tm2, 8)
  C(ssse3, 9)
  C(cnxtid, 10)
  C(fma, 12)
  C(cx16, 13)
  C(xtpr, 14)
  C(pdcm, 15)
  C(pcid, 17)
  C(dca, 18)
  C(sse41, 19)
  C(sse42, 20)
  C(x2apic, 21)
  C(movbe, 22)
  C(popcnt, 23)
  C(tscdeadline, 24)
  C(aes, 25)
  C(xsave, 26)
  C(osxsave, 27)
  C(avx, 28)
  C(f16c, 29)
  C(rdrand, 30)
#undef C

#define D(name, bit) X(name, f1d_, bit)
  D(fpu, 0)
  D(vme, 1)
  D(de, 2)
  D(pse, 3)
  D(tsc, 4)
  D(msr, 5)
  D(pae, 6)
  D(mce, 7)
  D(cx8, 8)
  D(apic, 9)
  D(sep, 11)
  D(mtrr, 12)
  D(pge, 13)
  D(mca, 14)
  D(cmov, 15)
  D(pat, 16)
  D(pse36, 17)
  D(psn, 18)
  D(clfsh, 19)
  D(ds, 21)
  D(acpi, 22)
  D(mmx, 23)
  D(fxsr, 24)
  D(sse, 25)
  D(sse2, 26)
  D(ss, 27)
  D(htt, 28)
  D(tm, 29)
  D(pbe, 31)
#undef D

// cpuid(7): Extended Features.
#define B(name, bit) X(name, f7b_, bit)
  B(bmi1, 3)
  B(hle, 4)
  B(avx2, 5)
  B(smep, 7)
  B(bmi2, 8)
  B(erms, 9)
  B(invpcid, 10)
  B(rtm, 11)
  B(mpx, 14)
  B(avx512f, 16)
  B(avx512dq, 17)
  B(rdseed, 18)
  B(adx, 19)
  B(smap, 20)
  B(avx512ifma, 21)
  B(pcommit, 22)
  B(clflushopt, 23)
  B(clwb, 24)
  B(avx512pf, 26)
  B(avx512er, 27)
  B(avx512cd, 28)
  B(sha, 29)
  B(avx512bw, 30)
  B(avx512vl, 31)
#undef B

#define E(name, bit) X(name, f7c_, bit)
  E(prefetchwt1, 0)
  E(avx512vbmi, 1)
#undef E

#undef X

 private:
  CAFFE2_API static uint32_t f1c_;
  CAFFE2_API static uint32_t f1d_;
  CAFFE2_API static uint32_t f7b_;
  CAFFE2_API static uint32_t f7c_;
};

} // namespace caffe2
