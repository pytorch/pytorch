#ifndef THC_ATOMICS_INC
#define THC_ATOMICS_INC

#include <THC/THC.h>
#include <TH/THHalf.h>
#include <THC/THCNumerics.cuh>
#include <ATen/ATen.h>

template <typename T, size_t n>
struct AtomicAddIntegerImpl;

template<typename T>
struct AtomicAddIntegerImpl<T, 1> {
  inline __device__ void operator()(T *address, T val) {
    uint32_t * address_as_ui =
        (uint32_t *) (address - ((size_t)address & 3));
    uint32_t old = *address_as_ui;
    uint32_t shift = (((size_t)address & 3) * 8);
    uint32_t sum;
    uint32_t assumed;

    do {
      assumed = old;
      sum = val + T((old >> shift) & 0xff);
      old = (old & ~(0x000000ff << shift)) | (sum << shift);
      old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
  }
};

template<typename T>
struct AtomicAddIntegerImpl<T, 2> {
  inline __device__ void operator()(T *address, T val) {
    uint32_t * address_as_ui =
        (uint32_t *) ((char *)address - ((size_t)address & 2));
    uint32_t old = *address_as_ui;
    uint32_t sum;
    uint32_t newval;
    uint32_t assumed;

    do {
      assumed = old;
      sum = val + (size_t)address & 2 ? T(old >> 16) : T(old & 0xffff);
      newval = (size_t)address & 2 ? (old & 0xffff) | (sum << 16) : (old & 0xffff0000) | sum;
      old = atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);
  }
};

template<typename T>
struct AtomicAddIntegerImpl<T, 4> {
  inline __device__ void operator()(T *address, T val) {
    uint32_t * address_as_ui = (uint32_t *) (address);
    uint32_t old = *address_as_ui;
    uint32_t newval;
    uint32_t assumed;

    do {
      assumed = old;
      newval = val +  (T)old;
      old = atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);
  }
};

template<typename T>
struct AtomicAddIntegerImpl<T, 8> {
  inline __device__ void operator()(T *address, T val) {
    unsigned long long * address_as_ui = (unsigned long long *) (address);
    unsigned long long old = *address_as_ui;
    unsigned long long newval;
    unsigned long long assumed;

    do {
      assumed = old;
      newval = val +  (T)old;
      old = atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);
  }
};

static inline __device__ void atomicAdd(uint8_t *address, uint8_t val) {
  AtomicAddIntegerImpl<uint8_t, sizeof(uint8_t)>()(address, val);
}

static inline  __device__ void atomicAdd(int8_t *address, int8_t val) {
  AtomicAddIntegerImpl<int8_t, sizeof(int8_t)>()(address, val);
}

static inline  __device__ void atomicAdd(int16_t *address, int16_t val) {
  AtomicAddIntegerImpl<int16_t, sizeof(int16_t)>()(address, val);
}

static inline __device__ void atomicAdd(int64_t *address, int64_t val) {
  AtomicAddIntegerImpl<int64_t, sizeof(int64_t)>()(address, val);
}

static inline  __device__ void atomicAdd(at::Half *address, at::Half val) {
  #if ((CUDA_VERSION < 10000) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
    unsigned int * address_as_ui =
      (unsigned int *) ((char *)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do {
      assumed = old;
      at::Half hsum;
      hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
      hsum = THCNumerics<at::Half>::add(hsum, val);
      old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16) : (old & 0xffff0000) | hsum.x;
      old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
  #else
    atomicAdd(reinterpret_cast<__half*>(address), val);
  #endif

}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600 || CUDA_VERSION < 8000)
// from CUDA C Programmic Guide
static inline  __device__  void atomicAdd(double *address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                    __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
} while (assumed != old);
}
#elif !defined(__CUDA_ARCH__) && (CUDA_VERSION < 8000) || defined(__HIP_PLATFORM_HCC__)

/* Note [hip-clang differences to hcc]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * The upcoming hip-clang compiler for ROCm differs from hcc in a few details.
 * It exports the __HIP__ macro, we can hence differentiate between hcc and
 * hip-clang. In the below, hcc only received support for atomicAdd with double
 * typing after work week 18312. hip-clang had support from the first version.
 * In general, the code-visible differences between hip-clang and hcc will be
 * minimal.
 */

#if defined(__HIP_PLATFORM_HCC__) && __hcc_workweek__ < 18312 && !__HIP__
  // This needs to be defined for the host side pass
  static inline  __device__  void atomicAdd(double *address, double val) { }
#endif
#endif

#endif // THC_ATOMICS_INC
