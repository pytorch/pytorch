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
    size_t offset = (size_t)address & 3;
    uint32_t * address_as_ui = (uint32_t *)((char *)address - offset);
    uint32_t old = *address_as_ui;
    uint32_t shift = offset * 8;
    uint32_t old_byte;
    uint32_t newval;
    uint32_t assumed;

    do {
      assumed = old;
      old_byte = (old >> shift) & 0xff;
      // preserve size in initial cast. Casting directly to uint32_t pads 
      // negative signed values with 1's (e.g. signed -1 = unsigned ~0).
      newval = static_cast<uint8_t>(THCNumerics<T>::add(val, old_byte));
      newval = (old & ~(0x000000ff << shift)) | (newval << shift);
      old = atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);
  }
};

template<typename T>
struct AtomicAddIntegerImpl<T, 2> {
  inline __device__ void operator()(T *address, T val) {
    size_t offset = (size_t)address & 2;
    uint32_t * address_as_ui = (uint32_t *)((char *)address - offset);
    bool is_32_align = offset;
    uint32_t old = *address_as_ui;
    uint32_t old_bytes;
    uint32_t newval;
    uint32_t assumed;

    do {
      assumed = old;
      old_bytes = is_32_align ? old >> 16 : old & 0xffff;
      // preserve size in initial cast. Casting directly to uint32_t pads 
      // negative signed values with 1's (e.g. signed -1 = unsigned ~0).
      newval = static_cast<uint16_t>(THCNumerics<T>::add(val, old_bytes));
      newval = is_32_align ? (old & 0xffff) | (newval << 16) : (old & 0xffff0000) | newval;
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

static inline __device__ void gpuAtomicAdd(uint8_t *address, uint8_t val) {
  AtomicAddIntegerImpl<uint8_t, sizeof(uint8_t)>()(address, val);
}

static inline  __device__ void gpuAtomicAdd(int8_t *address, int8_t val) {
  AtomicAddIntegerImpl<int8_t, sizeof(int8_t)>()(address, val);
}

static inline  __device__ void gpuAtomicAdd(int16_t *address, int16_t val) {
  AtomicAddIntegerImpl<int16_t, sizeof(int16_t)>()(address, val);
}

static inline __device__ void gpuAtomicAdd(int32_t *address, int32_t val) {
  atomicAdd(address, val);
}

static inline __device__ void gpuAtomicAdd(int64_t *address, int64_t val) {
#ifdef __HIP_PLATFORM_HCC__
  __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
#else
  AtomicAddIntegerImpl<int64_t, sizeof(int64_t)>()(address, val);
#endif
}

static inline __device__ void gpuAtomicAdd(bool *address, bool val) {
  *address = address && val;
}

static inline  __device__ void gpuAtomicAdd(at::Half *address, at::Half val) {
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

static inline __device__ void gpuAtomicAdd(at::BFloat16 *address, at::BFloat16 val) {
    unsigned int * address_as_ui =
      (unsigned int *) ((char *)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do {
      assumed = old;
      at::BFloat16 bsum;
      bsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
      bsum = THCNumerics<at::BFloat16>::add(bsum, val);
      old = (size_t)address & 2 ? (old & 0xffff) | (bsum.x << 16) : (old & 0xffff0000) | bsum.x;
      old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600 || CUDA_VERSION < 8000)
// from CUDA C Programmic Guide
static inline __device__ void atomicAdd(double* address, double val)
#if defined(__clang__) && defined(__CUDA__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wgcc-compat"
    __attribute__((enable_if(true, "")))
#pragma GCC diagnostic pop
#endif
{
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

static inline __device__ void gpuAtomicAdd(double *address, double val) {
  atomicAdd(address, val);
}

static inline __device__ void gpuAtomicAdd(float *address, float val) {
  atomicAdd(address, val);
}

/* Note [gpuAtomicAdd vs atomicAdd]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * We are trying to standardize inside the PyTorch backend on using gpuAtomicAdd()
 * without a return. These may either be resolved through library functions or
 * implemented internally. Some extensions such as torchvision call atomicAdd()
 * directly and require non-library provided data type support. Only for these, we
 * continue to provide atomicAdd overloads. 
 */
static inline __device__ void atomicAdd(at::Half *address, at::Half val) {
  gpuAtomicAdd(address, val);
}

static inline __device__ void atomicAdd(at::BFloat16 *address, at::BFloat16 val) {
  gpuAtomicAdd(address, val);
}

static inline __device__ void atomicAdd(uint8_t *address, uint8_t val) {
  gpuAtomicAdd(address, val);
}

static inline  __device__ void atomicAdd(int8_t *address, int8_t val) {
  gpuAtomicAdd(address, val);
}

static inline  __device__ void atomicAdd(int16_t *address, int16_t val) {
  gpuAtomicAdd(address, val);
}

static inline __device__ void atomicAdd(int64_t *address, int64_t val) {
  gpuAtomicAdd(address, val);
}

static inline __device__ void atomicAdd(bool *address, bool val) {
  gpuAtomicAdd(address, val);
}

#endif // THC_ATOMICS_INC
