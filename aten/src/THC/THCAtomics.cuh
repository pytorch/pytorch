#ifndef THC_ATOMICS_INC
#define THC_ATOMICS_INC

#include <c10/util/complex.h>
#include <THC/THC.h>
#include <TH/THHalf.h>
#include <THC/THCNumerics.cuh>
#include <ATen/ATen.h>

template <typename T>
struct AtomicFPOp;

template <>
struct AtomicFPOp<at::Half> {
  template <typename func_t>
  inline __device__ at::Half operator() (at::Half *address, at::Half val, const func_t& func) {
    unsigned int * address_as_ui =
      (unsigned int *) ((char *)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    at::Half hsum;
    do {
      assumed = old;
      hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
      hsum = func(hsum, val);
      old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16) : (old & 0xffff0000) | hsum.x;
      old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    return hsum;
  }
};

template <>
struct AtomicFPOp<at::BFloat16> {
  template <typename func_t>
  inline __device__ at::BFloat16 operator() (at::BFloat16 *address, at::BFloat16 val, const func_t& func) {
    unsigned int * address_as_ui =
      (unsigned int *) ((char *)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    at::BFloat16 bsum;
    do {
      assumed = old;
      bsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
      bsum = func(bsum, val);
      old = (size_t)address & 2 ? (old & 0xffff) | (bsum.x << 16) : (old & 0xffff0000) | bsum.x;
      old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
    bsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    return bsum.x;
  }
};

template <>
struct AtomicFPOp<double> {
  template <typename func_t>
  inline __device__ double operator() (double * address, double val, const func_t& func) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;

    do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, func(val, assumed));
      // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
  }
};

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

static inline __device__ int32_t gpuAtomicAdd(int32_t *address, int32_t val) {
  return atomicAdd(address, val);
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

static inline  __device__ at::Half gpuAtomicAdd(at::Half *address, at::Half val) {
#if ((CUDA_VERSION < 10000) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
  return AtomicFPOp<at::Half>()(address, val,
                                [](at::Half hsum, at::Half val) {
                                  return THCNumerics<at::Half>::add(hsum, val);
                                });
#else
  return atomicAdd(reinterpret_cast<__half*>(address), val);
#endif
}

static inline __device__ at::BFloat16 gpuAtomicAdd(at::BFloat16 *address, at::BFloat16 val) {
  return AtomicFPOp<at::BFloat16>()(address, val,
                                    [](at::BFloat16 bsum, at::BFloat16 val) {
                                      return THCNumerics<at::BFloat16>::add(bsum, val);
                                    });
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600 || CUDA_VERSION < 8000)
// from CUDA C Programmic Guide
static inline __device__ double atomicAdd(double* address, double val)
#if defined(__clang__) && defined(__CUDA__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wgcc-compat"
    __attribute__((enable_if(true, "")))
#pragma GCC diagnostic pop
#endif
{

  return AtomicFPOp<double>()(address, val,
                              [](double val, unsigned long long int assumed) {
                                return __double_as_longlong(val + __longlong_as_double(assumed));
                              });
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
  static inline  __device__  double atomicAdd(double *address, double val) { }
#endif
#endif

static inline __device__ double gpuAtomicAdd(double *address, double val) {
  return atomicAdd(address, val);
}

static inline __device__ float gpuAtomicAdd(float *address, float val) {
  return atomicAdd(address, val);
}

template<typename T>
static inline __device__ void gpuAtomicAdd(c10::complex<T> *address, c10::complex<T> val) {
  gpuAtomicAdd(&address->real_, val.real_);
  gpuAtomicAdd(&address->imag_, val.imag_);
}

/* Note [gpuAtomicAdd vs atomicAdd]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Some extensions such as torchvision call atomicAdd()
 * directly and require non-library provided data type support. Only for these, we
 * continue to provide atomicAdd overloads.
 */
static inline __device__ at::Half atomicAdd(at::Half *address, at::Half val) {
  return gpuAtomicAdd(address, val);
}

static inline __device__ at::BFloat16 atomicAdd(at::BFloat16 *address, at::BFloat16 val) {
  return gpuAtomicAdd(address, val);
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

// Atomic multiplication implementation.

inline __device__ at::Half gpuAtomicMul(at::Half * address, at::Half val) {
  return AtomicFPOp<at::Half>()(address, val,
                                [](at::Half bsum, at::Half val) {
                                  return THCNumerics<at::Half>::mul(bsum, val);
                                });
}

inline __device__ at::BFloat16 gpuAtomicMul(at::BFloat16 * address, at::BFloat16 val) {
  return AtomicFPOp<at::BFloat16>()(address, val,
                                    [](at::BFloat16 bsum, at::BFloat16 val) {
                                      return THCNumerics<at::BFloat16>::mul(bsum, val);
                                    });
}

inline __device__ double gpuAtomicMul(double * address, double val) {
  return AtomicFPOp<double>()(address, val,
                              [](double val, unsigned long long int assumed) {
                                return __double_as_longlong(val * __longlong_as_double(assumed));
                              });
}

// Dont use a templated function for this since the addition function defaults to the CUDA built-in.
inline __device__ float gpuAtomicMul (float * address, float val) {
  unsigned int* address_as_ull = (unsigned int*)address;
  unsigned int old = *address_as_ull;
  unsigned int assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __float_as_int(val *
                                   __int_as_float(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __int_as_float(old);
}
#endif // THC_ATOMICS_INC
