
#pragma once

#include <c10/util/complex.h>
#include <THC/THC.h>
#include <TH/THHalf.h>
#include <THC/THCNumerics.cuh>
#include <ATen/ATen.h>

namespace at { namespace native { namespace atomic_ops {

template <typename operation, size_t n>
struct AtomicIntegerop;

using add_op = std::integral_constant<int, 0>;
using mul_op = std::integral_constant<int, 1>;

// Integer addition functions.
template<typename T>
struct AtomicIntegerop<T, 1> {
  template <typename func_t>
  inline __device__ void operator()(T *address, T val, const func_t& func) {
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
      newval = func(val, old_byte);//static_cast<uint8_t>(THCNumerics<T>::add(val, old_byte));
      newval = (old & ~(0x000000ff << shift)) | (newval << shift);
      old = atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);
  }
};

template<typename T>
struct AtomicIntegerop<T, 2> {
  template <typename func_t>
  inline __device__ void operator()(T *address, T val, const func_t& func) {
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
      newval = func(val, old_bytes);//static_cast<uint16_t>(THCNumerics<T>::add(val, old_bytes));
      newval = is_32_align ? (old & 0xffff) | (newval << 16) : (old & 0xffff0000) | newval;
      old = atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);
  }
};

template<typename T>
struct AtomicIntegerop<T,  4> {
template <typename func_t>
inline __device__ void operator()(T *address, T val, const func_t &func) {
  uint32_t * address_as_ui = (uint32_t *) (address);
  uint32_t old = *address_as_ui;
  uint32_t newval;
  uint32_t assumed;

  do {
    assumed = old;
    newval = func(val, old);
    old = atomicCAS(address_as_ui, assumed, newval);
  } while (assumed != old);
}
};

template<typename T>
struct AtomicIntegerop<T, 8> {
template <typename func_t>
inline __device__ void operator()(T *address, T val, const func_t& func) {
  unsigned long long * address_as_ui = (unsigned long long *) (address);
  unsigned long long old = *address_as_ui;
  unsigned long long newval;
  unsigned long long assumed;

  do {
    assumed = old;
    newval = func(val, old);
    old = atomicCAS(address_as_ui, assumed, newval);
  } while (assumed != old);
}
};

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
    
template <typename operation>
struct gpuAtomic;

template <>
struct gpuAtomic<add_op> {
  inline __device__ void operator() (uint8_t * address, uint8_t val) {
    AtomicIntegerop<uint8_t, sizeof(uint8_t)>()(address, val,
                                                [](uint8_t val, uint32_t old_byte) {
                                                  return static_cast<uint8_t>(THCNumerics<uint8_t>::add(val, old_byte));
                                                });
  }

  inline __device__ void operator() (int8_t * address, int8_t val) {
    AtomicIntegerop<int8_t, sizeof(int8_t)>()(address, val,
                                              [](int8_t val, uint32_t old_byte) {
                                                return static_cast<int8_t>(THCNumerics<int8_t>::add(val, old_byte));
                                              });
  }

  inline  __device__ void operator()(int16_t * address, int16_t val) {
    AtomicIntegerop<int16_t, sizeof(int16_t)>()(address, val,
                                                [](int16_t val, uint32_t old_byte) {
                                                  return static_cast<uint16_t>(THCNumerics<int16_t>::add(val, old_byte));
                                                });
  }

  inline __device__ int32_t operator() (int32_t * address, int32_t val) {
    return atomicAdd(address, val);
  }

  inline __device__ void operator() (int64_t * address, int64_t val) {
#ifdef __HIP_PLATFORM_HCC__
    __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
#else
    AtomicIntegerop<int64_t, sizeof(int64_t)>()(address, val,
                                                [](int64_t val, unsigned long long old){
                                                  return val + old;
                                                });
#endif
  }

  inline __device__ void operator() (bool * address, bool val) {
    *address = address && val;
  }

  inline  __device__ at::Half operator() (at::Half *address, at::Half val) {
#if ((CUDA_VERSION < 10000) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
    return AtomicFPOp<at::Half>()(address, val,
                                  [](at::Half hsum, at::Half val) {
                                    return THCNumerics<at::Half>::add(hsum, val);
                                  });
#else
    return atomicAdd(reinterpret_cast<__half*>(address), val);
#endif
  }

  inline __device__ at::BFloat16 operator() (at::BFloat16 *address, at::BFloat16 val) {
    return AtomicFPOp<at::BFloat16>()(address, val,
                                      [](at::BFloat16 bsum, at::BFloat16 val) {
                                        return THCNumerics<at::BFloat16>::add(bsum, val);
                                      });
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600 || CUDA_VERSION < 8000)
  // from CUDA C Programmic Guide
  inline __device__ double operator() (double* address, double val)
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
#endif

  // default double and float types call CUDA atomicAdd.
  template <typename T>
  inline __device__ T operator() (T * address, T val) {
    return atomicAdd(address, val);
  }

  template <typename T>
  inline __device__ void operator() (c10::complex<T> * address, c10::complex<T> val) {
    operator()(&address->real_, val.real_);
    operator()(&address->imag_, val.imag_);
  }
};

template <>
struct gpuAtomic<mul_op> {
  inline __device__ void operator() (uint8_t * address, uint8_t val) {
    AtomicIntegerop<uint8_t, sizeof(uint8_t)>()(address, val,
                                                [](uint8_t val, uint32_t old_byte) {
                                                  return static_cast<uint8_t>(THCNumerics<uint8_t>::mul(val, old_byte));
                                                });
  }

  inline __device__ void operator() (int8_t * address, int8_t val) {
    AtomicIntegerop<int8_t, sizeof(int8_t)>()(address, val,
                                              [](int8_t val, uint32_t old_byte) {
                                                return static_cast<int8_t>(THCNumerics<int8_t>::mul(val, old_byte));
                                              });
  }

  inline __device__ void operator() (int16_t * address, int16_t val) {
    AtomicIntegerop<int16_t, sizeof(int16_t)>()(address, val,
                                                [](int16_t val, uint32_t old_byte) {
                                                  return static_cast<uint16_t>(THCNumerics<int16_t>::mul(val, old_byte));
                                                });    
  }

  inline __device__ void operator() (int32_t * address, int32_t val) {
    AtomicIntegerop<int32_t, sizeof(int32_t)>()(address, val,
                                                [](int32_t val, uint32_t old) {
                                                  return val * (int32_t)old;
                                                });
  }

  inline __device__ void operator() (int64_t * address, int64_t val) {
    AtomicIntegerop<int64_t, sizeof(int64_t)>()(address, val,
                                                [](int64_t val, unsigned long long old) {
                                                  return val * (int64_t)old;
                                                });    
  }

  inline __device__ at::Half operator() (at::Half * address, at::Half val) {
    return AtomicFPOp<at::Half>()(address, val,
                                  [](at::Half bsum, at::Half val) {
                                    return THCNumerics<at::Half>::mul(bsum, val);
                                  });
  }

  inline __device__ at::BFloat16 operator() (at::BFloat16 * address, at::BFloat16 val) {
    return AtomicFPOp<at::BFloat16>()(address, val,
                                      [](at::BFloat16 bsum, at::BFloat16 val) {
                                        return THCNumerics<at::BFloat16>::mul(bsum, val);
                                      });    
  }

  inline __device__ void operator() (bool * address, bool val) {
    *address = address && val;
  }
    
  inline __device__ double operator() (double * address, double val) {
    return AtomicFPOp<double>()(address, val,
                         [](double val, unsigned long long int assumed) {
                           return __double_as_longlong(val * __longlong_as_double(assumed));
                         });
  }

  inline __device__ float operator() (float * address, float val) {
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
  }

  template <typename T>
  inline __device__ void operator() (c10::complex<T> * address, c10::complex<T> val) {
    operator()(&address->real_, val.real_);
    operator()(&address->imag_, val.imag_);
  }
};

}}}                             // namespace at::native::atomic_op
