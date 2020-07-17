#pragma once

#include <c10/util/complex.h>
#include <THC/THC.h>
#include <TH/THHalf.h>
#include <THC/THCNumerics.cuh>
#include <ATen/ATen.h>

template <typename operation, size_t n>
struct AtomicAddIntegerImplNew;

using add_op = std::integral_constant<int, 0>;
using sub_op = std::integral_constant<int, 1>;

// Integer addition functions.
template<typename T>
struct AtomicAddIntegerImplNew<T, 1> {
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
struct AtomicAddIntegerImplNew<T, 2> {
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
struct AtomicAddIntegerImplNew<T, 4> {
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
struct AtomicAddIntegerImplNew<T, 8> {
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

template <typename operation>
struct gpuAtomic;

template <>
struct gpuAtomic<add_op> {
  inline __device__ void operator() (uint8_t * address, uint8_t val) {
    AtomicAddIntegerImplNew<uint8_t, sizeof(uint8_t)>()(address, val);
  }

  inline __device__ void operator() (int8_t * address, int8_t val) {
    AtomicAddIntegerImplNew<int8_t, sizeof(int8_t)>()(address, val);
  }

  inline  __device__ void operator()(int16_t * address, int16_t val) {
    AtomicAddIntegerImplNew<int16_t, sizeof(int16_t)>()(address, val);
  }

  inline __device__ int32_t operator() (int32_t * address, int32_t val) {
    return atomicAdd(address, val);
  }

  inline __device__ void operator() (int64_t * address, int64_t val) {
#ifdef __HIP_PLATFORM_HCC__
    __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
#else
    AtomicAddIntegerImplNew<int64_t, sizeof(int64_t)>()(address, val);
#endif
  }

  inline __device__ void operator() (bool * address, bool val) {
    *address = address && val;
  }

  inline  __device__ at::Half operator() (at::Half *address, at::Half val) {
#if ((CUDA_VERSION < 10000) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
    unsigned int * address_as_ui =
      (unsigned int *) ((char *)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    at::Half hsum;
    do {
      assumed = old;
      hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
      hsum = THCNumerics<at::Half>::add(hsum, val);
      old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16) : (old & 0xffff0000) | hsum.x;
      old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    return hsum;
#else
    return atomicAdd(reinterpret_cast<__half*>(address), val);
#endif
  }

  inline __device__ at::BFloat16 operator() (at::BFloat16 *address, at::BFloat16 val) {
    unsigned int * address_as_ui =
      (unsigned int *) ((char *)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    at::BFloat16 bsum;
    do {
      assumed = old;
      bsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
      bsum = THCNumerics<at::BFloat16>::add(bsum, val);
      old = (size_t)address & 2 ? (old & 0xffff) | (bsum.x << 16) : (old & 0xffff0000) | bsum.x;
      old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
    bsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    return bsum.x;
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600 || CUDA_VERSION < 8000)
  // from CUDA C Programmic Guide
  inline __device__ void operator() (double* address, double val)
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
    return __longlong_as_double(old);
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

template<>
struct gpuAtomic<sub_op> {
  template <typename T>
  inline __device__ void operator() (T * address, T val) {
  }
};

// template <>
// struct gpuAtomic<add_op> {
//   template <typename T>
//   inline void operator() (T * address, T val) {
    
//   }
// };

// template <>
// struct gpuAtomic<sub_op> {
//   inline void operator() (uint8_t * address, uint8_t val) {

//   }
// };



