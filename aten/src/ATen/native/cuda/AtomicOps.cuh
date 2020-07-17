#pragma once

#include <c10/util/complex.h>
#include <THC/THC.h>
#include <TH/THHalf.h>
#include <THC/THCNumerics.cuh>
#include <ATen/ATen.h>

template <typename operation, size_t n>
struct AtomicAddIntegerImpl;

using add_op = std::integral_constant<int, 0>;
using sub_op = std::integral_constant<int, 1>;

template <typename operation, size_t n>

template <>
struct AtomicAddIntegerImpl<typename operation, 1> {
  template <typename T>
  inline __device__ void operator() (T *address, T val) {
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
