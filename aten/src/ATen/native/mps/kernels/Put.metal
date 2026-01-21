#include <c10/metal/atomic.h>
#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void put_kernel(
    device T* output [[buffer(0)]],
    constant T* source [[buffer(1)]],
    constant int64_t* indices [[buffer(2)]],
    constant int64_t& numel [[buffer(3)]],
    constant int64_t& output_numel [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= (uint)numel) {
    return;
  }
  
  int64_t idx = indices[tid];
  // Wrap negative indices
  if (idx < 0) {
    idx += output_numel;
  }
  
  // Bounds check
  if (idx >= 0 && idx < output_numel) {
    output[idx] = source[tid];
  }
}

// Accumulate kernel using AtomicType for all supported types
template <typename T>
kernel void put_accumulate_kernel(
    device c10::metal::AtomicType_t<T>* output [[buffer(0)]],
    constant T* source [[buffer(1)]],
    constant int64_t* indices [[buffer(2)]],
    constant int64_t& numel [[buffer(3)]],
    constant int64_t& output_numel [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= (uint)numel) {
    return;
  }
  
  int64_t idx = indices[tid];
  // Wrap negative indices
  if (idx < 0) {
    idx += output_numel;
  }
  
  // Bounds check
  if (idx >= 0 && idx < output_numel) {
    c10::metal::AtomicType<T>::atomic_add(&output[idx], source[tid]);
  }
}

// Non-atomic accumulate for types that don't support atomics
template <typename T>
kernel void put_accumulate_serial_kernel(
    device T* output [[buffer(0)]],
    constant T* source [[buffer(1)]],
    constant int64_t* indices [[buffer(2)]],
    constant int64_t& numel [[buffer(3)]],
    constant int64_t& output_numel [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  // This kernel runs with 1 thread only
  if (tid != 0) {
    return;
  }
  
  for (int64_t i = 0; i < numel; i++) {
    int64_t idx = indices[i];
    // Wrap negative indices
    if (idx < 0) {
      idx += output_numel;
    }
    
    // Bounds check
    if (idx >= 0 && idx < output_numel) {
      output[idx] += source[i];
    }
  }
}

#define INSTANTIATE_PUT(TYPE) \
  template [[host_name("put_" #TYPE)]] \
  kernel void put_kernel<TYPE>( \
      device TYPE*, constant TYPE*, constant int64_t*, \
      constant int64_t&, constant int64_t&, uint);

#define INSTANTIATE_PUT_ACCUMULATE(TYPE) \
  template [[host_name("put_accumulate_" #TYPE)]] \
  kernel void put_accumulate_kernel<TYPE>( \
      device c10::metal::AtomicType_t<TYPE>*, constant TYPE*, constant int64_t*, \
      constant int64_t&, constant int64_t&, uint);

#define INSTANTIATE_PUT_ACCUMULATE_SERIAL(TYPE) \
  template [[host_name("put_accumulate_serial_" #TYPE)]] \
  kernel void put_accumulate_serial_kernel<TYPE>( \
      device TYPE*, constant TYPE*, constant int64_t*, \
      constant int64_t&, constant int64_t&, uint);

// Basic types
INSTANTIATE_PUT(float)
INSTANTIATE_PUT(half)
INSTANTIATE_PUT(bfloat)
INSTANTIATE_PUT(int)
INSTANTIATE_PUT(uint)
INSTANTIATE_PUT(short)
INSTANTIATE_PUT(ushort)
INSTANTIATE_PUT(char)
INSTANTIATE_PUT(uchar)
INSTANTIATE_PUT(bool)
INSTANTIATE_PUT(long)
INSTANTIATE_PUT(ulong)

// Accumulate with AtomicType support for all numeric types
INSTANTIATE_PUT_ACCUMULATE(float)
INSTANTIATE_PUT_ACCUMULATE(half)
INSTANTIATE_PUT_ACCUMULATE(bfloat)
INSTANTIATE_PUT_ACCUMULATE(int)
INSTANTIATE_PUT_ACCUMULATE(uint)
INSTANTIATE_PUT_ACCUMULATE(short)
INSTANTIATE_PUT_ACCUMULATE(ushort)
INSTANTIATE_PUT_ACCUMULATE(char)
INSTANTIATE_PUT_ACCUMULATE(uchar)
INSTANTIATE_PUT_ACCUMULATE(long)
INSTANTIATE_PUT_ACCUMULATE(ulong)

// Serial fallback for types without native atomic support
INSTANTIATE_PUT_ACCUMULATE_SERIAL(float)
INSTANTIATE_PUT_ACCUMULATE_SERIAL(half)
INSTANTIATE_PUT_ACCUMULATE_SERIAL(bfloat)
INSTANTIATE_PUT_ACCUMULATE_SERIAL(int)
INSTANTIATE_PUT_ACCUMULATE_SERIAL(uint)
INSTANTIATE_PUT_ACCUMULATE_SERIAL(short)
INSTANTIATE_PUT_ACCUMULATE_SERIAL(ushort)
INSTANTIATE_PUT_ACCUMULATE_SERIAL(char)
INSTANTIATE_PUT_ACCUMULATE_SERIAL(uchar)
INSTANTIATE_PUT_ACCUMULATE_SERIAL(long)
INSTANTIATE_PUT_ACCUMULATE_SERIAL(ulong)
