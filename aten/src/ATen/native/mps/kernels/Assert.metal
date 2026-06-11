// `atomic.h` (alphabetically before `error.h`) pulls in <metal_atomic>, which
// provides the ::metal::atomic that error.h's ErrorMessages needs. Including it
// here keeps the kernel compiling under clang-format's SortIncludes.
#include <c10/metal/atomic.h>
#include <c10/metal/error.h>
#include <metal_stdlib>

using namespace metal;

// Only the element width matters, so we dispatch by size (1/2/4/8 bytes) rather
// than per-dtype: the assert fires when the element's bits are all zero. For
// integer/bool inputs this is exactly `value == 0`. For floating point it tests
// the bit pattern, so `-0.0` (nonzero bits) is treated as true and does not
// fire; the practical callers pass boolean/integer condition tensors.
template <typename T>
kernel void assert_async(
    device const T* input [[buffer(0)]],
    constant char* msg [[buffer(1)]],
    device c10::metal::ErrorMessages* error_buf [[buffer(2)]]) {
  if (input[0] == 0) {
    TORCH_REPORT_ERROR(error_buf, msg);
  }
}

#define REGISTER_ASSERT_ASYNC(WIDTH, DTYPE)                  \
  template [[host_name("assert_async_" #WIDTH)]] kernel void \
  assert_async<DTYPE>(                                       \
      device const DTYPE*, constant char*, device c10::metal::ErrorMessages*)

REGISTER_ASSERT_ASYNC(1, uchar);
REGISTER_ASSERT_ASYNC(2, ushort);
REGISTER_ASSERT_ASYNC(4, uint);
REGISTER_ASSERT_ASYNC(8, ulong);
