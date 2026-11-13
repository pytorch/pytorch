#include <metal_stdlib>
using namespace metal;

kernel void sleep(
    device long* out [[buffer(0)]],
    constant long& cycles [[buffer(1)]]) {
  long x = 0;
  // Perform a dependent chain of operations and store the result so that the
  // loop cannot be optimized away.
  for (long i = 0; i < cycles; ++i) {
    x = x * 6364136223846793005L + 1442695040888963407L;
  }
  out[0] = x;
}
