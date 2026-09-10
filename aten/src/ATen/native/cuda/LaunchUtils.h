#pragma once
#include <algorithm>

namespace at::native {

// returns 2**floor(log2(n))
static int lastPow2(unsigned int n) {
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  return std::max<int>(1, n - (n >> 1));
}

} // namespace at::native
