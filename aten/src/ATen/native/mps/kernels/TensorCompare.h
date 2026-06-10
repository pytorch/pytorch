#pragma once

#define ISIN_THREADS_PER_THREADGROUP static_cast<uint32_t>(128)
#define ISIN_TARGET_THREADGROUPS static_cast<int64_t>(4096)

struct IsinParams {
  uint32_t numel_elements;
  uint32_t numel_test;
  uint32_t num_chunks;
};

struct IsinSortedParams {
  uint32_t numel_test;
  bool invert;
};

template <typename T>
struct ClampScalarParams {
  T min;
  T max;
};
