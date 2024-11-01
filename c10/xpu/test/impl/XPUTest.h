#include <gtest/gtest.h>

#include <c10/util/irange.h>

static inline void initHostData(int* hostData, int numel) {
  for (const auto i : c10::irange(numel)) {
    hostData[i] = i;
  }
}

static inline void clearHostData(int* hostData, int numel) {
  for (const auto i : c10::irange(numel)) {
    hostData[i] = 0;
  }
}

static inline void validateHostData(int* hostData, int numel) {
  for (const auto i : c10::irange(numel)) {
    EXPECT_EQ(hostData[i], i);
  }
}
