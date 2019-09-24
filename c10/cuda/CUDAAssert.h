#pragma once

#include <cstdint>

namespace c10 { namespace cuda {

constexpr int MAX_ERROR_DETAILS_SIZE = 1024;

struct CUDAAssert {
  int32_t error;
  int32_t type;
  int8_t details[MAX_ERROR_DETAILS_SIZE];
};

struct CUDAAssertDetailIndexKernel {
  int64_t index;
  int64_t axis;
  int64_t size;
};

}}