#pragma once

#include <cstdint>

namespace c10 { namespace cuda {

constexpr int MAX_ERROR_DETAILS_SIZE = 1024;
constexpr int MAX_ASSERT_MSG_LENGTH = 1024;

struct CUDAAssert {
  int32_t error;
  int32_t type;
  uint32_t line;
  uint32_t _reserved;  // round to 8 byte boundary
  char message[MAX_ASSERT_MSG_LENGTH];
  char file[MAX_ASSERT_MSG_LENGTH];
  int8_t details[MAX_ERROR_DETAILS_SIZE];
};

struct CUDAAssertDetailIndexKernel {
  int64_t index;
  int64_t axis;
  int64_t size;
};

}}