#pragma once

#include <cstdint>

namespace c10 { namespace cuda {

enum class CUDAAssertKind : int32_t {
  ASSERTION_FAILED,
  INDEX_OUT_OF_BOUNDS,
  ZERO_DIVISION,
};

constexpr int MAX_ASSERT_MSG_LENGTH = 1024;
constexpr int MAX_ERROR_DETAILS_SIZE = 1024;

struct CUDAAssert {
  volatile int32_t error;
  CUDAAssertKind type;
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
