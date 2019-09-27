#pragma once

#include <cstdint>

namespace c10 { namespace cuda {

constexpr int MAX_ASSERT_MESSAGE_LENGTH = 1024;
constexpr int MAX_ASSERT_FILE_LENGTH = 1024;

// Types of asynchronous CUDA assert reports
enum class CUDAAssertKind : int32_t {
  ASSERTION_FAILED,
  INDEX_OUT_OF_BOUNDS,
  ZERO_DIVISION,
};

// Details about an invalid tensor index access operation
struct CUDAAssertDetailIndexError {
  int64_t index;
  int64_t axis;
  int64_t size;
};

// Union of error specific data captured with a failed assertions
union CUDAAssertDetails {
  CUDAAssertDetailIndexError index_error;
};

// Stream assertion error information
// The CUDA device must be synchronized before accessing other members than error_flag.
struct CUDAAssert {
  volatile int32_t error;    // flag: a non-zero value indicates an assert-error occured
  CUDAAssertKind kind;

  // call site information of assertion
  uint32_t line;
  char message[MAX_ASSERT_MESSAGE_LENGTH];
  char file[MAX_ASSERT_FILE_LENGTH];

  // error specific additional data
  CUDAAssertDetails details;
};

}}
