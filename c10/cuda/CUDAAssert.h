#pragma once

#include <cstdint>
#include <mutex>

namespace c10 {
namespace cuda {

constexpr size_t C10_ASSERT_BUFFER_SIZE = 512;
constexpr size_t C10_ASSERT_ARG_ALIGN_SIZE = sizeof(double);

// Global CUDA kernel assert and error codes

namespace AssertTag {

// MultinomialKernel.cu       1000-1999
enum class MultinomialKernel : int32_t {
  FIRST = 1000,
  LAST = FIRST + 999,
  _000 = FIRST + 0,
  _001 = FIRST + 1,
  _002 = FIRST + 2,
  _003 = FIRST + 3,
  _004 = FIRST + 4,
  _005 = FIRST + 5,
  _006 = FIRST + 6,
  _007 = FIRST + 7,
  _008 = FIRST + 8,
};

// BinaryOpsKernel.cu         2000-2999
enum class BinaryOpsKernel : int32_t {
  FIRST = 2000,
  LAST = FIRST + 999,
  _000 = FIRST + 0,
};

// IndexKernel.cu             3000-3999
enum class IndexKernel : int32_t {
  FIRST = 3000,
  LAST = FIRST + 999,
  _000 = FIRST + 0,
};

// THCTensorIndex.cu          4000-4999
enum class THCTensorIndex : int32_t {
  FIRST = 4000,
  LAST = FIRST + 999,
  _000 = FIRST + 0,
  _001 = FIRST + 1,
  _002 = FIRST + 2,
  _003 = FIRST + 3,
  _004 = FIRST + 4,
  _005 = FIRST + 5,
  _006 = FIRST + 6,
  _007 = FIRST + 7,
};

// ClassNLLCriterion.cu       5000-5999
enum class ClassNLLCriterion : int32_t {
  FIRST = 5000,
  LAST = FIRST + 999,
  _000 = FIRST + 0,
  _001 = FIRST + 1,
  _002 = FIRST + 2,
  _003 = FIRST + 3,
  _004 = FIRST + 4,
  _005 = FIRST + 5,
};

// Distributions.cu           6000-6999
enum class Distributions : int32_t {
  FIRST = 6000,
  LAST = FIRST + 999,
  _000 = FIRST + 0,
  _001 = FIRST + 1,
  _002 = FIRST + 2,
  _003 = FIRST + 3,
};

// EmbeddingBag.cu            7000-7999
enum class EmbeddingBag : int32_t {
  FIRST = 7000,
  LAST = FIRST + 999,
  _000 = FIRST + 0,
};

// FractionalMaxPoo2d.cu      8000-8999
enum class FractionalMaxPoo2d : int32_t {
  FIRST = 8000,
  LAST = FIRST + 999,
  _000 = FIRST + 0,
  _001 = FIRST + 1,
  _002 = FIRST + 2,
  _003 = FIRST + 3,
};

// SpatialClassNLLCriterion.cu  9000-9999
enum class SpatialClassNLLCriterion : int32_t {
  FIRST = 9000,
  LAST = FIRST + 999,
  _000 = FIRST + 0,
  _001 = FIRST + 1,
};

} // namespace AssertTag

// AssertKind defines the type of execption to throw when the assertion fails.
enum class AssertKind : int32_t {
  DEFAULT = 0,
  ZERO_DIVISION = 1,
  INDEX_ERROR = 2,
};

// This class holds the assert state associated with a single CUDA stream.
// The `error` field is set to a non-zero value with a CAS operation in the
// kernel when a C10_KERNEL_ASSERT() fails. A device sync should be executed
// prior to accessing the assert report in `buffer`.
struct CUDAAssert {
  volatile int32_t error; // non-zero indicates error, set in device kernel
  uint32_t length; // number of bytes of the assert report in buffer
  AssertKind kind;
  uint32_t line;
  bool persistent; // whether a hard device side assert was triggered
  std::mutex* mutex; // synchronize host side decoding & reset
  alignas(C10_ASSERT_ARG_ALIGN_SIZE) char buffer[C10_ASSERT_BUFFER_SIZE];
};

void checkAssertError(c10::cuda::CUDAAssert* assert_state);

} // namespace cuda
} // namespace c10
