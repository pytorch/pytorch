#pragma once

#include <cstdint>
#include <mutex>

namespace c10 {
namespace cuda {

constexpr size_t C10_ASSERT_BUFFER_SIZE = 512;
constexpr size_t C10_ASSERT_ARG_ALIGN_SIZE = sizeof(double);

// defines the type of execption to throw when the assertion fails.
enum class AssertError : int32_t {
  OK = 0,
  Error = 1,
  IndexError = 2,
};

enum class AssertFormat  : int32_t {
  Default,
  KernelAssertFailed,
  ZeroDivisionError,
  IndexOutOfBounds,
  MultinomialProbLessZero,
  MultinomialProbInf,
  MultinomialProbNaN,
};

enum class AssertSource : int32_t {
  MultinomialKernel,
  BinaryOpsKernel,
  IndexKernel,
  THCTensorIndex,
  ClassNLLCriterion,
  Distributions,
  EmbeddingBag,
  FractionalMaxPoo2d,
  SpatialClassNLLCriterion,
};

// This class holds the assert state associated with a single CUDA stream.
// The `error` field is set to a non-zero value with a CAS operation in the
// kernel when a C10_KERNEL_ASSERT() fails. A device sync should be executed
// prior to accessing the assert report in `buffer`.
struct CUDAAssert {
  volatile int32_t error; // non-zero indicates error, set in device kernel
  AssertFormat format; // format string id used to format the error message
  AssertSource file; // numerical source file identifier
  uint32_t line; // line number
  bool persistent; // whether a non-recoverable device side assert was triggered
  uint32_t length; // number of bytes in buffer (captured arguments)
  std::mutex* mutex; // synchronize host side decoding & reset
  alignas(C10_ASSERT_ARG_ALIGN_SIZE) char buffer[C10_ASSERT_BUFFER_SIZE];
};

void checkAssertError(c10::cuda::CUDAAssert* assert_state);

} // namespace cuda
} // namespace c10
