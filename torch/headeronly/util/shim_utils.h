#pragma once

#include <sstream>
#include <stdexcept>

#if defined(__GNUC__) || defined(__clang__)
#define TORCH_NOINLINE __attribute__((noinline))
#elif _MSC_VER
#define TORCH_NOINLINE __declspec(noinline)
#else
#define TORCH_NOINLINE
#endif

#define TORCH_SUCCESS 0
#define TORCH_FAILURE 1

TORCH_NOINLINE static void throw_exception(
    const char* call,
    const char* file,
    int64_t line) {
  std::stringstream ss;
  ss << call << " API call failed at " << file << ", line " << line;
  throw std::runtime_error(ss.str());
}

// This API is 100% inspired by AOTI_TORCH_ERROR_CODE_CHECK defined in
// pytorch/torch/csrc/inductor/aoti_runtime/utils.h to handle the returns
// of the APIs in the shim. We are genericizing this for more global use
// of the shim beyond AOTI, for examples, see torch/csrc/stable/ops.h.
#define TORCH_ERROR_CODE_CHECK(call)            \
  if ((call) != TORCH_SUCCESS) {                \
    throw_exception(#call, __FILE__, __LINE__); \
  }
