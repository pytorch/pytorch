#pragma once

#include <torch/headeronly/macros/Macros.h>

#include <sstream>
#include <stdexcept>

#define TORCH_SUCCESS 0
#define TORCH_FAILURE 1

namespace torch::headeronly::detail {
[[maybe_unused]] C10_NOINLINE static void throw_exception(
    const char* call,
    const char* file,
    int64_t line) {
  std::stringstream ss;
  ss << call << " API call failed at " << file << ", line " << line;
  throw std::runtime_error(ss.str());
}
} // namespace torch::headeronly::detail

// This API is 100% inspired by AOTI_TORCH_ERROR_CODE_CHECK defined in
// pytorch/torch/csrc/inductor/aoti_runtime/utils.h to handle the returns
// of the APIs in the shim. We are genericizing this for more global use
// of the shim beyond AOTI, for examples, see torch/csrc/stable/ops.h.
#define TORCH_ERROR_CODE_CHECK(call)                                       \
  if ((call) != TORCH_SUCCESS) {                                           \
    torch::headeronly::detail::throw_exception(#call, __FILE__, __LINE__); \
  }
