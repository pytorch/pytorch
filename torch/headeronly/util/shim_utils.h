#pragma once

#include <torch/headeronly/macros/Macros.h>

#include <sstream>
#include <stdexcept>

#define TORCH_SUCCESS 0
#define TORCH_FAILURE 1

HIDDEN_NAMESPACE_BEGIN(torch, headeronly, detail)
// Reports a failed shim call, led by the message it recorded, if any.
[[maybe_unused]] [[noreturn]] C10_NOINLINE static void throw_exception(
    const char* what,
    const char* call,
    const char* file,
    int64_t line) {
  const bool has_what = what != nullptr && what[0] != '\0';
  std::stringstream ss;
  if (has_what) {
    ss << what << " (originally from ";
  }
  ss << call << " API call failed at " << file << ", line " << line;
  if (has_what) {
    ss << ")";
  }
  // STD_TORCH_CHECK would stamp this helper's file and line over the
  // caller's, which the message above already carries.
  // @allow-raw-throw: keeps the caller's location in the message
  throw std::runtime_error(std::move(ss).str());
}

[[maybe_unused]] [[noreturn]] C10_NOINLINE static void throw_exception(
    const char* call,
    const char* file,
    int64_t line) {
  throw_exception(nullptr, call, file, line);
}
HIDDEN_NAMESPACE_END(torch, headeronly, detail)

// This API is 100% inspired by AOTI_TORCH_ERROR_CODE_CHECK defined in
// pytorch/torch/csrc/inductor/aoti_runtime/utils.h to handle the returns
// of the APIs in the shim. We are genericizing this for more global use
// of the shim beyond AOTI, for examples, see torch/csrc/stable/ops.h.
#define TORCH_ERROR_CODE_CHECK(call)                                       \
  if ((call) != TORCH_SUCCESS) {                                           \
    torch::headeronly::detail::throw_exception(#call, __FILE__, __LINE__); \
  }
