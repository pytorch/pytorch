#pragma once

#include <filesystem>
#include <optional>

#include <torch/csrc/inductor/aoti_runtime/interface.h>
#include <torch/csrc/inductor/aoti_runtime/model.h>

#include <c10/util/generic_math.h>
#include <torch/csrc/inductor/aoti_runtime/scalar_to_tensor.h>
using half = at::Half;
using bfloat16 = at::BFloat16;

// Round up to the nearest multiple of 64
[[maybe_unused]] inline int64_t align(int64_t nbytes) {
  return (nbytes + 64 - 1) & -64;
}

#if defined(_MSC_VER)
#define DISABLE_FUNCTION_OPTIMIZATION _Pragma("optimize( \"\", off )")
#elif defined(__clang__)
#define DISABLE_FUNCTION_OPTIMIZATION [[clang::optnone]]
#elif defined(__GNUC__)
#define DISABLE_FUNCTION_OPTIMIZATION [[gnu::optimize(0)]]
#endif
