#pragma once

#include <array>
#include <filesystem>
#include <optional>

#include <torch/csrc/inductor/aoti_runtime/interface.h>
#include <torch/csrc/inductor/aoti_runtime/model.h>

#include <c10/util/generic_math.h>
#include <torch/csrc/inductor/aoti_runtime/scalar_to_tensor.h>

// Provides torch::stable::detail::from / ::to and StableIValue, used by the
// wrapper.cpp's aoti_torch_call_dispatcher path. See pytorch#184195.
#include <torch/csrc/stable/stableivalue_conversions.h>

// Round up to the nearest multiple of 64
[[maybe_unused]] inline int64_t align(int64_t nbytes) {
  return (nbytes + 64 - 1) & -64;
}
