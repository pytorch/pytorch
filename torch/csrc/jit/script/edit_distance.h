#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <cstddef>

namespace torch {
namespace jit {
namespace script {

TORCH_API size_t ComputeEditDistance(
    const char* word1,
    const char* word2,
    size_t maxEditDistance);

}
} // namespace jit
} // namespace torch
