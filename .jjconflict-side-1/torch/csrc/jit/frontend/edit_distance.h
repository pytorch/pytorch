#pragma once

#include <torch/csrc/Export.h>
#include <cstddef>

namespace torch::jit {

TORCH_API size_t ComputeEditDistance(
    const char* word1,
    const char* word2,
    size_t maxEditDistance);

} // namespace torch::jit
