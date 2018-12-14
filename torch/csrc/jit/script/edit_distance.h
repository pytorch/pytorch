#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <cstddef>
#include <string>

namespace torch { namespace jit { namespace script {


TORCH_API size_t ComputeEditDistance(const std::string& word1, const std::string& word2,
    size_t maxEditDistance);

}}}
