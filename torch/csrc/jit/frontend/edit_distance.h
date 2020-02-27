
copy: fbcode/caffe2/torch/csrc/jit/frontend/edit_distance.h
copyrev: 9f89d2af0bdec85411b767bca7477ae733ac6d50

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
