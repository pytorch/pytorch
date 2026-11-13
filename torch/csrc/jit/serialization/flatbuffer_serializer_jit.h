#pragma once

#include <torch/csrc/jit/serialization/flatbuffer_serializer.h>

namespace torch::jit {

TORCH_API bool register_flatbuffer_all();

} // namespace torch::jit
