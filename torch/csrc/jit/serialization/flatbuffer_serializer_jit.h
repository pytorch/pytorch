#pragma once

#include <torch/csrc/jit/serialization/flatbuffer_serializer.h>

namespace torch {
namespace jit {

TORCH_API bool register_flatbuffer_all();

} // namespace jit
} // namespace torch
