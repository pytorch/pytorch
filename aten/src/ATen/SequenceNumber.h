#pragma once

#include <cstdint>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace at {

// A simple thread local enumeration, used to link forward and backward pass
// ops and is used by autograd and observers framework
namespace sequence_number {

TORCH_API uint64_t peek();
TORCH_API uint64_t get_and_increment();

} // namespace sequence_number
} // namespace at
