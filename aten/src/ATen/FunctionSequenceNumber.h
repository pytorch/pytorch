#pragma once

#include <cstdint>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace at {

// A simple thread local enumeration, used to link forward and backward pass
// ops and is used by autograd and observers framework

class TORCH_API FunctionSequenceNumber {
 public:
  static uint64_t peek();
  static uint64_t get_and_increment();
 private:
  FunctionSequenceNumber() {};
};

} // namespace at
