#pragma once

#include "torch/csrc/WindowsTorchApiMacro.h"

namespace torch { namespace autograd {

struct AnomalyMode {
  static bool is_enabled() {
    return _enabled;
  }
  static void set_enabled(bool enabled) {
    _enabled = enabled;
  }

private:
 TORCH_API static bool _enabled;
};


struct AnomalyMetadata {
  virtual ~AnomalyMetadata() = default;
  virtual void store_stack() = 0;
  virtual void print_stack() = 0;
};

}}
