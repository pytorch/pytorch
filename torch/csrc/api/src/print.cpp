#include <torch/print.h>
#include <ATen/core/Formatting.h>

namespace torch {

void set_printoptions(bool sci_mode) {
  at::set_printoptions(sci_mode);
}

} // namespace torch
