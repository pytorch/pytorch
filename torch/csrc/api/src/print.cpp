#include <ATen/core/Formatting.h>
#include <torch/print.h>

namespace torch {

void set_printoption_sci_mode(bool enabled) {
  at::set_printoption_sci_mode(enabled);
}

} // namespace torch
