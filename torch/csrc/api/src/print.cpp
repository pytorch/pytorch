#include <torch/print.h>
#include <ATen/core/Formatting.h>

namespace torch {

void set_print_sci_mode(bool enabled) {
  at::set_print_sci_mode(enabled);
}

} // namespace torch
