#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/metal/MetalPrepackOpContext.h>
#include <ATen/ATen.h>

bool logical_and(bool a, bool b) { return a && b; }

TORCH_LIBRARY(torch_library, m) {
  m.def("logical_and", &logical_and);
}
