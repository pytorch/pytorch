#include <torch/extension.h>

bool logical_and(bool a, bool b) { return a && b; }

TORCH_LIBRARY(torch_library, m) {
  m.def("logical_and", &logical_and);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
