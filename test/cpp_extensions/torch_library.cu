#include <torch/extension.h>

bool logical_and(bool a, bool b) { return a && b; }

TORCH_LIBRARY(torch_library, m) {
  m.def("logical_and", &logical_and);
}

struct CuaevComputer : torch::CustomClassHolder {};

TORCH_LIBRARY(cuaev, m) {
  m.class_<CuaevComputer>("CuaevComputer");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
