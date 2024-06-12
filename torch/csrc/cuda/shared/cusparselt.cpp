// The clang-tidy job seems to complain that it can't find cudnn.h without this.
// This file should only be compiled if this condition holds, so it should be
// safe.
#if defined(USE_CUDNN)
#include <torch/csrc/utils/pybind.h>

#include <array>
#include <tuple>

#ifdef USE_CUDNN
#include <cusparseLt.h>

namespace {

size_t getVersionInt() {
  return CUSPARSELT_VERSION;
}

} // namespace
#endif

namespace torch::cuda::shared {

void initCusparseltBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto cusparselt = m.def_submodule("_cusparselt", "libcusparselt.so bindings");

  // The runtime version check in python needs to distinguish cudnn from miopen
#ifdef USE_CUDNN
  cusparselt.attr("is_cuda") = true;
#else
  cusparselt.attr("is_cuda") = false;
#endif
  cusparselt.def("getVersionInt", getVersionInt);
}

} // namespace torch::cuda::shared
#endif
