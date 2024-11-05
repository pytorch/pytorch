#include <torch/csrc/utils/pybind.h>

#ifdef USE_CUSPARSELT
#include <cusparseLt.h>

namespace {

size_t getVersionInt() {
  return CUSPARSELT_VERSION;
}

} // namespace

namespace torch::cuda::shared {

void initCusparseltBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  auto cusparselt = m.def_submodule("_cusparselt", "libcusparselt.so bindings");
  cusparselt.def("getVersionInt", getVersionInt);
}

} // namespace torch::cuda::shared
#endif
