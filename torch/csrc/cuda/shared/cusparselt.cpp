#if defined(USE_CUSPARSELT)
#include <torch/csrc/utils/pybind.h>

#include <array>
#include <tuple>

namespace {
using version_tuple = std::tuple<size_t, size_t, size_t>;
}

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

  auto cusparselt = m.def_submodule("_cusparselt", "libcusparseLt.so bindings");

#ifdef USE_CUSPARSELT
  cusparselt.attr("is_cuda") = true;
#else
  cusparselt.attr("is_cuda") = false;
#endif

  cusparselt.def("getVersionInt", getVersionInt);
}

} // namespace torch::cuda::shared
#endif
