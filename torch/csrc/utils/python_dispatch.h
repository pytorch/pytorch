#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::impl::dispatch {

void initDispatchBindings(PyObject* module);

} // namespace torch::impl::dispatch
