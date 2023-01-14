#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace functorch {
namespace impl {

void initFuncTorchBindings(PyObject* module);

} // namespace impl
} // namespace functorch
} // namespace torch
