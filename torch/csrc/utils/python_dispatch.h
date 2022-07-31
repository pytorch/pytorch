#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace impl {
namespace dispatch {

void initDispatchBindings(PyObject* module);

}
} // namespace impl
} // namespace torch
