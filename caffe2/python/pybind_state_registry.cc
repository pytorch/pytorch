#include "caffe2/python/pybind_state_registry.h"

namespace caffe2 {
namespace python {

namespace py = pybind11;

C10_DEFINE_REGISTRY(PybindAdditionRegistry, PybindAddition, py::module&);

} // namespace python
} // namespace caffe2
