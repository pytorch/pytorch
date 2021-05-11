#include "caffe2/python/pybind_state_registry.h"

namespace caffe2 {
namespace python {

namespace py = pybind11;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_REGISTRY(PybindAdditionRegistry, PybindAddition, py::module&);

} // namespace python
} // namespace caffe2
