#include <gloo/common/error.h>
#include <pybind11/pybind11.h>

namespace gloo {
namespace python {

namespace py = pybind11;

PYBIND11_MODULE(python, m) {
  m.doc() = "Python interface for Gloo";
  py::register_exception<IoException>(m, "IoError");
}

} // namespace python
} // namespace gloo
