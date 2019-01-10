#include <pybind11/pybind11.h>

#include "caffe2/distributed/store_handler.h"

namespace caffe2 {
namespace python {

namespace py = pybind11;

PYBIND11_MODULE(python, m) {
  m.doc() = "Python interface for distributed Caffe2";

  py::register_exception<StoreHandlerTimeoutException>(
      m, "StoreHandlerTimeoutError");
}

} // namespace python
} // namespace caffe2
