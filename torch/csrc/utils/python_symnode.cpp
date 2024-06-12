#include <torch/csrc/utils/python_symnode.h>

namespace torch {

py::handle get_symint_class() {
  // NB: leak
  static py::handle symint_class =
      py::object(py::module::import("torch").attr("SymInt")).release();
  return symint_class;
}

py::handle get_symfloat_class() {
  // NB: leak
  static py::handle symfloat_class =
      py::object(py::module::import("torch").attr("SymFloat")).release();
  return symfloat_class;
}

py::handle get_symbool_class() {
  // NB: leak
  static py::handle symbool_class =
      py::object(py::module::import("torch").attr("SymBool")).release();
  return symbool_class;
}

} // namespace torch
