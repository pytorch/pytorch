#include <ATen/ViewMetaClasses.h>
#include <torch/csrc/functionalization/Module.h>

namespace torch::functionalization {

void initGenerated(PyObject* module) {
  auto functionalization = py::handle(module).cast<py::module>();
  $view_meta_bindings
}

} // namespace torch::functionalization
