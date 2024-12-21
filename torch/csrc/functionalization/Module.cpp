#include <torch/csrc/functionalization/Module.h>

#include <ATen/FunctionalizeFallbackKernel.h>

namespace torch::functionalization {

void initModule(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  // Create a `torch._C._functionalization` Python module.
  auto functionalization = m.def_submodule(
      "_functionalization", "functionalization related pybind.");

  // Binding for InverseReturnMode.
  py::enum_<at::functionalization::InverseReturnMode>(
      functionalization, "InverseReturnMode")
      .value("AlwaysView", at::functionalization::InverseReturnMode::AlwaysView)
      .value("NeverView", at::functionalization::InverseReturnMode::AlwaysView)
      .value(
          "ViewOrScatterInverse",
          at::functionalization::InverseReturnMode::AlwaysView);

  // Bindings for `ViewMeta` specializations manually implemented.
  create_binding_with_pickle<at::functionalization::resize__ViewMeta>(
      functionalization);
  create_binding_with_pickle<at::functionalization::_unsafe_view_ViewMeta>(
      functionalization);

  // Bindings for `ViewMeta` specializations automatically generated.
  initGenerated(functionalization.ptr());
}

} // namespace torch::functionalization
