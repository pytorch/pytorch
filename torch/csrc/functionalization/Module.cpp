#include <torch/csrc/functionalization/Module.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/FunctionalStorageImpl.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/FunctionalizeFallbackKernel.h>
#include <memory>

namespace torch::functionalization {

void initModule(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  // Create a `torch._C._functionalization` Python module.
  auto functionalization = m.def_submodule(
      "_functionalization", "functionalization related pybind.");

  // Retrieve the ViewMeta sequence of a given functional tensor.
  functionalization.def("get_view_meta_sequence", [](const at::Tensor& tensor) {
    TORCH_INTERNAL_ASSERT(
        at::functionalization::impl::isFunctionalTensor(tensor));
    auto impl = at::functionalization::impl::unsafeGetFunctionalWrapper(tensor);
    return impl->view_metas();
  });

  // Applies the given ViewMeta sequence to the given base.
  functionalization.def(
      "apply_view_meta_sequence",
      [](const at::Tensor& base,
         const std::vector<std::shared_ptr<at::functionalization::ViewMeta>>&
             sequence) {
        return at::functionalization::impl::apply_view_meta_sequence(
            base, sequence);
      });

  // Binding for InverseReturnMode.
  py::enum_<at::functionalization::InverseReturnMode>(
      functionalization, "InverseReturnMode")
      .value("AlwaysView", at::functionalization::InverseReturnMode::AlwaysView)
      .value("NeverView", at::functionalization::InverseReturnMode::NeverView)
      .value(
          "ViewOrScatterInverse",
          at::functionalization::InverseReturnMode::ViewOrScatterInverse);

  // Create bindings for the ViewMeta base class.
  //
  // Needed so that we can take a list of ViewMeta objects as parameter.
  // Specifically, in the Python-side, we will have a list of derived ViewMeta
  // classes. We need to tell pybind11 that all of those are, in fact, instances
  // of different ViewMeta sub-types.
  py::class_<
      at::functionalization::ViewMeta,
      std::shared_ptr<at::functionalization::ViewMeta>>(
      functionalization, "ViewMeta")
      .def_property_readonly(
          "has_symbolic_inputs",
          [](const std::shared_ptr<at::functionalization::ViewMeta>& meta) {
            return meta->has_symbolic_inputs;
          });

  // Bindings for `ViewMeta` specializations manually implemented.
  create_binding_with_pickle<at::functionalization::resize__ViewMeta>(
      functionalization);
  create_binding_with_pickle<at::functionalization::_unsafe_view_ViewMeta>(
      functionalization);

  // Bindings for `ViewMeta` specializations automatically generated.
  initGenerated(functionalization.ptr());
}

} // namespace torch::functionalization
