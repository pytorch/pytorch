#pragma once

#include <torch/detail/static.h>
#include <torch/types.h>

#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/types.h>
#include <torch/ordered_dict.h>

#include <iterator>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch {
namespace python {
namespace detail {
inline std::unordered_map<std::string, Tensor> ordered_dict_to_map(
    const OrderedDict<std::string, torch::Tensor>& dict) {
  auto pairs = dict.pairs();
  return {pairs.begin(), pairs.end()};
}
} // namespace detail

/// Adds method bindings for a pybind11 `class_` that binds an `nn::Module`
/// subclass.
///
/// Say you have a pybind11 class object created with `py::class_<Net>(m,
/// "Net")`. This function will add all the necessary `.def()` calls to bind the
/// `nn::Module` base class' methods, such as `train()`, `eval()` etc. into
/// Python. The exact list of supported methods and their Python signatures are:
/// - `train()`
/// - `eval()`
/// - `is_training() -> bool`
/// - `zero_grad()`
/// - `cuda()`
/// - `cpu()`
/// - `parameters() -> List<Tensor>`
/// - `named_parameters() -> Dict<String, Tensor>`
/// - `buffers() -> List<Tensor>`
/// - `named_buffers() -> Dict<String, Tensor>`
template <typename M, typename... Extra>
py::class_<M, Extra...> add_module_bindings(py::class_<M, Extra...> module) {
  return module.def("train", [](M& module) { module.train(); })
      .def("eval", [](M& module) { module.eval(); })
      .def("clone", [](M& module) { return module.clone(); })
      .def_property_readonly(
          "training", [](M& module) { return module.is_training(); })
      .def("zero_grad", [](M& module) { module.zero_grad(); })
      .def("cuda", [](M& module) { module.to(torch::kCUDA); })
      .def("cpu", [](M& module) { module.to(torch::kCPU); })
      .def("parameters", [](M& module) { return module.parameters(); })
      .def(
          "named_parameters",
          [](M& module) {
            return detail::ordered_dict_to_map(module.named_parameters());
          })
      .def("buffers", [](M& module) { return module.buffers(); })
      .def("named_buffers", [](M& module) {
        return detail::ordered_dict_to_map(module.named_buffers());
      });
}

/// Creates a pybind11 class object for an `nn::Module` subclass type and adds
/// default bindings.
///
/// After adding the default bindings, the class object is returned, such that
/// you can add more bindings.
///
/// Example usage:
/// \rst
/// .. code-block:: cpp
///
///   struct Net : torch::nn::Module {
///     Net(int in, int out) { }
///     torch::Tensor forward(torch::Tensor x) { return x; }
///   };
///
///   PYBIND11_MODULE(my_module, m) {
///     torch::python::bind_module<Net>(m, "Net")
///       .def(py::init<int, int>())
///       .def("forward", &Net::forward);
///  }
/// \endrst
template <typename M>
py::class_<M, std::shared_ptr<M>> bind_module(
    py::module module,
    const char* name) {
  return add_module_bindings(py::class_<M, std::shared_ptr<M>>(module, name));
}
} // namespace python
} // namespace torch
