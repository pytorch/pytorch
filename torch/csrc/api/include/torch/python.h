#pragma once

#include <torch/detail/static.h>
#include <torch/nn/module.h>
#include <torch/ordered_dict.h>
#include <torch/types.h>

#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>

#include <iterator>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch {
namespace python {
namespace detail {
inline Device py_object_to_device(py::object object) {
  PyObject* obj = object.ptr();
  if (THPDevice_Check(obj)) {
    return reinterpret_cast<THPDevice*>(obj)->device;
  }
  throw TypeError("Expected device");
}

inline Dtype py_object_to_dtype(py::object object) {
  PyObject* obj = object.ptr();
  if (THPDtype_Check(obj)) {
    return reinterpret_cast<THPDtype*>(obj)->scalar_type;
  }
  throw TypeError("Expected dtype");
}

template <typename ModuleType>
using PyModuleClass =
    py::class_<ModuleType, torch::nn::Module, std::shared_ptr<ModuleType>>;

} // namespace detail

/// Adds method bindings for a pybind11 `class_` that binds an `nn::Module`
/// subclass.
///
/// Say you have a pybind11 class object created with `py::class_<Net>(m,
/// "Net")`. This function will add all the necessary `.def()` calls to bind the
/// `nn::Module` base class' methods, such as `train()`, `eval()` etc. into
/// Python.
///
/// Users should prefer to use `bind_module` if possible.
template <typename ModuleType, typename... Extra>
py::class_<ModuleType, Extra...> add_module_bindings(
    py::class_<ModuleType, Extra...> module) {
  // clang-format off
  return module
      .def("train",
          [](ModuleType& module, bool mode) { module.train(mode); },
          py::arg("mode") = true)
      .def("eval", [](ModuleType& module) { module.eval(); })
      .def("clone", [](ModuleType& module) { return module.clone(); })
      .def_property_readonly(
          "training", [](ModuleType& module) { return module.is_training(); })
      .def("zero_grad", [](ModuleType& module) { module.zero_grad(); })
      .def("cuda", [](ModuleType& module) { module.to(torch::kCUDA); })
      .def("cpu", [](ModuleType& module) { module.to(torch::kCPU); })
      .def_property_readonly( "_parameters", [](ModuleType& module) {
            return module.named_parameters(/*recurse=*/false);
          })
      .def("parameters", [](ModuleType& module, bool recurse) {
            return module.parameters(recurse);
          },
          py::arg("recurse") = true)
      .def("named_parameters", [](ModuleType& module, bool recurse) {
            return module.named_parameters(recurse);
          },
          py::arg("recurse") = true)
      .def_property_readonly("_buffers", [](ModuleType& module) {
            return module.named_buffers(/*recurse=*/false);
          })
      .def("buffers", [](ModuleType& module, bool recurse) {
            return module.buffers(recurse); },
          py::arg("recurse") = true)
      .def("named_buffers", [](ModuleType& module, bool recurse) {
            return module.named_buffers(recurse);
          },
          py::arg("recurse") = true)
      .def_property_readonly(
          "_modules", [](ModuleType& module) { return module.children(); })
      .def("modules", [](ModuleType& module) { return module.modules(); })
      .def("named_modules",
          [](ModuleType& module, py::object /* unused */, std::string prefix) {
            return module.named_modules(std::move(prefix));
          },
          py::arg("memo") = py::none(),
          py::arg("prefix") = std::string())
      .def("children", [](ModuleType& module) { return module.children(); })
      .def("named_children",
          [](ModuleType& module) { return module.named_children(); })
      .def("to", [](ModuleType& module, py::object object, bool non_blocking) {
            if (THPDevice_Check(object.ptr())) {
              module.to(
                  reinterpret_cast<THPDevice*>(object.ptr())->device,
                  non_blocking);
            } else {
              module.to(detail::py_object_to_dtype(object), non_blocking);
            }
          },
          py::arg("dtype_or_device"),
          py::arg("non_blocking") = false)
      .def("to",
          [](ModuleType& module,
             py::object device,
             py::object dtype,
             bool non_blocking) {
            module.to(
                detail::py_object_to_device(device),
                detail::py_object_to_dtype(dtype),
                non_blocking);
          },
          py::arg("device"),
          py::arg("dtype"),
          py::arg("non_blocking") = false)
      .def("cuda", [](ModuleType& module) { module.to(kCUDA); })
      .def("cpu", [](ModuleType& module) { module.to(kCPU); })
      .def("float", [](ModuleType& module) { module.to(kFloat32); })
      .def("double", [](ModuleType& module) { module.to(kFloat64); })
      .def("half", [](ModuleType& module) { module.to(kFloat16); })
      .def("__str__", [](ModuleType& module) { return module.name(); })
      .def("__repr__", [](ModuleType& module) { return module.name(); });
  // clang-format on
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
template <typename ModuleType>
torch::disable_if_t<
    torch::detail::has_forward<ModuleType>::value,
    detail::PyModuleClass<ModuleType>>
bind_module(py::module module, const char* name) {
  return add_module_bindings(detail::PyModuleClass<ModuleType>(module, name));
}

/// Creates a pybind11 class object for an `nn::Module` subclass type and adds
/// default bindings.
///
/// After adding the default bindings, the class object is returned, such that
/// you can add more bindings.
///
/// If the class has a `forward()` method, it is automatically exposed as
/// `forward()` and `__call__` in Python.
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
template <
    typename ModuleType,
    typename =
        torch::enable_if_t<torch::detail::has_forward<ModuleType>::value>>
detail::PyModuleClass<ModuleType> bind_module(
    py::module module,
    const char* name) {
  return add_module_bindings(detail::PyModuleClass<ModuleType>(module, name))
      .def("forward", &ModuleType::forward)
      .def("__call__", &ModuleType::forward);
}
} // namespace python
} // namespace torch
