#pragma once

#include <torch/detail/static.h>
#include <torch/types.h>

#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>

#include <memory>

namespace torch {
namespace python {
namespace detail {
// The first template argument is the ModuleType we are binding, the second is
// the base class and the last is the holder type for the created object when
// going between C++ and Python.
template <typename ModuleType>
using ModulePybindClass =
    py::class_<ModuleType, torch::nn::Module, std::shared_ptr<ModuleType>>;
} // namespace detail

template <typename ModuleType>
torch::disable_if_t<
    torch::detail::has_forward<ModuleType>::value,
    detail::ModulePybindClass<ModuleType>>
bind_module(py::module module, const char* name) {
  return {module, name};
}

template <
    typename ModuleType,
    typename =
        torch::enable_if_t<torch::detail::has_forward<ModuleType>::value>>
detail::ModulePybindClass<ModuleType> bind_module(
    py::module module,
    const char* name) {
  return detail::ModulePybindClass<ModuleType>(module, name)
      .def("forward", &ModuleType::forward)
      .def("__call__", &ModuleType::forward);
}
} // namespace python
} // namespace torch
