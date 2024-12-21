#pragma once

#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::functionalization {

// Creates the default bindings for `ViewMeta` specializations.
//
// Defines a constructor using the types in `SerializableTuple`, as well
// as pickle methods.
template <class T>
void create_binding_with_pickle(py::module m) {
  py::class_<T, std::shared_ptr<T>>(m, T::name())
      .def(py::init<typename T::SerializableTuple>())
      .def(py::pickle(
          [](std::shared_ptr<T> meta) { return meta->to_serializable_tuple(); },
          [](const typename T::SerializableTuple& tpl) {
            return std::make_shared<T>(tpl);
          }));
}

void initModule(PyObject* module);
void initGenerated(PyObject* module);

} // namespace torch::functionalization
