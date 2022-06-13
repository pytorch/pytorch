#include <torch/csrc/autograd/python_enum_tag.h>
#include <pybind11/pybind11.h>
#include <ATen/core/enum_tag.h>

namespace py = pybind11;
namespace torch {
    namespace autograd {
    void initEnumTag(PyObject* module) {
        auto m = py::handle(module).cast<py::module>();
        py::enum_<at::Tag>(m, "Tag")
        ${enum_of_valid_tags}
        .export_values();
        m.doc() = "An Enum that contains tags that can be assigned to an operator registered in C++.";
    }
}}
