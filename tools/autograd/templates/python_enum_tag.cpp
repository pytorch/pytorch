#include <torch/csrc/autograd/python_enum_tag.h>
#include <pybind11/pybind11.h>
#include <ATen/core/enum_tag.h>

namespace py = pybind11;
namespace torch {
    namespace autograd {
    void initEnumTag(PyObject* module) {
        auto m = py::handle(module).cast<py::module>();
        ${enum_of_valid_tags}
    }
}}
