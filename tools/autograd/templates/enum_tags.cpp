#include <torch/csrc/autograd/enum_tags.h>
#include <pybind11/pybind11.h>
#include <ATen/core/enum_tags.h>

namespace py = pybind11;
namespace torch {
    namespace autograd {
    void initEnumTags(PyObject* module) {
        auto m = py::handle(module).cast<py::module>();
        ${enum_of_valid_tags}
    }
}}
