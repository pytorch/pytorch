#include <torch/csrc/autograd/python_enum_tag.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/pybind11.h>
#include <ATen/core/enum_tag.h>

namespace py = pybind11;
namespace torch::autograd {
    void initEnumTag(PyObject* module) {
        auto m = py::handle(module).cast<py::module>();
        py::native_enum<at::Tag>(m, "Tag", "enum.IntEnum")
          ${enum_of_valid_tags}
          .finalize();
        m.doc() = "An Enum that contains tags that can be assigned to an operator registered in C++.";
    }
}
