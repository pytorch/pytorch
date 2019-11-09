#include <torch/csrc/nested_tensor/python_nested_tensor.h>
#include <torch/csrc/tensor/python_tensor.h>

#include <structmember.h>

#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/cuda_enabled.h>
#include <torch/csrc/utils/cuda_lazy_init.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/csrc/utils/tensor_types.h>

#include <ATen/ATen.h>

#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace torch {
namespace nested_tensor {

using namespace at;
using namespace torch::autograd;
namespace py = pybind11;

std::vector<_ListNestedTensor> _ListNestedTensor::unbind_nested() {
  std::vector<_ListNestedTensor> result;
  size_t i = 0;
  for (_MetaNode child : _structure._children) {
    result.push_back(_ListNestedTensor(
        std::vector<at::Tensor>(_flat_tensors.begin() + i,
                                _flat_tensors.begin() + i +
                                    _num_tensor(child)),
        child));
    i += _num_tensor(child);
  }
  return result;
}

std::vector<py::object> _ListNestedTensor::unbind() {
  std::vector<py::object> result;
  if (nested_dim() == 1) {
    for (at::Tensor tensor : _flat_tensors) {
      result.push_back(py::reinterpret_borrow<py::object>(
          torch::autograd::utils::wrap(tensor)));
    }
  } else {
    std::vector<_ListNestedTensor> nts = unbind_nested();
    for (_ListNestedTensor nt : nts) {
      result.push_back(py::cast(nt));
    }
  }
  return result;
}

std::string _ListNestedTensor::__str__() {
  std::stringstream result;
  if (nested_dim() == 1) {
    for (auto tensor : _flat_tensors) {
      result << "  ";
      result << tensor;
      result << ",";
      result << std::endl;
    }
  } else {
      for(_ListNestedTensor nt : unbind_nested()) {
      result << "  ";
      result << nt.__str__();
      result << ",";
      result << std::endl;
    }
  }
  result << "])";
  return result.str();
}

void initialize_python_bindings() {
  auto obj = py::module::import("torch");
  auto m = py::handle(obj).cast<py::module>();

  py::class_<_ListNestedTensor>(m, "_ListNestedTensor")
      .def(py::init<std::vector<py::object>>())
      .def_property_readonly("dtype", &_ListNestedTensor::get_dtype)
      .def_property_readonly("layout", &_ListNestedTensor::get_layout)
      .def_property_readonly("device", &_ListNestedTensor::get_device)
      .def_property_readonly("requires_grad", &_ListNestedTensor::requires_grad)
      .def_property_readonly("grad", &_ListNestedTensor::grad)
      .def("detach", &_ListNestedTensor::detach)
      .def("backward", &_ListNestedTensor::backward)
      .def("requires_grad_", &_ListNestedTensor::requires_grad_)
      .def("element_size", &_ListNestedTensor::element_size)
      .def("unbind", &_ListNestedTensor::unbind)
      .def("nested_size", &_ListNestedTensor::nested_size)
      .def("nested_stride", &_ListNestedTensor::nested_stride)
      .def("is_pinned", &_ListNestedTensor::is_contiguous)
      .def("is_contiguous", &_ListNestedTensor::is_contiguous)
      .def("__len__", &_ListNestedTensor::__len__)
      .def("__str__", &_ListNestedTensor::__str__)
      .def("dim", &_ListNestedTensor::dim)
      .def("nested_dim", &_ListNestedTensor::nested_dim);
}
}
}
