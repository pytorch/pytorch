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

// void _ListNestedTensor::backward(_ListNestedTensor gradient, bool retain_graph,
//                                  bool create_graph) {
//   apply2_method(
//       unbind(), gradient.unbind(),
//       [retain_graph, create_graph](at::Tensor tensor1, at::Tensor tensor2) {
//         tensor1.backward(tensor2, retain_graph, create_graph);
//       });
// }

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
      .def("__len__", &_ListNestedTensor::__len__)
      .def("dim", &_ListNestedTensor::dim)
      .def("nested_dim", &_ListNestedTensor::nested_dim);
}
}
}
