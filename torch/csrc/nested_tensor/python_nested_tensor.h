#pragma once

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorTypeId.h>
#include <torch/csrc/Device.h>
#include <exception>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/tensor/python_tensor.h>
#include <typeinfo>

namespace torch {
namespace nested_tensor {

namespace py = pybind11;

static const at::Tensor get_first_tensor(std::vector<py::object> tensors) {
  try {
    return tensors[0].cast<const at::Tensor>();
  } catch (std::exception e) {
    return get_first_tensor(tensors[0].cast<std::vector<py::object>>());
  }
}

static int64_t get_nested_dim(std::vector<py::object> tensors) {
  try {
    at::Tensor t = tensors[0].cast<const at::Tensor>();
    return 1;
  } catch (std::exception e) {
    return 1 + get_nested_dim(tensors[0].cast<std::vector<py::object>>());
  }
}
// TODO: Operate on _ListNestedTensor using nested_dim instead.
template <class F>
static std::vector<py::object> map_method(std::vector<py::object> tensors,
        F method){
  try {
    std::vector<py::object> result_tensors;
    for (py::object item : tensors) {
      result_tensors.push_back(method(item.cast<at::Tensor>()));
    }
    return result_tensors;
  } catch (std::exception e) {
    std::vector<py::object> result_tensors;
    for (py::object item : tensors) {
      std::vector<py::object> map_method_result =
          map_method(item.cast<std::vector<py::object>>(), method);
      result_tensors.push_back(py::cast(map_method_result));
    }
    return result_tensors;
  }
}

struct TORCH_API _ListNestedTensor {
  _ListNestedTensor(std::vector<py::object> tensors)
      : _tensors(tensors), _first_tensor(get_first_tensor(_tensors)),
        _nested_dim(get_nested_dim(_tensors)) {}
  size_t element_size() { return _first_tensor.element_size(); }
  std::vector<py::object> unbind() { return _tensors; }
  _ListNestedTensor requires_grad_(bool requires_grad) {
    return _ListNestedTensor(map_method(_tensors, [requires_grad](at::Tensor tensor) -> py::object {
      return py::reinterpret_borrow<py::object>(
          torch::autograd::utils::wrap(tensor.requires_grad_(requires_grad)));
    }));
  }
  std::vector<py::object> nested_size() {
    return map_method(_tensors, [](at::Tensor tensor) -> py::object {
      return py::reinterpret_borrow<py::object>(
          torch::autograd::utils::wrap(tensor.sizes()));
    });
  }
  int64_t nested_dim() { return _nested_dim; }
  py::object get_dtype() {
    return py::reinterpret_borrow<py::object>(torch::autograd::utils::wrap(torch::getDtype(_first_tensor.scalar_type())));
  }
  py::object get_layout() {
    return py::reinterpret_borrow<py::object>(torch::autograd::utils::wrap(torch::getLayout(_first_tensor.type().backend())));
  }
  py::object get_device() {
    return py::reinterpret_borrow<py::object>(THPDevice_New(_first_tensor.device()));
  }
  bool requires_grad() {
    return _first_tensor.requires_grad();
  }

private:
  std::vector<py::object> _tensors;
  const at::Tensor _first_tensor;
  int64_t _nested_dim;
};

void initialize_python_bindings();

} // namespace nestedtensor
} // namespace torch
