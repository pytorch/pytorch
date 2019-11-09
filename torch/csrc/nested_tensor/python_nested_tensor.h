#pragma once

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorTypeId.h>
#include <exception>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/tensor/python_tensor.h>
#include <typeinfo>

namespace torch {
namespace nested_tensor {

namespace py = pybind11;

struct _ListNestedTensor;

// If is_leaf tensors are available, otherwise children.
struct _MetaNode {
  // _MetaNode() = delete;
  // _MetaNode(const _MetaNode&) = delete;
  // _MetaNode(_MetaNode&&) = delete;

  _MetaNode() {}
  _MetaNode(const std::vector<_MetaNode> children) : _children(children) {}
  const std::vector<_MetaNode> _children;
};

static size_t _num_tensor(const _MetaNode &meta_node) {
  size_t result = 0;
  for (size_t i = 0; i < meta_node._children.size(); i++) {
    result += _num_tensor(meta_node._children[i]);
  }
  return result;
}

static _MetaNode _get_meta_node(std::vector<py::object> tensors) {
  std::vector<_MetaNode> meta_nodes;
  try {
    for (py::object item : tensors) {
      item.cast<at::Tensor>();
      meta_nodes.push_back(_MetaNode());
    }
  } catch (std::exception e) {
    for (py::object item : tensors) {
      meta_nodes.push_back(
          _get_meta_node(item.cast<std::vector<py::object>>()));
    }
  }
  return _MetaNode(meta_nodes);
}

static const std::vector<at::Tensor>
_get_flat_tensors(std::vector<py::object> tensors) {
  try {
    std::vector<at::Tensor> last_tensors;
    for (py::object item : tensors) {
      last_tensors.push_back(item.cast<at::Tensor>());
    }
    return last_tensors;
  } catch (std::exception e) {
    std::vector<at::Tensor> flat_tensors;
    for (py::object item : tensors) {
      const std::vector<at::Tensor> item_flat_tensors =
          _get_flat_tensors(item.cast<std::vector<py::object>>());
      for (at::Tensor tensor : item_flat_tensors) {
        flat_tensors.push_back(tensor);
      }
    }
    return flat_tensors;
  }
}

template <class F>
static std::vector<at::Tensor>
map_flat_tensors(std::vector<at::Tensor> flat_tensors, F fn) {
  std::vector<at::Tensor> result_tensors;
  for (at::Tensor tensor : flat_tensors) {
    result_tensors.push_back(fn(tensor));
  }
  return result_tensors;
}

// TODO: Eventually allow construction from a list of _BufferNestedTensors.

struct TORCH_API _ListNestedTensor {
  _ListNestedTensor() = delete;
  // _ListNestedTensor(const _ListNestedTensor&) = delete;
  // _ListNestedTensor(_ListNestedTensor&&) = delete;

  _ListNestedTensor(std::vector<py::object> tensors)
      : _ListNestedTensor(_get_flat_tensors(tensors), _get_meta_node(tensors)) {
  }
  _ListNestedTensor(const std::vector<at::Tensor> flat_tensors,
                    _MetaNode structure)
      : _flat_tensors(flat_tensors), _structure(structure),
        _first_tensor(_flat_tensors[0]) {}
  size_t element_size() { return _flat_tensors[0].element_size(); }
  _ListNestedTensor grad() {
    return _ListNestedTensor(
        map_flat_tensors(
            _flat_tensors,
            [](at::Tensor tensor) -> at::Tensor { return tensor.grad(); }),
        _structure);
  }
  _ListNestedTensor detach() {
    return _ListNestedTensor(
        map_flat_tensors(
            _flat_tensors,
            [](at::Tensor tensor) -> at::Tensor { return tensor.detach(); }),
        _structure);
  }
  _ListNestedTensor requires_grad_(bool requires_grad) {
    return _ListNestedTensor(
        map_flat_tensors(_flat_tensors,
                         [requires_grad](at::Tensor tensor) -> at::Tensor {
                           return tensor.requires_grad_(requires_grad);
                         }),
        _structure);
  }
  void backward(_ListNestedTensor gradient, bool retain_graph,
                bool create_graph) {
    auto gradient_tensors = gradient._flat_tensors;
    for (size_t i = 0; i < _flat_tensors.size(); i++) {
      _flat_tensors[i].backward(gradient_tensors[i], retain_graph,
                                create_graph);
    }
  }
  // Only works if nested_dim() higher than 1.
  std::vector<_ListNestedTensor> unbind_nested();
  std::vector<py::object> unbind();
  template <class F> std::vector<py::object> map_fn(F fn) {
    std::vector<py::object> result;
    if (nested_dim() == 1) {
      for (at::Tensor tensor : get_flat_tensors()) {
        result.push_back(fn(tensor));
      }
    } else {
      for (_ListNestedTensor nti : unbind_nested()) {
        result.push_back(py::cast(nti.map_fn(fn)));
      }
    }
    return result;
  }
  std::vector<py::object> nested_size() {
    return map_fn([](at::Tensor tensor) -> py::object {
      return py::reinterpret_borrow<py::object>(
          torch::autograd::utils::wrap(tensor.sizes()));
    });
  }
  std::vector<py::object> nested_stride() {
    return map_fn([](at::Tensor tensor) -> py::object {
      return py::reinterpret_borrow<py::object>(
          torch::autograd::utils::wrap(tensor.strides()));
    });
  }
  int64_t __len__() { return _structure._children.size(); }
  int64_t nested_dim() {
    const _MetaNode *start_structure = &_structure;
    int64_t depth = 1;
    while (start_structure->_children.size()) {
      depth++;
      start_structure = &start_structure->_children[0];
    }
    return depth;
  }
  py::object get_dtype() {
    return py::reinterpret_borrow<py::object>(torch::autograd::utils::wrap(
        torch::getDtype(_first_tensor.scalar_type())));
  }
  py::object get_layout() {
    return py::reinterpret_borrow<py::object>(torch::autograd::utils::wrap(
        torch::getLayout(_first_tensor.type().backend())));
  }
  py::object get_device() {
    return py::reinterpret_borrow<py::object>(
        THPDevice_New(_first_tensor.device()));
  }
  bool requires_grad() { return _first_tensor.requires_grad(); }
  int64_t dim() { return _first_tensor.dim() + nested_dim(); }
  bool is_pinned() { return _first_tensor.is_pinned(); }
  bool is_contiguous() { return true; }
  const _MetaNode get_structure() { return _structure; }
  const std::vector<at::Tensor> &get_flat_tensors() { return _flat_tensors; }
  std::string __str__();

private:
  const std::vector<at::Tensor> _flat_tensors;
  const _MetaNode _structure;
  const at::Tensor _first_tensor;
};

template <class R, class F> std::vector<R> map_fn(const _ListNestedTensor, F);

void initialize_python_bindings();

} // namespace nestedtensor
} // namespace torch
