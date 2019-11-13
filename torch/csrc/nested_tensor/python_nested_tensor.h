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
#include <torch/include/torch/csrc/Exceptions.h>
#include <typeinfo>

namespace torch {
namespace nested_tensor {

namespace py = pybind11;

struct _ListNestedTensor;

struct _VariableNode {
  _VariableNode() {}
  _VariableNode(torch::autograd::Variable variable) : _variable(variable) {}

  torch::autograd::Variable _variable;
};

PyObject *_ListNestedTensorVariable_Wrap(_ListNestedTensor);

THP_API PyObject *_ListNestedTensorVariableClass;

// If is_leaf tensors are available, otherwise children.
struct _NestedNode {
  // _NestedNode() = delete;
  // _NestedNode(const _NestedNode&) = delete;
  // _NestedNode(_NestedNode&&) = delete;

  _NestedNode() {}
  _NestedNode(const std::vector<_NestedNode> children) : _children(children) {}
  _NestedNode(_VariableNode variable_node) : _variable_node(variable_node) {}
  const std::vector<_NestedNode> _children;
  _VariableNode _variable_node;
};

static size_t _num_tensor(const _NestedNode &meta_node) {
  size_t result = 0;
  for (size_t i = 0; i < meta_node._children.size(); i++) {
    result += _num_tensor(meta_node._children[i]);
  }
  return result;
}

static int64_t _numel(const _NestedNode &meta_node) {
  if (meta_node._children.size() == 0) {
    return meta_node._variable_node._variable.numel();
  } else {
    int64_t result = 0;
    for (size_t i = 0; i < meta_node._children.size(); i++) {
      result += _numel(meta_node._children[i]);
    }
    return result;
  }
}

static _NestedNode _get_structure(PyObject *tensors) {
  if (THPVariable_Check(tensors)) {
    auto variable_ = THPVariable_Unpack(tensors);
    auto variable = make_variable_view(variable_, variable_);
    return _NestedNode(_VariableNode(variable));
  } else {
    std::vector<_NestedNode> meta_nodes;
    Py_ssize_t i, n;
    n = PyObject_Length(tensors);
    PyObject *item;
    if (n < 0) {
      throw python_error();
    }
    for (i = 0; i < n; i++) {
      item = PyList_GetItem(tensors, i);
      _NestedNode node = _get_structure(item);
      meta_nodes.push_back(node);
    }
    return _NestedNode(meta_nodes);
  }
}

static torch::autograd::Variable _get_first_variable(_NestedNode nested_node) {
  const _NestedNode *start = &nested_node;
  while (start->_children.size()) {
    start = &start->_children[0];
  }
  return start->_variable_node._variable;
}

static std::vector<at::IntArrayRef> _get_flat_sizes(_NestedNode nested_node) {
  if (nested_node._children.size() == 0) {
    return std::vector<at::IntArrayRef>(
        {nested_node._variable_node._variable.sizes()});
  } else {
    std::vector<at::IntArrayRef> flat_sizes;
    for (size_t i = 0; i < nested_node._children.size(); i++) {
      auto flat_sizes_i = _get_flat_sizes(nested_node._children[i]);
      for (size_t j = 0; j < flat_sizes_i.size(); j++) {
        flat_sizes.push_back(flat_sizes_i[j]);
      }
    }
    return flat_sizes;
  }
}

template <class F> static _NestedNode map(_NestedNode nested_node, F fn) {
  if (nested_node._children.size() == 0) {
    _NestedNode new_nested_node(
        _VariableNode(fn(nested_node._variable_node._variable)));
    return new_nested_node;
  } else {
    std::vector<_NestedNode> new_children;
    for (size_t i = 0; i < nested_node._children.size(); i++) {
      new_children.push_back(_NestedNode(map(nested_node._children[i], fn)));
    }
    return _NestedNode(new_children);
  }
}

template <class F>
static void apply2(_NestedNode nested_node1, _NestedNode nested_node2, F fn) {
  if (nested_node1._children.size() == 0) {
    fn(nested_node1._variable_node._variable,
       nested_node2._variable_node._variable);
  } else {
    for (size_t i = 0; i < nested_node1._children.size(); i++) {
      apply2(nested_node1._children[i], nested_node2._children[i], fn);
    }
  }
}

TORCH_API extern PyTypeObject _ListNestedTensorVariableType;

// TODO: Eventually allow construction from a list of _BufferNestedTensors.

struct TORCH_API _ListNestedTensor {
  _ListNestedTensor() = delete;
  // _ListNestedTensor(_ListNestedTensor& other) : _structure(other._structure),
  // _first_variable(_get_first_variable(_structure)) {}
  // _ListNestedTensor(_ListNestedTensor&&) = delete;

  // _ListNestedTensor(std::vector<py::object> tensors)
  //     : _ListNestedTensor(_get_structure(tensors)) {}
  _ListNestedTensor(_NestedNode structure)
      : _structure(structure),
        _first_variable(_get_first_variable(_structure)) {}
  int64_t element_size() { return _first_variable.element_size(); }
  // py::tuple size(int64_t dim) { return py::make_tuple(py::none(),
  // py::none()); }
  _ListNestedTensor pin_memory() {
    return _ListNestedTensor(
        map(_structure, [](at::Tensor tensor) -> at::Tensor {
          return tensor.pin_memory();
        }));
  }
  _ListNestedTensor grad() {
    return _ListNestedTensor(
        map(_structure,
            [](at::Tensor tensor) -> at::Tensor { return tensor.grad(); }));
  }
  _ListNestedTensor detach() {
    return _ListNestedTensor(
        map(_structure,
            [](at::Tensor tensor) -> at::Tensor { return tensor.detach(); }));
  }
  _ListNestedTensor requires_grad_(bool requires_grad) {
    return _ListNestedTensor(
        map(_structure, [requires_grad](at::Tensor tensor) -> at::Tensor {
          return tensor.requires_grad_(requires_grad);
        }));
  }
  // void backward(_ListNestedTensor gradient, bool retain_graph,
  //               bool create_graph) {
  //   apply2(_structure, gradient.get_structure(),
  //          [retain_graph, create_graph](at::Tensor &tensor1,
  //                                       const at::Tensor &tensor2) {
  //            tensor1.backward(tensor2, retain_graph, create_graph);
  //          });
  // }
  // // Only works if nested_dim() higher than 1.
  // std::vector<py::object> nested_size();
  // std::vector<py::object> nested_stride();
  // int64_t __len__() { return _structure._children.size(); }
  int64_t nested_dim() {
    const _NestedNode *start_structure = &_structure;
    int64_t depth = 0;
    while (start_structure->_children.size()) {
      depth++;
      start_structure = &start_structure->_children[0];
    }
    return depth;
  }
  // py::object get_dtype() {
  //   return py::reinterpret_borrow<py::object>(torch::autograd::utils::wrap(
  //       torch::getDtype(_first_tensor.scalar_type())));
  // }
  // py::object get_layout() {
  //   return py::reinterpret_borrow<py::object>(torch::autograd::utils::wrap(
  //       torch::getLayout(_first_tensor.type().backend())));
  // }
  // py::object get_device() {
  //   return py::reinterpret_borrow<py::object>(
  //       THPDevice_New(_first_tensor.device()));
  // }
  bool requires_grad() { return _first_variable.requires_grad(); }
  int64_t dim() { return _first_variable.dim() + nested_dim(); }
  int64_t numel() { return _numel(_structure); }
  bool is_pinned() { return _first_variable.is_pinned(); }
  bool is_contiguous() { return false; }
  _NestedNode get_structure() { return _structure; }
  // TODO: Implement these and call into them isntead of implementing them
  // separately in Variable dispatch functions.
  // std::vector<py::object> unbind();
  // std::string __str__();
  // std::string __repr__();

private:
  const _NestedNode _structure;
  const torch::autograd::Variable _first_variable;
};

struct TORCH_API _ListNestedTensorVariable {
  PyObject_HEAD
      /* Type-specific fields go here. */
      _ListNestedTensor cdata;
};

// template <class R, class F> std::vector<R> map_fn(const _ListNestedTensor,
// F);

void initialize_python_bindings();

} // namespace nestedtensor
} // namespace torch
