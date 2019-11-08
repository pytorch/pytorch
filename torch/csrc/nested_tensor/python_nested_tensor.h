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
  _MetaNode(std::vector<_MetaNode> children)
      : is_leaf(false), _children(children) {}
  _MetaNode(std::vector<at::Tensor> tensors)
      : is_leaf(true), _tensors(tensors) {}
  bool is_leaf;
  std::vector<_MetaNode> _children;
  std::vector<at::Tensor> _tensors;
};

static size_t _num_tensor(_MetaNode meta_node) {
  if (meta_node.is_leaf) {
    return meta_node._tensors.size();
  } else {
    size_t result = 0;
    for (_MetaNode child : meta_node._children) {
      result += _num_tensor(child);
    }
    return result;
  }
}

static _MetaNode _get_meta_node(std::vector<py::object> tensors) {
  try {
    std::vector<at::Tensor> last_tensors;
    for (py::object item : tensors) {
      last_tensors.push_back(item.cast<at::Tensor>());
    }
    return _MetaNode(last_tensors);
  } catch (std::exception e) {
    std::vector<_MetaNode> meta_nodes;
    for (py::object item : tensors) {
      meta_nodes.push_back(
          _get_meta_node(item.cast<std::vector<py::object>>()));
    }
    return _MetaNode(meta_nodes);
  }
}

static std::vector<at::Tensor>
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
      std::vector<at::Tensor> item_flat_tensors =
          _get_flat_tensors(item.cast<std::vector<py::object>>());
      flat_tensors.insert(flat_tensors.end(), item_flat_tensors.begin(),
                          item_flat_tensors.end());
    }
    return flat_tensors;
  }
}

// static const at::Tensor get_first_tensor(std::vector<py::object> tensors) {
//   try {
//     return tensors[0].cast<const at::Tensor>();
//   } catch (std::exception e) {
//     return get_first_tensor(tensors[0].cast<std::vector<py::object>>());
//   }
// }

// static int64_t get_nested_dim(std::vector<py::object> tensors) {
//   try {
//     at::Tensor t = tensors[0].cast<const at::Tensor>();
//     return 1;
//   } catch (std::exception e) {
//     return 1 + get_nested_dim(tensors[0].cast<std::vector<py::object>>());
//   }
// }

template <class F>
static std::vector<at::Tensor>
map_flat_tensors(std::vector<at::Tensor> flat_tensors, F fn) {
  std::vector<at::Tensor> result_tensors;
  for (at::Tensor tensor : flat_tensors) {
    result_tensors.push_back(fn(tensor));
  }
  return result_tensors;
}

template <class F>
std::vector<py::object> map_meta_node(_MetaNode meta_node, F fn) {
  std::vector<py::object> result;
  if (meta_node.is_leaf) {
    for (at::Tensor tensor : meta_node._tensors) {
      result.push_back(fn(tensor));
    }
  } else {
    for (_MetaNode child : meta_node._children) {
      result.push_back(py::cast(map_meta_node(child, fn)));
    }
  }
  return result;
}

// TODO: Operate on _ListNestedTensor using nested_dim instead.
// TODO: Operate on flat list of Tensors, add nested_size constrctor
// TODO: Eventually allow construction from a list of _BufferNestedTensors.
// template <class F>
// static std::vector<at::Tensor> flat_map_method(std::vector<at::Tensor>
// tensors,
//                                                F method) {
//   std::vector<at::Tensor> result;
//   for (at::Tensor tensor : tensors) {
//     result.push_back(method(tensor));
//   }
//   return result;
//
//   //  try {
//   //    std::vector<py::object> result_tensors;
//   //    for (py::object item : tensors) {
//   //      result_tensors.push_back(method(item.cast<at::Tensor>()));
//   //    }
//   //    return result_tensors;
//   //  } catch (std::exception e) {
//   //    std::vector<py::object> result_tensors;
//   //    for (py::object item : tensors) {
//   //      std::vector<py::object> map_method_result =
//   //          map_method(item.cast<std::vector<py::object>>(), method);
//   //      result_tensors.push_back(py::cast(map_method_result));
//   //    }
//   //    return result_tensors;
//   //  }
// }

// template <class F>
// static void flat_apply2_method(std::vector<at::Tensor> tensors1,
//                                std::vector<at::Tensor> tensors2, F method) {
//   for (size_t i = 0; i < tensors1.size(); i++) {
//     method(tensors1[i], tensors2[i]);
//   }
// }

struct TORCH_API _ListNestedTensor {
  _ListNestedTensor(std::vector<py::object> tensors)
      : _ListNestedTensor(_get_flat_tensors(tensors), _get_meta_node(tensors)) {
  }
  _ListNestedTensor(std::vector<at::Tensor> flat_tensors, _MetaNode structure)
      : _flat_tensors(flat_tensors), _structure(structure) {
    _first_tensor = _flat_tensors[0];
  }
  //      : _tensors(tensors), _first_tensor(get_first_tensor(_tensors)),
  //        _nested_dim(get_nested_dim(_tensors)), _structure(get_meta_node) {}
  size_t element_size() { return _flat_tensors[0].element_size(); }
  // std::vector<py::object> unbind() { return _tensors; }
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
    auto gradient_tensors = gradient.get_flat_tensors();
    for (size_t i = 0; i < _flat_tensors.size(); i++) {
      _flat_tensors[i].backward(gradient_tensors[i], retain_graph,
                                create_graph);
    }
  }
  std::vector<py::object> unbind() {
    std::vector<py::object> result;
    if (nested_dim() == 1) {
      for (at::Tensor tensor : _flat_tensors) {
        result.push_back(py::reinterpret_borrow<py::object>(
            torch::autograd::utils::wrap(tensor)));
      }
    } else {
      size_t i = 0;
      for (_MetaNode child : _structure._children) {
        result.push_back(py::cast(_ListNestedTensor(
            std::vector<at::Tensor>(_flat_tensors.begin() + i,
                                    _flat_tensors.begin() + i +
                                        _num_tensor(child)),
            child)));
        i += _num_tensor(child);
      }
      return result;
    }
  }
  std::vector<py::object> nested_size() {
    return map_meta_node(_structure, [](at::Tensor tensor) -> py::object {
      return py::reinterpret_borrow<py::object>(
          torch::autograd::utils::wrap(tensor.sizes()));
    });
  }
  std::vector<py::object> nested_stride() {
    return map_meta_node(_structure, [](at::Tensor tensor) -> py::object {
      return py::reinterpret_borrow<py::object>(
          torch::autograd::utils::wrap(tensor.strides()));
    });
  }
  int64_t __len__() {
    return _flat_tensors.size() ? nested_dim() == 1
                                : _structure._children.size();
  }
  int64_t nested_dim() {
    if (_structure.is_leaf) {
      return 1;
    } else {
      _MetaNode start_structure = _structure;
      int64_t depth = 1;
      while (not start_structure.is_leaf) {
        depth++;
        start_structure = start_structure._children[0];
      }
      return depth;
    }
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
  std::vector<at::Tensor> get_flat_tensors() { return _flat_tensors; }

private:
  std::vector<at::Tensor> _flat_tensors;
  at::Tensor _first_tensor;
  _MetaNode _structure;
};

void initialize_python_bindings();

} // namespace nestedtensor
} // namespace torch
