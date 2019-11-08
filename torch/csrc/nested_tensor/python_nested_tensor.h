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
  _MetaNode(const std::vector<const _MetaNode> children)
      : is_leaf(false), _children(children) {}
  _MetaNode(const std::vector<at::Tensor> tensors)
      : is_leaf(true), _tensors(tensors) {}
  bool is_leaf;
  const std::vector<const _MetaNode> _children;
  const std::vector<at::Tensor> _tensors;
};

static const _MetaNode get_meta_node(std::vector<py::object> tensors) {
  try {
    std::vector<at::Tensor> last_tensors(tensors.size());
    for (py::object item : tensors) {
      last_tensors.push_back(item.cast<at::Tensor>());
    }
    return _MetaNode(last_tensors);
  } catch (std::exception e) {
    std::vector<const _MetaNode> meta_nodes;
    for (py::object item : tensors) {
      meta_nodes.push_back(get_meta_node(item.cast<std::vector<py::object>>()));
    }
    return _MetaNode(meta_nodes);
  }
}

static const std::vector<at::Tensor> get_flat_tensors(std::vector<py::object> tensors) {
  try {
    std::vector<at::Tensor> last_tensors(tensors.size());
    for (py::object item : tensors) {
      last_tensors.push_back(item.cast<at::Tensor>());
    }
    return last_tensors;
  } catch (std::exception e) {
    std::vector<at::Tensor> flat_tensors;
    for (py::object item : tensors) {
      for (at::Tensor tensor :
           get_flat_tensors(item.cast<std::vector<py::object>>())) {
        flat_tensors.push_back(tensor);
      }
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

static int64_t get_nested_dim(std::vector<py::object> tensors) {
  try {
    at::Tensor t = tensors[0].cast<const at::Tensor>();
    return 1;
  } catch (std::exception e) {
    return 1 + get_nested_dim(tensors[0].cast<std::vector<py::object>>());
  }
}

template <class F>
static const std::vector<at::Tensor>
map_flat_tensors(const std::vector<at::Tensor> flat_tensors, F fn) {
  std::vector<at::Tensor> result_tensors;
  for (at::Tensor tensor : flat_tensors) {
    result_tensors.push_back(fn(tensor));
  }
  return result_tensors;
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
      : _flat_tensors(get_flat_tensors(tensors)),
        _structure(get_meta_node(tensors)) {}
  _ListNestedTensor(const std::vector<at::Tensor> flat_tensors,
                    const _MetaNode structure)
      : _flat_tensors(flat_tensors), _structure(structure) {}
  //      : _tensors(tensors), _first_tensor(get_first_tensor(_tensors)),
  //        _nested_dim(get_nested_dim(_tensors)), _structure(get_meta_node) {}
  size_t element_size() { return _flat_tensors[0].element_size(); }
  std::vector<py::object> unbind() { return _tensors; }
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
  void backward(_ListNestedTensor gradient, bool retain_graph,
                bool create_graph) {
    for (int64_t i = 0; i < _flat_tensors.size(); i++) {

    for (at::Tensor tensor : _flat_tensors) {
      tensor
  _ListNestedTensor requires_grad_(bool requires_grad) {
    return _ListNestedTensor(
        map_method(_tensors, [requires_grad](at::Tensor tensor) -> py::object {
          return py::reinterpret_borrow<py::object>(
              torch::autograd::utils::wrap(
                  tensor.requires_grad_(requires_grad)));
        }));
  }
  std::vector<py::object> nested_size() {
    return map_method(_tensors, [](at::Tensor tensor) -> py::object {
      return py::reinterpret_borrow<py::object>(
          torch::autograd::utils::wrap(tensor.sizes()));
    });
  }
  std::vector<py::object> nested_stride() {
    return map_method(_tensors, [](at::Tensor tensor) -> py::object {
      return py::reinterpret_borrow<py::object>(
          torch::autograd::utils::wrap(tensor.strides()));
    });
  }
  int64_t __len__() { return _tensors.size(); }
  int64_t nested_dim() { return _nested_dim; }
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
  const std::vector<const at::Tensor> get_flat_tensors { return _flat_tensors; }

private:
  const std::vector<at::Tensor> _flat_tensors;
  const _MetaNode _structure;
  int64_t _nested_dim;
};

void initialize_python_bindings();

} // namespace nestedtensor
} // namespace torch
