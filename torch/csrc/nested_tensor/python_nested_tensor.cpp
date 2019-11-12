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

template <class F> std::vector<py::object> map_fn(_ListNestedTensor nt, F fn) {
  std::vector<py::object> result;
  if (nt.nested_dim() == 1) {
    for (_NestedNode node : nt.get_structure()._children) {
      result.push_back(fn(node._tensor_node._tensor));
    }
  } else {
    for (_ListNestedTensor nti : nt.get_structure()._children) {
      result.push_back(py::cast(map_fn(nti, fn)));
    }
  }
  return result;
}

std::vector<py::object> _ListNestedTensor::nested_size() {
  return map_fn(*this, [](at::Tensor tensor) -> py::object {
    return py::reinterpret_borrow<py::object>(
        torch::autograd::utils::wrap(tensor.sizes()));
  });
}

std::vector<py::object> _ListNestedTensor::nested_stride() {
  return map_fn(*this, [](at::Tensor tensor) -> py::object {
    return py::reinterpret_borrow<py::object>(
        torch::autograd::utils::wrap(tensor.strides()));
  });
}

std::vector<py::object> _ListNestedTensor::unbind() {
  std::vector<py::object> result;
  if (nested_dim() == 1) {
    for (_NestedNode node : _structure._children) {
      result.push_back(py::reinterpret_borrow<py::object>(
          torch::autograd::utils::wrap(node._tensor_node._tensor)));
    }
  } else {
    for (_NestedNode node : _structure._children) {
      result.push_back(py::cast(_ListNestedTensor(node)));
    }
  }
  return result;
}

std::string _ListNestedTensor::__str__() {
  std::stringstream result;
  if (nested_dim() == 1) {
    for (_NestedNode node : _structure._children) {
      result << "  ";
      result << node._tensor_node._tensor;
      result << ",";
      result << std::endl;
    }
  } else {
    for (_NestedNode node : _structure._children) {
      _ListNestedTensor nt(node);
      result << "  ";
      result << nt.__str__();
      result << ",";
      result << std::endl;
    }
  }
  result << "])";
  return result.str();
}

std::string _ListNestedTensor::__repr__() {
  std::stringstream result;
  if (nested_dim() == 1) {
    for (_NestedNode node : _structure._children) {
      result << "  ";
      // TODO: There appears to be no difference between
      // __str__ and __repr__ for torch.Tensor.
      result << node._tensor_node._tensor;
      result << ",";
      result << std::endl;
    }
  } else {
    for (_NestedNode node : _structure._children) {
      _ListNestedTensor nt(node);
      result << "  ";
      result << nt.__repr__();
      result << ",";
      result << std::endl;
    }
  }
  result << "])";
  return result.str();
}

// PyTypeObject *_init_ListNestedTensor2TypeObject(PyTypeObject &type) {
//   // TODO: Necessary for NestedTensor as well?
//   // NOTE: we don't use the typical static declaration of PyTypeObject because
//   // we need to initialize as many types as there are VariableType instances.
//   // The typical PyVarObject_HEAD_INIT(nullptr, 0) is described in the Python
//   // documentation: it initializes the refcnt to 1 and the other object header
//   // fields to zero.
//   memset(&type, 0, sizeof(PyTypeObject));
//   ((PyObject *)&type)->ob_refcnt = 1;
//   ((PyObject *)&type)->ob_type = &_ListNestedTensor2Type;
//   type.tp_basicsize = sizeof(_ListNestedTensor2Type);
//   type.tp_flags = Py_TPFLAGS_DEFAULT;
//   type.tp_name = "torch._ListNestedTensor2";
//   type.tp_new = PyType_GenericNew;
// 
//   // type.tp_doc = "Custom objects";
//   // type.tp_itemsize = 0;
// 
//   // type.tp_basicsize = sizeof(_ListNestedTensor2Type);
//   // type.tp_flags = Py_TPFLAGS_DEFAULT;
//   // type.tp_name = "_ListNestedTensor2";
//   // type.tp_new = PyType_GenericNew;
//   if (PyType_Ready(&type) < 0) {
//     throw python_error();
//   }
//   return &type;
// }

PyTypeObject _ListNestedTensor2Type = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch._ListNestedTensor2", /* tp_name */
    sizeof(_ListNestedTensor2), /* tp_basicsize */
    0,                          /* tp_itemsize */
    nullptr,                    /* tp_dealloc */
    nullptr,                    /* tp_print */
    nullptr,                    /* tp_getattr */
    nullptr,                    /* tp_setattr */
    nullptr,                    /* tp_reserved */
    nullptr,                    /* tp_repr */
    nullptr,                    /* tp_as_number */
    nullptr,                    /* tp_as_sequence */
    nullptr,                    /* tp_as_mapping */
    nullptr,                    /* tp_hash  */
    nullptr,                    /* tp_call */
    nullptr,                    /* tp_str */
    nullptr,                    /* tp_getattro */
    nullptr,                    /* tp_setattro */
    nullptr,                    /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,         /* tp_flags */
    nullptr,                    /* tp_doc */
    nullptr,                    /* tp_traverse */
    nullptr,                    /* tp_clear */
    nullptr,                    /* tp_richcompare */
    0,                          /* tp_weaklistoffset */
    nullptr,                    /* tp_iter */
    nullptr,                    /* tp_iternext */
    nullptr,                    /* tp_methods */
    nullptr,                    /* tp_members */
    nullptr,                    /* tp_getset */
    nullptr,                    /* tp_base */
    nullptr,                    /* tp_dict */
    nullptr,                    /* tp_descr_get */
    nullptr,                    /* tp_descr_set */
    0,                          /* tp_dictoffset */
    nullptr,                    /* tp_init */
    nullptr,                    /* tp_alloc */
    nullptr,                    /* tp_new */
};

void initialize_python_bindings() {
  PyObject *m;
  if (PyType_Ready(&_ListNestedTensor2Type) < 0) {
    throw python_error();
  }

  m = PyImport_ImportModule("torch");
  if (!m) {
    throw python_error();
  }

  Py_INCREF(&_ListNestedTensor2Type);
  if (PyModule_AddObject(m, "_ListNestedTensor2", (PyObject *)&_ListNestedTensor2Type) < 0) {
    Py_DECREF(&_ListNestedTensor2Type);
    Py_DECREF(m);
    throw python_error();
  }

  // auto obj = py::module::import("torch");
  // auto m = py::handle(obj).cast<py::module>();

  // py::class_<_ListNestedTensor>(m, "_ListNestedTensor")
  //     .def(py::init<std::vector<py::object>>())
  //     .def_property_readonly("dtype", &_ListNestedTensor::get_dtype)
  //     .def_property_readonly("layout", &_ListNestedTensor::get_layout)
  //     .def_property_readonly("device", &_ListNestedTensor::get_device)
  //     .def_property_readonly("requires_grad",
  //     &_ListNestedTensor::requires_grad)
  //     .def_property_readonly("grad", &_ListNestedTensor::grad)
  //     .def("detach", &_ListNestedTensor::detach)
  //     .def("pin_memory", &_ListNestedTensor::pin_memory)
  //     .def("backward", &_ListNestedTensor::backward)
  //     .def("requires_grad_", &_ListNestedTensor::requires_grad_)
  //     .def("element_size", &_ListNestedTensor::element_size)
  //     .def("size", &_ListNestedTensor::size)
  //     .def("unbind", &_ListNestedTensor::unbind)
  //     .def("nested_size", &_ListNestedTensor::nested_size)
  //     .def("nested_stride", &_ListNestedTensor::nested_stride)
  //     .def("is_pinned", &_ListNestedTensor::is_contiguous)
  //     .def("is_contiguous", &_ListNestedTensor::is_contiguous)
  //     .def("__len__", &_ListNestedTensor::__len__)
  //     .def("__str__", &_ListNestedTensor::__str__)
  //     .def("dim", &_ListNestedTensor::dim)
  //     .def("numel", &_ListNestedTensor::numel)
  //     .def("nested_dim", &_ListNestedTensor::nested_dim);
}
}
}
