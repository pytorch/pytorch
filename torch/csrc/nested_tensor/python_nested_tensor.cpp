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

PyObject *_ListNestedTensorVariableClass = nullptr;

// template <class F> std::vector<py::object> map_fn(_ListNestedTensor nt, F fn)
// {
//  std::vector<py::object> result;
//  if (nt.nested_dim() == 1) {
//    for (_NestedNode node : nt.get_structure()._children) {
//      result.push_back(fn(node._tensor_node._tensor));
//    }
//  } else {
//    for (_ListNestedTensor nti : nt.get_structure()._children) {
//      result.push_back(py::cast(map_fn(nti, fn)));
//    }
//  }
//  return result;
// }

// std::vector<py::object> _ListNestedTensor::nested_size() {
//   return map_fn(*this, [](at::Tensor tensor) -> py::object {
//     return py::reinterpret_borrow<py::object>(
//         torch::autograd::utils::wrap(tensor.sizes()));
//   });
// }

// std::vector<py::object> _ListNestedTensor::nested_stride() {
//   return map_fn(*this, [](at::Tensor tensor) -> py::object {
//     return py::reinterpret_borrow<py::object>(
//         torch::autograd::utils::wrap(tensor.strides()));
//   });
// }
//

PyObject *_ListNestedTensorVariable_unbind(PyObject *self_) {
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  auto children = self.get_structure()._children;
  PyObject *return_list = PyList_New(children.size());
  if (return_list == NULL) {
    throw python_error();
  }
  for (size_t i = 0; i < children.size(); i++) {
    if (children[i]._children.size() == 0) {
      if (PyList_SetItem(
              return_list, i,
              THPVariable_Wrap(children[i]._variable_node._variable)) == -1) {
        throw python_error();
      }
    } else {
      if (PyList_SetItem(return_list, i, _ListNestedTensorVariable_Wrap(
                                             _ListNestedTensor(children[i]))) ==
          -1) {
        throw python_error();
      }
    }
  }
  return return_list;
}

std::string _NestedNode___str__(const _NestedNode &nested_node) {
  std::stringstream result;
  if (nested_node._children.size() == 0) {
    PyObject *objectsRepresentation =
        PyObject_Str(THPVariable_Wrap(nested_node._variable_node._variable));
    result << PyBytes_AsString(PyUnicode_AsUTF8String(objectsRepresentation));
    return result.str();
  } else {
    result << "nested_tensor([";
    result << std::endl;
    for (_NestedNode node : nested_node._children) {
      result << "  ";
      result << _NestedNode___str__(node);
      result << ",";
      result << std::endl;
    }
    result << "])";
    return result.str();
  }
}

PyObject *_ListNestedTensorVariable___str__(PyObject *self_) {
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  std::string str = _NestedNode___str__(self.get_structure());
  return PyUnicode_FromStringAndSize(str.c_str(), str.size());
}

PyObject *_ListNestedTensorVariable_requires_grad(PyObject *self_) {
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return torch::autograd::utils::wrap(self.requires_grad());
}

PyObject *_ListNestedTensorVariable_is_pinned(PyObject *self_) {
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return torch::autograd::utils::wrap(self.is_pinned());
}

PyObject *_ListNestedTensorVariable_is_contiguous(PyObject *self_) {
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return torch::autograd::utils::wrap(self.is_contiguous());
}

PyObject *_ListNestedTensorVariable_dim(PyObject *self_) {
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return torch::autograd::utils::wrap(self.dim());
}

PyObject *_ListNestedTensorVariable_numel(PyObject *self_) {
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return torch::autograd::utils::wrap(self.numel());
}

PyObject *_ListNestedTensorVariable___repr__(PyObject *self_) {
  // NOTE: Don't delete this. repr is an important concept, this
  // implementation is just faulty due to torch.Tensor.__repr__
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  // TODO: Assuming that there is no difference in __str__ and __repr__ for
  // torch.Tensor.
  std::string str = _NestedNode___str__(self.get_structure());
  return PyUnicode_FromStringAndSize(str.c_str(), str.size());
}

static void _ListNestedTensorVariable_dealloc(_ListNestedTensorVariable *self) {
  // PyObject_GC_UnTrack(self);
  // THPVariable_clear(self);
  self->cdata.~_ListNestedTensor();
  Py_TYPE(self)->tp_free((PyObject *)self);
}

// Creates a new Python object for a Variable. The Variable must not already
// have a PyObject* associated with it.
static PyObject *
_ListNestedTensorVariable_NewWithVar(PyTypeObject *type,
                                     _ListNestedTensor nested_tensor) {
  PyObject *obj = type->tp_alloc(type, 0);
  if (obj) {
    auto v = (_ListNestedTensorVariable *)obj;
    new (&v->cdata) _ListNestedTensor(std::move(nested_tensor));
    // v->cdata.set_pyobj(obj);
    return obj;
  } else {
    throw python_error();
  }
}

static PyObject *_ListNestedTensorVariable_pynew(PyTypeObject *type,
                                                 PyObject *args,
                                                 PyObject *kwargs) {
  PyObject *listObj;
  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &listObj)) {
    throw std::runtime_error("invalid arguments");
  }
  return _ListNestedTensorVariable_NewWithVar(
      type, std::move(_ListNestedTensor(_get_structure(listObj))));
}

// inline Tensor dispatch_sigmoid(const Tensor & self, Tensor out) {
//
//   AutoNoGIL no_gil;
//   return at::sigmoid_out(out, self);
// }
//
// static PyObject * THPVariable_sigmoid(PyObject* self_, PyObject* args)
// {
//   HANDLE_TH_ERRORS
//
//   auto& self = reinterpret_cast<THPVariable*>(self_)->cdata;
//   return wrap(dispatch_sigmoid(self));
//   END_HANDLE_TH_ERRORS
// }
//
PyObject *_ListNestedTensorVariable_Wrap(_ListNestedTensor var) {
  return _ListNestedTensorVariable_NewWithVar(
      (PyTypeObject *)_ListNestedTensorVariableClass, std::move(var));
}

static PyObject *_ListNestedTensorVariable_element_size(PyObject *self_) {
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return torch::autograd::utils::wrap(self.element_size());
}

static PyObject *_ListNestedTensorVariable_pin_memory(PyObject *self_) {
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return _ListNestedTensorVariable_Wrap(self.pin_memory());
}

static PyObject *_ListNestedTensorVariable_grad(PyObject *self_) {
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return _ListNestedTensorVariable_Wrap(self.grad());
}

static PyObject *_ListNestedTensorVariable_detach(PyObject *self_) {
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  return _ListNestedTensorVariable_Wrap(self.detach());
}

static PyObject *_ListNestedTensorVariable_requires_grad_(PyObject *self_,
                                                          PyObject *bool_arg) {
  if (not PyBool_Check(bool_arg)) {
    throw std::runtime_error("Argument must be bool.");
  }
  auto &self = reinterpret_cast<_ListNestedTensorVariable *>(self_)->cdata;
  if (PyObject_IsTrue(bool_arg)) {
    return _ListNestedTensorVariable_Wrap(self.requires_grad_(true));
  } else {
    return _ListNestedTensorVariable_Wrap(self.requires_grad_(false));
  }
}

static PyMethodDef _ListNestedTensorVariable_methods[] = {
    {"element_size", (PyCFunction)_ListNestedTensorVariable_element_size,
     METH_NOARGS, "Return element size."},
    {"pin_memory", (PyCFunction)_ListNestedTensorVariable_pin_memory,
     METH_NOARGS, "Pins memory."},
    {"grad", (PyCFunction)_ListNestedTensorVariable_grad, METH_NOARGS,
     "Returns grad."},
    {"detach", (PyCFunction)_ListNestedTensorVariable_detach, METH_NOARGS,
     "Detaches and returns."},
    {"requires_grad_", (PyCFunction)_ListNestedTensorVariable_requires_grad_,
     METH_O, "requires_grad_ and returns."},
    {"requires_grad", (PyCFunction)_ListNestedTensorVariable_requires_grad,
     METH_NOARGS, "Returns requires_grad."},
    {"is_pinned", (PyCFunction)_ListNestedTensorVariable_is_pinned, METH_NOARGS,
     "Returns is_pinned."},
    {"is_contiguous", (PyCFunction)_ListNestedTensorVariable_is_contiguous,
     METH_NOARGS, "Returns is_contiguous."},
    {"numel", (PyCFunction)_ListNestedTensorVariable_numel, METH_NOARGS,
     "Returns numel."},
    {"dim", (PyCFunction)_ListNestedTensorVariable_dim, METH_NOARGS,
     "Returns dim."},
    {"unbind", (PyCFunction)_ListNestedTensorVariable_unbind, METH_NOARGS,
     "Returns unbound components."},
    {NULL} /* Sentinel */
};

// TODO: "Py_TPFLAGS_DEFAULT enables all memebers defined until Python 3.3"
// https://docs.python.org/3/extending/newtypes_tutorial.html
// Does that mean it won't work before Python 3.3?
PyTypeObject _ListNestedTensorVariableType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch._ListNestedTensor", /* tp_name */
    sizeof(_ListNestedTensorVariable),             /* tp_basicsize */
    0,                                             /* tp_itemsize */
    (destructor)_ListNestedTensorVariable_dealloc, /* tp_dealloc */
    nullptr,                                       /* tp_print */
    nullptr,                                       /* tp_getattr */
    nullptr,                                       /* tp_setattr */
    nullptr,                                       /* tp_reserved */
    (reprfunc)_ListNestedTensorVariable___repr__,  /* tp_repr */
    nullptr,                                       /* tp_as_number */
    nullptr,                                       /* tp_as_sequence */
    nullptr,                                       /* tp_as_mapping */
    nullptr,                                       /* tp_hash  */
    nullptr,                                       /* tp_call */
    (reprfunc)_ListNestedTensorVariable___str__,   /* tp_str */
    nullptr,                                       /* tp_getattro */
    nullptr,                                       /* tp_setattro */
    nullptr,                                       /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                            /* tp_flags */
    nullptr,                                       /* tp_doc */
    nullptr,                                       /* tp_traverse */
    nullptr,                                       /* tp_clear */
    nullptr,                                       /* tp_richcompare */
    0,                                             /* tp_weaklistoffset */
    nullptr,                                       /* tp_iter */
    nullptr,                                       /* tp_iternext */
    _ListNestedTensorVariable_methods,             /* tp_methods */
    nullptr,                                       /* tp_members */
    nullptr,                                       /* tp_getset */
    nullptr,                                       /* tp_base */
    nullptr,                                       /* tp_dict */
    nullptr,                                       /* tp_descr_get */
    nullptr,                                       /* tp_descr_set */
    0,                                             /* tp_dictoffset */
    nullptr,                                       /* tp_init */
    nullptr,                                       /* tp_alloc */
    _ListNestedTensorVariable_pynew,               /* tp_new */
};

void initialize_python_bindings() {
  PyObject *m;
  if (PyType_Ready(&_ListNestedTensorVariableType) < 0) {
    throw python_error();
  }

  m = PyImport_ImportModule("torch");
  if (!m) {
    throw python_error();
  }

  Py_INCREF(&_ListNestedTensorVariableType);
  if (PyModule_AddObject(m, "_ListNestedTensor",
                         (PyObject *)&_ListNestedTensorVariableType) < 0) {
    Py_DECREF(&_ListNestedTensorVariableType);
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
