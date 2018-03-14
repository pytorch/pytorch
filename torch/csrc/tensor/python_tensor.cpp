#include "python_tensor.h"

#include <structmember.h>
#include <pybind11/pybind11.h>
#include <sstream>

#include "torch/csrc/assertions.h"
#include "torch/csrc/Dtype.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/utils/cuda_lazy_init.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/utils/tensor_new.h"
#include "torch/csrc/utils/tensor_types.h"

namespace torch { namespace tensor {

using namespace at;
using namespace torch::autograd;

struct PyTensorType {
  PyTypeObject py_type;
  at::Type* aten_type;
  THPDtype *dtype;
  // The base tensor type i.e. `torch.Tensor`. All tensors are pass isinstance
  // checks on the base type.
  bool is_base_type;
  char name[64];
};

static_assert(std::is_standard_layout<PyTensorType>::value, "PyTensorType must be standard layout");

static PyTensorType* default_tensor_type;

static void py_bind_tensor_types(const std::vector<PyTensorType>& tensor_types);

static void py_bind_torch_storage(const PyTensorType& py_type);

static TypeError unavailable_type(const PyTensorType& type) {
  const char* cuda_msg = type.dtype->is_cuda ? ". Torch not compiled with CUDA enabled." : "";
  return TypeError("type %s not available%s", type.name, cuda_msg);
}

static PyObject* Tensor_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  HANDLE_TH_ERRORS
  auto& tensor_type = *((PyTensorType*)type);
  if (!tensor_type.aten_type) {
    throw unavailable_type(tensor_type);
  }
  if (tensor_type.dtype->is_cuda) {
    torch::utils::cuda_lazy_init();
  }
  return THPVariable_Wrap(torch::utils::legacy_tensor_ctor(*tensor_type.aten_type, args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject* Tensor_instancecheck(PyTensorType* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (THPVariable_Check(arg)) {
    if (self->is_base_type) {
      // Every tensor is treated as an instance of torch.Tensor
      Py_RETURN_TRUE;
    }
    auto& var = ((THPVariable*)arg)->cdata;
    if (&var.type() == self->aten_type) {
      Py_RETURN_TRUE;
    }
  }
  Py_RETURN_FALSE;
  END_HANDLE_TH_ERRORS
}

PyObject *Tensor_dtype(PyTensorType* self) {
  return torch::autograd::utils::wrap(self->dtype);
}

PyObject *Tensor_is_cuda(PyTensorType* self) {
  if (self->dtype->is_cuda) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

PyObject *Tensor_is_sparse(PyTensorType *self) {
  if (self->dtype->is_sparse) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

static struct PyMethodDef metaclass_methods[] = {
  {"__instancecheck__", (PyCFunction)Tensor_instancecheck, METH_O, NULL},
  {NULL}
};

typedef PyObject *(*getter)(PyObject *, void *);

static struct PyGetSetDef metaclass_properties[] = {
  {"dtype",        (getter)Tensor_dtype, nullptr, nullptr, nullptr},
  {"is_cuda",      (getter)Tensor_is_cuda, nullptr, nullptr, nullptr},
  {"is_sparse",    (getter)Tensor_is_sparse, nullptr, nullptr, nullptr},
  {nullptr}
};

static PyTypeObject metaclass;

static void py_initialize_metaclass(PyTypeObject& metaclass) {
  ((PyObject*)&metaclass)->ob_refcnt = 1;
  metaclass.tp_basicsize = sizeof(PyTypeObject);
  metaclass.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  metaclass.tp_methods = metaclass_methods;
  metaclass.tp_getset = metaclass_properties;
  metaclass.tp_name = "torch.tensortype";
  metaclass.tp_base = &PyType_Type;
  if (PyType_Ready(&metaclass) < 0) {
    throw python_error();
  }
}

static void py_initialize_tensor_type(PyTypeObject& type, const char* name, PyObject* tp_dict) {
  // NOTE: we don't use the typical static declaration of PyTypeObject because
  // we need to initialize as many types as there are VariableType instances.
  // The typical PyVarObject_HEAD_INIT(NULL, 0) is described in the Python
  // documentation: it initializes the refcnt to 1 and the other object header
  // fields to zero.
  memset(&type, 0, sizeof(PyTypeObject));
  ((PyObject*)&type)->ob_refcnt = 1;
  ((PyObject*)&type)->ob_type = &metaclass;
  type.tp_basicsize = sizeof(PyTensorType);
  type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  type.tp_name = name;
  type.tp_new = Tensor_new;
  if (PyType_Ready(&type) < 0) {
    throw python_error();
  }
  if (PyDict_Merge(type.tp_dict, tp_dict, 0) < 0) {
    throw python_error();
  }
}

static const char* get_module(Backend backend) {
  switch (backend) {
    case kCPU: return "torch";
    case kCUDA: return "torch.cuda";
    case kSparseCPU: return "torch.sparse";
    case kSparseCUDA: return "torch.cuda.sparse";
    default: runtime_error("invalid backend: %s", toString(backend));
  }
}

static std::string get_name(Backend backend, ScalarType scalarType) {
  std::ostringstream ss;
  ss << get_module(backend) << "." << at::toString(scalarType) << "Tensor";
  return ss.str();
}

static THPObjectPtr get_storage_obj(const PyTensorType& py_type) {
  if (!py_type.aten_type) {
    throw TypeError("tried to get storage object for invalid Type %s", py_type.name);
  }
  auto module_name = get_module(py_type.aten_type->backend());
  auto module_obj = THPObjectPtr(PyImport_ImportModule(module_name));
  if (!module_obj) throw python_error();

  auto storage_name = std::string(at::toString(py_type.aten_type->scalarType())) + "Storage";
  THPObjectPtr storage(PyObject_GetAttrString(module_obj.get(), storage_name.c_str()));
  if (!storage.get()) {
    throw TypeError("couldn't find storage object %s", storage_name.c_str());
  }
  return storage;
}

static void set_type(PyTensorType& type_obj, Backend backend, ScalarType scalarType) {
  auto baseType = globalContext().type_registry[static_cast<int>(backend)][static_cast<int>(scalarType)].get();
  type_obj.aten_type = baseType ? torch::autograd::VariableType::getType(*baseType) : nullptr;
  type_obj.dtype = torch::getDtype(backend, scalarType);
}

static void set_name(PyTensorType& type_obj, const std::string& name) {
  size_t n = sizeof(type_obj.name);
  strncpy(type_obj.name, name.c_str(), n);
  type_obj.name[n - 1] = '\0';
}

static THPObjectPtr get_variable_dict() {
  auto autograd = THPObjectPtr(PyImport_ImportModule("torch.autograd"));
  if (!autograd) throw python_error();

  auto variable_class = THPObjectPtr(PyObject_GetAttrString(autograd.get(), "Variable"));
  if (!variable_class) throw python_error();

  auto variable_type = (PyTypeObject*)variable_class.get();
  TORCH_ASSERTM(variable_type->tp_base, "missing base type for Variable");

  auto res = THPObjectPtr(PyDict_New());
  if (!res) throw python_error();

  if (PyDict_Merge(res.get(), variable_type->tp_dict, 0) < 0) {
    throw python_error();
  }
  if (PyDict_Merge(res.get(), variable_type->tp_base->tp_dict, 0) < 0) {
    throw python_error();
  }

  return res;
}

static std::vector<PyTensorType> tensor_types;

static void initialize_aten_types(std::vector<PyTensorType>& tensor_types) {
  // includes CUDA types even when PyTorch is not built with CUDA
  auto declared_types = torch::utils::all_declared_types();
  tensor_types.resize(declared_types.size() + 1);

  for (size_t i = 0, end = declared_types.size(); i != end; i++) {
    auto& tensor_type = tensor_types[i];
    Backend backend = declared_types[i].first;
    ScalarType scalar_type = declared_types[i].second;
    set_type(tensor_type, backend, scalar_type);
    set_name(tensor_type, get_name(backend, scalar_type));
  }

  // The type object for torch.Tensor is at the end.
  default_tensor_type = &tensor_types.back();
  set_type(*default_tensor_type, kCPU, kFloat);
  set_name(*default_tensor_type, "torch.Tensor");
  default_tensor_type->is_base_type = true;
}

void initialize_python_bindings() {
  // Initialize the at::Type* pointers, name, and properties of the PyTensorType
  // vector. After this call, the vector must not be resized.
  initialize_aten_types(tensor_types);

  // Initialize the Python metaclass for the torch.Tensor, torch.FloatTensor,
  // etc. types. The metaclass handles __instancecheck__ checks and binds the
  // dtype property on the type objects.
  py_initialize_metaclass(metaclass);

  // Get the tp_dict of the Variable class. We copy function definitions
  // onto each Tensor type object so that they can be accessed via e.g.
  // `torch.Tensor.add`.
  auto var_dict = get_variable_dict();

  // Initialize each Python type object torch.FloatTensor, torch.DoubleTensor,
  // etc. and the "default" type object torch.Tensor.
  for (auto& tensor_type : tensor_types) {
    py_initialize_tensor_type(tensor_type.py_type, tensor_type.name, var_dict.get());
  }

  // Add the type objects to their corresponding modules. e.g. torch.FloatTensor
  // is added to the `torch` module as `FloatTensor`. Also add all the type
  // objects to the set torch._tensor_classes.
  py_bind_tensor_types(tensor_types);

  // Add torch.Storage corresponding to the default tensor type.
  py_bind_torch_storage(tensor_types.back());
}

static void py_bind_tensor_types(const std::vector<PyTensorType>& tensor_types) {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module) throw python_error();

  auto tensor_classes = THPObjectPtr(PyObject_GetAttrString(torch_module.get(), "_tensor_classes"));
  if (!tensor_classes) throw python_error();

  for (auto& tensor_type : tensor_types) {
    auto name = std::string(tensor_type.name);
    auto idx = name.rfind(".");
    auto type_name = name.substr(idx + 1);
    auto module_name = name.substr(0, idx);

    auto module_obj = THPObjectPtr(PyImport_ImportModule(module_name.c_str()));
    if (!module_obj) throw python_error();

    PyObject* type_obj = (PyObject*)&tensor_type;
    Py_INCREF(type_obj);
    if (PyModule_AddObject(module_obj.get(), type_name.c_str(), type_obj) < 0) {
      throw python_error();
    }
    if (PySet_Add(tensor_classes.get(), type_obj) < 0) {
      throw python_error();
    }
  }
}

static void py_bind_torch_storage(const PyTensorType& py_type) {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module) throw python_error();

  if (PyModule_AddObject(torch_module.get(), "Storage", get_storage_obj(py_type).release()) < 0) {
    throw python_error();
  }
}

static bool PyTensorType_Check(PyObject* obj) {
  auto it = std::find_if(tensor_types.begin(), tensor_types.end(),
    [obj](const PyTensorType& x) {
      return (PyObject*)&x == obj;
    });
  return it != tensor_types.end();
}

static PyTensorType& get_tensor_type(THPDtype *obj) {
  auto it = std::find_if(tensor_types.begin(), tensor_types.end(),
    [obj](const PyTensorType& x) {
      return x.dtype == obj;
    });
  if (it == tensor_types.end()) {
    throw TypeError("invalid dtype object");
  }
  return *it;
}

void set_default_tensor_type(const at::Type& type) {
  set_type(*default_tensor_type, type.backend(), type.scalarType());
}

void py_set_default_tensor_type(PyObject* obj) {
  PyTensorType *type;
  if (PyTensorType_Check(obj)) {
    type = (PyTensorType*)obj;
  } else if (THPDtype_Check(obj)) {
    type = &get_tensor_type((THPDtype*)obj);
  } else {
    throw TypeError("invalid type object");
  }
  if (!type->aten_type) {
    throw unavailable_type(*type);
  }

  if (!at::isFloatingType(type->aten_type->scalarType())) {
    throw TypeError("only floating-point types are supported as the default type");
  }

  if (type->aten_type->is_sparse()) {
    throw TypeError("only dense types are supported as the default type");
  }

  // get the storage first, so if it doesn't exist we don't change the default tensor type
  THPObjectPtr storage = get_storage_obj(*type);
  set_default_tensor_type(*type->aten_type);

  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module) throw python_error();

  if (PyObject_SetAttrString(torch_module.get(), "Storage", storage) != 0) {
    // technically, we should undo the change of default tensor type.
    throw python_error();
  }
}

at::Type& get_default_tensor_type() {
  TORCH_ASSERT(default_tensor_type && default_tensor_type->aten_type);
  return *default_tensor_type->aten_type;
}

}} // namespace torch::tensor
