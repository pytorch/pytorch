#include "python_tensor.h"

#include <structmember.h>
#include <mutex>
#include <pybind11/pybind11.h>

#include "torch/csrc/assertions.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/utils/tensor_new.h"

namespace torch { namespace tensor {

using namespace at;
using namespace torch::autograd;

struct PyTensorType {
  PyTypeObject py_type;
  at::Type* aten_type;
  bool is_cuda;
  bool is_sparse;
  bool is_default;
  std::string name;
};

static_assert(std::is_standard_layout<PyTensorType>::value, "PyTensorType must be standard layout");

static PyTensorType* default_tensor_type;
static std::once_flag init_cuda_flag;

static void py_bind_tensor_types(const std::vector<PyTensorType>& tensor_types);

static PyObject* Tensor_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  HANDLE_TH_ERRORS
  auto& tensor_type = *((PyTensorType*)type);
  if (tensor_type.is_cuda) {
    std::call_once(init_cuda_flag, []() {
      pybind11::module::import("torch.cuda").attr("init")();
    });
  }
  return THPVariable_Wrap(torch::utils::legacy_tensor_ctor(*tensor_type.aten_type, args, kwargs));
  END_HANDLE_TH_ERRORS
}

static PyObject* Tensor_instancecheck(PyTensorType* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (THPVariable_Check(arg)) {
    if (self->is_default) {
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

static struct PyMethodDef metaclass_methods[] = {
  {"__instancecheck__", (PyCFunction)Tensor_instancecheck, METH_O, NULL},
  {NULL}
};

static struct PyMemberDef metaclass_members[] = {
  {(char*)"is_cuda", T_BOOL, offsetof(PyTensorType, is_cuda), READONLY, NULL},
  {(char*)"is_sparse", T_BOOL, offsetof(PyTensorType, is_sparse), READONLY, NULL},
  {NULL}
};

static PyTypeObject metaclass;

static void py_initialize_metaclass(PyTypeObject& metaclass) {
  metaclass.ob_base.ob_base.ob_refcnt = 1;
  metaclass.tp_basicsize = sizeof(PyTypeObject);
  metaclass.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  metaclass.tp_methods = metaclass_methods;
  metaclass.tp_members = metaclass_members;
  metaclass.tp_name = "torch.tensortype";
  metaclass.tp_base = &PyType_Type;
  if (PyType_Ready(&metaclass) < 0) {
    throw python_error();
  }
}

static void py_initialize_tensor_type(PyTypeObject& type, const char* name, PyObject* tp_dict) {
  memset(&type, 0, sizeof(PyTypeObject));
  type.ob_base.ob_base.ob_refcnt = 1;
  type.ob_base.ob_base.ob_type = &metaclass;
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

static std::string get_name(Type& aten_type) {
  std::string name = at::toString(aten_type.scalarType());
  name += "Tensor";
  std::string module_name = get_module(aten_type.backend());
  return module_name + "." + name;
}

static void set_type(PyTensorType& type_obj, Type& aten_type) {
  auto backend = aten_type.backend();
  type_obj.aten_type = &aten_type;
  type_obj.is_cuda = backend == kCUDA || backend == kSparseCUDA;
  type_obj.is_sparse = backend == kSparseCPU || backend == kSparseCUDA;
}

static PyObject* get_variable_dict() {
  auto autograd = THPObjectPtr(PyImport_ImportModule("torch.autograd"));
  if (!autograd) throw python_error();

  auto variable_class = THPObjectPtr(PyObject_GetAttrString(autograd.get(), "Variable"));
  if (!variable_class) throw python_error();

  return ((PyTypeObject*)variable_class.get())->tp_dict;
}

static std::vector<PyTensorType> tensor_types;

static void initialize_aten_types(std::vector<PyTensorType>& tensor_types) {
  auto var_types = VariableType::allTypes();
  tensor_types.resize(var_types.size() + 1);
  for (size_t i = 0; i < var_types.size(); i++) {
    set_type(tensor_types[i], *var_types[i]);
    tensor_types[i].name = get_name(*var_types[i]);
  }
  set_type(tensor_types.back(), *VariableType::getType(CPU(kFloat)));
  tensor_types.back().name = "torch.Tensor";
  tensor_types.back().is_default = true;
}

void initialize_python_bindings(PyObject* module) {
  // Initialize the at::Type* pointers, name, and properties of the PyTensorType
  // vector. After this call, the vector must not be resized.
  initialize_aten_types(tensor_types);

  // Initialize the Python metaclass for the torch.Tensor, torch.FloatTensor,
  // etc. types. The metaclass handles __instancecheck__ checks and binds the
  // propeties is_cuda and is_sparse on the type objects.
  py_initialize_metaclass(metaclass);

  // Get the tp_dict of the Variable Python class. We copy function definitions
  // onto each Tensor type object so that they can be accessed via e.g.
  // `torch.Tensor.add`.
  PyObject* var_dict = get_variable_dict();

  // Initialize each Python type object torch.FloatTensor, torch.DoubleTensor,
  // etc. and the "default" type object torch.Tensor.
  for (auto& tensor_type : tensor_types) {
    const char* name = tensor_type.name.c_str();
    py_initialize_tensor_type(tensor_type.py_type, name, var_dict);
  }

  // The type object for torch.Tensor is at the end.
  default_tensor_type = &tensor_types.back();

  // Add the type objects to their corresponding modules. e.g. torch.FloatTensor
  // is added to the `torch` module as `FloatTensor`. Also add all the type
  // objects to the set torch._tensor_classes.
  py_bind_tensor_types(tensor_types);
}

static void py_bind_tensor_types(const std::vector<PyTensorType>& tensor_types) {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module) throw python_error();

  auto tensor_classes = THPObjectPtr(PyObject_GetAttrString(torch_module.get(), "_tensor_classes"));
  if (!tensor_classes) throw python_error();

  for (auto& tensor_type : tensor_types) {
    auto idx = tensor_type.name.rfind(".");
    auto type_name = tensor_type.name.substr(idx + 1);
    auto module_name = tensor_type.name.substr(0, idx);

    auto module_obj = THPObjectPtr(PyImport_ImportModule(module_name.c_str()));
    if (!module_obj) throw python_error();

    PyObject* type_obj = (PyObject*)&tensor_type;
    Py_INCREF(type_obj);
    PyModule_AddObject(module_obj.get(), type_name.c_str(), type_obj);

    if (PySet_Add(tensor_classes.get(), type_obj) < 0) {
      throw python_error();
    }
  }
}

static bool PyTensorType_Check(PyObject* obj) {
  auto it = std::find_if(tensor_types.begin(), tensor_types.end(),
    [obj](const PyTensorType& x) {
      return (PyObject*)&x == obj;
    });
  return it != tensor_types.end();
}

static at::Type* THPDefaultATenType;

void set_default_tensor_type(PyObject* type_obj) {
  if (!PyTensorType_Check(type_obj)) {
    throw TypeError("invalid type object");
  }
  set_type(*default_tensor_type, *((PyTensorType*)type_obj)->aten_type);
  THPDefaultATenType = default_tensor_type->aten_type;
}

at::Type& get_default_tensor_type() {
  TORCH_ASSERT(THPDefaultATenType);
  return *THPDefaultATenType;
}

}} // namespace torch::tensor
