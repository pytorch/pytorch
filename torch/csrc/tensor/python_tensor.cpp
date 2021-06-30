#include <torch/csrc/tensor/python_tensor.h>

#include <structmember.h>
#include <pybind11/pybind11.h>

#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
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

namespace torch { namespace tensors {

using namespace at;
using namespace torch::autograd;

struct PyTensorType {
  PyTypeObject py_type;
  THPDtype* dtype;
  THPLayout* layout;
  bool is_cuda;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays)
  char name[64];
  int backend;
  int scalar_type;

  Backend get_backend() const {
    return static_cast<Backend>(backend);
  }

  DispatchKey get_dispatch_key() const {
    return backendToDispatchKey(static_cast<Backend>(backend));
  }

  ScalarType get_scalar_type() const {
    return static_cast<ScalarType>(scalar_type);
  }
};

static_assert(std::is_standard_layout<PyTensorType>::value, "PyTensorType must be standard layout");

// This is always an instance of VariableType
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static PyTensorType* default_tensor_type;

static void py_bind_tensor_types(const std::vector<PyTensorType*>& tensor_types);

static TypeError unavailable_type(const PyTensorType& type) {
  return TypeError("type %s not available. Torch not compiled with CUDA enabled.", type.name);
}

static PyObject* Tensor_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  HANDLE_TH_ERRORS
  auto& tensor_type = *((PyTensorType*)type);
  if (tensor_type.is_cuda && !torch::utils::cuda_enabled()) {
    throw unavailable_type(tensor_type);
  }
  return THPVariable_Wrap(torch::utils::legacy_tensor_ctor(tensor_type.get_dispatch_key(), tensor_type.get_scalar_type(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

// TODO: Deprecate this instancecheck entirely.  It's here to make
// instanceof(t, torch.FloatTensor) work, but we are not going to keep
// adding torch.QuantizedIntTensor classes for every new tensor type
// we add...
static PyObject* Tensor_instancecheck(PyObject* _self, PyObject* arg) {
  HANDLE_TH_ERRORS
  auto self = (PyTensorType*)_self;
  if (THPVariable_Check(arg)) {
    const auto& var = THPVariable_Unpack(arg);
    // NB: This is a little unfortunate, in that if I do an isinstance check
    // against torch.cuda.FloatTensor, this will immediately initialize CUDA.
    // I originally thought that it would not be possible for aten_type_ to
    // be nullptr if you had a tensor of some type, in which case you can
    // skip initializing aten_type(), but TestAutograd.test_type_conversions
    // seems to violate this property (for whatever reason.)
    //
    // TODO: Stop using legacyExtractDispatchKey here (probably need to build
    // in instanceof checking to Tensor class itself)
    if (legacyExtractDispatchKey(var.key_set()) == self->get_dispatch_key() &&
        var.scalar_type() == static_cast<ScalarType>(self->scalar_type)) {
      Py_RETURN_TRUE;
    }
  }
  Py_RETURN_FALSE;
  END_HANDLE_TH_ERRORS
}

PyObject *Tensor_dtype(PyTensorType* self, void *unused) {
  return torch::autograd::utils::wrap(self->dtype);
}

PyObject *Tensor_layout(PyTensorType* self, void *unused) {
  return torch::autograd::utils::wrap(self->layout);
}

PyObject *Tensor_is_cuda(PyTensorType* self, void *unused) {
  if (self->is_cuda) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

PyObject *Tensor_is_sparse(PyTensorType *self, void *unused) {
  if (self->layout->layout == at::Layout::Strided) {
    Py_RETURN_FALSE;
  } else {
    Py_RETURN_TRUE;
  }
}

PyObject *Tensor_is_sparse_csr(PyTensorType *self, void *unused) {
  if (self->layout->layout == at::Layout::SparseCsr) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays)
static struct PyMethodDef metaclass_methods[] = {
  {"__instancecheck__", Tensor_instancecheck, METH_O, nullptr},
  {nullptr}
};

typedef PyObject *(*getter)(PyObject *, void *);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays)
static struct PyGetSetDef metaclass_properties[] = {
  {"dtype",        (getter)Tensor_dtype, nullptr, nullptr, nullptr},
  {"layout",       (getter)Tensor_layout, nullptr, nullptr, nullptr},
  {"is_cuda",      (getter)Tensor_is_cuda, nullptr, nullptr, nullptr},
  {"is_sparse",    (getter)Tensor_is_sparse, nullptr, nullptr, nullptr},
  {"is_sparse_csr",(getter)Tensor_is_sparse_csr, nullptr, nullptr, nullptr},
  {nullptr}
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static PyTypeObject metaclass = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch.tensortype",                          /* tp_name */
  sizeof(PyTypeObject)                         /* tp_basicsize */
};

static void py_initialize_metaclass(PyTypeObject& metaclass) {
  metaclass.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  metaclass.tp_methods = metaclass_methods;
  metaclass.tp_getset = metaclass_properties;
  metaclass.tp_base = &PyType_Type;
  if (PyType_Ready(&metaclass) < 0) {
    throw python_error();
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static PyTypeObject tensor_type_prototype = {
  PyVarObject_HEAD_INIT(&metaclass, 0)
  nullptr,                                     /* tp_name */
  sizeof(PyTensorType)                         /* tp_basicsize */
};

static void py_initialize_tensor_type(PyTypeObject& type, const char* name, PyObject* tp_dict) {
  // NOTE: we don't use the typical static declaration of PyTypeObject because
  // we need to initialize as many types as there are VariableType instances.
  // We copy the basic object fields from a prototype definition and initialize
  // the remaining fields below.
  memcpy(&type, &tensor_type_prototype, sizeof(PyTypeObject));
  // Subclassing from torch.<ScalarType>Tensor isn't supported.
  // (Py_TPFLAGS_BASETYPE omitted). Subclassing torch.Tensor still allowed.
  type.tp_flags = Py_TPFLAGS_DEFAULT;
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
    case Backend::CPU: return "torch";
    case Backend::CUDA: return "torch.cuda";
    case Backend::SparseCPU: return "torch.sparse";
    case Backend::SparseCUDA: return "torch.cuda.sparse";
    default: AT_ERROR("invalid backend: ", toString(backend));
  }
}

static std::string get_name(Backend backend, ScalarType scalarType) {
  std::ostringstream ss;
  ss << get_module(backend) << "." << toString(scalarType) << "Tensor";
  return ss.str();
}

static THPObjectPtr get_storage_obj(PyTensorType* type) {
  auto module_name = get_module(type->get_backend());
  auto module_obj = THPObjectPtr(PyImport_ImportModule(module_name));
  if (!module_obj) throw python_error();

  auto storage_name = std::string(toString(type->get_scalar_type())) + "Storage";
  THPObjectPtr storage(PyObject_GetAttrString(module_obj.get(), storage_name.c_str()));
  if (!storage.get()) {
    throw TypeError("couldn't find storage object %s", storage_name.c_str());
  }
  return storage;
}

static void set_type(PyTensorType& type_obj, Backend backend, ScalarType scalarType) {
  // This field is lazily initialized from backend and scalar_type
  type_obj.backend = static_cast<int>(backend);
  type_obj.scalar_type = static_cast<int>(scalarType);
  type_obj.layout = torch::getTHPLayout(layout_from_backend(backend));
  type_obj.dtype = torch::getTHPDtype(scalarType);
  type_obj.is_cuda = (backend == at::Backend::CUDA || backend == at::Backend::SparseCUDA);
}

static void set_name(PyTensorType& type_obj, const std::string& name) {
  size_t n = sizeof(type_obj.name);
  strncpy(type_obj.name, name.c_str(), n);
  type_obj.name[n - 1] = '\0';
}

static THPObjectPtr get_tensor_dict() {
  auto torch = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch) throw python_error();

  auto tensor_class = THPObjectPtr(PyObject_GetAttrString(torch, "Tensor"));
  if (!tensor_class) throw python_error();

  auto tensor_type = (PyTypeObject*)tensor_class.get();
  TORCH_CHECK(tensor_type->tp_base, "missing base type for Tensor");

  auto res = THPObjectPtr(PyDict_New());
  if (!res) throw python_error();

  if (PyDict_Merge(res.get(), tensor_type->tp_dict, 0) < 0) {
    throw python_error();
  }
  if (PyDict_Merge(res.get(), tensor_type->tp_base->tp_dict, 0) < 0) {
    throw python_error();
  }

  return res;
}

// A note about the lifetime of the various PyTensorType: normally
// PyTypeObject instances are statically allocated, but we want to create them
// dynamically at init time, because their exact number depends on
// torch::utils::all_declared_types(). The memory for each PyTensorType is
// allocated by initialize_aten_types() and never freed: technically it's a
// leak, but it's not a problem since we want them to be alive for the whole time
// of the process anyway.
//
// An alternative is to use a std::vector<PyTensorType> instead, and let
// std::vector to manage the lifetime of its items. This is problematic
// though, because it means that the memory of PyTensorType is deallocated at
// some point during the exit: if by chance we have another global destructor
// and/or atexit() function which tries to access the PyTensorTypes, we risk
// an use-after-free error. This happens for example if we embed CPython and
// call Py_Finalize inside an atexit() function which was registered before
// importing torch.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static std::vector<PyTensorType*> tensor_types;

void set_default_tensor_type(PyTensorType* type) {
  if (!at::isFloatingType(type->get_scalar_type())) {
    throw TypeError("only floating-point types are supported as the default type");
  }
  if (type->get_backend() == Backend::Undefined) {
    throw TypeError("default type cannot be undefined");
  }
  if (isSparse(type->get_backend())) {
    throw TypeError("only dense types are supported as the default type");
  }

  // get the storage first, so if it doesn't exist we don't change the default tensor type
  THPObjectPtr storage = get_storage_obj(type);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  default_tensor_type = type;
  at::set_default_dtype(scalarTypeToTypeMeta(type->get_scalar_type()));

  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module) throw python_error();

  if (PyObject_SetAttrString(torch_module.get(), "Storage", storage) != 0) {
    // technically, we should undo the change of default tensor type.
    throw python_error();
  }
}

static void initialize_aten_types(std::vector<PyTensorType*>& tensor_types) {
  // includes CUDA types even when PyTorch is not built with CUDA
  auto declared_types = torch::utils::all_declared_types();
  tensor_types.resize(declared_types.size());

  for (size_t i = 0, end = declared_types.size(); i != end; i++) {
    tensor_types[i] = new PyTensorType();
    auto& tensor_type = *tensor_types[i];
    Backend backend = declared_types[i].first;
    ScalarType scalar_type = declared_types[i].second;
    set_type(tensor_type, backend, scalar_type);
    set_name(tensor_type, get_name(backend, scalar_type));

    // Use torch.float32 as the default tensor type
    if (backend == Backend::CPU && scalar_type == at::kFloat) {
      set_default_tensor_type(&tensor_type);
    }
  }
}

void initialize_python_bindings() {
  // Initialize the at::Type* pointers, name, and properties of the PyTensorType
  // vector. After this call, the vector must not be resized.
  initialize_aten_types(tensor_types);

  // Initialize the Python metaclass for the torch.FloatTensor, etc. types.
  // The metaclass handles __instancecheck__ checks and binds the dtype property
  // on the type objects.
  py_initialize_metaclass(metaclass);

  // Get the tp_dict of the Variable class. We copy function definitions
  // onto each Tensor type object so that they can be accessed via e.g.
  // `torch.FloatTensor.add`.
  auto tensor_dict = get_tensor_dict();

  // Initialize each Python type object torch.FloatTensor, torch.DoubleTensor, etc.
  for (auto& tensor_type : tensor_types) {
    py_initialize_tensor_type(tensor_type->py_type, tensor_type->name, tensor_dict.get());
  }

  // Add the type objects to their corresponding modules. e.g. torch.FloatTensor
  // is added to the `torch` module as `FloatTensor`. Also add all the type
  // objects to the set torch._tensor_classes.
  py_bind_tensor_types(tensor_types);
}

static void py_bind_tensor_types(const std::vector<PyTensorType*>& tensor_types) {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module) throw python_error();

  auto tensor_classes = THPObjectPtr(PyObject_GetAttrString(torch_module.get(), "_tensor_classes"));
  if (!tensor_classes) throw python_error();

  for (auto& tensor_type : tensor_types) {
    auto name = std::string(tensor_type->name);
    auto idx = name.rfind('.');
    auto type_name = name.substr(idx + 1);
    auto module_name = name.substr(0, idx);

    auto module_obj = THPObjectPtr(PyImport_ImportModule(module_name.c_str()));
    if (!module_obj) throw python_error();

    PyObject* type_obj = (PyObject*)tensor_type;
    Py_INCREF(type_obj);
    if (PyModule_AddObject(module_obj.get(), type_name.c_str(), type_obj) < 0) {
      throw python_error();
    }
    if (PySet_Add(tensor_classes.get(), type_obj) < 0) {
      throw python_error();
    }
  }
}

static bool PyTensorType_Check(PyObject* obj) {
  auto it = std::find_if(tensor_types.begin(), tensor_types.end(),
    [obj](PyTensorType *x) {
      return (PyObject*)x == obj;
    });
  return it != tensor_types.end();
}

void py_set_default_tensor_type(PyObject* obj) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  PyTensorType *type;
  if (PyTensorType_Check(obj)) {
    type = (PyTensorType*)obj;
  } else {
    throw TypeError("invalid type object");
  }
  if (type->is_cuda && !torch::utils::cuda_enabled()) {
    throw unavailable_type(*type);
  }
  set_default_tensor_type(type);
}

void py_set_default_dtype(PyObject* obj) {
  if (THPDtype_Check(obj)) {
    auto scalar_type = ((THPDtype*)obj)->scalar_type;
    auto backend = default_tensor_type->get_backend();
    auto it = std::find_if(tensor_types.begin(), tensor_types.end(),
      [backend, scalar_type](PyTensorType *x) {
        return x->get_backend() == backend && x->get_scalar_type() == scalar_type;
      });
    set_default_tensor_type(*it);
  } else {
    throw TypeError("invalid dtype object");
  }
}

c10::DispatchKey get_default_dispatch_key() {
  AT_ASSERT(default_tensor_type);
  return default_tensor_type->get_dispatch_key();
}

ScalarType get_default_scalar_type() {
  return typeMetaToScalarType(get_default_dtype());
}

}} // namespace torch::tensors
