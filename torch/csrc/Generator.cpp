#include <torch/csrc/Generator.h>

#include <structmember.h>
#include <ATen/ATen.h>
#include <ATen/CPUGenerator.h>

#include <TH/TH.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/utils/tensor_types.h>
#include "torch/csrc/utils/python_arg_parser.h"
#include <torch/csrc/autograd/generated/variable_factories.h>

#ifdef USE_CUDA
#include <THC/THCTensorRandom.h>
#include <ATen/CUDAGenerator.h>
#endif

using namespace at;
using namespace torch;

PyObject *THPGeneratorClass = nullptr;

PyObject * THPGenerator_initDefaultGenerator(at::Generator cdata)
{
  auto type = (PyTypeObject*)THPGeneratorClass;
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  if (!self) throw python_error();
  auto self_ = reinterpret_cast<THPGenerator*>(self.get());
  self_->cdata = cdata;
  return self.release();
}

static void THPGenerator_dealloc(THPGenerator* self)
{
  self->cdata->set_pyobj(nullptr);
  self->cdata.~Generator();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject * THPGenerator_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "Generator(Device device=None)"
  });
  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto device = r.deviceWithDefault(0, at::Device(at::kCPU));

  THPGeneratorPtr self((THPGenerator *)type->tp_alloc(type, 0));
#ifdef USE_CUDA
  if (device.type() == at::kCPU) {
    self->cdata = make_generator<CPUGenerator>();
  } else if (device.type() == at::kCUDA){
    self->cdata = make_generator<CUDAGenerator>(device.index());
  } else {
    AT_ERROR("Device type ", c10::DeviceTypeName(device.type()),
             " is not supported for torch.Generator() api.");
  }
#else
  TORCH_CHECK(device.type() == at::kCPU,
              "Device type ", c10::DeviceTypeName(device.type()),
              " is not supported for torch.Generator() api.");
  self->cdata = make_generator<CPUGenerator>();
#endif
  return (PyObject*)self.release();
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_getState(THPGenerator *self, PyObject *noargs)
{
  using namespace torch::autograd;
  HANDLE_TH_ERRORS
  Variable var = torch::empty({0}, at::device(at::kCPU).dtype(at::kByte));
  if (self->cdata->device().type() == at::kCPU) {
    THByteTensor_getRNGState(self->cdata, (THByteTensor*)(var.unsafeGetTensorImpl()));
  } else {
#ifdef USE_CUDA
    TORCH_INTERNAL_ASSERT(self->cdata->device().type() == at::kCUDA);
    THCRandom_getRNGState(self->cdata, (THByteTensor*)(var.unsafeGetTensorImpl()));
#else 
    TORCH_INTERNAL_ASSERT(false, "PyTorch not compiled with CUDA");
#endif 
  }
  return THPVariable_Wrap(std::move(var));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_setState(THPGenerator *self, PyObject *_new_state)
{
  using namespace torch::autograd;
  HANDLE_TH_ERRORS
  if (!THPVariable_Check(_new_state)) {
    throw TypeError("expected a torch.ByteTensor, but got %s", Py_TYPE(_new_state)->tp_name);
  }
  auto& tensor = ((THPVariable*)_new_state)->cdata;
  if (tensor.layout() != kStrided || tensor.device().type() != kCPU || tensor.scalar_type() != kByte) {
    auto type_name = torch::utils::options_to_string(tensor.options());
    throw TypeError("expected a torch.ByteTensor, but got %s", type_name.c_str());
  }
  if (self->cdata->device().type() == at::kCPU) {
    THByteTensor_setRNGState(self->cdata, (THByteTensor*)tensor.unsafeGetTensorImpl());
  } else {
#ifdef USE_CUDA
    TORCH_INTERNAL_ASSERT(self->cdata->device().type() == at::kCUDA);
    THCRandom_setRNGState(self->cdata, (THByteTensor*)tensor.unsafeGetTensorImpl());
#else 
    TORCH_INTERNAL_ASSERT(false, "PyTorch not compiled with CUDA");
#endif 
  }
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_manualSeed(THPGenerator *self, PyObject *seed)
{
  HANDLE_TH_ERRORS
  auto generator = self->cdata;
  THPUtils_assert(THPUtils_checkLong(seed), "manual_seed expected a long, "
          "but got %s", THPUtils_typename(seed));
  // See Note [Acquire lock when using random generators]
  std::lock_guard<std::mutex> lock(generator->mutex_);
  generator->set_current_seed(THPUtils_unpackLong(seed));
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_seed(THPGenerator *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  // See Note [Acquire lock when using random generators]
  std::lock_guard<std::mutex> lock(self->cdata->mutex_);
  uint64_t seed_val = self->cdata->seed();
  return THPUtils_packUInt64(seed_val);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_initialSeed(THPGenerator *self, PyObject *noargs)
{
  HANDLE_TH_ERRORS
  return THPUtils_packUInt64(self->cdata->current_seed());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_get_device(THPGenerator *self, void *unused) {
  HANDLE_TH_ERRORS
  return THPDevice_New(self->cdata->device());
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef THPGenerator_properties[] = {
  {"device", (getter)THPGenerator_get_device, nullptr, nullptr, nullptr},
  {nullptr}
};

static PyMethodDef THPGenerator_methods[] = {
  {"get_state",       (PyCFunction)THPGenerator_getState,       METH_NOARGS,  nullptr},
  {"set_state",       (PyCFunction)THPGenerator_setState,       METH_O,       nullptr},
  {"manual_seed",     (PyCFunction)THPGenerator_manualSeed,     METH_O,       nullptr},
  {"seed",            (PyCFunction)THPGenerator_seed,           METH_NOARGS,  nullptr},
  {"initial_seed",    (PyCFunction)THPGenerator_initialSeed,    METH_NOARGS,  nullptr},
  {nullptr}
};

static struct PyMemberDef THPGenerator_members[] = {
  {(char*)"_cdata", T_ULONGLONG, offsetof(THPGenerator, cdata), READONLY, nullptr},
  {nullptr}
};

PyTypeObject THPGeneratorType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch._C.Generator",                   /* tp_name */
  sizeof(THPGenerator),                        /* tp_basicsize */
  0,                                           /* tp_itemsize */
  (destructor)THPGenerator_dealloc,            /* tp_dealloc */
  0,                                           /* tp_vectorcall_offset */
  nullptr,                                     /* tp_getattr */
  nullptr,                                     /* tp_setattr */
  nullptr,                                     /* tp_reserved */
  nullptr,                                     /* tp_repr */
  nullptr,                                     /* tp_as_number */
  nullptr,                                     /* tp_as_sequence */
  nullptr,                                     /* tp_as_mapping */
  nullptr,                                     /* tp_hash  */
  nullptr,                                     /* tp_call */
  nullptr,                                     /* tp_str */
  nullptr,                                     /* tp_getattro */
  nullptr,                                     /* tp_setattro */
  nullptr,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,    /* tp_flags */
  nullptr,                                     /* tp_doc */
  nullptr,                                     /* tp_traverse */
  nullptr,                                     /* tp_clear */
  nullptr,                                     /* tp_richcompare */
  0,                                           /* tp_weaklistoffset */
  nullptr,                                     /* tp_iter */
  nullptr,                                     /* tp_iternext */
  THPGenerator_methods,                        /* tp_methods */
  THPGenerator_members,                        /* tp_members */
  THPGenerator_properties,                     /* tp_getset */
  nullptr,                                     /* tp_base */
  nullptr,                                     /* tp_dict */
  nullptr,                                     /* tp_descr_get */
  nullptr,                                     /* tp_descr_set */
  0,                                           /* tp_dictoffset */
  nullptr,                                     /* tp_init */
  nullptr,                                     /* tp_alloc */
  THPGenerator_pynew,                          /* tp_new */
};

bool THPGenerator_init(PyObject *module)
{
  THPGeneratorClass = (PyObject*)&THPGeneratorType;
  if (PyType_Ready(&THPGeneratorType) < 0)
    return false;
  Py_INCREF(&THPGeneratorType);
  PyModule_AddObject(module, "Generator", (PyObject *)&THPGeneratorType);
  return true;
}

void set_pyobj(const Generator& self, PyObject* pyobj) {
  TORCH_CHECK(self.defined(), "cannot call set_pyobj() on undefined generator");
  self->set_pyobj(pyobj);
}

PyObject* pyobj(const Generator& self) {
  TORCH_CHECK(self.defined(), "cannot call pyobj() on undefined generator");
  return self->pyobj();
}

PyObject * THPGenerator_Wrap(Generator gen)
{
  if (!gen.defined()) {
    Py_RETURN_NONE;
  }

  if (auto obj = pyobj(gen)) {
    Py_INCREF(obj);
    return obj;
  }

  return THPGenerator_NewWithVar((PyTypeObject *)THPGeneratorClass, std::move(gen));
}

// Creates a new Python object for a Generator. The Generator must not already
// have a PyObject* associated with it.
PyObject* THPGenerator_NewWithVar(PyTypeObject* type, Generator gen)
{
  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    auto g = (THPGenerator*) obj;
    new (&g->cdata) Generator(std::move(gen));
    set_pyobj(g->cdata, obj);
  }
  return obj;
}
