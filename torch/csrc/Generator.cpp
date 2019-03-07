#include <torch/csrc/Generator.h>

#include <structmember.h>
#include <ATen/ATen.h>
#include <ATen/CPUGenerator.h>

#include <TH/TH.h>
#include <random>
#include <torch/csrc/THP.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/utils/tensor_types.h>
#include "torch/csrc/utils/python_arg_parser.h"
#include <torch/csrc/autograd/generated/variable_factories.h>

using namespace at;
using namespace torch;

PyObject *THPGeneratorClass = nullptr;

const char *doc_string = 
"Generator(device='cpu', default=False)\n"
" Creates and returns a generator object which manages the state of the algorithm that\n"
"produces pseudo random numbers. Used as a keyword argument in many random tensors such\n"
"as normal_, randn etc.\n"
" Keyword arguments:\n"
"    device (:class:`torch.device`, optional): the desired device for the generator.\n"
"        Default: `torch.device('cpu')`.\n"
"    default (bool, optional): If using the default CPU/CUDA generator\n"
"        Default: `False`.\n"
" Example::\n"
"    >>> g_cpu = torch.Generator()\n"
"    >>> g_cpu_default = torch.Generator(default=True)\n"
"    >>> g_cuda = torch.Generator(device='cuda')\n"
"    >>> g_cuda_default = torch.Generator(device='cuda', default=True)\n"
"    >>> g_cuda_default_1 = torch.Generator(device='cuda:1', default=True)\n";

static void THPGenerator_dealloc(THPGenerator* self)
{
  if (self->owner) {
    delete self->cdata;
  }
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject * THPGenerator_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "Generator(Device device=None, bool default=False)"
  });
  torch::ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto device = r.deviceWithDefault(0, at::Device(at::kCPU));
  auto is_default_generator = r.toBool(1);

  THPGeneratorPtr self((THPGenerator *)type->tp_alloc(type, 0));
  if(is_default_generator) {
    self->cdata = at::detail::getDefaultCPUGenerator().get();
    self->owner = false;
  }else{
    if(device.type() == at::kCPU) {
      self->cdata = new CPUGenerator();
    }  
    self->owner = true;
  }
  return (PyObject*)self.release();
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_getState(THPGenerator *self)
{
  using namespace torch::autograd;
  HANDLE_TH_ERRORS
  Variable var = torch::empty({0}, at::device(at::kCPU).dtype(at::kByte));
  THByteTensor_getRNGState(self->cdata, (THByteTensor*)(var.data().unsafeGetTensorImpl()));
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
  auto& tensor = ((THPVariable*)_new_state)->cdata.data();
  auto& tensor_type = at::globalContext().getNonVariableType(tensor.type().backend(), tensor.scalar_type());
  if (tensor_type != CPU(kByte)) {
    auto type_name = torch::utils::type_to_string(tensor_type);
    throw TypeError("expected a torch.ByteTensor, but got %s", type_name.c_str());
  }
  THByteTensor_setRNGState(self->cdata, (THByteTensor*)tensor.unsafeGetTensorImpl());
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
  // See Note [Thread-safety and Generators]
  std::lock_guard<std::mutex> lock(generator->mutex_);
  generator->set_current_seed(THPUtils_unpackLong(seed));
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_seed(THPGenerator *self)
{
  HANDLE_TH_ERRORS
  std::random_device rd;
  uint64_t seed_val;
  // std::random_device might not work always
  // in those case use chrono
  if (rd.entropy() != 0) {
    // limit to 53 bits to ensure unique representation in double
    seed_val = ((((uint64_t)rd()) << 32) + rd()) & 0x1FFFFFFFFFFFFF;
  }
  else {
    seed_val = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    seed_val = ((((uint64_t)seed_val) << 32) + seed_val) & 0x1FFFFFFFFFFFFF;
  }
  // See Note [Thread-safety and Generators]
  std::lock_guard<std::mutex> lock(self->cdata->mutex_);
  self->cdata->set_current_seed(seed_val);
  return THPUtils_packUInt64(seed_val);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_initialSeed(THPGenerator *self)
{
  HANDLE_TH_ERRORS
  // See Note [Thread-safety and Generators]
  std::lock_guard<std::mutex> lock(self->cdata->mutex_);
  return THPUtils_packUInt64(self->cdata->current_seed());
  END_HANDLE_TH_ERRORS
}

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
  "torch._C.Generator",                  /* tp_name */
  sizeof(THPGenerator),                  /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THPGenerator_dealloc,      /* tp_dealloc */
  nullptr,                                     /* tp_print */
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
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  doc_string,                                  /* tp_doc */
  nullptr,                                     /* tp_traverse */
  nullptr,                                     /* tp_clear */
  nullptr,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  nullptr,                                     /* tp_iter */
  nullptr,                                     /* tp_iternext */
  THPGenerator_methods,                  /* tp_methods */
  THPGenerator_members,                  /* tp_members */
  nullptr,                                     /* tp_getset */
  nullptr,                                     /* tp_base */
  nullptr,                                     /* tp_dict */
  nullptr,                                     /* tp_descr_get */
  nullptr,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  nullptr,                                     /* tp_init */
  nullptr,                                     /* tp_alloc */
  THPGenerator_pynew,                    /* tp_new */
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
