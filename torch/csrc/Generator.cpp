#include "Generator.h"

#include <structmember.h>
#include <ATen/ATen.h>

#include <stdbool.h>
#include <TH/TH.h>
#include <random>
#include "THP.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/utils/tensor_types.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/autograd/generated/variable_factories.h"

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
    self->cdata = &at::globalContext().getDefaultGenerator(device.type(), device.index());
  }else{
    self->cdata = new Generator(at::detail::createGenerator(device.type(), device.index()));
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
  if (tensor.type() != CPU(kByte)) {
    auto type_name = torch::utils::type_to_string(tensor.type());
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
  generator->setCurrentSeed(THPUtils_unpackLong(seed));
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
  self->cdata->setCurrentSeed(seed_val);
  return THPUtils_packUInt64(seed_val);
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_initialSeed(THPGenerator *self)
{
  HANDLE_TH_ERRORS
  return THPUtils_packUInt64(self->cdata->getCurrentSeed());
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
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  doc_string,                            /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  THPGenerator_methods,                  /* tp_methods */
  THPGenerator_members,                  /* tp_members */
  0,                                     /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
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
