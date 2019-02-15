#include <torch/csrc/Generator.h>

#include <structmember.h>
#include <ATen/ATen.h>

#include <TH/TH.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/utils/tensor_types.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

using namespace at;
using namespace torch;

PyObject *THPGeneratorClass = nullptr;

PyObject * THPGenerator_New()
{
  PyObject *args = PyTuple_New(0);
  if (!args) {
    PyErr_SetString(PyExc_RuntimeError, "Could not create a new generator object - "
        "failed to allocate argument tuple");
    return nullptr;
  }
  PyObject *result = PyObject_Call((PyObject*)THPGeneratorClass, args, nullptr);
  Py_DECREF(args);
  return result;
}

PyObject * THPGenerator_NewWithGenerator(at::Generator& cdata)
{
  auto type = (PyTypeObject*)THPGeneratorClass;
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  if (!self) throw python_error();
  auto self_ = reinterpret_cast<THPGenerator*>(self.get());
  self_->cdata = &cdata;
  return self.release();
}

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
  if ((args && PyTuple_Size(args) != 0) || kwargs) {
    THPUtils_setError("torch.Generator constructor doesn't accept any arguments");
    return nullptr;
  }
  THPGeneratorPtr self((THPGenerator *)type->tp_alloc(type, 0));
  // having to pick a specific type rather than just a backend here is strange,
  // but we don't really have fully fledged backend objects.
  self->cdata = at::CPU(at::kFloat).generator().release();
  self->owner = true;
  return (PyObject*)self.release();
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_getState(THPGenerator *self)
{
  using namespace torch::autograd;
  HANDLE_TH_ERRORS
  THGenerator *generator = THPGenerator_TH_CData(self);
  Variable var = torch::empty({0}, at::device(at::kCPU).dtype(at::kByte));
  THByteTensor_getRNGState(generator, (THByteTensor*)(var.data().unsafeGetTensorImpl()));
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
  auto& tensor_type = at::globalContext().getNonVariableType(tensor.type().backend(), tensor.type().scalarType());
  if (tensor_type != CPU(kByte)) {
    auto type_name = torch::utils::type_to_string(tensor_type);
    throw TypeError("expected a torch.ByteTensor, but got %s", type_name.c_str());
  }
  THGenerator *generator = THPGenerator_TH_CData(self);
  THByteTensor_setRNGState(generator, (THByteTensor*)tensor.unsafeGetTensorImpl());
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
  generator->manualSeed(THPUtils_unpackLong(seed));
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_seed(THPGenerator *self)
{
  HANDLE_TH_ERRORS
  return THPUtils_packUInt64(self->cdata->seed());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPGenerator_initialSeed(THPGenerator *self)
{
  HANDLE_TH_ERRORS
  return THPUtils_packUInt64(self->cdata->initialSeed());
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
  nullptr,                                  /* tp_doc */
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
