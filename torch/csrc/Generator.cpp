#include "Generator.h"

#include <structmember.h>
#include <ATen/ATen.h>

#include <stdbool.h>
#include <TH/TH.h>
#include "THP.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/utils/tensor_types.h"

using namespace at;
using namespace torch;

PyObject *THPGeneratorClass = NULL;

PyObject * THPGenerator_New()
{
  PyObject *args = PyTuple_New(0);
  if (!args) {
    PyErr_SetString(PyExc_RuntimeError, "Could not create a new generator object - "
        "failed to allocate argument tuple");
    return NULL;
  }
  PyObject *result = PyObject_Call((PyObject*)THPGeneratorClass, args, NULL);
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
    return NULL;
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
  Variable var = VariableType::getType(CPU(kByte))->tensor();
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
  if (tensor.type() != CPU(kByte)) {
    auto type_name = torch::utils::type_to_string(tensor.type());
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
  {"get_state",       (PyCFunction)THPGenerator_getState,       METH_NOARGS,  NULL},
  {"set_state",       (PyCFunction)THPGenerator_setState,       METH_O,       NULL},
  {"manual_seed",     (PyCFunction)THPGenerator_manualSeed,     METH_O,       NULL},
  {"seed",            (PyCFunction)THPGenerator_seed,           METH_NOARGS,  NULL},
  {"initial_seed",    (PyCFunction)THPGenerator_initialSeed,    METH_NOARGS,  NULL},
  {NULL}
};

static struct PyMemberDef THPGenerator_members[] = {
  {(char*)"_cdata", T_ULONGLONG, offsetof(THPGenerator, cdata), READONLY, NULL},
  {NULL}
};

PyTypeObject THPGeneratorType = {
  PyVarObject_HEAD_INIT(NULL, 0)
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
  NULL,                                  /* tp_doc */
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
