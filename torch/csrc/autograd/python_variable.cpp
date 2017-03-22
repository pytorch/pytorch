#include "torch/csrc/autograd/python_variable.h"

#include <structmember.h>

#include "THP.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Types.h"
#include "torch/csrc/autograd/python_cpp_function.h"
#include "torch/csrc/autograd/python_hook.h"
#include "torch/csrc/cuda/AutoGPU.h"
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/Exceptions.h"
#include <THPP/tensors/THTensor.hpp>


using namespace torch::autograd;

PyObject *THPVariableClass = NULL;

static PyObject* THPVariable_NewWithVar(PyTypeObject* type, std::shared_ptr<Variable> var)
{
  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    auto v = (THPVariable*) obj;
    new (&v->cdata) std::shared_ptr<Variable>(std::move(var));
  }
  return obj;
}

PyObject * THPVariable_Wrap(const std::shared_ptr<Variable>& var)
{
  if (!var) {
    Py_RETURN_NONE;
  } else if (var->pyobj) {
    Py_INCREF(var->pyobj);
  } else {
    var->pyobj = THPVariable_NewWithVar((PyTypeObject *)THPVariableClass, var);
  }
  return var->pyobj;
}

// This function DOES NOT steal a reference to data and creator
// To create a leaf Variable pass NULL as creator.
PyObject * THPVariable_New(PyObject *data, PyObject *creator, bool requires_grad, bool is_volatile)
{
  THPUtils_assert(THPModule_isTensor(data), "data must be a Tensor");
  THPUtils_assert(!creator || THPFunction_Check(creator), "creator must be a Function");
  auto v = std::make_shared<Variable>(torch::createTensor(data), requires_grad, is_volatile);
  PyObject* obj = THPVariable_NewWithVar((PyTypeObject*)THPVariableClass, v);
  if (obj) {
    v->pyobj = obj;
    v->creator = THPFunction_asFunction((THPFunction*)creator);
    ((THPVariable*)obj)->data = data;
    Py_INCREF(data);
  }
  return obj;
}

// This function DOES NOT steal a reference to data
PyObject * THPVariable_NewVolatile(PyObject *data)
{
  return THPVariable_New(data, nullptr, false, true);
}

static int THPVariable_traverse(THPVariable *self, visitproc visit, void *arg)
{
  Py_VISIT(self->data);
  Py_VISIT(self->backward_hooks);
  if (self->cdata) {
    if (auto fn = dynamic_cast<PyFunction*>(self->cdata->creator.get())) {
      Py_VISIT(fn->obj);
    }
    for (auto& hook : self->cdata->pre_hooks) {
      if (auto pyhook = dynamic_cast<PyFunctionPreHook*>(hook.get())) {
        Py_VISIT(pyhook->dict);
      }
    }
  }
  return 0;
}

static int THPVariable_clear(THPVariable *self)
{
  Py_CLEAR(self->data);
  Py_CLEAR(self->backward_hooks);
  if (self->cdata) {
    self->cdata->pyobj = nullptr;
  }
  self->cdata.reset();
  return 0;
}

static void THPVariable_dealloc(THPVariable* self)
{
  PyObject_GC_UnTrack(self);
  THPVariable_clear(self);
  self->cdata.~shared_ptr<Variable>();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject *THPVariable_pynew(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  THPObjectPtr _data;
  PyObject *data = NULL;
  PyObject *creator = NULL;
  char is_volatile = 0;
  char requires_grad = 0;

  const char *accepted_args[] = {"data", "creator", "volatile", "requires_grad", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OObb", (char**)accepted_args,
      &data, &creator, &is_volatile, &requires_grad))
    return NULL;

  if (creator == Py_None)
    creator = NULL;

  if (data == NULL || data == Py_None) {
    // For legacy serialization code, create an empty tensor temporarily.
    thpp::THTensor<float> tensor;
    _data = torch::createPyObject(tensor);
    data = _data.get();
  }

  THPUtils_assert(!(is_volatile && requires_grad),
          "Variable can't be volatile and require_grad at the same time!");
  THPUtils_assert(!creator || THPFunction_Check(creator),
          "Variable creator has to be a Function object or None, but got %s",
          THPUtils_typename(creator));
  THPUtils_assert(THPModule_isTensor(data), "Variable data has to "
          "be a tensor, but got %s", THPUtils_typename(data));

  auto var = std::make_shared<Variable>(torch::createTensor(data), requires_grad, is_volatile);
  PyObject* self = THPVariable_NewWithVar(type, var);
  if (self) {
    var->pyobj = self;
    var->creator = THPFunction_asFunction((THPFunction*)creator);
    ((THPVariable*)self)->cdata = var;
    ((THPVariable*)self)->data = data;
    Py_INCREF(data);
  }

  return self;
}

int THPVariable_pyinit(PyObject *self, PyObject *args, PyObject *kwds)
{
  // Ensures that calls to Variable() and subclasses contain data argument.
  // The 'data' argument is optional in __new__ to handle legacy serialized
  // Variables.
  PyObject *data;
  PyObject *creator = NULL;
  char is_volatile = 0;
  char requires_grad = 0;

  const char *accepted_args[] = {"data", "creator", "volatile", "requires_grad", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|Obb", (char**)accepted_args,
      &data, &creator, &is_volatile, &requires_grad))
    return -1;

  return 0;
}

typedef PyObject *(*getter)(PyObject *, void *);
typedef int (*setter)(PyObject *, PyObject *, void *);

PyObject *THPVariable_get_version(THPVariable *self)
{
  auto& var = *self->cdata;
  return PyInt_FromLong(**var.version_counter);
}

PyObject *THPVariable_get_creator(THPVariable *self)
{
  auto& var = *self->cdata;
  if (!var.creator) {
    Py_RETURN_NONE;
  }
  return functionToPyObject(var.creator);
}

int THPVariable_set_creator(THPVariable *self, PyObject *obj)
{
  THPUtils_assertRet(-1, obj == Py_None, "_creator can be only set to None");
  self->cdata->creator = nullptr;
  return 0;
}

PyObject * THPVariable_get_data(THPVariable *self)
{
  if (!self->data) {
    self->data = torch::createPyObject(*self->cdata->data);
  }
  Py_INCREF(self->data);
  return self->data;
}

int THPVariable_set_data(THPVariable *self, PyObject *data)
{
  THPUtils_assertRet(-1, THPModule_isTensor(data), "Variable data has to "
      "be a tensor, but got %s", THPUtils_typename(data));
  Py_INCREF(data);
  Py_XDECREF(self->data);
  self->data = data;
  auto& var = *self->cdata;
  auto tensor = torch::createTensor(data);
  var.data.swap(tensor);
  return 0;
}

PyObject *THPVariable_get_raw_grad(THPVariable *self)
{
  auto& var = *self->cdata;
  if (!var.grad) {
    Py_RETURN_NONE;
  }
  return THPVariable_Wrap(var.grad);
}

int THPVariable_set_raw_grad(THPVariable *self, PyObject *data)
{
  auto& var = *self->cdata;
  if (data == Py_None) {
    var.grad.reset();
    return 0;
  }
  THPUtils_assertRet(-1, THPVariable_Check(data),
      "expected Variable or None (got %s)", THPUtils_typename(data));
  var.grad = ((THPVariable*)data)->cdata;
  return 0;
}

PyObject *THPVariable_get_grad(THPVariable *self)
{
  auto& var = *self->cdata;
  if (!var.grad) {
    Py_RETURN_NONE;
  }
  return THPVariable_Wrap(var.grad);
}

PyObject *THPVariable_get_volatile(THPVariable *self)
{
  auto& var = *self->cdata;
  return PyBool_FromLong(var.is_volatile);
}

int THPVariable_set_volatile(THPVariable *self, PyObject *obj)
{
  THPUtils_assertRet(-1, PyBool_Check(obj), "volatile must be a bool");
  THPUtils_assertRet(-1, !self->cdata->creator,
      "volatile can only be set on leaf variables");
  auto& var = *self->cdata;
  var.is_volatile = (obj == Py_True);
  return 0;
}

PyObject *THPVariable_get_output_nr(THPVariable *self)
{
  auto& var = *self->cdata;
  return PyInt_FromLong(var.output_nr);
}

PyObject *THPVariable_get_requires_grad(THPVariable *self)
{
  auto& var = *self->cdata;
  return PyBool_FromLong(var.requires_grad);
}

int THPVariable_set_requires_grad(THPVariable *self, PyObject *obj)
{
  THPUtils_assertRet(-1, PyBool_Check(obj), "requires_grad must be a bool");
  auto& var = *self->cdata;
  if (var.creator) {
    const char *hint = "";
    if (obj == Py_False) {
      hint = " If you want to use a computed variable in a subgraph "
             "that doesn't require differentiation use "
             "var_no_grad = var.detach().";
    }
    THPUtils_setError("you can only change requires_grad flags of leaf variables.%s", hint);
    return -1;
  }
  var.requires_grad = (obj == Py_True);
  return 0;
}

PyObject *THPVariable_get_backwards_hooks(THPVariable *self)
{
  if (self->backward_hooks) {
    Py_INCREF(self->backward_hooks);
    return self->backward_hooks;
  }
  Py_RETURN_NONE;
}

int THPVariable_set_backwards_hooks(THPVariable *self, PyObject *obj)
{
  if (obj == Py_None) {
    obj = nullptr;
  }
  Py_XINCREF(obj);
  Py_XDECREF(self->backward_hooks);
  self->backward_hooks = obj;
  self->cdata->pre_hooks.clear();
  if (obj) {
    self->cdata->pre_hooks.emplace_back(new PyFunctionPreHook(obj, 0));
  }
  return 0;
}

static struct PyGetSetDef THPVariable_properties[] = {
  {"_version", (getter)THPVariable_get_version, NULL, NULL, NULL},
  {"creator", (getter)THPVariable_get_creator, NULL, NULL, NULL},
  {"_creator", (getter)THPVariable_get_creator, (setter)THPVariable_set_creator, NULL, NULL},
  {"data", (getter)THPVariable_get_data, (setter)THPVariable_set_data, NULL, NULL},
  {"_grad", (getter)THPVariable_get_raw_grad, (setter)THPVariable_set_raw_grad, NULL, NULL},
  {"grad", (getter)THPVariable_get_grad, NULL, NULL, NULL},
  {"volatile", (getter)THPVariable_get_volatile, (setter)THPVariable_set_volatile, NULL, NULL},
  {"output_nr", (getter)THPVariable_get_output_nr, NULL, NULL, NULL},
  {"requires_grad", (getter)THPVariable_get_requires_grad, (setter)THPVariable_set_requires_grad, NULL, NULL},
  {"_backward_hooks", (getter)THPVariable_get_backwards_hooks, (setter)THPVariable_set_backwards_hooks, NULL, NULL},
  {NULL}
};

PyTypeObject THPVariableType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C._VariableBase",              /* tp_name */
  sizeof(THPVariable),                   /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THPVariable_dealloc,       /* tp_dealloc */
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
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC, /* tp_flags */
  NULL,                                  /* tp_doc */
  (traverseproc)THPVariable_traverse,    /* tp_traverse */
  (inquiry)THPVariable_clear,            /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  0,                                     /* tp_methods */
  0,                                     /* tp_members */
  THPVariable_properties,                /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  THPVariable_pyinit,                    /* tp_init */
  0,                                     /* tp_alloc */
  THPVariable_pynew                      /* tp_new */
};

bool THPVariable_initModule(PyObject *module)
{
  if (PyType_Ready(&THPVariableType) < 0)
    return false;
  Py_INCREF(&THPVariableType);
  PyModule_AddObject(module, "_VariableBase", (PyObject *)&THPVariableType);
  return true;
}
