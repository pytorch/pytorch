#include "torch/csrc/autograd/python_variable.h"

#include <structmember.h>

#include "THP.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Types.h"
#include "torch/csrc/autograd/python_cpp_function.h"
#include "torch/csrc/autograd/python_hook.h"
#include "torch/csrc/autograd/functions/accumulate_grad.h"
#include "torch/csrc/cuda/AutoGPU.h"
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/Exceptions.h"


using namespace torch::autograd;

PyObject *THPVariableClass = NULL;

static PyObject* THPVariable_NewWithVar(PyTypeObject* type, std::shared_ptr<Variable> var)
{
  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    auto v = (THPVariable*) obj;
    new (&v->cdata) std::shared_ptr<Variable>(std::move(var));
    if (auto fn = dynamic_cast<PyFunction*>(v->cdata->grad_fn.get())) {
      // Create a new reference to the THPFunction. This ensures that ref count
      // of the THPFunction is at least the number of referring THPVariables.
      v->cdata->grad_fn = THPFunction_asFunction((THPFunction*)fn->obj);
    }
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
    THPVariable* py_var = (THPVariable*)var->pyobj;
    py_var->data = torch::createPyObject(var->data);
  }
  return var->pyobj;
}

// This function DOES NOT steal a reference to data
PyObject * THPVariable_NewWithFunction(PyObject *data, const std::shared_ptr<torch::autograd::Function>& grad_fn)
{
  THPUtils_assert(THPModule_isTensor(data), "data must be a Tensor");
  auto v = std::make_shared<Variable>(torch::createTensor(data), grad_fn->is_executable, false);
  v->grad_fn = grad_fn;
  PyObject* obj = THPVariable_NewWithVar((PyTypeObject*)THPVariableClass, v);
  if (obj) {
    v->pyobj = obj;
    Py_INCREF(data);
    ((THPVariable*)obj)->data = data;
  }
  return obj;
}

// This function DOES NOT steal a reference to data
PyObject * THPVariable_NewVolatile(PyObject *data)
{
  auto v = std::make_shared<Variable>(torch::createTensor(data), false, true);
  PyObject* obj = THPVariable_NewWithVar((PyTypeObject*)THPVariableClass, v);
  if (obj) {
    v->pyobj = obj;
    ((THPVariable*)obj)->data = data;
    Py_INCREF(data);
  }
  return obj;
}

// This function DOES NOT steal a reference to data
PyObject * THPVariable_NewLeaf(PyObject *data)
{
  auto v = std::make_shared<Variable>(torch::createTensor(data), false, false);
  PyObject* obj = THPVariable_NewWithVar((PyTypeObject*)THPVariableClass, v);
  if (obj) {
    v->pyobj = obj;
    ((THPVariable*)obj)->data = data;
    Py_INCREF(data);
  }
  return obj;
}

static int THPVariable_traverse(THPVariable *self, visitproc visit, void *arg)
{
  Py_VISIT(self->data);
  Py_VISIT(self->backward_hooks);
  if (self->cdata) {
    if (auto fn = dynamic_cast<PyFunction*>(self->cdata->grad_fn.get())) {
      Py_VISIT(fn->obj);
    }
    for (auto& hook : self->cdata->hooks) {
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
    if (auto grad_acc = self->cdata->grad_accumulator.lock()) {
      grad_acc->pre_hooks.clear();
    }
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
  PyObject *grad_fn = NULL;
  char is_volatile = 0;
  char requires_grad = 0;

  const char *accepted_args[] = {"data", "requires_grad", "volatile", "_grad_fn", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ObbO", (char**)accepted_args,
      &data, &requires_grad, &is_volatile, &grad_fn))
    return NULL;

  if (grad_fn == Py_None)
    grad_fn = NULL;

  if (data == NULL || data == Py_None) {
    // For legacy serialization code, create an empty tensor temporarily.
    at::Tensor tensor = at::CPU(at::kFloat).tensor();
    _data = torch::createPyObject(tensor);
    data = _data.get();
  }

  THPUtils_assert(!(is_volatile && requires_grad),
          "Variable can't be volatile and require_grad at the same time!");
  THPUtils_assert(!grad_fn || THPFunction_Check(grad_fn),
          "Variable _grad_fn has to be a Function object or None, but got %s",
          THPUtils_typename(grad_fn));
  THPUtils_assert(THPModule_isTensor(data), "Variable data has to "
          "be a tensor, but got %s", THPUtils_typename(data));

  std::shared_ptr<Variable> var;
  if (grad_fn) {
    var = std::make_shared<Variable>(torch::createTensor(data), THPFunction_asFunction((THPFunction*)grad_fn));
  } else {
    var = std::make_shared<Variable>(torch::createTensor(data), requires_grad, is_volatile);
  }
  PyObject* self = THPVariable_NewWithVar(type, var);
  if (self) {
    var->pyobj = self;
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
  PyObject *grad_fn = NULL;
  char is_volatile = 0;
  char requires_grad = 0;

  const char *accepted_args[] = {"data", "requires_grad", "volatile", "_grad_fn", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ObbO", (char**)accepted_args,
      &data, &requires_grad, &is_volatile, &grad_fn))
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

PyObject *THPVariable_get_grad_fn(THPVariable *self)
{
  auto& var = *self->cdata;
  if (!var.grad_fn) {
    Py_RETURN_NONE;
  }
  return functionToPyObject(var.grad_fn);
}

int THPVariable_set_grad_fn(THPVariable *self, PyObject *obj)
{
  THPUtils_assertRet(-1, obj == Py_None, "_grad_fn can be only set to None");
  self->cdata->grad_fn = nullptr;
  return 0;
}

PyObject *THPVariable_is_leaf(THPVariable *self)
{
  return PyBool_FromLong(!self->cdata->grad_fn);
}

PyObject * THPVariable_get_data(THPVariable *self)
{
  if (!self->data) {
    self->data = torch::createPyObject(self->cdata->data);
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

PyObject *THPVariable_get_grad(THPVariable *self)
{
  auto& var = *self->cdata;
  if (!var.grad) {
    Py_RETURN_NONE;
  }
  return THPVariable_Wrap(var.grad);
}

int THPVariable_set_grad(THPVariable *self, PyObject *other)
{
  auto& var = *self->cdata;
  if (other == Py_None) {
    var.grad.reset();
    return 0;
  }

  THPUtils_assertRet(-1, THPVariable_Check(other),
      "expected Variable or None (got %s)", THPUtils_typename(other));
  THPUtils_assertRet(-1, self != (THPVariable*)other,
      "can't assign Variable as its own grad");

  auto& other_var = ((THPVariable*)other)->cdata;

  // Make sure the data is ok
  THPUtils_assertRet(-1, other_var->data.type().ID() == var.data.type().ID(),
      "assigned grad has data of a different type");
  THPUtils_assertRet(-1, other_var->data.type().isCuda() == var.data.type().isCuda(),
      "assigned grad has data located on a different device");
  if (var.data.type().isCuda()) {
    THPUtils_assertRet(-1, other_var->data.get_device() == var.data.get_device(),
        "assigned grad has data located on a different device");
  }
  THPUtils_assertRet(-1, other_var->data.sizes().vec() == var.data.sizes().vec(),
      "assigned grad has data of a different size");

  var.grad = other_var;
  if (auto grad_acc = var.grad_accumulator.lock()) {
    ((AccumulateGrad*)grad_acc.get())->variable_grad = other_var;
  }
  return 0;
}

PyObject *THPVariable_get_volatile(THPVariable *self)
{
  auto& var = *self->cdata;
  return PyBool_FromLong(var.is_volatile);
}

int THPVariable_set_volatile(THPVariable *self, PyObject *obj)
{
  THPUtils_assertRet(-1, PyBool_Check(obj), "volatile must be a bool");
  THPUtils_assertRet(-1, !self->cdata->grad_fn,
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
  if (var.grad_fn) {
    const char *hint = "";
    if (obj == Py_False) {
      hint = " If you want to use a computed variable in a subgraph "
             "that doesn't require differentiation use "
             "var_no_grad = var.detach().";
    }
    THPUtils_setError("you can only change requires_grad flags of leaf variables.%s", hint);
    return -1;
  }
  var.requires_grad = obj == Py_True;
  if (auto grad_accumulator = var.grad_accumulator.lock()) {
    grad_accumulator->is_executable = var.requires_grad;
  }
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
  self->cdata->hooks.clear();
  if (obj) {
    self->cdata->hooks.emplace_back(new PyFunctionPreHook(obj, 0));
  }
  return 0;
}

static struct PyGetSetDef THPVariable_properties[] = {
  {"_version", (getter)THPVariable_get_version, NULL, NULL, NULL},
  {"grad_fn", (getter)THPVariable_get_grad_fn, NULL, NULL, NULL},
  {"_grad_fn", (getter)THPVariable_get_grad_fn, (setter)THPVariable_set_grad_fn, NULL, NULL},
  {"is_leaf", (getter)THPVariable_is_leaf, NULL, NULL, NULL},
  {"data", (getter)THPVariable_get_data, (setter)THPVariable_set_data, NULL, NULL},
  {"_grad", (getter)THPVariable_get_grad, (setter)THPVariable_set_grad, NULL, NULL}, // only for legacy reasons
  {"grad", (getter)THPVariable_get_grad, (setter)THPVariable_set_grad, NULL, NULL},
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
