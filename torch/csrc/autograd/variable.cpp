#include <Python.h>
#include <structmember.h>

#include "THP.h"

PyObject *THPVariableClass = NULL;

constexpr size_t CACHE_SIZE = 100000;
static THPVariable *cached_variables[CACHE_SIZE];
static size_t num_cached;

// This helper steals a reference to data and creator
static inline THPVariable * pop_cache(PyObject *data, PyObject *creator, char requires_grad)
{
  THPVariable *self = cached_variables[--num_cached];
  PyObject_Init((PyObject*)self, Py_TYPE(self));
  PyObject_GC_Track(self);

  self->is_volatile = 0;
  self->version_counter = new THPVariableVersion();
  self->grad = NULL;
  self->backward_hooks = NULL;
  self->requires_grad = requires_grad;

  self->data = data;
  self->creator = creator;
  return self;
}

// This function DOES NOT steal a reference to data
PyObject * THPVariable_NewVolatile(PyObject *data)
{
  THPVariable *variable;
  if (num_cached > 0) {
    Py_INCREF(data);
    variable = pop_cache(data, NULL, 0);
  } else {
    variable = (THPVariable*)PyObject_CallFunctionObjArgs(THPVariableClass, data, NULL);
  }
  ((THPVariable*)variable)->is_volatile = 1;
  return (PyObject*)variable;
}

// This function DOES NOT steal a reference to data and creator
PyObject * THPVariable_New(PyObject *data, PyObject *creator, char requires_grad)
{
  if (num_cached > 0) {
    Py_INCREF(data);
    Py_INCREF(creator);
    return (PyObject*)pop_cache(data, creator, requires_grad);
  }
  return PyObject_CallFunction(THPVariableClass, "OObb", data, creator, (char)0, requires_grad);
}

static int THPVariable_traverse(THPVariable *self, visitproc visit, void *arg)
{
  Py_VISIT(self->creator);
  Py_VISIT(self->data);
  Py_VISIT(self->grad);
  Py_VISIT(self->backward_hooks);
  return 0;
}

static int THPVariable_clear(THPVariable *self)
{
  Py_CLEAR(self->creator);
  Py_CLEAR(self->data);
  Py_CLEAR(self->grad);
  Py_CLEAR(self->backward_hooks);
  return 0;
}

static void THPVariable_dealloc(THPVariable* self)
{
  PyObject_GC_UnTrack(self);
  Py_XDECREF(self->creator);
  Py_XDECREF(self->data);
  Py_XDECREF(self->grad);
  Py_XDECREF(self->backward_hooks);
  delete self->version_counter;
  self->version_counter = nullptr;

  // We don't want to cache any subclasses
  if ((PyObject*)Py_TYPE(self) == THPVariableClass && num_cached < CACHE_SIZE) {
    cached_variables[num_cached++] = self;
    // Variable class is defined in Python code, and as such has a
    // Py_TPFLAGS_HEAPTYPE flag set, so python DECREFs the class at each
    // object dealloc.
    Py_INCREF(Py_TYPE(self));
  } else {
    Py_TYPE(self)->tp_free((PyObject*)self);
  }
}

PyObject *THPVariable_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  THPVariable *self;
  if ((PyObject*)type != THPVariableClass || num_cached == 0) {
    self = (THPVariable*)type->tp_alloc(type, 0);
    if (!self) return NULL;
    self->version_counter = new THPVariableVersion();
  } else {
    self = pop_cache(NULL, NULL, 0);
  }
  return (PyObject*)self;
}

int THPVariable_init(THPVariable *self, PyObject *args, PyObject *kwargs)
{
  const char *accepted_args[] = {"data", "creator", "volatile", "requires_grad", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Obb", (char**)accepted_args,
      &self->data, &self->creator, &self->is_volatile,
      &self->requires_grad))
    return -1;
  Py_INCREF(self->data);
  if (self->creator == Py_None)
    self->creator = NULL;
  Py_XINCREF(self->creator);
  THPUtils_assertRet(-1, !self->creator || THPFunction_Check(self->creator),
          "Variable creator has to be a Function object or None, but got %s",
          THPUtils_typename(self->creator));
  THPUtils_assertRet(-1, THPModule_isTensor(self->data), "Variable data has to "
          "be a tensor, but got %s", THPUtils_typename(self->data));
  return 0;
}

PyObject * THPVariable_getstate(THPVariable *self)
{
  THPUtils_assert(!self->creator, "serialization of non-leaf variables is not "
      "implemented yet");
  THPObjectPtr state = PyTuple_New(5);
  if (!state)
    return NULL;

  Py_INCREF(self->data);
  PyTuple_SET_ITEM(state.get(), 0, self->data);

  PyObject *grad = self->grad ? self->grad : Py_None;
  Py_INCREF(grad);
  PyTuple_SET_ITEM(state.get(), 1, grad);

  PyObject *backward_hooks = self->backward_hooks ? self->backward_hooks : Py_None;
  Py_INCREF(backward_hooks);
  PyTuple_SET_ITEM(state.get(), 2, backward_hooks);

  PyTuple_SET_ITEM(state.get(), 3, PyBool_FromLong(self->requires_grad));
  PyTuple_SET_ITEM(state.get(), 4, PyBool_FromLong(self->is_volatile));

  return state.release();
}

PyObject * THPVariable_setstate(THPVariable *self, PyObject *state)
{
  THPUtils_assert(!self->creator, "__setstate__ can be only called on leaf "
      "variables");
  THPUtils_assert(PyTuple_Check(state), "__setstate__ expects state to be a "
      "tuple");
  Py_ssize_t size = PyTuple_GET_SIZE(state);
  THPUtils_assert(size == 5, "__setstate__ expects state tuple to have 5 "
      "elements, but it has %d", size);

#define LOAD(NAME, IDX)                                                        \
  Py_XDECREF(self->NAME);                                                      \
  self->NAME = PyTuple_GET_ITEM(state, IDX) == Py_None ? NULL : PyTuple_GET_ITEM(state, IDX); \
  Py_XINCREF(self->NAME);
  THPUtils_assert(THPModule_isTensor(PyTuple_GET_ITEM(state, 0)), "first "
          "element of variable state tuple has to be a tensor");
  LOAD(data, 0);

  LOAD(grad, 1);
  LOAD(backward_hooks, 2);
#undef LOAD

  PyObject *requires_grad_obj = PyTuple_GET_ITEM(state, 3);
  PyObject *is_volatile_obj = PyTuple_GET_ITEM(state, 4);
  THPUtils_assert(PyBool_Check(requires_grad_obj), "requires_grad "
      "found in state was expected to be a bool, but got %s",
      THPUtils_typename(requires_grad_obj));
  THPUtils_assert(PyBool_Check(is_volatile_obj), "is_volatile "
      "found in state was expected to be a bool, but got %s",
      THPUtils_typename(is_volatile_obj));
  self->requires_grad= requires_grad_obj == Py_True ? 1 : 0;
  self->is_volatile = is_volatile_obj == Py_True ? 1 : 0;

  Py_RETURN_NONE;
}

typedef PyObject *(*getter)(PyObject *, void *);
typedef int (*setter)(PyObject *, PyObject *, void *);

PyObject *THPVariable_get_version(THPVariable *self)
{
  return PyInt_FromLong(**self->version_counter);
}

static struct PyGetSetDef THPVariable_properties[] = {
  {"_version", (getter)THPVariable_get_version, NULL, NULL, NULL},
  {NULL}
};

static struct PyMemberDef THPVariable_members[] = {
  {(char*)"creator",        T_OBJECT,   offsetof(THPVariable, creator), 0, NULL},
  {(char*)"data",           T_OBJECT,   offsetof(THPVariable, data), 0, NULL},
  {(char*)"_grad",          T_OBJECT,   offsetof(THPVariable, grad), 0, NULL},
  {(char*)"volatile",       T_BOOL,     offsetof(THPVariable, is_volatile), 0, NULL},
  {(char*)"output_nr",      T_INT,      offsetof(THPVariable, output_nr), 0, NULL},
  {(char*)"_backward_hooks",T_OBJECT,   offsetof(THPVariable, backward_hooks), 0, NULL},
  {(char*)"_requires_grad", T_BOOL,     offsetof(THPVariable, requires_grad), 0, NULL},
  {NULL}
};

static struct PyMethodDef THPVariable_methods[] = {
  {"__getstate__", (PyCFunction)THPVariable_getstate, METH_NOARGS, NULL},
  {"__setstate__", (PyCFunction)THPVariable_setstate, METH_O, NULL},
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
  THPVariable_methods,                   /* tp_methods */
  THPVariable_members,                   /* tp_members */
  THPVariable_properties,                /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  (initproc)THPVariable_init,            /* tp_init */
  0,                                     /* tp_alloc */
  THPVariable_new                        /* tp_new */
};


bool THPVariable_initModule(PyObject *module)
{
  if (PyType_Ready(&THPVariableType) < 0)
    return false;
  Py_INCREF(&THPVariableType);
  PyModule_AddObject(module, "_VariableBase", (PyObject *)&THPVariableType);
  return true;
}
