#include "torch/csrc/autograd/python_variable.h"

#include <structmember.h>

#include "THP.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Types.h"
#include "torch/csrc/autograd/python_cpp_function.h"
#include "torch/csrc/autograd/python_hook.h"
#include "torch/csrc/autograd/python_variable_indexing.h"
#include "torch/csrc/autograd/functions/accumulate_grad.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/cuda/AutoGPU.h"
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/Size.h"
#include "torch/csrc/autograd/variable.h"

using namespace at;
using namespace torch::autograd;

PyObject *THPVariableClass = NULL;

// Creates a new Python object for a Variable. The Variable must not already
// have a PyObject* associated with it.
static PyObject* THPVariable_NewWithVar(PyTypeObject* type, Variable var)
{
  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    auto v = (THPVariable*) obj;
    new (&v->cdata) Variable(std::move(var));
    v->cdata.get()->pyobj = obj;
    if (auto fn = dynamic_cast<PyFunction*>(v->cdata.grad_fn().get())) {
      // Create a new reference to the THPFunction. This ensures that ref count
      // of the THPFunction is at least the number of referring THPVariables.
      v->cdata.get()->_grad_fn = THPFunction_asFunction((THPFunction*)fn->obj);
    }
  }
  return obj;
}

PyObject * THPVariable_Wrap(Variable var)
{
  if (!var.defined()) {
    Py_RETURN_NONE;
  }

  if (var.dim() == 0) {
    throw std::runtime_error("Variable API does not support Scalars");
  }

  if (auto obj = var.get()->pyobj) {
    Py_INCREF(obj);
    return obj;
  }

  return THPVariable_NewWithVar((PyTypeObject *)THPVariableClass, std::move(var));
}

// This function DOES NOT steal a reference to data
PyObject * THPVariable_NewVolatile(PyObject *data)
{
  Variable v = make_variable(torch::createTensor(data), false, true);
  PyObject* obj = THPVariable_NewWithVar((PyTypeObject*)THPVariableClass, std::move(v));
  if (obj) {
    ((THPVariable*)obj)->data = data;
    Py_INCREF(data);
  }
  return obj;
}

// This function DOES NOT steal a reference to data
PyObject * THPVariable_NewLeaf(PyObject *data)
{
  Variable v = make_variable(torch::createTensor(data));
  PyObject* obj = THPVariable_NewWithVar((PyTypeObject*)THPVariableClass, std::move(v));
  if (obj) {
    ((THPVariable*)obj)->data = data;
    Py_INCREF(data);
  }
  return obj;
}

static int THPVariable_traverse(THPVariable *self, visitproc visit, void *arg)
{
  Py_VISIT(self->data);
  Py_VISIT(self->backward_hooks);
  if (self->cdata.defined()) {
    // Only visit this if we actually own it (no one else use the shared pointer)
    auto& grad_fn = self->cdata.grad_fn();
    if (grad_fn.use_count() == 1) {
      if (auto fn = dynamic_cast<PyFunction*>(grad_fn.get())) {
        Py_VISIT(fn->obj);
      } else {
        // visit hooks in C++ implemented autograd functions
        for (auto& hook : grad_fn->pre_hooks) {
          if (auto pyhook = dynamic_cast<PyFunctionPreHook*>(hook.get())) {
            Py_VISIT(pyhook->dict);
          }
        }
        for (auto& hook : grad_fn->post_hooks) {
          if (auto pyhook = dynamic_cast<PyFunctionPostHook*>(hook.get())) {
            Py_VISIT(pyhook->dict);
          }
        }
      }
    }
    for (auto& hook : self->cdata.hooks()) {
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
  if (self->cdata.defined()) {
    if (auto grad_acc = self->cdata.get()->grad_accumulator.lock()) {
      grad_acc->pre_hooks.clear();
    }
    self->cdata.get()->pyobj = nullptr;
  }
  self->cdata.reset();
  return 0;
}

static void THPVariable_dealloc(THPVariable* self)
{
  PyObject_GC_UnTrack(self);
  THPVariable_clear(self);
  self->cdata.~Variable();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject *THPVariable_pynew(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  THPObjectPtr _data;
  PyObject *data = NULL;
  PyObject *grad_fn = NULL;
  char is_volatile = 0;
  char requires_grad = 0;
  const char* name = NULL;

  const char *accepted_args[] = {"data", "requires_grad", "volatile", "_grad_fn", "name", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ObbOz", (char**)accepted_args,
      &data, &requires_grad, &is_volatile, &grad_fn, &name))
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

  Variable var;
  if (grad_fn) {
    auto grad_fn_ = THPFunction_asFunction((THPFunction*)grad_fn);
    var = make_variable(torch::createTensor(data), grad_fn_);
  } else {
    var = make_variable(torch::createTensor(data), requires_grad, is_volatile);
  }

  if (name)
    var.name() = std::string(name);

  PyObject* self = THPVariable_NewWithVar(type, std::move(var));
  if (self) {
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
  const char* name = NULL;

  const char *accepted_args[] = {"data", "requires_grad", "volatile", "_grad_fn", "name", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ObbOz", (char**)accepted_args,
      &data, &requires_grad, &is_volatile, &grad_fn, &name))
    return -1;

  return 0;
}

typedef PyObject *(*getter)(PyObject *, void *);
typedef int (*setter)(PyObject *, PyObject *, void *);

PyObject *THPVariable_get_version(THPVariable *self)
{
  HANDLE_TH_ERRORS
  auto& var = self->cdata;
  return PyInt_FromLong(var.current_version());
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_get_grad_fn(THPVariable *self)
{
  HANDLE_TH_ERRORS
  auto& var = self->cdata;
  if (!var.grad_fn()) {
    Py_RETURN_NONE;
  }
  return functionToPyObject(var.grad_fn());
  END_HANDLE_TH_ERRORS
}

int THPVariable_set_grad_fn(THPVariable *self, PyObject *obj)
{
  HANDLE_TH_ERRORS
  THPUtils_assertRet(-1, obj == Py_None, "_grad_fn can be only set to None");
  self->cdata.get()->_grad_fn = nullptr;
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

PyObject *THPVariable_is_leaf(THPVariable *self)
{
  HANDLE_TH_ERRORS
  return PyBool_FromLong(!self->cdata.grad_fn());
  END_HANDLE_TH_ERRORS
}

PyObject * THPVariable_get_data(THPVariable *self)
{
  HANDLE_TH_ERRORS
  if (!self->data) {
    self->data = torch::createPyObject(self->cdata.data());
  }
  Py_XINCREF(self->data);
  return self->data;
  END_HANDLE_TH_ERRORS
}

namespace {

// XXX: This is a hack to access private TensorImpl::type_
// http://bloglitb.blogspot.com/2011/12/access-to-private-members-safer.html
// This is currently needed because module.float() changes the type of the
// data field of each variable. We should fix this and not allow changing the
// type of var.data.

template<typename Tag, typename Tag::type M>
struct Rob {
  friend typename Tag::type get(Tag) {
    return M;
  }
};

struct TensorImpl_Type {
  typedef Type* TensorImpl::*type;
  friend type get(TensorImpl_Type);
};

template struct Rob<TensorImpl_Type, &TensorImpl::type_>;

}

int THPVariable_set_data(THPVariable *self, PyObject *data)
{
  HANDLE_TH_ERRORS
  THPUtils_assertRet(-1, THPModule_isTensor(data), "Variable data has to "
      "be a tensor, but got %s", THPUtils_typename(data));
  Py_INCREF(data);
  Py_XDECREF(self->data);
  self->data = data;
  Tensor tensor = torch::createTensor(data);
  if (&self->cdata.data().type() != &tensor.type()) {
    // we change the type of var.data so we must change the type of var
    auto newType = VariableImpl::getType(tensor);
    self->cdata.get()->*get(TensorImpl_Type()) = newType;
  }
  self->cdata.data() = tensor;
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

PyObject *THPVariable_get_grad(THPVariable *self)
{
  HANDLE_TH_ERRORS
  return THPVariable_Wrap(self->cdata.grad());
  END_HANDLE_TH_ERRORS
}

int THPVariable_set_grad(THPVariable *self, PyObject *other)
{
  HANDLE_TH_ERRORS
  auto& var = self->cdata;
  if (other == Py_None) {
    var.grad().reset();
    return 0;
  }

  THPUtils_assertRet(-1, THPVariable_Check(other),
      "expected Variable or None (got %s)", THPUtils_typename(other));
  THPUtils_assertRet(-1, self != (THPVariable*)other,
      "can't assign Variable as its own grad");

  auto& data = var.data();
  auto& other_var = ((THPVariable*)other)->cdata;
  auto& other_data = other_var.data();

  // Make sure the data is ok
  THPUtils_assertRet(-1, other_data.type().ID() == data.type().ID(),
      "assigned grad has data of a different type");
  THPUtils_assertRet(-1, other_data.type().is_cuda() == data.type().is_cuda(),
      "assigned grad has data located on a different device");
  if (data.type().is_cuda()) {
    THPUtils_assertRet(-1, other_data.get_device() == data.get_device(),
        "assigned grad has data located on a different device");
  }
  THPUtils_assertRet(-1, other_data.sizes().vec() == data.sizes().vec(),
      "assigned grad has data of a different size");

  var.grad() = other_var;
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

PyObject *THPVariable_get_volatile(THPVariable *self)
{
  HANDLE_TH_ERRORS
  auto& var = self->cdata;
  return PyBool_FromLong(var.is_volatile());
  END_HANDLE_TH_ERRORS
}

int THPVariable_set_volatile(THPVariable *self, PyObject *obj)
{
  HANDLE_TH_ERRORS
  THPUtils_assertRet(-1, PyBool_Check(obj), "volatile must be a bool");
  THPUtils_assertRet(-1, !self->cdata.grad_fn(),
      "volatile can only be set on leaf variables");
  self->cdata.is_volatile() = (obj == Py_True);
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

PyObject *THPVariable_get_output_nr(THPVariable *self)
{
  HANDLE_TH_ERRORS
  return PyInt_FromLong(self->cdata.output_nr());
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_get_requires_grad(THPVariable *self)
{
  HANDLE_TH_ERRORS
  return PyBool_FromLong(self->cdata.requires_grad());
  END_HANDLE_TH_ERRORS
}

int THPVariable_set_requires_grad(THPVariable *self, PyObject *obj)
{
  HANDLE_TH_ERRORS
  THPUtils_assertRet(-1, PyBool_Check(obj), "requires_grad must be a bool");
  auto& var = self->cdata;
  if (var.grad_fn()) {
    const char *hint = "";
    if (obj == Py_False) {
      hint = " If you want to use a computed variable in a subgraph "
             "that doesn't require differentiation use "
             "var_no_grad = var.detach().";
    }
    THPUtils_setError("you can only change requires_grad flags of leaf variables.%s", hint);
    return -1;
  }
  var.requires_grad() = (obj == Py_True);
  if (auto grad_accumulator = var.get()->grad_accumulator.lock()) {
    grad_accumulator->is_executable = var.requires_grad();
  }
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

PyObject *THPVariable_get_name(THPVariable* self)
{
  if (self->cdata.name() == "")
    Py_RETURN_NONE;
  return THPUtils_packString(self->cdata.name().c_str());
}

PyObject *THPVariable_get_backwards_hooks(THPVariable *self)
{
  HANDLE_TH_ERRORS
  if (self->backward_hooks) {
    Py_INCREF(self->backward_hooks);
    return self->backward_hooks;
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

int THPVariable_set_backwards_hooks(THPVariable *self, PyObject *obj)
{
  HANDLE_TH_ERRORS
  if (obj == Py_None) {
    obj = nullptr;
  }
  Py_XINCREF(obj);
  Py_XDECREF(self->backward_hooks);
  self->backward_hooks = obj;
  self->cdata.hooks().clear();
  if (obj) {
    self->cdata.hooks().emplace_back(new PyFunctionPreHook(obj, 0));
  }
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

PyObject *THPVariable_get_base(THPVariable *self)
{
  HANDLE_TH_ERRORS
  if (self->cdata.is_view()) {
    return THPVariable_Wrap(self->cdata.base());
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_get_shape(THPVariable *self)
{
  HANDLE_TH_ERRORS
  auto& self_ = self->cdata;
  auto sizes = self_.sizes();
  return THPSize_New(sizes.size(), (int64_t *)sizes.data());
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_is_cuda(THPVariable *self)
{
  HANDLE_TH_ERRORS
  auto& self_ = self->cdata;
  return torch::autograd::utils::wrap(self_.is_cuda());
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef THPVariable_properties[] = {
  {"_version", (getter)THPVariable_get_version, NULL, NULL, NULL},
  {"grad_fn", (getter)THPVariable_get_grad_fn, NULL, NULL, NULL},
  {"_grad_fn", (getter)THPVariable_get_grad_fn, (setter)THPVariable_set_grad_fn, NULL, NULL},
  {"is_leaf", (getter)THPVariable_is_leaf, NULL, NULL, NULL},
  {"data", (getter)THPVariable_get_data, (setter)THPVariable_set_data, NULL, NULL},
  {"_grad", (getter)THPVariable_get_grad, (setter)THPVariable_set_grad, NULL, NULL}, // only for legacy reasons
  {"grad", (getter)THPVariable_get_grad, (setter)THPVariable_set_grad, NULL, NULL},
  {"_base", (getter)THPVariable_get_base, NULL, NULL, NULL},
  {"volatile", (getter)THPVariable_get_volatile, (setter)THPVariable_set_volatile, NULL, NULL},
  {"output_nr", (getter)THPVariable_get_output_nr, NULL, NULL, NULL},
  {"requires_grad", (getter)THPVariable_get_requires_grad, (setter)THPVariable_set_requires_grad, NULL, NULL},
  {"_backward_hooks", (getter)THPVariable_get_backwards_hooks, (setter)THPVariable_set_backwards_hooks, NULL, NULL},
  {"name", (getter)THPVariable_get_name, NULL, NULL, NULL},
  {"shape", (getter)THPVariable_get_shape, NULL, NULL, NULL},
  {"is_cuda", (getter)THPVariable_is_cuda, NULL, NULL, NULL},
  {NULL}
};

static PyMappingMethods THPVariable_as_mapping = {
  THPVariable_length,
  THPVariable_getitem,
  THPVariable_setitem,
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
  &THPVariable_as_mapping,               /* tp_as_mapping */
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

namespace torch { namespace autograd {

extern PyMethodDef variable_methods[];

}}

bool THPVariable_initModule(PyObject *module)
{
  static std::vector<PyMethodDef> methods;
  THPUtils_addPyMethodDefs(methods, torch::autograd::variable_methods);
  THPVariableType.tp_methods = methods.data();
  if (PyType_Ready(&THPVariableType) < 0)
    return false;
  Py_INCREF(&THPVariableType);
  PyModule_AddObject(module, "_VariableBase", (PyObject *)&THPVariableType);
  return true;
}
