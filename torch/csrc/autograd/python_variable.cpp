#include "torch/csrc/autograd/python_variable.h"

#include "THP.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/Size.h"
#include "torch/csrc/Types.h"
#include "torch/csrc/autograd/edge.h"
#include "torch/csrc/autograd/python_cpp_function.h"
#include "torch/csrc/autograd/python_hook.h"
#include "torch/csrc/autograd/python_variable_indexing.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/functions/accumulate_grad.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/cuda/AutoGPU.h"
#include "torch/csrc/jit/tracer_state.h"
#include "torch/csrc/tensor/python_tensor.h"
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/python_strings.h"

#include <ATen/ATen.h>

#include <list>
#include <memory>
#include <structmember.h>

using namespace at;
using namespace torch::autograd;

PyObject *THPVariableClass = nullptr;

static const char* VOLATILE_WARNING =
    "volatile was removed and now has no effect. Use "
    "`with torch.no_grad():` instead.";

// Creates a new Python object for a Variable. The Variable must not already
// have a PyObject* associated with it.
static PyObject* THPVariable_NewWithVar(PyTypeObject* type, Variable var)
{
  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    auto v = (THPVariable*) obj;
    new (&v->cdata) Variable(std::move(var));
    v->cdata.set_pyobj(obj);
    if (auto fn = dynamic_cast<PyFunction*>(v->cdata.grad_fn_unsafe())) {
      // Create a new reference to the THPFunction. This ensures that ref count
      // of the THPFunction is at least the number of referring THPVariables.
      const auto output_nr = v->cdata.output_nr();
      auto grad_fn = THPFunction_asFunction((THPFunction*)fn->obj);
      v->cdata.set_gradient_edge({std::move(grad_fn), output_nr});
    }
  }
  return obj;
}

PyObject * THPVariable_Wrap(Variable var)
{
  if (!var.defined()) {
    Py_RETURN_NONE;
  }

  if (auto obj = var.pyobj()) {
    Py_INCREF(obj);
    return obj;
  }

  return THPVariable_NewWithVar((PyTypeObject *)THPVariableClass, std::move(var));
}

static int THPVariable_traverse(THPVariable *self, visitproc visit, void *arg)
{
  Py_VISIT(self->backward_hooks);
  // We don't want to traverse the grad_fn, even if the Variable owns it and the
  // shared pointer's use count is 1. This is because we would need to treat
  // the grad_fn as part of the Python state and hold the GIL sometimes when
  // grad_fn's shared_ptr is copied, otherwise a race condition with the Python
  // GC could occur. Holding the GIL when the shared_ptr is copied adds
  // undesirable complexity/overhead.
  //
  // When hooks, a Variable, and its grad_fn are involved in a Python reference
  // cycle, because we're not traversing the grad_fn, the reference cycle will
  // in fact leak.
  //
  // See https://gist.github.com/zou3519/7ac92b84dd7d206dcc6eae55fee8372c
  // for more details about the race condition involving traversing the grad_fn
  // and the python GC.
  if (self->cdata.defined()) {
    for (const auto& hook : self->cdata.hooks()) {
      if (auto pyhook = dynamic_cast<PyFunctionPreHook*>(hook.get())) {
        Py_VISIT(pyhook->dict);
      }
    }
  }
  return 0;
}

static int THPVariable_clear(THPVariable *self)
{
  Py_CLEAR(self->backward_hooks);
  if (self->cdata.defined()) {
    if (auto grad_acc = self->cdata.try_get_grad_accumulator()) {
      grad_acc->pre_hooks().clear();
    }
    self->cdata.set_pyobj(nullptr);
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
  HANDLE_TH_ERRORS
  THPObjectPtr _data;
  PyObject *data = nullptr;
  PyObject *grad_fn = nullptr;
  char is_volatile = 0;
  char requires_grad = 0;
  const char* name = nullptr;

  const char *accepted_args[] = {"data", "requires_grad", "volatile", "_grad_fn", "name", nullptr};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ObbOz", (char**)accepted_args,
      &data, &requires_grad, &is_volatile, &grad_fn, &name))
    return nullptr;

  if (grad_fn == Py_None)
    grad_fn = nullptr;

  if (is_volatile) {
    PyErr_WarnEx(PyExc_UserWarning, VOLATILE_WARNING, 1);
  }

  THPUtils_assert(!(is_volatile && requires_grad),
          "Variable can't be volatile and require_grad at the same time!");
  THPUtils_assert(!grad_fn || THPFunction_Check(grad_fn),
          "Variable _grad_fn has to be a Function object or None, but got %s",
          THPUtils_typename(grad_fn));
  Tensor tensor;
  if (!data || data == Py_None) {
    // For legacy serialization code, create an empty tensor. This is also used
    // by nn.Parameter() with no arguments.
    auto var = torch::tensor::get_default_tensor_type().tensor();
    tensor = static_cast<Variable&>(var).data();
  } else if (THPModule_isTensor(data)) {
    tensor = torch::createTensor(data);
  } else if (THPVariable_Check(data)) {
    tensor = ((THPVariable*)data)->cdata.data();
  } else {
    throw torch::TypeError("Variable data has to be a tensor, but got %s",
        THPUtils_typename(data));
  }

  Variable var;
  if (grad_fn) {
    auto grad_fn_ = THPFunction_asFunction((THPFunction*)grad_fn);
    Edge edge(grad_fn_, grad_fn_->bump_inputs());
    var = make_variable(std::move(tensor), std::move(edge));
  } else {
    var = make_variable(std::move(tensor), requires_grad);
  }

  if (name) {
    var.set_name(name);
  }

  return THPVariable_NewWithVar(type, std::move(var));
  END_HANDLE_TH_ERRORS
}

int THPVariable_pyinit(PyObject *self, PyObject *args, PyObject *kwds)
{
  // Ensures that calls to Variable() and subclasses contain data argument.
  // The 'data' argument is optional in __new__ to handle legacy serialized
  // Variables.
  PyObject *data;
  PyObject *grad_fn = nullptr;
  char is_volatile = 0;
  char requires_grad = 0;
  const char* name = nullptr;

  const char *accepted_args[] = {"data", "requires_grad", "volatile", "_grad_fn", "name", nullptr};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ObbOz", (char**)accepted_args,
      &data, &requires_grad, &is_volatile, &grad_fn, &name))
    return -1;

  return 0;
}

typedef PyObject *(*getter)(PyObject *, void *);
typedef int (*setter)(PyObject *, PyObject *, void *);

PyObject *THPVariable_get_cdata(THPVariable *self)
{
  HANDLE_TH_ERRORS
  auto& var = self->cdata;
  return PyLong_FromVoidPtr(var.unsafeGetTH(false));
  END_HANDLE_TH_ERRORS
}

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

static int THPVariable_set_grad_fn(THPVariable *self, PyObject *obj)
{
  HANDLE_TH_ERRORS
  THPUtils_assertRet(-1, obj == Py_None, "_grad_fn can be only set to None");
  self->cdata.detach_();
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

static PyObject *THPVariable_is_leaf(THPVariable *self)
{
  HANDLE_TH_ERRORS
  return PyBool_FromLong(!self->cdata.grad_fn());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_get_data(THPVariable *self)
{
  HANDLE_TH_ERRORS
  return THPVariable_Wrap(make_variable(self->cdata.data(), false));
  END_HANDLE_TH_ERRORS
}

int THPVariable_set_data(THPVariable *self, PyObject *data)
{
  HANDLE_TH_ERRORS
  if (!THPVariable_Check(data)) {
    throw torch::TypeError("Variable data has to be a tensor, but got %s", Py_TYPE(data)->tp_name);
  }
  Tensor tensor = THPVariable_UnpackData(data);
  if (self->cdata.data().type() != tensor.type()) {
    // we change the type of var.data so we must change the type of var
    auto newType = VariableType::getType(tensor);
    self->cdata.temporary_hack_set_type(newType);
  }
  self->cdata.data() = std::move(tensor);
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

PyObject *THPVariable_get_grad(THPVariable *self)
{
  HANDLE_TH_ERRORS
  return THPVariable_Wrap(self->cdata.grad());
  END_HANDLE_TH_ERRORS
}

int THPVariable_set_grad(THPVariable *self, PyObject *py_grad)
{
  HANDLE_TH_ERRORS
  auto& var = self->cdata;
  if (py_grad == Py_None) {
    var.reset_grad();
    return 0;
  }

  THPUtils_assertRet(-1, THPVariable_Check(py_grad),
      "expected Variable or None (got %s)", THPUtils_typename(py_grad));
  THPUtils_assertRet(-1, self != (THPVariable*)py_grad,
      "can't assign Variable as its own grad");

  auto& grad = ((THPVariable*)py_grad)->cdata;
  auto& sparseType = var.type().toBackend(var.is_cuda() ? kSparseCUDA : kSparseCPU);

  THPUtils_assertRet(-1, grad.type() == var.type() || grad.type() == sparseType,
      "assigned grad has data of a different type");
  if (var.type().is_cuda()) {
    THPUtils_assertRet(-1, grad.get_device() == var.get_device(),
        "assigned grad has data located on a different device");
  }
  THPUtils_assertRet(-1, grad.sizes().equals(var.sizes()),
      "assigned grad has data of a different size");

  var.grad() = grad;
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

PyObject *THPVariable_get_volatile(THPVariable *self)
{
  const char* msg = "volatile was removed (Variable.volatile is always False)";
  PyErr_WarnEx(PyExc_UserWarning, msg, 1);
  Py_RETURN_FALSE;
}

int THPVariable_set_volatile(THPVariable *self, PyObject *obj)
{
  return PyErr_WarnEx(PyExc_UserWarning, VOLATILE_WARNING, 1);
}

PyObject *THPVariable_get_output_nr(THPVariable *self)
{
  HANDLE_TH_ERRORS
  const auto output_nr = static_cast<long>(self->cdata.output_nr());
  return PyInt_FromLong(output_nr);
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
  if (!var.is_leaf()) {
    const char *hint = "";
    if (obj == Py_False) {
      hint = " If you want to use a computed variable in a subgraph "
             "that doesn't require differentiation use "
             "var_no_grad = var.detach().";
    }
    THPUtils_setError("you can only change requires_grad flags of leaf variables.%s", hint);
    return -1;
  }
  var.set_requires_grad(obj == Py_True);
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
  self->cdata.clear_hooks();
  if (obj) {
    self->cdata.add_hook(std::make_shared<PyFunctionPreHook>(obj, 0));
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

PyObject *THPVariable_is_sparse(THPVariable *self)
{
  HANDLE_TH_ERRORS
  auto& self_ = self->cdata;
  return torch::autograd::utils::wrap(self_.is_sparse());
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_dtype(THPVariable *self)
{
  HANDLE_TH_ERRORS
  auto& self_ = self->cdata;
  return torch::autograd::utils::wrap(torch::getDtype(self_.type()));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef THPVariable_properties[] = {
  {"_cdata", (getter)THPVariable_get_cdata, nullptr, nullptr, nullptr},
  {"_version", (getter)THPVariable_get_version, nullptr, nullptr, nullptr},
  {"grad_fn", (getter)THPVariable_get_grad_fn, nullptr, nullptr, nullptr},
  {"_grad_fn", (getter)THPVariable_get_grad_fn, (setter)THPVariable_set_grad_fn, nullptr, nullptr},
  {"is_leaf", (getter)THPVariable_is_leaf, nullptr, nullptr, nullptr},
  {"data", (getter)THPVariable_get_data, (setter)THPVariable_set_data, nullptr, nullptr},
  {"_grad", (getter)THPVariable_get_grad, (setter)THPVariable_set_grad, nullptr, nullptr}, // only for legacy reasons
  {"grad", (getter)THPVariable_get_grad, (setter)THPVariable_set_grad, nullptr, nullptr},
  {"_base", (getter)THPVariable_get_base, nullptr, nullptr, nullptr},
  {"volatile", (getter)THPVariable_get_volatile, (setter)THPVariable_set_volatile, nullptr, nullptr},
  {"output_nr", (getter)THPVariable_get_output_nr, nullptr, nullptr, nullptr},
  {"requires_grad", (getter)THPVariable_get_requires_grad, (setter)THPVariable_set_requires_grad, nullptr, nullptr},
  {"_backward_hooks", (getter)THPVariable_get_backwards_hooks, (setter)THPVariable_set_backwards_hooks, nullptr, nullptr},
  {"name", (getter)THPVariable_get_name, nullptr, nullptr, nullptr},
  {"shape", (getter)THPVariable_get_shape, nullptr, nullptr, nullptr},
  {"is_cuda", (getter)THPVariable_is_cuda, nullptr, nullptr, nullptr},
  {"is_sparse", (getter)THPVariable_is_sparse, nullptr, nullptr, nullptr},
  {"dtype", (getter)THPVariable_dtype, NULL, NULL, NULL},
  {nullptr}
};

static PyMappingMethods THPVariable_as_mapping = {
  THPVariable_length,
  THPVariable_getitem,
  THPVariable_setitem,
};

PyTypeObject THPVariableType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
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
  nullptr,                               /* tp_doc */
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
extern void initTorchFunctions(PyObject *module);

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
  torch::autograd::initTorchFunctions(module);
  return true;
}
