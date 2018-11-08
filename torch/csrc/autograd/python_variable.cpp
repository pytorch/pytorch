#include "torch/csrc/autograd/python_variable.h"

#include "THP.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/Device.h"
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
#include "torch/csrc/autograd/utils/python_error_messages.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/tensor/python_tensor.h"
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/cuda_lazy_init.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/tensor_new.h"
#include "torch/csrc/jit/tracer.h"

#include <ATen/ATen.h>

#include <structmember.h>
#include <memory>
#include <utility>
#include <vector>

using namespace at;
using namespace torch;
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
  if (self->cdata.defined() && self->cdata.get_autograd_meta()) {
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

static PyObject *THPVariable_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  jit::tracer::warn("torch.Tensor", jit::tracer::WARN_CONSTRUCTOR);
  auto& default_type = torch::tensors::get_default_tensor_type();
  auto tensor = torch::utils::legacy_tensor_ctor(default_type, args, kwargs);
  return THPVariable_NewWithVar(type, std::move(tensor));
  END_HANDLE_TH_ERRORS
}

// Instantiates a subclass of torch.Tensor. Used by nn.Parameter()
static PyObject* THPVariable_make_subclass(PyObject* _ignored, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_make_subclass(PyObject* cls, Tensor data, bool require_grad=False)",
  });
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  PyObject* cls = r.pyobject(0);
  if (!PyType_Check(cls)) {
    throw TypeError("cls must be a type (got %s)", Py_TYPE(cls)->tp_name);
  }
  auto& data = as_variable_ref(r.tensor(1)).data();
  auto var = make_variable(data, r.toBool(2));
  return THPVariable_NewWithVar((PyTypeObject*)cls, std::move(var));
  END_HANDLE_TH_ERRORS
}

typedef PyObject *(*getter)(PyObject *, void *);
typedef int (*setter)(PyObject *, PyObject *, void *);

PyObject *THPVariable_get_cdata(THPVariable *self)
{
  HANDLE_TH_ERRORS
  auto& var = self->cdata;
  return PyLong_FromVoidPtr(var.data().unsafeGetTensorImpl());
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
  self->cdata.set_data(THPVariable_UnpackData(data));
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
    var.grad().reset();
    return 0;
  }

  THPUtils_assertRet(-1, THPVariable_Check(py_grad),
      "expected Variable or None (got %s)", THPUtils_typename(py_grad));
  THPUtils_assertRet(-1, self != (THPVariable*)py_grad,
      "can't assign Variable as its own grad");

  auto& grad = ((THPVariable*)py_grad)->cdata;
  bool gradIsSparse = false;
  auto backend = var.is_cuda() ? Backend::SparseCUDA : Backend::SparseCPU;
  auto typeOpt = at::globalContext().getNonVariableTypeOpt(backend, var.type().scalarType());  
  if (typeOpt) {
       auto& sparseType = at::globalContext().getNonVariableType(backend, var.type().scalarType());
       gradIsSparse = grad.type() == sparseType;
  }

  THPUtils_assertRet(-1, grad.type() == var.type() || gradIsSparse,
      "assigned grad has data of a different type");
  if (var.is_cuda()) {
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
  auto requires_grad = (obj == Py_True);
  if (!var.is_leaf()) {
    THPUtils_setError(autograd::utils::requires_grad_leaf_error(obj == Py_True).c_str());
    return -1;
  }
  if (requires_grad && !var.is_floating_point()) {
    THPUtils_setError("only Tensors of floating point dtype can require gradients");
    return -1;
  }
  var.set_requires_grad(requires_grad);
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
  return THPSize_New(self->cdata);
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

static PyObject *THPVariable_dtype(THPVariable *self)
{
  HANDLE_TH_ERRORS
  auto& self_ = self->cdata;
  return torch::autograd::utils::wrap(torch::getDtype(self_.type().scalarType()));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_layout(THPVariable* self) {
  HANDLE_TH_ERRORS
  auto& self_ = self->cdata;
  return torch::autograd::utils::wrap(torch::getLayout(self_.type().backend()));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_device(THPVariable* self) {
  HANDLE_TH_ERRORS
  return THPDevice_New(torch::tensors::getDevice(self->cdata));
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
  {"dtype", (getter)THPVariable_dtype, nullptr, nullptr, nullptr},
  {"layout", (getter)THPVariable_layout, nullptr, nullptr, nullptr},
  {"device", (getter)THPVariable_device, nullptr, nullptr, nullptr},
  {nullptr}
};

static PyMappingMethods THPVariable_as_mapping = {
  THPVariable_length,
  THPVariable_getitem,
  THPVariable_setitem,
};

static PyMethodDef extra_methods[] = {
  {"_make_subclass", (PyCFunction)THPVariable_make_subclass, METH_STATIC | METH_VARARGS | METH_KEYWORDS, nullptr},
  {nullptr}
};

PyTypeObject THPVariableType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch._C._TensorBase",                /* tp_name */
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
  0,                                     /* tp_init */
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
  THPUtils_addPyMethodDefs(methods, extra_methods);
  THPVariableType.tp_methods = methods.data();
  if (PyType_Ready(&THPVariableType) < 0)
    return false;
  Py_INCREF(&THPVariableType);
  PyModule_AddObject(module, "_TensorBase",   (PyObject *)&THPVariableType);
  torch::autograd::initTorchFunctions(module);
  return true;
}
