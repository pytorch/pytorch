#include <torch/csrc/autograd/python_variable.h>

#include <torch/csrc/THP.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Size.h>
#include <torch/csrc/Types.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/csrc/autograd/python_hook.h>
#include <torch/csrc/autograd/python_variable_indexing.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/autograd/utils/error_messages.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/tensor/python_tensor.h>
#include <torch/csrc/utils/auto_gil.h>
#include <torch/csrc/utils/cuda_lazy_init.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/csrc/jit/tracer.h>
#include <ATen/core/EnableNamedTensor.h>
#ifdef BUILD_NAMEDTENSOR
#include <ATen/NamedTensorUtils.h>
#endif

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>

#include <structmember.h>
#include <memory>
#include <utility>
#include <vector>

using namespace at;
using namespace torch;
using namespace torch::autograd;

namespace py = pybind11;

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
    torch::autograd::impl::set_pyobj(v->cdata, obj);
  }
  return obj;
}

PyObject * THPVariable_Wrap(Variable var)
{
  if (!var.defined()) {
    Py_RETURN_NONE;
  }

  if (auto obj = torch::autograd::impl::pyobj(var)) {
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
    for (const auto& hook : torch::autograd::impl::hooks(self->cdata)) {
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
    if (auto grad_acc = torch::autograd::impl::try_get_grad_accumulator(self->cdata)) {
      grad_acc->pre_hooks().clear();
    }
    // We must clear the pyobj field in the base C++ Variable, to ensure
    // that if we attempt to pass the Variable to Python, we don't
    // attempt to reuse the (now-dead) PyObject.
    //
    // One non-obvious consequence of this: if you have a tensor x, you
    // take its id(), and then you let it become dead in Python, if you
    // get another reference to the tensor in Python later (because you
    // passed it from C++ to Python), you'll get a *different* id() the
    // second time around.  So you better make sure that if you're using
    // id() to keep track of Tensors, you better make sure their Python
    // objects stay live, buster!  See
    // https://github.com/pytorch/pytorch/issues/22884 for an example of
    // this actually showing up.
    torch::autograd::impl::set_pyobj(self->cdata, nullptr);
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
  auto tensor = torch::utils::legacy_tensor_ctor(torch::tensors::get_default_tensor_type_id(), torch::tensors::get_default_scalar_type(), args, kwargs);
  return THPVariable_NewWithVar(type, std::move(tensor));
  END_HANDLE_TH_ERRORS
}

// Instantiates a subclass of torch.Tensor. Used by nn.Parameter()
static PyObject* THPVariable_make_subclass(PyObject* _ignored, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_make_subclass(PyObject* cls, Tensor data, bool require_grad=False)",
  });
  ParsedArgs<3> parsed_args{};
  auto r = parser.parse(args, kwargs, parsed_args);
  PyObject* cls = r.pyobject(0);
  if (!PyType_Check(cls)) {
    throw TypeError("cls must be a type (got %s)", Py_TYPE(cls)->tp_name);
  }
  auto data = as_variable_ref(r.tensor(1)).detach();
  // We set `data`'s `allow_tensor_metadata_change` to true here, because we want to
  // allow the following use case for backward compatibility:
  //
  // ```python
  // rnn = torch.nn.RNN(100, 100, 2)
  // # The following calls `torch._cudnn_rnn_flatten_weight(rnn._flat_weights, ...)`,
  // # which changes storage of `rnn`'s weights in-place
  // rnn.flatten_parameters()
  // ```
  data.unsafeGetTensorImpl()->set_allow_tensor_metadata_change(true);
  auto var = data.set_requires_grad(r.toBool(2));
  return THPVariable_NewWithVar((PyTypeObject*)cls, std::move(var));
  END_HANDLE_TH_ERRORS
}

typedef PyObject *(*getter)(PyObject *, void *);
typedef int (*setter)(PyObject *, PyObject *, void *);

PyObject *THPVariable_get_T(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  auto& var = self->cdata;
  return THPVariable_Wrap(var.numpy_T());
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_get_cdata(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  auto& var = self->cdata;
  return PyLong_FromVoidPtr(var.unsafeGetTensorImpl());
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_get_version(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  auto& var = self->cdata;
  return PyInt_FromLong(var.current_version());
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_get_grad_fn(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  auto& var = self->cdata;
  if (!var.grad_fn()) {
    Py_RETURN_NONE;
  }
  return functionToPyObject(var.grad_fn());
  END_HANDLE_TH_ERRORS
}

static int THPVariable_set_grad_fn(THPVariable *self, PyObject *obj, void *unused)
{
  HANDLE_TH_ERRORS
  THPUtils_assertRet(-1, obj, "Deletion of _grad_fn not allowed. Detach tensor instead!");
  THPUtils_assertRet(-1, obj == Py_None, "_grad_fn can be only set to None");
  self->cdata.detach_();
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

static PyObject *THPVariable_is_leaf(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  return PyBool_FromLong(!self->cdata.grad_fn());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_get_data(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  auto var = self->cdata.variable_data();
  return THPVariable_Wrap(var);
  END_HANDLE_TH_ERRORS
}

int THPVariable_set_data(THPVariable *self, PyObject *data, void *unused)
{
  HANDLE_TH_ERRORS
  THPUtils_assertRet(-1, data, "Deleting tensor data is not allowed. Delete tensor instead!");
  if (!THPVariable_Check(data)) {
    throw torch::TypeError("Variable data has to be a tensor, but got %s", Py_TYPE(data)->tp_name);
  }

  self->cdata.set_data(THPVariable_Unpack(data));
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

PyObject *THPVariable_get_grad(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  return THPVariable_Wrap(self->cdata.grad());
  END_HANDLE_TH_ERRORS
}

int THPVariable_set_grad(THPVariable *self, PyObject *py_grad, void *unused)
{
  HANDLE_TH_ERRORS
  auto& var = self->cdata;
  if (!py_grad || py_grad == Py_None) {
    var.grad().reset();
    return 0;
  }

  THPUtils_assertRet(-1, THPVariable_Check(py_grad),
      "expected Variable or None (got %s)", THPUtils_typename(py_grad));
  THPUtils_assertRet(-1, self != (THPVariable*)py_grad,
      "can't assign Variable as its own grad");

  auto& grad = ((THPVariable*)py_grad)->cdata;
  bool gradIsSparse = (var.dtype() == grad.dtype() &&
                       var.device().type() == grad.device().type() &&
                       grad.layout() == kSparse);
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

PyObject *THPVariable_get_volatile(THPVariable *self, void *unused)
{
  const char* msg = "volatile was removed (Variable.volatile is always False)";
  PyErr_WarnEx(PyExc_UserWarning, msg, 1);
  Py_RETURN_FALSE;
}

int THPVariable_set_volatile(THPVariable *self, PyObject *obj, void *unused)
{
  return PyErr_WarnEx(PyExc_UserWarning, VOLATILE_WARNING, 1);
}

PyObject *THPVariable_get_output_nr(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  const auto output_nr = static_cast<long>(self->cdata.output_nr());
  return PyInt_FromLong(output_nr);
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_get_requires_grad(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  return PyBool_FromLong(self->cdata.requires_grad());
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_get_ndim(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  return PyInt_FromLong(self->cdata.dim());
  END_HANDLE_TH_ERRORS
}

#ifdef BUILD_NAMEDTENSOR
PyObject *THPVariable_get_names(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  // The long-term plan is to return a list of (python) torch.Dimname.
  // However, for now, return a list of string.
  size_t size = self->cdata.dim();
  THPObjectPtr tuple(PyTuple_New(size));
  if (!tuple) throw python_error();

  const auto dimnames = self->cdata.names();
  for (size_t i = 0; i < size; ++i) {
    PyObject* str;
    if (dimnames[i].type() == at::NameType::WILDCARD) {
      // PyTuple_SET_ITEM steals a reference to the object. When the tuple is
      // deallocated, it'll decrement the refcount on Py_None, which is bad.
      // To avoid this, we "create" a new reference to Py_None by increasing
      // the refcount.
      // Sources:
      // - https://docs.python.org/3/c-api/tuple.html#c.PyTuple_SetItem
      // - https://stackoverflow.com/questions/16400600/how-to-return-a-tuple-containing-a-none-value-from-the-c-api
      Py_INCREF(Py_None);
      str = Py_None;
    } else {
      str = THPUtils_packString(dimnames[i].symbol().toUnqualString());
      if (!str) throw python_error();
    }
    PyTuple_SET_ITEM(tuple.get(), i, str);
  }
  return tuple.release();
  END_HANDLE_TH_ERRORS
}

int THPVariable_set_names(THPVariable *self, PyObject *names) {
  HANDLE_TH_ERRORS
  auto& var = self->cdata;
  if (names == Py_None) {
    at::internal_set_names_inplace(var, at::nullopt);
  } else {
    THPUtils_assertRet(-1,
        THPUtils_checkDimnameList(names),
        "names must either be None or a tuple of dim names");
    at::internal_set_names_inplace(var, torch::parseDimnameList(names));
  }
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}
#endif

int THPVariable_set_requires_grad(THPVariable *self, PyObject *obj, void *unused)
{
  HANDLE_TH_ERRORS
  THPUtils_assertRet(-1, obj && PyBool_Check(obj), "requires_grad must be a bool");
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

PyObject *THPVariable_get_name(THPVariable* self, void *unused)
{
  if (self->cdata.name() == "")
    Py_RETURN_NONE;
  return THPUtils_packString(self->cdata.name().c_str());
}

PyObject *THPVariable_get_backwards_hooks(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  if (self->backward_hooks) {
    Py_INCREF(self->backward_hooks);
    return self->backward_hooks;
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

int THPVariable_set_backwards_hooks(THPVariable *self, PyObject *obj, void *unused)
{
  HANDLE_TH_ERRORS
  THPUtils_assertRet(-1, obj, "Deletion of _backwards_hooks not allowed!");
  if (obj == Py_None) {
    obj = nullptr;
  }
  Py_XINCREF(obj);
  Py_XDECREF(self->backward_hooks);
  self->backward_hooks = obj;
  torch::autograd::impl::clear_hooks(self->cdata);
  if (obj) {
    torch::autograd::impl::add_hook(self->cdata, std::make_shared<PyFunctionPreHook>(obj, 0));
  }
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

PyObject *THPVariable_get_base(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  if (self->cdata.is_view()) {
    return THPVariable_Wrap(self->cdata.base());
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_get_shape(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  return THPSize_New(self->cdata);
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_is_cuda(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  auto& self_ = self->cdata;
  return torch::autograd::utils::wrap(self_.is_cuda());
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_is_sparse(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  auto& self_ = self->cdata;
  return torch::autograd::utils::wrap(self_.is_sparse());
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_is_mkldnn(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  auto& self_ = self->cdata;
  return torch::autograd::utils::wrap(self_.is_mkldnn());
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_is_quantized(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  auto& self_ = self->cdata;
  return torch::autograd::utils::wrap(self_.is_quantized());
  END_HANDLE_TH_ERRORS
}

static PyObject *THPVariable_dtype(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  auto& self_ = self->cdata;
  return torch::autograd::utils::wrap(torch::getDtype(self_.scalar_type()));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_layout(THPVariable* self, void *unused) {
  HANDLE_TH_ERRORS
  auto& self_ = self->cdata;
  return torch::autograd::utils::wrap(torch::getLayout(self_.type().backend()));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_device(THPVariable* self, void *unused) {
  HANDLE_TH_ERRORS
  return THPDevice_New(self->cdata.device());
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef THPVariable_properties[] = {
  {"T", (getter)THPVariable_get_T, nullptr, nullptr, nullptr},
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
  {"is_mkldnn", (getter)THPVariable_is_mkldnn, nullptr, nullptr, nullptr},
  {"is_quantized", (getter)THPVariable_is_quantized, nullptr, nullptr, nullptr},
  {"dtype", (getter)THPVariable_dtype, nullptr, nullptr, nullptr},
  {"layout", (getter)THPVariable_layout, nullptr, nullptr, nullptr},
  {"device", (getter)THPVariable_device, nullptr, nullptr, nullptr},
  {"ndim", (getter)THPVariable_get_ndim, nullptr, nullptr, nullptr},
#ifdef BUILD_NAMEDTENSOR
  {"names", (getter)THPVariable_get_names, (setter)THPVariable_set_names, nullptr, nullptr},
#endif
  {nullptr}
};

static PyMappingMethods THPVariable_as_mapping = {
  THPVariable_length,
  THPVariable_getitem,
  THPVariable_setitem,
};

static PyMethodDef extra_methods[] = {
  {"_make_subclass", (PyCFunction)(void(*)(void))THPVariable_make_subclass, METH_STATIC | METH_VARARGS | METH_KEYWORDS, nullptr},
  {nullptr}
};

PyTypeObject THPVariableType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch._C._TensorBase",                      /* tp_name */
  sizeof(THPVariable),                         /* tp_basicsize */
  0,                                           /* tp_itemsize */
  (destructor)THPVariable_dealloc,             /* tp_dealloc */
  0,                                           /* tp_vectorcall_offset */
  nullptr,                                     /* tp_getattr */
  nullptr,                                     /* tp_setattr */
  nullptr,                                     /* tp_reserved */
  nullptr,                                     /* tp_repr */
  nullptr,                                     /* tp_as_number */
  nullptr,                                     /* tp_as_sequence */
  &THPVariable_as_mapping,                     /* tp_as_mapping */
  nullptr,                                     /* tp_hash  */
  nullptr,                                     /* tp_call */
  nullptr,                                     /* tp_str */
  nullptr,                                     /* tp_getattro */
  nullptr,                                     /* tp_setattro */
  nullptr,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC, /* tp_flags */
  nullptr,                                     /* tp_doc */
  (traverseproc)THPVariable_traverse,          /* tp_traverse */
  (inquiry)THPVariable_clear,                  /* tp_clear */
  nullptr,                                     /* tp_richcompare */
  0,                                           /* tp_weaklistoffset */
  nullptr,                                     /* tp_iter */
  nullptr,                                     /* tp_iternext */
  nullptr,                                     /* tp_methods */
  nullptr,                                     /* tp_members */
  THPVariable_properties,                      /* tp_getset */
  nullptr,                                     /* tp_base */
  nullptr,                                     /* tp_dict */
  nullptr,                                     /* tp_descr_get */
  nullptr,                                     /* tp_descr_set */
  0,                                           /* tp_dictoffset */
  nullptr,                                     /* tp_init */
  nullptr,                                     /* tp_alloc */
  THPVariable_pynew                            /* tp_new */
};

namespace torch { namespace autograd {

extern PyMethodDef variable_methods[];
extern void initTorchFunctions(PyObject *module);

void initTensorImplConversion(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  m.def("_wrap_tensor_impl", [](void* ptr) {
    auto p = c10::intrusive_ptr<c10::TensorImpl, at::UndefinedTensorImpl>::
        unsafe_reclaim_from_nonowning(static_cast<c10::TensorImpl*>(ptr));
    TORCH_CHECK(p.defined(), "Can't wrap undefined tensor");
    auto tensor = at::Tensor::wrap_tensor_impl(std::move(p));
    return py::cast(std::move(tensor));
  });
  // set on the module level to avoid mixing pybind and plain CPython extensions
  m.def("_tensor_impl_raw_handle", [](torch::autograd::Variable* t) -> void* {
    // We return a raw non-owning pointer here, we rely on surrounding
    // code to keep the original tensor alive
    return t->getIntrusivePtr().get();
  });
}
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
  torch::autograd::initTensorImplConversion(module);
  return true;
}
