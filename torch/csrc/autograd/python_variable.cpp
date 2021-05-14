#include <torch/csrc/autograd/python_variable.h>

#include <torch/csrc/THP.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Size.h>
#include <torch/csrc/Types.h>
#include <torch/csrc/autograd/autograd.h>
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
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/cuda_lazy_init.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <ATen/NamedTensorUtils.h>
#include <c10/util/DeadlockDetection.h>

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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyObject *THPVariableClass = nullptr;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyObject *ParameterClass = nullptr;

// clang-tidy gets confused by static const
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static const char* VOLATILE_WARNING =
    "volatile was removed and now has no effect. Use "
    "`with torch.no_grad():` instead.";

#ifdef USE_DEPLOY
// used only in libtorch_deployinterpreter.so
// there are muliple copies of the python interpreter that
// can shared Tensors, so rather than use their internal pointer
// to a PyObject use a library-local map.
static std::unordered_map<void*, PyObject*> impl_to_pyobj;

void set_pyobj(const Variable& self, PyObject* pyobj) {
  TORCH_CHECK(self.defined(), "cannot call set_pyobj() on undefined tensor");
  void* key = self.unsafeGetTensorImpl();
  if (!pyobj) {
    impl_to_pyobj.erase(key);
    return;
  }
  impl_to_pyobj[key] = pyobj;
}

PyObject* pyobj(const Variable& self) {
  TORCH_CHECK(self.defined(), "cannot call pyobj() on undefined tensor");
  auto it = impl_to_pyobj.find(self.unsafeGetTensorImpl());
  return it == impl_to_pyobj.end() ? nullptr : it->second;
}
#else
void set_pyobj(const Variable& self, PyObject* pyobj) {
  TORCH_CHECK(self.defined(), "cannot call set_pyobj() on undefined tensor");
  self.unsafeGetTensorImpl()->set_pyobj(pyobj);
}

PyObject* pyobj(const Variable& self) {
  TORCH_CHECK(self.defined(), "cannot call pyobj() on undefined tensor");
  return self.unsafeGetTensorImpl()->pyobj();
}
#endif

// Creates a new Python object for a Variable. The Variable must not already
// have a PyObject* associated with it.
static PyObject* THPVariable_NewWithVar(PyTypeObject* type, Variable var)
{
  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    auto v = (THPVariable*) obj;
    new (&v->cdata) Variable(std::move(var));
    set_pyobj(v->cdata, obj);
  }
  return obj;
}

PyObject * THPVariable_Wrap(Variable var)
{
  if (!var.defined()) {
    Py_RETURN_NONE;
  }

  if (auto obj = pyobj(var)) {
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
  const auto& tensor = THPVariable_Unpack(self);
  if (tensor.defined()) {
    for (const auto& hook : torch::autograd::impl::hooks(tensor)) {
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
  const auto& tensor = THPVariable_Unpack(self);
  if (tensor.defined()) {
    if (auto grad_acc = torch::autograd::impl::try_get_grad_accumulator(tensor)) {
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
    set_pyobj(self->cdata, nullptr);
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

PyObject *THPVariable_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs);

// Instantiates a subclass of self with the same data.
static PyObject* THPVariable_as_subclass(PyObject* _self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  const auto& self = THPVariable_Unpack(_self);
  static PythonArgParser parser({
    "as_subclass(PyObject* cls)",
  });
  ParsedArgs<1> parsed_args{};
  auto r = parser.parse(_self, args, kwargs, parsed_args);
  PyObject* cls = r.pyobject(0);
  if (!PyType_Check(cls)) {
    throw torch::TypeError("cls must be a type (got %s)", Py_TYPE(cls)->tp_name);
  }
  return THPVariable_NewWithVar((PyTypeObject*)cls, self.alias());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_make_subclass(PyObject* _ignored, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_make_subclass(PyObject* cls, Tensor data, bool require_grad=False)",
  });
  ParsedArgs<3> parsed_args{};
  auto r = parser.parse(args, kwargs, parsed_args);
  PyObject* cls = r.pyobject(0);
  if (!PyType_Check(cls)) {
    throw torch::TypeError("cls must be a type (got %s)", Py_TYPE(cls)->tp_name);
  }
  auto data = r.tensor(1).detach();
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
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "T");
  }
  const auto& var = THPVariable_Unpack(self);
  return THPVariable_Wrap(var.numpy_T());
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_get_cdata(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "_cdata");
  }
  const auto& var = THPVariable_Unpack(self);
  return PyLong_FromVoidPtr(var.unsafeGetTensorImpl());
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_get_version(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "_version");
  }
  const auto& var = THPVariable_Unpack(self);
  return PyInt_FromLong(var._version());
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_get_grad_fn(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "grad_fn");
  }
  const auto& var = THPVariable_Unpack(self);
  if (!var.grad_fn()) {
    Py_RETURN_NONE;
  }
  return functionToPyObject(var.grad_fn());
  END_HANDLE_TH_ERRORS
}

static int THPVariable_set_grad_fn(THPVariable *self, PyObject *obj, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_setter(self, "_grad_fn", obj);
  }
  THPUtils_assertRet(-1, obj, "Deletion of _grad_fn not allowed. Detach tensor instead!");
  THPUtils_assertRet(-1, obj == Py_None, "_grad_fn can be only set to None");
  THPVariable_Unpack(self).detach_();
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

static PyObject *THPVariable_is_leaf(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "is_leaf");
  }
  return PyBool_FromLong(!THPVariable_Unpack(self).grad_fn());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_get_data(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "data");
  }
  const auto& var = THPVariable_Unpack(self).variable_data();
  return THPVariable_Wrap(var);
  END_HANDLE_TH_ERRORS
}

int THPVariable_set_data(THPVariable *self, PyObject *data, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_setter(self, "data", data);
  }
  THPUtils_assertRet(-1, data, "Deleting tensor data is not allowed. Delete tensor instead!");
  if (!THPVariable_Check(data)) {
    throw torch::TypeError("Variable data has to be a tensor, but got %s", Py_TYPE(data)->tp_name);
  }

  THPVariable_Unpack(self).set_data(THPVariable_Unpack(data));
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

PyObject *THPVariable_get_grad(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "grad");
  }
  return THPVariable_Wrap(THPVariable_Unpack(self).grad());
  END_HANDLE_TH_ERRORS
}

int THPVariable_set_grad(THPVariable *self, PyObject *py_grad, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_setter(self, "grad", py_grad);
  }
  const auto& var = THPVariable_Unpack(self);
  if (!py_grad || py_grad == Py_None) {
    var.mutable_grad().reset();
    return 0;
  }

  THPUtils_assertRet(-1, THPVariable_Check(py_grad),
      "expected Variable or None (got %s)", THPUtils_typename(py_grad));
  THPUtils_assertRet(-1, self != (THPVariable*)py_grad,
      "can't assign Variable as its own grad");

  const auto& grad = THPVariable_Unpack(py_grad);
  bool gradIsSparse = (var.dtype() == grad.dtype() &&
                       var.device().type() == grad.device().type() &&
                       grad.layout() == kSparse);
  THPUtils_assertRet(-1, grad.options().type_equal(var.options()) || gradIsSparse,
      "assigned grad has data of a different type");
  if (var.is_cuda()) {
    THPUtils_assertRet(-1, grad.get_device() == var.get_device(),
        "assigned grad has data located on a different device");
  }
  THPUtils_assertRet(-1, grad.sizes().equals(var.sizes()),
      "assigned grad has data of a different size");

  var.mutable_grad() = grad;
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

PyObject *THPVariable_get_volatile(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "volatile");
  }
  const char* msg = "volatile was removed (Variable.volatile is always False)";
  auto r = PyErr_WarnEx(PyExc_UserWarning, msg, 1);
  if (r != 0) throw python_error();
  Py_RETURN_FALSE;
  END_HANDLE_TH_ERRORS
}

int THPVariable_set_volatile(THPVariable *self, PyObject *obj, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_setter(self, "volatile", obj);
  }
  auto r = PyErr_WarnEx(PyExc_UserWarning, VOLATILE_WARNING, 1);
  if (r != 0) throw python_error();
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

PyObject *THPVariable_get_output_nr(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "output_nr");
  }
  const auto output_nr = static_cast<long>(THPVariable_Unpack(self).output_nr());
  return PyInt_FromLong(output_nr);
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_get_requires_grad(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "requires_grad");
  }
  return PyBool_FromLong(THPVariable_Unpack(self).requires_grad());
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_get_ndim(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "ndim");
  }
  return PyInt_FromLong(THPVariable_Unpack(self).dim());
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_get_names(PyObject *self, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function_getter((THPVariable*)self, "names");
  }
  // The long-term plan is to return a list of (python) torch.Dimname.
  // However, for now, return a list of string.
  const auto& tensor = THPVariable_Unpack(self);
  size_t size = tensor.dim();
  THPObjectPtr tuple(PyTuple_New(size));
  if (!tuple) throw python_error();

  const auto dimnames = tensor.names();
  for (size_t i = 0; i < size; ++i) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
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

int THPVariable_set_names(PyObject *self, PyObject *names, void *unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function_setter((THPVariable*)self, "names", names);
  }
  const auto& var = THPVariable_Unpack(self);
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

int THPVariable_set_requires_grad(THPVariable *self, PyObject *obj, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_setter(self, "requires_grad", obj);
  }
  THPUtils_assertRet(-1, obj && PyBool_Check(obj), "requires_grad must be a bool");
  const auto& var = THPVariable_Unpack(self);
  auto requires_grad = (obj == Py_True);
  if (!var.is_leaf()) {
    THPUtils_setError(autograd::utils::requires_grad_leaf_error(obj == Py_True).c_str());
    return -1;
  }
  if (requires_grad && !isDifferentiableType(at::typeMetaToScalarType((var.dtype())))) {
    THPUtils_setError("only Tensors of floating point and complex dtype can require gradients");
    return -1;
  }
  var.set_requires_grad(requires_grad);
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

PyObject *THPVariable_get_name(THPVariable* self, void *unused)
{
  if (check_has_torch_function((PyObject *)self)) {
    HANDLE_TH_ERRORS
    return handle_torch_function_getter(self, "name");
    END_HANDLE_TH_ERRORS
  }
  const auto& tensor = THPVariable_Unpack(self);
  if (tensor.name() == "")
    Py_RETURN_NONE;
  return THPUtils_packString(tensor.name().c_str());
}

PyObject *THPVariable_get_backwards_hooks(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "_backward_hooks");
  }
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
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_setter(self, "_backward_hooks", obj);
  }
  THPUtils_assertRet(-1, obj, "Deletion of _backwards_hooks not allowed!");
  if (obj == Py_None) {
    obj = nullptr;
  }
  Py_XINCREF(obj);
  Py_XDECREF(self->backward_hooks);
  self->backward_hooks = obj;
  const auto& tensor = THPVariable_Unpack(self);
  torch::autograd::impl::clear_hooks(tensor);
  if (obj) {
    torch::autograd::impl::add_hook(tensor, std::make_shared<PyFunctionPreHook>(obj, 0));
  }
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

PyObject *THPVariable_get_base(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "_base");
  }
  const auto& tensor = THPVariable_Unpack(self);
  if (tensor.is_view()) {
    return THPVariable_Wrap(tensor._base());
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

#ifndef USE_DEPLOY
// This code is only used for asserts, so it is OK to skip it entirely from
// deploy interpreters (in which case we will just skip the safety check).  For
// a more precise check, it would be necessary to test that we are not holding
// the GIL for *all* active torch deploy interpreters.  There is not really any
// reason to do this.
struct ConcretePythonGILHooks : public c10::impl::PythonGILHooks {
  bool check_python_gil() const override {
    return Py_IsInitialized() && PyGILState_Check();
  };
};
// During process destruction, python_gil_hooks will get destructed, making
// further virtual calls on the object invalid.  By the ordering of declarations
// in this file, the registerer will get destructed first, removing the
// externally visible reference to the object.  Assuming at this point in time,
// there aren't other threads racing to read out the hooks, subsequent calls
// into GIL hooks will hit a nullptr and gracefully no-op the asserts (as
// desired, since at process shutdown time the Python interpreter is definitely
// dead).
//
// An alternative way to reduce the risk of python_gil_hooks going prematurely
// dead would be to leak it at destruction time.  I didn't do that because
// it's annoying to write the Registerer class for this case.
ConcretePythonGILHooks python_gil_hooks;
static c10::impl::PythonGILHooksRegisterer python_gil_hooks_registerer(&python_gil_hooks);
#endif

PyObject *THPVariable_get_shape(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "shape");
  }
  return THPSize_New(THPVariable_Unpack(self));
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_is_cuda(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "is_cuda");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_cuda());
  END_HANDLE_TH_ERRORS
}

PyObject* THPVariable_is_xpu(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_xpu");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_xpu());
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_is_sparse(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "is_sparse");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_sparse());
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_is_sparse_csr(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "is_sparse_csr");
  }
  auto& self_ = self->cdata;
  return torch::autograd::utils::wrap(self_.is_sparse_csr());
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_is_mkldnn(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "is_mkldnn");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_mkldnn());
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_is_mlc(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "is_mlc");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_mlc());
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_is_vulkan(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "is_vulkan");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_vulkan());
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_is_quantized(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "is_quantized");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_quantized());
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_is_meta(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "is_meta");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_meta());
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_is_complex(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "is_complex");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_complex());
  END_HANDLE_TH_ERRORS
}

static PyObject *THPVariable_dtype(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "dtype");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(torch::getTHPDtype(self_.scalar_type()));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_layout(THPVariable* self, void *unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "layout");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(torch::getTHPLayout(self_.layout()));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPVariable_device(THPVariable* self, void *unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "device");
  }
  return THPDevice_New(THPVariable_Unpack(self).device());
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_get_real(THPVariable* self, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "real");
  }
  auto& self_ = THPVariable_Unpack(self);
  auto real = at::real(self_);
  return THPVariable_Wrap(real);
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_get_imag(THPVariable* self, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "imag");
  }
  auto& self_ = THPVariable_Unpack(self);
  auto imag = at::imag(self_);
  return THPVariable_Wrap(imag);
  END_HANDLE_TH_ERRORS
}

int THPVariable_set_real(THPVariable *self, THPVariable *real, void *unused)
{
  HANDLE_TH_ERRORS
  auto& self_ = THPVariable_Unpack(self);
  auto self_real = at::real(self_);
  self_real.copy_(THPVariable_Unpack(real));
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

int THPVariable_set_imag(THPVariable* self, THPVariable *imag, void *unused)
{
  HANDLE_TH_ERRORS
  auto& self_ = THPVariable_Unpack(self);
  auto self_imag = at::imag(self_);
  self_imag.copy_(THPVariable_Unpack(imag));
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

// properties are registered here because we are currently only able to bind them
// manually. TODO: make declarable in native_functions
// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static struct PyGetSetDef THPVariable_properties[] = {
  {"T", (getter)THPVariable_get_T, nullptr, nullptr, nullptr},
  {"_cdata", (getter)THPVariable_get_cdata, nullptr, nullptr, nullptr},
  {"_version", (getter)THPVariable_get_version, nullptr, nullptr, nullptr},
  {"grad_fn", (getter)THPVariable_get_grad_fn, nullptr, nullptr, nullptr},
  {"_grad_fn", (getter)THPVariable_get_grad_fn, (setter)THPVariable_set_grad_fn, nullptr, nullptr},
  {"is_leaf", (getter)THPVariable_is_leaf, nullptr, nullptr, nullptr},
  {"data", (getter)THPVariable_get_data, (setter)THPVariable_set_data, nullptr, nullptr},
  {"_grad", (getter)THPVariable_get_grad, (setter)THPVariable_set_grad, nullptr, nullptr}, // Allows the python class to override .grad
  {"grad", (getter)THPVariable_get_grad, (setter)THPVariable_set_grad, nullptr, nullptr},
  {"_base", (getter)THPVariable_get_base, nullptr, nullptr, nullptr},
  {"volatile", (getter)THPVariable_get_volatile, (setter)THPVariable_set_volatile, nullptr, nullptr},
  {"output_nr", (getter)THPVariable_get_output_nr, nullptr, nullptr, nullptr},
  {"requires_grad", (getter)THPVariable_get_requires_grad, (setter)THPVariable_set_requires_grad, nullptr, nullptr},
  {"_backward_hooks", (getter)THPVariable_get_backwards_hooks, (setter)THPVariable_set_backwards_hooks, nullptr, nullptr},
  {"name", (getter)THPVariable_get_name, nullptr, nullptr, nullptr},
  {"shape", (getter)THPVariable_get_shape, nullptr, nullptr, nullptr},
  {"is_cuda", (getter)THPVariable_is_cuda, nullptr, nullptr, nullptr},
  {"is_xpu", (getter)THPVariable_is_xpu, nullptr, nullptr, nullptr},
  {"is_sparse", (getter)THPVariable_is_sparse, nullptr, nullptr, nullptr},
  {"is_sparse_csr", (getter)THPVariable_is_sparse_csr, nullptr, nullptr, nullptr},
  {"is_mkldnn", (getter)THPVariable_is_mkldnn, nullptr, nullptr, nullptr},
  {"is_mlc", (getter)THPVariable_is_mlc, nullptr, nullptr, nullptr},
  {"is_vulkan", (getter)THPVariable_is_vulkan, nullptr, nullptr, nullptr},
  {"is_complex", (getter)THPVariable_is_complex, nullptr, nullptr, nullptr},
  {"is_quantized", (getter)THPVariable_is_quantized, nullptr, nullptr, nullptr},
  {"is_meta", (getter)THPVariable_is_meta, nullptr, nullptr, nullptr},
  {"dtype", (getter)THPVariable_dtype, nullptr, nullptr, nullptr},
  {"layout", (getter)THPVariable_layout, nullptr, nullptr, nullptr},
  {"device", (getter)THPVariable_device, nullptr, nullptr, nullptr},
  {"ndim", (getter)THPVariable_get_ndim, nullptr, nullptr, nullptr},
  {"names", (getter)THPVariable_get_names, (setter)THPVariable_set_names, nullptr, nullptr},
  {"real", (getter)THPVariable_get_real, (setter)THPVariable_set_real, nullptr, nullptr},
  {"imag", (getter)THPVariable_get_imag, (setter)THPVariable_set_imag, nullptr, nullptr},
  {nullptr}
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static PyMappingMethods THPVariable_as_mapping = {
  THPVariable_length,
  THPVariable_getitem,
  THPVariable_setitem,
};

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static PyMethodDef extra_methods[] = {
  {"as_subclass", castPyCFunctionWithKeywords(THPVariable_as_subclass),
    METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_make_subclass", castPyCFunctionWithKeywords(THPVariable_make_subclass),
    METH_STATIC | METH_VARARGS | METH_KEYWORDS, nullptr},
  {nullptr}
};

/* From https://github.com/python/cpython/blob/v3.7.0/Modules/xxsubtype.c
   If compiled as a shared library instead, some compilers don't allow addresses
   of Python objects defined in other libraries to be used in static
   initializers here.  The DEFERRED_ADDRESS macro is used to tag the slots where
   such addresses appear; the module init function must fill in the tagged slots
   at runtime.  The argument is for documentation -- the macro ignores it.
*/
#define DEFERRED_ADDRESS(ADDR) nullptr

struct THPVariableMeta {
  PyHeapTypeObject base;
};

int THPVariableMetaType_init(PyObject *cls, PyObject *args, PyObject *kwargs);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyTypeObject THPVariableMetaType = {
  PyVarObject_HEAD_INIT(DEFERRED_ADDRESS(&PyType_Type), 0)
  "torch._C._TensorMeta",                      /* tp_name */
  sizeof(THPVariableMeta),                     /* tp_basicsize */
  0,                                           /* tp_itemsize */
  nullptr,                                     /* tp_dealloc */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                           /* tp_vectorcall_offset */
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
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,    /* tp_flags */
  nullptr,                                     /* tp_doc */
  nullptr,                                     /* tp_traverse */
  nullptr,                                     /* tp_clear */
  nullptr,                                     /* tp_richcompare */
  0,                                           /* tp_weaklistoffset */
  nullptr,                                     /* tp_iter */
  nullptr,                                     /* tp_iternext */
  nullptr,                                     /* tp_methods */
  nullptr,                                     /* tp_members */
  nullptr,                                     /* tp_getset */
  DEFERRED_ADDRESS(&PyType_Type),              /* tp_base */
  nullptr,                                     /* tp_dict */
  nullptr,                                     /* tp_descr_get */
  nullptr,                                     /* tp_descr_set */
  0,                                           /* tp_dictoffset */
  THPVariableMetaType_init,                    /* tp_init */
  nullptr,                                     /* tp_alloc */
  nullptr,                                     /* tp_new */
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyTypeObject THPVariableType = {
  PyVarObject_HEAD_INIT(&THPVariableMetaType, 0)
  "torch._C._TensorBase",                      /* tp_name */
  sizeof(THPVariable),                         /* tp_basicsize */
  0,                                           /* tp_itemsize */
  (destructor)THPVariable_dealloc,             /* tp_dealloc */
  // NOLINTNEXTLINE(modernize-use-nullptr)
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
  // Although new is provided here, it is illegal to call this with cls ==
  // THPVariableMeta.  Instead, subclass it first and then construct it
  THPVariable_pynew,                           /* tp_new */
};

PyObject *THPVariable_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  TORCH_CHECK(type != &THPVariableType, "Cannot directly construct _TensorBase; subclass it and then construct that");
  jit::tracer::warn("torch.Tensor", jit::tracer::WARN_CONSTRUCTOR);
  auto tensor = torch::utils::legacy_tensor_ctor(torch::tensors::get_default_dispatch_key(), torch::tensors::get_default_scalar_type(), args, kwargs);
  return THPVariable_NewWithVar(type, std::move(tensor));
  END_HANDLE_TH_ERRORS
}


int THPVariableMetaType_init(PyObject *cls, PyObject *args, PyObject *kwargs) {
  if (PyType_Type.tp_init(cls, args, kwargs) < 0) {
    return -1;
  }
  // TODO: put in custom tp_dealloc
  return 0;
}

namespace torch { namespace autograd {

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
extern PyMethodDef variable_methods[];
extern void initTorchFunctions(PyObject *module);

void initTensorImplConversion(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  m.def("_wrap_tensor_impl", [](void* ptr) {
    auto p = c10::intrusive_ptr<c10::TensorImpl, at::UndefinedTensorImpl>::
        unsafe_reclaim_from_nonowning(static_cast<c10::TensorImpl*>(ptr));
    TORCH_CHECK(p.defined(), "Can't wrap undefined tensor");
    auto tensor = at::Tensor::wrap_tensor_impl(std::move(p));
    // NOLINTNEXTLINE(performance-move-const-arg)
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
  THPVariableMetaType.tp_base = &PyType_Type;
  if (PyType_Ready(&THPVariableMetaType) < 0)
    return false;
  Py_INCREF(&THPVariableMetaType);
  PyModule_AddObject(module, "_TensorMeta",   (PyObject *)&THPVariableMetaType);

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
