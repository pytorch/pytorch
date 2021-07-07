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
#include <c10/util/irange.h>

#include <torch/library.h>
#include <torch/csrc/jit/python/pybind_utils.h>


#include <ATen/ATen.h>
#include <pybind11/pybind11.h>

#include <structmember.h>
#include <cstdint>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

using namespace at;
using namespace torch;
using namespace torch::autograd;

namespace {

std::string concrete_name_fn(const c10::impl::PyInterpreter* self) {
  std::stringstream ss;
  ss << self;
  return ss.str();
}

void concrete_decref_fn(const c10::impl::PyInterpreter* self, PyObject* pyobj) {
  // Leak the pyobj if not initialized.  This can happen if we are running
  // exit handlers that are destructing tensors with residual (owned)
  // PyObjects stored in them.
  if (!Py_IsInitialized())
    return;

  pybind11::gil_scoped_acquire gil;
  if (Py_REFCNT(pyobj) > 1) {
    // It's still alive!  This can happen if a weak ref resurrected
    // the PyObject without flipping ownership.  At this point it is
    // too late to rescue the object, so just stub out the PyObject
    // so that it fails on subsequent uses.  Don't raise an error here;
    // you're probably in a destructor.
    TORCH_WARN(
      "Deallocating Tensor that still has live PyObject references.  "
      "This probably happened because you took out a weak reference to "
      "Tensor and didn't call _fix_weakref() after dereferencing it.  "
      "Subsequent accesses to this tensor via the PyObject will now fail."
    );
    ((THPVariable*)pyobj)->cdata = MaybeOwned<Variable>();
  }
  Py_DECREF(pyobj);
};

c10::intrusive_ptr<TensorImpl> concrete_detach_fn(const c10::impl::PyInterpreter*, const c10::TensorImpl* self);
void concrete_dispatch_fn(const c10::impl::PyInterpreter*, const c10::OperatorHandle& op, torch::jit::Stack* stack);

class PyInterpreterHolder {
 public:
  PyInterpreterHolder()
      : impl_(new c10::impl::PyInterpreter(
            &concrete_name_fn,
            &concrete_decref_fn,
            &concrete_detach_fn,
            &concrete_dispatch_fn)) {}
  // NB: intentionally leaks the memory
  ~PyInterpreterHolder() {
    impl_->disarm();
  }
  c10::impl::PyInterpreter* get() const noexcept {
    return impl_;
  }

 private:
  c10::impl::PyInterpreter* impl_;
};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyInterpreterHolder self_interpreter;

} // anonymous namespace

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

static bool check_has_torch_dispatch(PyObject *obj) {
  PyTypeObject *tp = Py_TYPE(obj);
  return (
    !THPVariable_CheckTypeExact(tp) &&
    // TODO: test if Python key is disabled
    PyObject_FastGetAttrString(obj, "__torch_dispatch__").ptr() != nullptr
  );
}

// Creates a new Python object for a Variable.  The status parameter
// specifies what the interpreter tag status on the object is; for
// example, if you ran check_pyobj, the return optional of this object
// tells you if the tensor was already tagged or not so you can pass
// TAGGED_BY_US or MAYBE_UNINITIALIZED; in other cases, you know where
// var came from and can directly assert that it's DEFINITELY_UNINITIALIZED.
// It's ALWAYS safe (albeit slower) to call this with MAYBE_UNINITIALIZED.
static PyObject* THPVariable_NewWithVar(
    PyTypeObject* type,
    Variable _var,
    c10::impl::PyInterpreterStatus status) {
  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    auto v = (THPVariable*) obj;
    // TODO: named constructor to avoid default initialization
    new (&v->cdata) MaybeOwned<Variable>();
    v->cdata = MaybeOwned<Variable>::owned(std::move(_var));
    const auto& var = THPVariable_Unpack(v);
    var.unsafeGetTensorImpl()->init_pyobj(self_interpreter.get(), obj, status);
    if (check_has_torch_dispatch(obj)) {
      var.unsafeGetTensorImpl()->set_python_dispatch(true);
    }
  }
  return obj;
}

// TODO: Make this take Variable by const reference
PyObject * THPVariable_Wrap(Variable var)
{
  if (!var.defined()) {
    Py_RETURN_NONE;
  }

  c10::optional<PyObject*> mb_obj =
      var.unsafeGetTensorImpl()->check_pyobj(self_interpreter.get());
  c10::impl::PyInterpreterStatus status;
  if (mb_obj.has_value()) {
    auto obj = *mb_obj;
    if (obj) {
      if (var.unsafeGetTensorImpl()->owns_pyobj()) {
        // C++ owns the Python object; this implies there weren't any other
        // owning references to the Python object.  Since we're making the
        // object "live" again on Python side, let's flip back the ownership
        // (Python owns C++) as it would now be unsound to deallocate the C++
        // object if all C++ references go to zero
        var.unsafeGetTensorImpl()->set_owns_pyobj(false);
        reinterpret_cast<THPVariable*>(obj)->cdata =
            MaybeOwned<Variable>::owned(std::move(var));
        // NB: incref is not necessary, because we are "stealing" the previous
        // ownership from the Variable to return it here for the wrap
        return obj;
      }
      Py_INCREF(obj);
      return obj;
    }
    // TODO: a better invariant is that if we tagged, we MUST have a valid
    // PyObject.  That's PyObject preservation
    // (https://github.com/pytorch/pytorch/pull/56017).  Prior to this PR
    // being a thing, the PyObject field will get cleared when all references
    // to the Python object are removed.
    status = c10::impl::PyInterpreterStatus::TAGGED_BY_US;
  } else {
    // Assumption: if a Tensor has been shared across threads, this induces
    // a refcount bump.  Therefore, if the use count 1, we are the sole thread
    // with access to this tensor and no race is possible.
    if (var.use_count() <= 1) {
      status = c10::impl::PyInterpreterStatus::DEFINITELY_UNINITIALIZED;
    } else {
      status = c10::impl::PyInterpreterStatus::MAYBE_UNINITIALIZED;
    }
  }
  return THPVariable_NewWithVar(
      (PyTypeObject*)THPVariableClass, std::move(var), status);
}

static int THPVariable_clear(THPVariable* self) {
  Py_CLEAR(self->backward_hooks);
  const auto& tensor = THPVariable_Unpack(self);
  if (tensor.defined()) {
    // Two situations to consider:
    //    PyObject -owns-> Tensor
    //        unsafeIsBorrowed() is FALSE.  We're obligated to look through
    //        Tensor to break references.  Clearing cdata must induce the
    //        destruction of the C++ Tensor.  If there were other references
    //        to C++ tensor, the Python object would have been resurrected
    //        by flipping the ownership.
    //    Tensor -owns-> PyObject
    //        unsafeIsBorrowed() is TRUE.  We're deallocating the PyObject
    //        because Tensor asked us to (it's already destructing).

    if (!self->cdata.unsafeIsBorrowed()) {
      // TODO: empirically, on OS X this assert appears to be untrue
      // In test_py_tensors_multi_async_call - ProcessGroupRpcTestWithSpawn
      // distributed/rpc/test_process_group_agent.py
      //
      //  libc++abi.dylib: terminating with uncaught exception of type
      //  c10::Error: !tensor.unsafeGetTensorImpl()->owns_pyobj()INTERNAL ASSERT
      //  FAILED at "../torch/csrc/autograd/python_variable.cpp":171, please
      //  report a bug to PyTorch. Exception raised from THPVariable_clear at
      //  ../torch/csrc/autograd/python_variable.cpp:171 (most recent call
      //  first): frame #0: c10::Error::Error(c10::SourceLocation,
      //  std::__1::basic_string<char, std::__1::char_traits<char>,
      //  std::__1::allocator<char> >) + 98 (0x1158a0442 in libc10.dylib) frame
      //  #1: c10::detail::torchCheckFail(char const*, char const*, unsigned
      //  int, char const*) + 205 (0x11589ed3d in libc10.dylib) frame #2:
      //  c10::detail::torchInternalAssertFail(char const*, char const*,
      //  unsigned int, char const*, c10::detail::CompileTimeEmptyString) + 9
      //  (0x1141e3f89 in libtorch_python.dylib) frame #3:
      //  THPVariable_clear(THPVariable*) + 412 (0x1148a547c in
      //  libtorch_python.dylib) frame #4:
      //  THPVariable_subclass_dealloc(_object*) + 453 (0x1148a5035 in
      //  libtorch_python.dylib) frame #5: (anonymous
      //  namespace)::concrete_decref_fn(c10::impl::PyInterpreter const*,
      //  _object*) + 53 (0x1148a5ea5 in libtorch_python.dylib) frame #6:
      //  c10::TensorImpl::release_resources() + 182 (0x11588c4a6 in
      //  libc10.dylib) frame #7:
      //  c10::MaybeOwned<at::Tensor>::operator=(c10::MaybeOwned<at::Tensor>&&)
      //  + 91 (0x11488c11b in libtorch_python.dylib) frame #8:
      //  THPVariable_subclass_dealloc(_object*) + 607 (0x1148a50cf in
      //  libtorch_python.dylib) <omitting python frames> frame #47: start + 1
      //  (0x7fff6ffc7cc9 in libdyld.dylib) frame #48: 0x0 + 4 (0x4 in ???)
      // TORCH_INTERNAL_ASSERT(!tensor.unsafeGetTensorImpl()->owns_pyobj());
      if (auto grad_acc =
              torch::autograd::impl::try_get_grad_accumulator(tensor)) {
        grad_acc->pre_hooks().clear();
      }
    }
  }
  self->cdata = MaybeOwned<Variable>();
  return 0;
}

// returns true if successfully rezzed; if so, cancel the
// rest of deallocation
static bool THPVariable_tryResurrect(THPVariable* self) {
  const auto& tensor = THPVariable_Unpack(self);

  // Is this true or not???  Triggered by TestAutograd.test_variable_traverse
  // TORCH_INTERNAL_ASSERT(tensor.defined());

  // Check if there are other C++ owners
  if (tensor.use_count() <= 1) {
    return false;
  }

  // There are other C++ owners of the tensor.  Flip ownership
  // so that C++ owns this Python object, and cancel deallocation.
  TORCH_INTERNAL_ASSERT(!tensor.unsafeGetTensorImpl()->owns_pyobj());

  tensor.unsafeGetTensorImpl()->set_owns_pyobj(true);

// Resurrect the Python object.  This is something CPython does
// internally occasionally, see
// https://github.com/python/cpython/blob/b98eba5bc2ffbe7a0ed49d540ebc4f756ae61985/Objects/object.c#L248-L259
// so we just copy the pattern here.  Note that we don't have to worry
// about saving and restoring the refcount (as the quoted code does)
// because we actually DO need to reset the refcount to one here, we
// can't assume that some other code has taken care of it.
// NB: this will overreport _Py_RefTotal but based on inspection of object.c
// there is no way to avoid this
#ifdef Py_TRACE_REFS
  _Py_AddToAllObjects(reinterpret_cast<PyObject *>(self), 1);
#endif
  Py_INCREF(self);

  // Flip THPVariable to be non-owning
  // (near use-after-free miss here: fresh MaybeOwned is created breaking
  // reference on Tensor in struct BEFORE we overwrite the old one)
  self->cdata = MaybeOwned<Variable>::borrowed(tensor);

  // NB: At this point, tensor *could* be dead (e.g., some other C++ thread
  // decrefed it.)  At this point, it is probably waiting on the GIL to
  // deallocate the Python object and will kill self, BUT NOT YET.

  return true;
}

PyObject *THPVariable_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs);

static PyObject* THPVariable_fix_weakref(PyObject* self, PyObject* noargs) {
  const auto& var = THPVariable_Unpack(self);
  THPVariable_Wrap(var);
  Py_RETURN_NONE;
}

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
  return THPVariable_NewWithVar(
      (PyTypeObject*)cls,
      self.alias(),
      c10::impl::PyInterpreterStatus::DEFINITELY_UNINITIALIZED);
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
  auto data =
      r.tensor(1).detach(); // creates a fresh Tensor (DEFINITELY_UNINITIALIZED)
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
  data.set_requires_grad(r.toBool(2));
  return THPVariable_NewWithVar(
      (PyTypeObject*)cls,
      std::move(data),
      c10::impl::PyInterpreterStatus::DEFINITELY_UNINITIALIZED);
  END_HANDLE_TH_ERRORS
}

typedef PyObject *(*getter)(PyObject *, void *);
typedef int (*setter)(PyObject *, PyObject *, void *);

PyObject *THPVariable_get_python_dispatch(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  const auto& var = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(var.unsafeGetTensorImpl()->is_python_dispatch());
  END_HANDLE_TH_ERRORS
}

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
  if(THPVariable_Unpack(self).requires_grad()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject *THPVariable_retains_grad(THPVariable *self, void *unused)
{
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject *)self)) {
    return handle_torch_function_getter(self, "retains_grad");
  }
  if(THPVariable_Unpack(self).retains_grad()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
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
  for (const auto i : c10::irange(size)) {
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
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
ConcretePythonGILHooks python_gil_hooks;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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
  auto& self_ = THPVariable_Unpack(self);
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
  {"_python_dispatch", (getter)THPVariable_get_python_dispatch, nullptr, nullptr, nullptr},
  {"T", (getter)THPVariable_get_T, nullptr, nullptr, nullptr},
  {"_cdata", (getter)THPVariable_get_cdata, nullptr, nullptr, nullptr},
  {"_version", (getter)THPVariable_get_version, nullptr, nullptr, nullptr},
  {"grad_fn", (getter)THPVariable_get_grad_fn, nullptr, nullptr, nullptr},
  {"_grad_fn", (getter)THPVariable_get_grad_fn, (setter)THPVariable_set_grad_fn, nullptr, nullptr},
  {"is_leaf", (getter)THPVariable_is_leaf, nullptr, nullptr, nullptr},
  {"retains_grad", (getter)THPVariable_retains_grad, nullptr, nullptr, nullptr},
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
  {"_fix_weakref", THPVariable_fix_weakref,
    METH_NOARGS, nullptr},
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
    PyVarObject_HEAD_INIT(
        &THPVariableMetaType,
        0) "torch._C._TensorBase", /* tp_name */
    sizeof(THPVariable), /* tp_basicsize */
    0, /* tp_itemsize */
    // This is unspecified, because it is illegal to create a THPVariableType
    // directly.  Subclasses will have their tp_dealloc set appropriately
    // by the metaclass
    nullptr, /* tp_dealloc */
    // NOLINTNEXTLINE(modernize-use-nullptr)
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    &THPVariable_as_mapping, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
        Py_TPFLAGS_HAVE_GC, /* tp_flags */
    nullptr, /* tp_doc */
    // Also set by metaclass
    nullptr, /* tp_traverse */
    (inquiry)THPVariable_clear, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    nullptr, /* tp_methods */
    nullptr, /* tp_members */
    THPVariable_properties, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    // Although new is provided here, it is illegal to call this with cls ==
    // THPVariableMeta.  Instead, subclass it first and then construct it
    THPVariable_pynew, /* tp_new */
};

PyObject *THPVariable_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  TORCH_CHECK(type != &THPVariableType, "Cannot directly construct _TensorBase; subclass it and then construct that");
  jit::tracer::warn("torch.Tensor", jit::tracer::WARN_CONSTRUCTOR);
  auto tensor = torch::utils::legacy_tensor_ctor(torch::tensors::get_default_dispatch_key(), torch::tensors::get_default_scalar_type(), args, kwargs);
  // WARNING: tensor is NOT guaranteed to be a fresh tensor; e.g., if it was
  // given a raw pointer that will refcount bump
  return THPVariable_NewWithVar(
      type,
      std::move(tensor),
      c10::impl::PyInterpreterStatus::MAYBE_UNINITIALIZED);
  END_HANDLE_TH_ERRORS
}

static void clear_slots(PyTypeObject* type, PyObject* self) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  Py_ssize_t i, n;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  PyMemberDef* mp;

  n = Py_SIZE(type);
  mp = PyHeapType_GET_MEMBERS((PyHeapTypeObject*)type);
  for (i = 0; i < n; i++, mp++) {
    if (mp->type == T_OBJECT_EX && !(mp->flags & READONLY)) {
      char* addr = (char*)self + mp->offset;
      PyObject* obj = *(PyObject**)addr;
      // NOLINTNEXTLINE(modernize-use-nullptr)
      if (obj != NULL) {
        // NOLINTNEXTLINE(modernize-use-nullptr)
        *(PyObject**)addr = NULL;
        Py_DECREF(obj);
      }
    }
  }
}

// NB: this is not the tp_dealloc on THPVariable; instead, its the dealloc
// on subclasses.  It's never valid to construct a THPVariable so it's not
// necessary to implement the dealloc for that case
void THPVariable_subclass_dealloc(PyObject* self) {
  if (THPVariable_tryResurrect((THPVariable*)self))
    return;

  // This is like a crappy version of subtype_dealloc.
  // Unfortunately, we cannot directly delegate to
  // subtype_dealloc as it will start walking the parent
  // chain *starting with* the type of self, which will cause
  // us to go back to our custom dealloc.
  //
  // We have to replicate the subtype_dealloc logic to ensure
  // that finalizers are handled correctly
  PyTypeObject* type = Py_TYPE(self);
  TORCH_INTERNAL_ASSERT(type->tp_flags & Py_TPFLAGS_HEAPTYPE);
  TORCH_INTERNAL_ASSERT(PyType_IS_GC(type), "GC types not implemented");

  PyObject_GC_UnTrack(self);
  // TODO: consider using trash can

  bool has_finalizer = type->tp_finalize || type->tp_del;

  if (type->tp_finalize) {
    PyObject_GC_Track(self);
    if (PyObject_CallFinalizerFromDealloc(self) < 0) {
      /* Resurrected */
      return;
    }
    PyObject_GC_UnTrack(self);
  }

  // base test is unnecessary as THPVariable does not set this
  if (type->tp_weaklistoffset) {
    PyObject_ClearWeakRefs(self);
  }

  if (type->tp_del) {
    PyObject_GC_Track(self);
    type->tp_del(self);
    if (self->ob_refcnt > 0) {
      /* Resurrected */
      return;
    }
    PyObject_GC_UnTrack(self);
  }

  if (has_finalizer) {
    /* New weakrefs could be created during the finalizer call.
       If this occurs, clear them out without calling their
       finalizers since they might rely on part of the object
       being finalized that has already been destroyed. */
    if (type->tp_weaklistoffset) {
      /* Modeled after GET_WEAKREFS_LISTPTR() */
      PyWeakReference** list =
          (PyWeakReference**)PyObject_GET_WEAKREFS_LISTPTR(self);
      while (*list)
        _PyWeakref_ClearRef(*list);
    }
  }

  // Clear all slots until we get to base class THPVariableType
  {
    PyTypeObject* base = type;
    while (base != &THPVariableType) {
      if (Py_SIZE(base)) {
        clear_slots(base, self);
      }
      base = base->tp_base;
      TORCH_INTERNAL_ASSERT(base);
    }
  }

  // All Python defined classes have __dict__
  if (C10_LIKELY(type->tp_dictoffset)) {
    PyObject** dictptr = _PyObject_GetDictPtr(self);
    // NOLINTNEXTLINE(modernize-use-nullptr)
    if (dictptr != NULL) {
      PyObject* dict = *dictptr;
      // NOLINTNEXTLINE(modernize-use-nullptr)
      if (dict != NULL) {
        Py_DECREF(dict);
        // NOLINTNEXTLINE(modernize-use-nullptr)
        *dictptr = NULL;
      }
    }
  }

  // subtype_dealloc allows for this but we don't
  TORCH_INTERNAL_ASSERT(Py_TYPE(self) == type);

  // Finally clear out the base THPVariable
  THPVariable_clear((THPVariable*)self);
  ((THPVariable*)self)->cdata.~MaybeOwned<Variable>();
  Py_TYPE(self)->tp_free(self);

  // Python defined subclasses should always be on the heap
  TORCH_INTERNAL_ASSERT(type->tp_flags & Py_TPFLAGS_HEAPTYPE);
  Py_DECREF(type);
}

/// NOTE [ PyObject Traversal ]
///
/// PyObjects that are wrapping c++ objects can lead to non-trivial traverse logic
/// and it can be tricky to know what to traverse and when. This note tries to
/// clarify what is the danger here and a simple algorithm to choose how to write
/// the tp_traverse and tp_clear functions.
/// If you're not already familiar with how the CPython GC works, you should read this
/// in-depth description: https://devguide.python.org/garbage_collector/
///
/// The complexity for us comes from the fact that some c++ shared_ptr objects
/// own references to python objects and are also owned both by other python objects
/// and c++ objects. This means that to allow the GC to collect all cycles, we need to
/// properly implement the traverse/clear methods that take into account these C++
/// ownership links.
///
/// The main danger here comes from the fact that, while all python-related code is
/// thread safe wrt the GC execution (thanks to the GIL), other threads might be using
/// our C++ objects arbitrarily which can lead to shared_ptr ref count going up or down
/// in between the different traverse/clear invocations.
/// The one constraint we add here that is not explicitly mentioned in the GC description
/// above is that for a given GC run (meaning while the GIL is held), the traverse/clear
/// pair should never report different ownership relations: if traverse visited a given
/// PyObject, then the clear within that same GC run must still be the sole owner and
/// clear that PyObject.
///
/// A more mechanical algorithm to know what to traverse/clear is as follows:
///   - Any field on this PyObject that contains a strong reference to another PyObject
///     must be visited and cleared. An example of that is the "backward_hooks" field of
///     the THPVariable.
///   - Any field that contains a C++ object that is uniquely owned by this PyObject (either
///     a unique_ptr or a shared_ptr with use_count==1) should have all the PyObject it owns
///     visited and cleared. An example would be here the tensor hooks.
///   - If that uniquely owned C++ object also uniquely owns other C++ objects, these should be
///     visited and cleared as well if they contain any PyObject.
///
/// Caveat: to avoid slow runtime, we limit the depth of this exploration of C++ objects in
/// practice and we do not, for example, go through the whole autograd graph, even if it is
/// uniquely owned. This is a known place where users can create noncollectable cycles as described
/// in: https://github.com/pytorch/pytorch/issues/7343
///

static int traverse_slots(
    PyTypeObject* type,
    PyObject* self,
    visitproc visit,
    void* arg) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  Py_ssize_t i, n;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  PyMemberDef* mp;

  n = Py_SIZE(type);
  mp = PyHeapType_GET_MEMBERS((PyHeapTypeObject*)type);
  for (i = 0; i < n; i++, mp++) {
    if (mp->type == T_OBJECT_EX) {
      char* addr = (char*)self + mp->offset;
      PyObject* obj = *(PyObject**)addr;
      // NOLINTNEXTLINE(modernize-use-nullptr)
      if (obj != NULL) {
        int err = visit(obj, arg);
        if (err)
          return err;
      }
    }
  }
  return 0;
}

static int THPVariable_subclass_traverse(
    PyObject* self,
    visitproc visit,
    void* arg) {
  // If the tensor is eligible to be resurrected, don't traverse it; instead
  // treat all of its references as a root (as they WOULD be a root since we
  // can treat the inbound C++ references as root owners).
  //
  // This works because unlike conventional GCs, Python's GC operates in two
  // phases: first it uses traverse to discover roots, and then it uses traverse
  // to do reachability.  Bypassing traverse during root discovery forces Python
  // to treat self as a root for everything it refers to.  For a full
  // explanation of the algorithm see
  // https://devguide.python.org/garbage_collector/
  //
  // NB: if we don't hold an owning reference to the underlying Tensor, it is
  // possible that the underlying Tensor has already gone dead.  In that case,
  // it's not safe to access it.  But it's also safe to traverse, because if
  // the underlying Tensor *is* live, then root discovery will determine that
  // self is live, and nothing will get GC'ed anyway (resurrection cannot happen
  // if the C++ objects owns the PyObject)
  THPVariable* var = reinterpret_cast<THPVariable*>(self);
  if (!var->cdata.unsafeIsBorrowed()) {
    const auto& tensor = THPVariable_Unpack(self);
    if (tensor.defined() && tensor.use_count() > 1)
      return 0;
  }

  // Crappy version of subtype_traverse; same deal as
  // THPVariable_subclass_dealloc

  PyTypeObject* type = Py_TYPE(self);
  // Traverse slots until we get to base class THPVariableType
  {
    PyTypeObject* base = type;
    while (base != &THPVariableType) {
      if (Py_SIZE(base)) {
        int err = traverse_slots(base, self, visit, arg);
        if (err)
          return err;
      }
      base = base->tp_base;
      TORCH_INTERNAL_ASSERT(base);
    }
  }

  // All Python defined classes have __dict__
  if (C10_LIKELY(type->tp_dictoffset)) {
    PyObject** dictptr = _PyObject_GetDictPtr(self);
    if (dictptr && *dictptr)
      Py_VISIT(*dictptr);
  }

  TORCH_INTERNAL_ASSERT(type->tp_flags & Py_TPFLAGS_HEAPTYPE);
  Py_VISIT(type);

  // Finally traverse THPVariable special stuff
  Py_VISIT(var->backward_hooks);
  if (!var->cdata.unsafeIsBorrowed()) {
    const auto& tensor = THPVariable_Unpack(var);
    if (tensor.defined()) {
      // WARNING: The grad_fn traversal logic is very subtle, if you change this,
      // be very careful not to re-introduce this bug:
      // https://gist.github.com/zou3519/7ac92b84dd7d206dcc6eae55fee8372c

      // We ensure that we follow NOTE [ PyObject Traversal ] he by checking that this
      // python object is the sole owner of the underlying Tensor and that this Tensor
      // is the sole owner of its grad_fn.
      // In this case, the only way to get a new reference to the grad_fn is by using
      // this python object, which requires the GIL to be accessed.
      // Note that this is only valid as long as user don't share non-owning references
      // across different threads (which is crazy and should never be done).

      if (tensor.use_count() == 1) {
        auto autograd_meta = torch::autograd::impl::get_autograd_meta(tensor);
        if (autograd_meta) {
          // Do NOT call grad_fn() here as that might trigger a recompute
          const auto& grad_fn = autograd_meta->grad_fn_;
          if (grad_fn && grad_fn.use_count() == 1) {
            // All Node can have a pyobj (stored in "pyobj_")
            Py_VISIT(grad_fn->pyobj());
            // PyNode are special as they also have an "obj" field
            if (auto py_node_fn = dynamic_cast<PyNode*>(grad_fn.get())) {
              Py_VISIT(py_node_fn->obj);
            }
          }
        }
      }

      for (const auto& hook : torch::autograd::impl::hooks(tensor)) {
        if (auto pyhook = dynamic_cast<PyFunctionPreHook*>(hook.get())) {
          Py_VISIT(pyhook->dict);
        }
      }
    }
  }

  return 0;
}

int THPVariableMetaType_init(PyObject *cls, PyObject *args, PyObject *kwargs) {
  if (PyType_Type.tp_init(cls, args, kwargs) < 0) {
    return -1;
  }
  ((PyTypeObject*)cls)->tp_dealloc = (destructor)THPVariable_subclass_dealloc;
  ((PyTypeObject*)cls)->tp_traverse =
      (traverseproc)THPVariable_subclass_traverse;
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

namespace {

bool isPythonTensor(const Tensor& tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(c10::DispatchKey::Python);
}

void concrete_dispatch_fn(const c10::impl::PyInterpreter*, const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto num_returns = schema.returns().size();

  const auto num_arguments = schema.arguments().size();
  auto arguments = torch::jit::pop(*stack, num_arguments);

  // Parse the name into namespace and name (no overload_name)
  // TODO: put this into the library
  const auto& qualified_name = op.operator_name().name;
  auto pos = qualified_name.find("::");
  TORCH_INTERNAL_ASSERT(pos != std::string::npos, qualified_name);
  // Make me some null terminated strings
  std::string ns_str = qualified_name.substr(0, pos);
  const char* ns = ns_str.c_str();
  const char* func_name = qualified_name.c_str() + pos + strlen("::");

  // The plan: convert all the arguments back into PyObjects,
  // extracting out the tensor handles, then call
  // handle_torch_function_no_python_arg_parser
  // NB: at the point arguments are pushed to the stack, ALL defaults
  // are already present

  py::gil_scoped_acquire g;

  std::vector<py::handle> overloaded_args;
  auto args = py::reinterpret_steal<py::object>(PyTuple_New(num_arguments));
  // TODO: actually populate kwargs sometimes?  At the moment, every argument
  // just gets passed positionally
  py::dict kwargs;
  // For now, overloads get coalesced.  Might be easier for users if they get
  // overload resolution but is more complicated (need to expose separate
  // functions per overload)
  py::handle torch_api_function = py::module::import("torch").attr("ops").attr(ns).attr(func_name);
  std::string module_name_str = "torch.ops." + ns_str;

  for (int64_t idx = 0; idx < arguments.size(); idx++) {
    auto& ivalue = arguments[idx];
    // Search for Tensors (as they may have the torch functions we need)
    if (ivalue.isTensor()) {
      const auto& tensor = ivalue.toTensor();
      if (isPythonTensor(tensor)) {
        overloaded_args.emplace_back(py::cast(tensor));
      }
    } else if (ivalue.isList()) {
      const auto& list = ivalue.toListRef();
      for (int64_t jdx = 0; jdx < list.size(); jdx++) {
        const auto& nv = list[jdx];
        if (nv.isTensor()) {
          const auto& tensor = nv.toTensor();
          if (isPythonTensor(tensor)) {
            overloaded_args.emplace_back(py::cast(tensor));
          }
        }
      }
    }
    PyTuple_SET_ITEM(args.ptr(), idx, torch::jit::toPyObject(std::move(ivalue)).release().ptr());
  }

  auto out = py::reinterpret_steal<py::object>(handle_torch_function_no_python_arg_parser(
    overloaded_args,
    args.ptr(),
    kwargs.ptr(),
    func_name,
    torch_api_function.ptr(),
    module_name_str.c_str(),
    "__torch_dispatch__"
  ));

  if (op.schema().returns().size() == 1) {
    torch::jit::push(stack, torch::jit::toIValue(out.ptr(), op.schema().returns()[0].type()));
  } else {
    auto outs = py::cast<py::sequence>(out);
    for (unsigned idx = 0; idx < outs.size(); idx++) {
      torch::jit::push(stack, torch::jit::toIValue(outs[idx].ptr(), op.schema().returns()[idx].type()));
    }
  }
}

c10::intrusive_ptr<TensorImpl> concrete_detach_fn(const c10::impl::PyInterpreter*, const c10::TensorImpl* self) {
  pybind11::gil_scoped_acquire gil;

  // Setup the arguments expected for the detach call
  std::vector<py::handle> overloaded_args;
  // TODO: there should be a shorter way to spell this
  // TODO: fix the constness of target
  Tensor self_t = Tensor(c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>::unsafe_reclaim_from_nonowning(const_cast<c10::TensorImpl*>(self)));
  auto self_p = py::reinterpret_steal<py::object>(THPVariable_Wrap(self_t));
  overloaded_args.emplace_back(self_p);
  auto args = py::reinterpret_steal<py::object>(PyTuple_New(1));
  PyTuple_SET_ITEM(args.ptr(), 0, self_p.release().ptr());

  py::dict kwargs;

  auto out = py::reinterpret_steal<py::object>(handle_torch_function_no_python_arg_parser(
    overloaded_args,
    args.ptr(),
    kwargs.ptr(),
    "detach",
    py::module::import("torch").attr("ops").attr("aten").attr("detach").ptr(),
    "torch.ops.aten",
    "__torch_dispatch__"
  ));

  TORCH_CHECK(THPVariable_Check(out.ptr()), "detach returned invalid type ", py::detail::get_fully_qualified_tp_name(Py_TYPE(out.ptr())), ", expected Tensor");
  const Tensor& res_t = THPVariable_Unpack(out.ptr());
  return res_t.getIntrusivePtr();
}

} // anonymous namespace
