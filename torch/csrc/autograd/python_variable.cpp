#include <ATen/NamedTensorUtils.h>
#include <ATen/core/PythonFallbackKernel.h>
#include <ATen/core/PythonOpRegistrationTrampoline.h>
#include <c10/core/DeviceType.h>
#include <c10/core/SafePyObject.h>
#include <c10/core/impl/GPUTrace.h>
#include <c10/core/impl/HermeticPyObjectTLS.h>
#include <c10/core/impl/PythonDispatcherTLS.h>
#include <c10/util/DeadlockDetection.h>
#include <c10/util/irange.h>
#include <pybind11/pytypes.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Size.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/Types.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/csrc/autograd/python_hook.h>
#include <torch/csrc/autograd/python_variable_indexing.h>
#include <torch/csrc/autograd/utils/error_messages.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/tensor/python_tensor.h>
#include <torch/csrc/utils/cuda_lazy_init.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_dispatch.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/tensor_memoryformats.h>
#include <torch/csrc/utils/tensor_new.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/torch_dispatch_mode.h>
#include <torch/library.h>

#include <ATen/ATen.h>

#include <c10/core/SymIntArrayRef.h>
#include <structmember.h>
#include <cstdint>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

using namespace at;
using namespace torch;
using namespace torch::autograd;

std::pair<py::object, py::dict> parseIValuesToPyArgsKwargs(
    const c10::OperatorHandle& op,
    const std::vector<c10::IValue>& arguments) {
  TORCH_CHECK(
      PyGILState_Check(),
      "GIL must be held before you call parseIValuesToPyArgsKwargs");
  const auto& schema = op.schema();
  py::dict kwargs;
  // About all the pointers:
  //
  // f(int x, int y = 0, *, int z = 0)
  //                                  ^- arguments.size()
  //                        ^- kwarg_only_start
  //          ^- positional_default_start
  //   ^- 0

  // Find the split point between kwarg-only and regular.  Since most functions
  // don't have kwarg-only arguments, it is more efficient to scan from the
  // right (but ideally, this would just be precomputed in FunctionSchema
  // itself).  (NB: minus one in the loop is because we're testing if the
  // *next* argument is kwarg-only before we advance the starting index)
  int64_t kwarg_only_start = arguments.size();
  for (; kwarg_only_start > 0; kwarg_only_start--) {
    const auto& arg = schema.arguments()[kwarg_only_start - 1];
    if (!arg.kwarg_only()) {
      break;
    }
  }

  // Find the first positional argument that isn't defaulted
  auto is_default = [&](int64_t idx) -> bool {
    const auto& arg = schema.arguments()[idx];
    if (!arg.default_value().has_value()) {
      return false;
    }
    const auto& default_ivalue = *arg.default_value();
    const auto& ivalue = arguments[idx];
    if (default_ivalue != ivalue) {
      return false;
    }
    return true;
  };

  int64_t positional_default_start = kwarg_only_start;
  for (; positional_default_start > 0; positional_default_start--) {
    if (!is_default(positional_default_start - 1)) {
      break;
    }
  }

  auto args =
      py::reinterpret_steal<py::object>(PyTuple_New(positional_default_start));

  auto schemaAwareToPyObject = [&](int64_t idx) -> py::object {
    const auto& arg = schema.arguments()[idx];
    auto match = [&](c10::TypeKind kind) {
      const auto& t = arg.real_type();
      if (t->kind() == kind)
        return true;
      if (auto opt_t = t->cast<c10::OptionalType>()) {
        if (opt_t->getElementType()->kind() == kind)
          return true;
      }
      return false;
    };
    if (arguments[idx].isNone()) {
      return py::none();
    } else if (match(c10::ScalarTypeType::Kind)) {
      auto* obj =
          getTHPDtype(static_cast<c10::ScalarType>(arguments[idx].toInt()));
      return py::reinterpret_borrow<py::object>(
          reinterpret_cast<PyObject*>(obj));
    } else if (match(c10::LayoutType::Kind)) {
      auto* obj =
          getTHPLayout(static_cast<c10::Layout>(arguments[idx].toInt()));
      return py::reinterpret_borrow<py::object>(
          reinterpret_cast<PyObject*>(obj));
    } else if (match(c10::MemoryFormatType::Kind)) {
      return py::cast(static_cast<c10::MemoryFormat>(arguments[idx].toInt()));
    } else {
      return torch::jit::toPyObject(arguments[idx]);
    }
  };

  // Populate positional arguments
  for (const auto idx : c10::irange(positional_default_start)) {
    PyTuple_SET_ITEM(
        args.ptr(), idx, schemaAwareToPyObject(idx).release().ptr());
  }

  // Populate keyword arguments
  for (const auto idx : c10::irange(kwarg_only_start, arguments.size())) {
    // But don't populate default keyword arguments
    if (is_default(idx))
      continue;
    const auto& arg = schema.arguments()[idx];
    kwargs[py::cast(arg.name())] = schemaAwareToPyObject(idx);
  }
  return std::make_pair(std::move(args), std::move(kwargs));
}

void pushPyOutToStack(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    py::object out,
    const char* msg) {
  TORCH_CHECK(
      PyGILState_Check(), "GIL must be held before you call pushPyOutToStack");
  auto schema_returns = op.schema().returns();
  const auto num_returns = schema_returns.size();
  if (num_returns == 0) {
    // Check that we got a None return from Python. Anything else is an error.
    TORCH_CHECK(
        out.is_none(),
        "Expected ",
        msg,
        " for ",
        op.operator_name(),
        " to return None but it returned something else instead.");
  } else if (num_returns == 1) {
    torch::jit::push(
        stack, torch::jit::toIValue(out.ptr(), schema_returns[0].type()));
  } else {
    auto outs = py::cast<py::sequence>(out);
    for (const auto idx : c10::irange(outs.size())) {
      torch::jit::push(
          stack,
          torch::jit::toIValue(outs[idx].ptr(), schema_returns[idx].type()));
    }
  }
}

namespace {

// NB: This is a macro and not a template function (like it was before)
// because passing in constexpr char* as template argument breaks some
// versions of MSVC that are being used internally at Meta.
// MSVC 14.16.27023 (vs2017_15.9)
#define CONCRETE_TRACE_CUDA(func_name, ...)                           \
  at::impl::MaybeSetTLSOnEntryGuard guard;                            \
  if (Py_IsInitialized()) {                                           \
    pybind11::gil_scoped_acquire gil;                                 \
    try {                                                             \
      py::module mod = py::module::import("torch.utils._cuda_trace"); \
      py::object hook = mod.attr(func_name).attr("fire_callbacks");   \
      hook(__VA_ARGS__);                                              \
    } catch (const std::exception& e) {                               \
      LOG(ERROR) << "CUDA trace hook execution failed: " << e.what(); \
    }                                                                 \
  }

struct ConcretePyInterpreterVTable final
    : public c10::impl::PyInterpreterVTable {
  std::string name() const override;

  void decref(PyObject* pyobj, bool is_tensor) const override;

  c10::intrusive_ptr<TensorImpl> detach(const TensorImpl* self) const override;

  void dispatch(const c10::OperatorHandle& op, torch::jit::Stack* stack)
      const override;
  void python_dispatcher(
      const c10::OperatorHandle& op,
      c10::DispatchKeySet,
      torch::jit::Stack* stack) const override;
  // NB: this is defined in python_dispatch.cpp
  void python_op_registration_trampoline(
      const c10::OperatorHandle& op,
      c10::DispatchKey key,
      torch::jit::Stack* stack) const override {
    torch::impl::dispatch::python_op_registration_trampoline_impl(
        op, key, stack);
  }

  bool is_contiguous(const TensorImpl* self, at::MemoryFormat) const override;
  bool is_strides_like(const TensorImpl* self, at::MemoryFormat) const override;
  bool is_non_overlapping_and_dense(const TensorImpl* self) const override;
  c10::Device device(const TensorImpl* self) const override;
  int64_t dim(const TensorImpl* self) const override;
  c10::IntArrayRef strides(const TensorImpl* self) const override;
  c10::IntArrayRef sizes(const TensorImpl* self) const override;
  c10::SymIntArrayRef sym_sizes(const TensorImpl* self) const override;
  c10::Layout layout(const TensorImpl* self) const override;
  c10::SymInt sym_numel(const TensorImpl* self) const override;
  c10::SymIntArrayRef sym_strides(const TensorImpl* self) const override;
  c10::SymInt sym_storage_offset(const TensorImpl* self) const override;

  void trace_gpu_event_creation(uintptr_t event) const override {
    CONCRETE_TRACE_CUDA("CUDAEventCreationCallbacks", event);
  }
  void trace_gpu_event_deletion(uintptr_t event) const override {
    CONCRETE_TRACE_CUDA("CUDAEventDeletionCallbacks", event);
  }
  void trace_gpu_event_record(uintptr_t event, uintptr_t stream)
      const override {
    CONCRETE_TRACE_CUDA("CUDAEventRecordCallbacks", event, stream);
  }
  void trace_gpu_event_wait(uintptr_t event, uintptr_t stream) const override {
    CONCRETE_TRACE_CUDA("CUDAEventWaitCallbacks", event, stream);
  }
  void trace_gpu_memory_allocation(uintptr_t ptr) const override {
    CONCRETE_TRACE_CUDA("CUDAMemoryAllocationCallbacks", ptr);
  }
  void trace_gpu_memory_deallocation(uintptr_t ptr) const override {
    CONCRETE_TRACE_CUDA("CUDAMemoryDeallocationCallbacks", ptr);
  }
  void trace_gpu_stream_creation(uintptr_t stream) const override {
    CONCRETE_TRACE_CUDA("CUDAStreamCreationCallbacks", stream);
  }
  void trace_gpu_device_synchronization() const override {
    CONCRETE_TRACE_CUDA("CUDADeviceSynchronizationCallbacks");
  }
  void trace_gpu_stream_synchronization(uintptr_t stream) const override {
    CONCRETE_TRACE_CUDA("CUDAStreamSynchronizationCallbacks", stream);
  }
  void trace_gpu_event_synchronization(uintptr_t event) const override {
    CONCRETE_TRACE_CUDA("CUDAEventSynchronizationCallbacks", event);
  }

  static ConcretePyInterpreterVTable* instance() {
    static ConcretePyInterpreterVTable s;
    return &s;
  }
};

// NOTE [PyInterpreter::decref takes an `is_tensor` arg]
// Before calling PyInterpreter::decref, we must statically know if the
// pyobj is a Tensor or not.
// - If it is a tensor, we need to be careful about PyObject resurrection
// - If it is not a tensor, we can freely decref
// One alternative to this is using PyObject_IsInstance
// to get at this information. However, we don't want to risk an incorrect
// `__instancecheck__` changing the semantics here.
void ConcretePyInterpreterVTable::decref(PyObject* pyobj, bool is_tensor)
    const {
  // Leak the pyobj if not initialized.  This can happen if we are running
  // exit handlers that are destructing tensors with residual (owned)
  // PyObjects stored in them.
  if (!Py_IsInitialized())
    return;

  pybind11::gil_scoped_acquire gil;
  // Two possibilities:
  // 1. We are decref-ing a tensor. Then we must be careful about
  // PyObject resurrection (this only applies to Tensors, see
  // THPVariable_clear).
  // 2. We are decref-ing some other Python object. We don't do
  // PyObject resurrection on non-Tensors, so we just carry on as usual
  if (is_tensor) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        !c10::impl::HermeticPyObjectTLS::get_state());
  }
  if (is_tensor && Py_REFCNT(pyobj) > 1) {
    // It's still alive!  This can happen if a weak ref resurrected
    // the PyObject without flipping ownership.  At this point it is
    // too late to rescue the object, so just stub out the PyObject
    // so that it fails on subsequent uses.  Don't raise an error here;
    // you're probably in a destructor.
    TORCH_WARN(
        "Deallocating Tensor that still has live PyObject references.  "
        "This probably happened because you took out a weak reference to "
        "Tensor and didn't call _fix_weakref() after dereferencing it.  "
        "Subsequent accesses to this tensor via the PyObject will now fail.");
    ((THPVariable*)pyobj)->cdata = MaybeOwned<Variable>();
  }
  Py_DECREF(pyobj);
};

class PyInterpreterHolder {
 public:
  PyInterpreterHolder()
      : impl_(new c10::impl::PyInterpreter(
            ConcretePyInterpreterVTable::instance())) {
    is_main_interpreter_ =
        at::impl::PythonOpRegistrationTrampoline::registerInterpreter(impl_);
  }
  // NB: intentionally leaks the PyInterpreter, as there may still be
  // references to it that are live, living in objects that aren't being
  // destructed while Python is being cleaned up.
  ~PyInterpreterHolder() {
    impl_->disarm();
  }
  c10::impl::PyInterpreter* get() const noexcept {
    return impl_;
  }
  bool is_main_interpreter() const noexcept {
    return is_main_interpreter_;
  }

 private:
  c10::impl::PyInterpreter* impl_;
  bool is_main_interpreter_;
};
PyInterpreterHolder self_interpreter;

c10::TensorImpl::SizesStridesPolicy parseSizesStridesPolicyArgument(
    c10::string_view arg) {
  if (arg == "strides") {
    return c10::TensorImpl::SizesStridesPolicy::CustomStrides;
  }

  if (arg == "sizes") {
    return c10::TensorImpl::SizesStridesPolicy::CustomSizes;
  }

  TORCH_CHECK_VALUE(
      false,
      "Unknown sizes_strides_policy: ",
      arg,
      "; expected 'strides' or 'sizes'");
}
} // anonymous namespace

c10::impl::PyInterpreter* getPyInterpreter() {
  return self_interpreter.get();
}

bool isMainPyInterpreter() {
  return self_interpreter.is_main_interpreter();
}

std::string ConcretePyInterpreterVTable::name() const {
  std::stringstream ss;
  ss << getPyInterpreter();
  return ss.str();
}

PyObject* THPVariableClass = nullptr;

PyObject* ParameterClass = nullptr;

static PyObject* THPVariable_NewWithVar(
    PyTypeObject* type,
    Variable _var,
    c10::impl::PyInterpreterStatus status);

// clang-tidy gets confused by static const
static const char* VOLATILE_WARNING =
    "volatile was removed and now has no effect. Use "
    "`with torch.no_grad():` instead.";

static bool check_has_torch_dispatch(PyObject* obj) {
  PyTypeObject* tp = Py_TYPE(obj);
  if (THPVariable_CheckTypeExact(tp)) {
    return false;
  }
  py::object attr = PyObject_FastGetAttrString(obj, "__torch_dispatch__");
  return (
      attr.ptr() != nullptr &&
      attr.ptr() != torch::disabled_torch_dispatch_impl());
}

// NOLINTNEXTLINE
static PyObject* device_to_py_class_[static_cast<size_t>(
    c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)];

void registerPythonTensorClass(
    const std::string& device,
    PyObject* python_tensor_class) {
  c10::Device dev(device);

  TORCH_CHECK(
      dev.type() == kXLA, "Only the python class for XLA can be overriden");
  if (device_to_py_class_[static_cast<size_t>(dev.type())] != nullptr) {
    TORCH_WARN(
        "Overriding a previously registered python class for ", dev.str());
  }

  device_to_py_class_[static_cast<size_t>(dev.type())] = python_tensor_class;
}

static PyObject* getPythonTensorClass(c10::Device d) {
  return device_to_py_class_[static_cast<size_t>(d.type())];
}

void activateCUDATrace() {
  c10::impl::GPUTrace::set_trace(self_interpreter.get());
}

// TODO: Make this take Variable by const reference
PyObject* THPVariable_Wrap(at::TensorBase var) {
  if (!var.defined()) {
    Py_RETURN_NONE;
  }

  if (c10::impl::HermeticPyObjectTLS::get_state()) {
    return THPVariable_NewWithVar(
        (PyTypeObject*)THPVariableClass,
        std::move(var),
        c10::impl::PyInterpreterStatus::DEFINITELY_UNINITIALIZED);
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

  if (C10_LIKELY(var.device().type() != c10::kXLA)) {
    return THPVariable_NewWithVar(
        (PyTypeObject*)THPVariableClass, std::move(var), status);
  }

  if (auto clazz = getPythonTensorClass(var.device())) {
    return THPVariable_NewWithVar((PyTypeObject*)clazz, std::move(var), status);
  }

  return THPVariable_NewWithVar(
      (PyTypeObject*)THPVariableClass, std::move(var), status);
}

bool isResurrectable(THPVariable* self) {
  // We want to divide this check into 2 cases.

  // 1. C++ owns PyObject (in this case, self->cdata.unsafeIsBorrowed() is
  // true). You might think that in this case, it is impossible for tp_clear to
  // be called: surely the C++ reference to the PyObject is keeping it live? And
  // you'd be right! In fact, when C++ owns the PyObject, we have an invariant
  // that the refcount on the PyObject should be precisely one (because if you
  // take out another reference to the PyObject, we're supposed to flip the
  // ownership pointer back). In reality, you can violate this invariant
  // temporarily with weak references, so we don't test for it in asserts.

  // 2. PyObject owns C++ (in this case, self->cdata.unsafeIsBorrowed() is
  // false). In this case, tp_clear can get called if the PyObject is referenced
  // from a dead cycle, and nowhere else. But if resurrection did not occur,
  // then the reference to C++ from the PyObject must be the ONLY reference to
  // the C++ object.
  if (self->cdata.unsafeIsBorrowed()) {
    return false;
  }
  auto const& tensor = THPVariable_Unpack(self);
  // Check if this is hermetic. If it is, no resurrection.
  if (tensor.unsafeGetTensorImpl()->check_pyobj(self_interpreter.get()) !=
      c10::make_optional((PyObject*)self)) {
    return false;
  }
  if (!tensor.defined() || tensor.use_count() <= 1) {
    return false;
  }
  return true;
}

// returns true if successfully rezzed; if so, cancel the
// rest of deallocation
static bool THPVariable_tryResurrect(THPVariable* self) {
  const auto& tensor = THPVariable_Unpack(self);

  if (!isResurrectable(self)) {
    return false;
  }

  // At this point, we are definitely going to resurrect the tensor. So, the
  // tensor better be defined :)
  TORCH_INTERNAL_ASSERT(tensor.defined());

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
  _Py_AddToAllObjects(reinterpret_cast<PyObject*>(self), 1);
#endif
  Py_INCREF(self);

  // Flip THPVariable to be non-owning
  // (near use-after-free miss here: fresh MaybeOwned is created breaking
  // reference on Tensor in struct BEFORE we overwrite the old one)
  TORCH_INTERNAL_ASSERT(!c10::impl::HermeticPyObjectTLS::get_state());
  self->cdata = MaybeOwned<Variable>::borrowed(tensor);

  // NB: At this point, tensor *could* be dead (e.g., some other C++ thread
  // decrefed it.)  At this point, it is probably waiting on the GIL to
  // deallocate the Python object and will kill self, BUT NOT YET.

  return true;
}

static int THPVariable_clear(THPVariable* self) {
  // Is it OK for an object to still be live after running
  // tp_clear? Yes. When Python is breaking reference cycles, it can't assume
  // that an object will dealloc after it's cleared.  The source code explicitly
  // handles this case:
  // https://github.com/python/cpython/blob/4e661cd69164318c1f871faa476c68a04092ddc4/Modules/gcmodule.c#L1010-L1025

  // Note that we don't need to actually resurrect here. There are 2 cases:
  // 1. The PyObject is not part of a reference cycle. In this case, we don't
  // need to do anything. The GC will move on to try and break the reference
  // cycle on another object, which will eventually trigger tp_dealloc (and thus
  // resurrection).

  // 2. The PyObject is part of a reference cycle. This case should not actually
  // be possible, due to the logic in our tp_traverse
  // (THPVariable_subclass_traverse).

  // In fact, resurrecting here breaks the invariant that "C++ owns Python only
  // when PyObject's refcount would otherwise be 0". Most immediately, as we're
  // merely breaking reference cycles here, there can be other references to the
  // PyObject. *However*, if other objects in the refcycle resurrect, then we
  // will be in a state where the PyObject has multiple Python references, yet
  // C++ owns the PyObject.

  // See https://github.com/pytorch/pytorch/pull/75933 for more discussion.
  if (isResurrectable((THPVariable*)self)) {
    return 0;
  }
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

    if (!self->cdata.unsafeIsBorrowed() &&
        tensor.unsafeGetTensorImpl()->check_pyobj(self_interpreter.get()) ==
            c10::make_optional((PyObject*)self)) {
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
  TORCH_INTERNAL_ASSERT(!isResurrectable((THPVariable*)self));
  {
    // MapAllocator can take significant time to release large tensors;
    // release the GIL here to avoid impacting main thread perf.
    pybind11::gil_scoped_release no_gil;
    self->cdata = MaybeOwned<Variable>();
  }
  return 0;
}

int THPFunction_traverse(THPFunction* self, visitproc visit, void* arg) {
  TORCH_INTERNAL_ASSERT(
      false, "Tensor tp_traverse function was not overriden properly");
  return 0;
}

PyObject* THPVariable_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs);

static PyObject* THPVariable_fix_weakref(PyObject* self, PyObject* noargs) {
  const auto& var = THPVariable_Unpack(self);
  THPVariable_Wrap(var);
  Py_RETURN_NONE;
}

// Instantiates a subclass of self with the same data.
static PyObject* THPVariable_as_subclass(
    PyObject* _self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  const auto& self = THPVariable_Unpack(_self);
  static PythonArgParser parser({
      "as_subclass(PyObject* cls)",
  });
  ParsedArgs<1> parsed_args{};
  auto r = parser.parse(_self, args, kwargs, parsed_args);
  PyObject* cls = r.pyobject(0);
  if (!PyType_Check(cls)) {
    throw torch::TypeError(
        "cls must be a type (got %s)", Py_TYPE(cls)->tp_name);
  }
  return THPVariable_NewWithVar(
      (PyTypeObject*)cls,
      self.alias(),
      c10::impl::PyInterpreterStatus::DEFINITELY_UNINITIALIZED);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_make_subclass(
    PyObject* _ignored,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
      "_make_subclass(PyObject* cls, Tensor data, bool require_grad=False, *, c10::string_view? dispatch_sizes_strides_policy=None, bool dispatch_device=False, bool dispatch_layout=False, Device? device_for_backend_keys=None)",
  });
  ParsedArgs<7> parsed_args{};
  auto r = parser.parse(args, kwargs, parsed_args);
  PyObject* cls = r.pyobject(0);
  if (!PyType_Check(cls)) {
    throw torch::TypeError(
        "cls must be a type (got %s)", Py_TYPE(cls)->tp_name);
  }
  // guard completely turns off torch dispatch modes, doesn't just pop off the
  // stack
  torch_dispatch_mode::StashTorchDispatchStackGuard td_g;
  c10::impl::DisablePythonDispatcher dpd_g;
  auto data =
      r.tensor(1).detach(); // creates a fresh Tensor (DEFINITELY_UNINITIALIZED)
  // We set `data`'s `allow_tensor_metadata_change` to true here, because we
  // want to allow the following use case for backward compatibility:
  //
  // ```python
  // rnn = torch.nn.RNN(100, 100, 2)
  // # The following calls `torch._cudnn_rnn_flatten_weight(rnn._flat_weights,
  // ...)`, # which changes storage of `rnn`'s weights in-place
  // rnn.flatten_parameters()
  // ```
  data.unsafeGetTensorImpl()->set_allow_tensor_metadata_change(true);
  data.set_requires_grad(r.toBool(2));
  const auto sizes_strides_policy = r.stringViewOptional(3);
  if (sizes_strides_policy.has_value()) {
    data.unsafeGetTensorImpl()->set_python_custom_sizes_strides(
        parseSizesStridesPolicyArgument(*sizes_strides_policy));
  }
  if (r.toBool(4)) {
    data.unsafeGetTensorImpl()->set_python_custom_device(true);
  }
  if (r.toBool(5)) {
    data.unsafeGetTensorImpl()->set_python_custom_layout(true);
  }
  if (!r.isNone(6)) {
    data.unsafeGetTensorImpl()->_change_backend_component_keys(r.device(6));
  }

  return THPVariable_NewWithVar(
      (PyTypeObject*)cls,
      std::move(data),
      c10::impl::PyInterpreterStatus::DEFINITELY_UNINITIALIZED);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_make_wrapper_subclass(
    PyObject*,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // NB: pin_memory doesn't actually do anything
  // TODO: strides variant?
  static PythonArgParser parser({
      "_make_wrapper_subclass(PyObject* cls, IntArrayRef size, *, IntArrayRef? strides=None, "
      "int64_t? storage_offset=None, MemoryFormat? memory_format=None, ScalarType dtype=None, "
      "Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False, "
      "c10::string_view? dispatch_sizes_strides_policy=None, bool dispatch_device=False, bool dispatch_layout=False)",
      "_make_wrapper_subclass(PyObject* cls, SymIntArrayRef size, SymIntArrayRef strides, "
      "SymInt? storage_offset=None, MemoryFormat? memory_format=None, ScalarType dtype=None, "
      "Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False, "
      "c10::string_view? dispatch_sizes_strides_policy=None, bool dispatch_device=False, bool dispatch_layout=False)",
  });
  ParsedArgs<13> parsed_args{};
  auto r = parser.parse(args, kwargs, parsed_args);
  PyObject* cls = r.pyobject(0);

  TORCH_CHECK_TYPE(
      PyType_Check(cls),
      "cls must be a type (got ",
      Py_TYPE(cls)->tp_name,
      ")");

  // This is an important safety check; without it, the default behavior will be
  // to continue on to the underlying CPU/CUDA kernel advertised by the dispatch
  // key, which will immediately segfault because the data pointer is null.  By
  // forcing users to define __torch_dispatch__ we ensure this does not happen
  // TODO: This check is not complete; because the user can disable torch
  // dispatch and then go again, triggering segfault.  TBH I'm thinking I want
  // to delete this function entirely
  py::object attr = PyObject_FastGetAttrString(cls, "__torch_dispatch__");
  TORCH_CHECK_TYPE(
      attr.ptr() != nullptr &&
          attr.ptr() != torch::disabled_torch_dispatch_impl(),
      ((PyTypeObject*)cls)->tp_name,
      " must define __torch_dispatch__");

  const auto options = TensorOptions()
                           .dtype(r.scalartype(5))
                           .device(r.device(7))
                           .layout(r.layoutOptional(6))
                           // NB: long standing issue, requires_grad is not
                           // respected here; you have to set it post facto, see
                           // https://github.com/pytorch/pytorch/issues/26428
                           // .requires_grad(r.toBool(7))
                           .pinned_memory(r.toBool(8));

  // don't bother releasing GIL here, as we are not allocating any nontrivial
  // data
  // TODO: for_blob produces non-resizable tensors, we might want this to be
  // resizable (have to define a custom allocator in that case)
  Tensor tensor;
  if (r.idx == 0) {
    tensor = at::for_blob(nullptr, r.intlist(1))
                 .strides(r.intlistOptional(2))
                 .storage_offset(r.toInt64Optional(3))
                 .context(nullptr, [](void* ctx) {})
                 .target_device(
                     options.device()) // TODO: this shouldn't be necessary if
                                       // it came from options
                 .options(options)
                 .make_tensor();

    const auto sizes_strides_policy = r.stringViewOptional(10);
    if (sizes_strides_policy.has_value()) {
      tensor.unsafeGetTensorImpl()->set_python_custom_sizes_strides(
          parseSizesStridesPolicyArgument(*sizes_strides_policy));
    }
  } else {
    AutoDispatchBelowADInplaceOrView guard{}; // TODO: Remove.
    tracer::impl::NoTracerDispatchMode tracer_guard{};

    // We shouldn't need storage
    Storage storage{Storage::use_byte_size_t{}, 0, at::DataPtr{}};

    tensor = at::detail::make_tensor<TensorImpl>(
        std::move(storage), options.computeDispatchKey(), options.dtype());

    auto sym_sizes = r.symintlist(1);
    auto sym_strides = r.symintlist(2);
    auto sym_storage_offset = r.toSymIntOptional(3);

    TensorImpl* tensor_impl = tensor.unsafeGetTensorImpl();

    tensor_impl->set_sizes_and_strides(
        sym_sizes, sym_strides, sym_storage_offset.value_or(0));

    const auto sizes_strides_policy = r.stringViewOptional(10);
    if (sizes_strides_policy.has_value()) {
      TORCH_CHECK(
          false,
          "Setting sizes_strides_policy isn't suppored for this overload")
    }
  }

  tensor.set_requires_grad(r.toBool(9));

  if (r.toBool(11)) {
    tensor.unsafeGetTensorImpl()->set_python_custom_device(true);
  }
  if (r.toBool(12)) {
    tensor.unsafeGetTensorImpl()->set_python_custom_layout(true);
  }

  return THPVariable_NewWithVar(
      (PyTypeObject*)cls,
      std::move(tensor),
      c10::impl::PyInterpreterStatus::DEFINITELY_UNINITIALIZED);
  END_HANDLE_TH_ERRORS
}

typedef PyObject* (*getter)(PyObject*, void*);
typedef int (*setter)(PyObject*, PyObject*, void*);

PyObject* THPVariable_get_python_dispatch(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  const auto& var = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(
      var.unsafeGetTensorImpl()->is_python_dispatch());
  END_HANDLE_TH_ERRORS
}

// CRTP base class to implement the python bindings for a Tensor property in
// PyTorch A class that implements a property is expected to have:
// - static constexpr const char* name;
//   - This variable should hold the Python name of the property
// - static Tensor fn(const Tensor&);
//   - This function calls the relevant ATen on the tensor
template <typename T>
struct GetterBase {
  static PyObject* getter(THPVariable* self, void* /*unused*/) {
    HANDLE_TH_ERRORS
    if (check_has_torch_function((PyObject*)self)) {
      return handle_torch_function_getter(self, T::name);
    }
    return THPVariable_Wrap(T::fn(THPVariable_Unpack(self)));
    END_HANDLE_TH_ERRORS
  }
};

struct PropertyT : GetterBase<PropertyT> {
  static constexpr const char* name = "T";
  static Tensor fn(const Tensor& t) {
    return t.numpy_T();
  }
};

struct PropertyH : GetterBase<PropertyH> {
  static constexpr const char* name = "H";
  static Tensor fn(const Tensor& t) {
    return t.matrix_H();
  }
};

struct PropertymT : GetterBase<PropertymT> {
  static constexpr const char* name = "mT";
  static Tensor fn(const Tensor& t) {
    return t.mT();
  }
};

struct PropertymH : GetterBase<PropertymH> {
  static constexpr const char* name = "mH";
  static Tensor fn(const Tensor& t) {
    return t.mH();
  }
};

struct PropertyData : GetterBase<PropertyData> {
  static constexpr const char* name = "data";
  static Tensor fn(const Tensor& t) {
    return t.variable_data();
  }
};

struct PropertyGrad : GetterBase<PropertyGrad> {
  static constexpr const char* name = "grad";
  static Tensor fn(const Tensor& t) {
    return t.grad();
  }
};

struct PropertyReal : GetterBase<PropertyReal> {
  static constexpr const char* name = "real";
  static Tensor fn(const Tensor& t) {
    return at::real(t);
  }
};

struct PropertyImag : GetterBase<PropertyImag> {
  static constexpr const char* name = "imag";
  static Tensor fn(const Tensor& t) {
    return at::imag(t);
  }
};

PyObject* THPVariable_get_cdata(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "_cdata");
  }
  const auto& var = THPVariable_Unpack(self);
  return PyLong_FromVoidPtr(var.unsafeGetTensorImpl());
  END_HANDLE_TH_ERRORS
}

PyObject* THPVariable_get_version(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "_version");
  }
  const auto& var = THPVariable_Unpack(self);
  return PyInt_FromLong(var._version());
  END_HANDLE_TH_ERRORS
}

PyObject* THPVariable_get_grad_fn(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "grad_fn");
  }
  const auto& var = THPVariable_Unpack(self);
  if (!var.grad_fn()) {
    Py_RETURN_NONE;
  }
  return functionToPyObject(var.grad_fn());
  END_HANDLE_TH_ERRORS
}

static int THPVariable_set_grad_fn(
    THPVariable* self,
    PyObject* obj,
    void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_setter(self, "_grad_fn", obj);
  }
  THPUtils_assertRet(
      -1, obj, "Deletion of _grad_fn not allowed. Detach tensor instead!");
  THPUtils_assertRet(-1, obj == Py_None, "_grad_fn can be only set to None");
  THPVariable_Unpack(self).detach_();
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

static PyObject* THPVariable_is_leaf(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_leaf");
  }
  return PyBool_FromLong(!THPVariable_Unpack(self).grad_fn());
  END_HANDLE_TH_ERRORS
}

int THPVariable_set_data(THPVariable* self, PyObject* data, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_setter(self, "data", data);
  }
  THPUtils_assertRet(
      -1, data, "Deleting tensor data is not allowed. Delete tensor instead!");
  if (!THPVariable_Check(data)) {
    throw torch::TypeError(
        "Variable data has to be a tensor, but got %s", Py_TYPE(data)->tp_name);
  }

  THPVariable_Unpack(self).set_data(THPVariable_Unpack(data));
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

int THPVariable_set_grad(THPVariable* self, PyObject* py_grad, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_setter(self, "grad", py_grad);
  }
  const auto& var = THPVariable_Unpack(self);
  if (!py_grad || py_grad == Py_None) {
    var.mutable_grad().reset();
    return 0;
  }

  TORCH_CHECK_TYPE(
      THPVariable_Check(py_grad),
      "assigned grad expected to be a Tensor or None but got grad of type",
      THPUtils_typename(py_grad));
  THPUtils_assertRet(
      -1,
      self != (THPVariable*)py_grad,
      "can't assign Variable as its own grad");

  const auto& grad = THPVariable_Unpack(py_grad);
  bool gradIsSparse =
      (var.dtype() == grad.dtype() &&
       var.device().type() == grad.device().type() && grad.layout() == kSparse);
  THPUtils_assertRet(
      -1,
      grad.options().type_equal(var.options()) || gradIsSparse,
      "assigned grad has data of a different type");
  if (var.is_cuda()) {
    THPUtils_assertRet(
        -1,
        grad.get_device() == var.get_device(),
        "assigned grad has data located on a different device");
  }
  THPUtils_assertRet(
      -1,
      grad.sym_sizes().equals(var.sym_sizes()),
      "assigned grad has data of a different size");

  var.mutable_grad() = grad;
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

PyObject* THPVariable_get_volatile(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "volatile");
  }
  const char* msg = "volatile was removed (Variable.volatile is always False)";
  auto r = PyErr_WarnEx(PyExc_UserWarning, msg, 1);
  if (r != 0)
    throw python_error();
  Py_RETURN_FALSE;
  END_HANDLE_TH_ERRORS
}

int THPVariable_set_volatile(THPVariable* self, PyObject* obj, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_setter(self, "volatile", obj);
  }
  auto r = PyErr_WarnEx(PyExc_UserWarning, VOLATILE_WARNING, 1);
  if (r != 0)
    throw python_error();
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

PyObject* THPVariable_get_output_nr(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "output_nr");
  }
  const auto output_nr =
      static_cast<long>(THPVariable_Unpack(self).output_nr());
  return PyInt_FromLong(output_nr);
  END_HANDLE_TH_ERRORS
}

PyObject* THPVariable_get_requires_grad(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "requires_grad");
  }
  if (THPVariable_Unpack(self).requires_grad()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPVariable_retains_grad(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "retains_grad");
  }
  if (THPVariable_Unpack(self).retains_grad()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPVariable_get_ndim(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "ndim");
  }
  return PyInt_FromLong(THPVariable_Unpack(self).dim());
  END_HANDLE_TH_ERRORS
}

PyObject* THPVariable_get_names(PyObject* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function_getter((THPVariable*)self, "names");
  }
  // The long-term plan is to return a list of (python) torch.Dimname.
  // However, for now, return a list of string.
  const auto& tensor = THPVariable_Unpack(self);
  size_t size = tensor.dim();
  THPObjectPtr tuple(PyTuple_New(size));
  if (!tuple)
    throw python_error();

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
      // -
      // https://stackoverflow.com/questions/16400600/how-to-return-a-tuple-containing-a-none-value-from-the-c-api
      Py_INCREF(Py_None);
      str = Py_None;
    } else {
      str = THPUtils_packString(dimnames[i].symbol().toUnqualString());
      if (!str)
        throw python_error();
    }
    PyTuple_SET_ITEM(tuple.get(), i, str);
  }
  return tuple.release();
  END_HANDLE_TH_ERRORS
}

int THPVariable_set_names(PyObject* self, PyObject* names, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function_setter((THPVariable*)self, "names", names);
  }
  const auto& var = THPVariable_Unpack(self);
  if (names == Py_None) {
    at::internal_set_names_inplace(var, at::nullopt);
  } else {
    THPUtils_assertRet(
        -1,
        THPUtils_checkDimnameList(names),
        "names must either be None or a tuple of dim names");
    at::internal_set_names_inplace(var, torch::parseDimnameList(names));
  }
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

int THPVariable_set_requires_grad(
    THPVariable* self,
    PyObject* obj,
    void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_setter(self, "requires_grad", obj);
  }
  THPUtils_assertRet(
      -1, obj && PyBool_Check(obj), "requires_grad must be a bool");
  const auto& var = THPVariable_Unpack(self);
  auto requires_grad = (obj == Py_True);
  if (!var.is_leaf()) {
    THPUtils_setError(
        autograd::utils::requires_grad_leaf_error(obj == Py_True).c_str());
    return -1;
  }
  if (requires_grad &&
      !isDifferentiableType(at::typeMetaToScalarType((var.dtype())))) {
    THPUtils_setError(
        "only Tensors of floating point and complex dtype can require gradients");
    return -1;
  }
  var.set_requires_grad(requires_grad);
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

PyObject* THPVariable_get_name(THPVariable* self, void* unused) {
  if (check_has_torch_function((PyObject*)self)) {
    HANDLE_TH_ERRORS
    return handle_torch_function_getter(self, "name");
    END_HANDLE_TH_ERRORS
  }
  const auto& tensor = THPVariable_Unpack(self);
  if (tensor.name() == "")
    Py_RETURN_NONE;
  return THPUtils_packString(tensor.name().c_str());
}

PyObject* THPVariable_get_backwards_hooks(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "_backward_hooks");
  }
  if (self->backward_hooks) {
    Py_INCREF(self->backward_hooks);
    return self->backward_hooks;
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

int THPVariable_set_backwards_hooks(
    THPVariable* self,
    PyObject* obj,
    void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
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
    torch::autograd::impl::add_hook(
        tensor, std::make_shared<PyFunctionTensorPreHook>(obj, 0));
  }
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

PyObject* THPVariable_get_base(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "_base");
  }
  const auto& tensor = THPVariable_Unpack(self);
  if (tensor.is_view()) {
    return THPVariable_Wrap(tensor._base());
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPVariable_get_shape(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "shape");
  }
  return THPSize_NewFromSymSizes(THPVariable_Unpack(self));
  END_HANDLE_TH_ERRORS
}

PyObject* THPVariable_is_cpu(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_cpu");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_cpu());
  END_HANDLE_TH_ERRORS
}

PyObject* THPVariable_is_cuda(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_cuda");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_cuda());
  END_HANDLE_TH_ERRORS
}

PyObject* THPVariable_is_ipu(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_ipu");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_ipu());
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

PyObject* THPVariable_is_sparse(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_sparse");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_sparse());
  END_HANDLE_TH_ERRORS
}

PyObject* THPVariable_is_sparse_csr(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_sparse_csr");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_sparse_csr());
  END_HANDLE_TH_ERRORS
}

PyObject* THPVariable_is_mkldnn(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_mkldnn");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_mkldnn());
  END_HANDLE_TH_ERRORS
}

PyObject* THPVariable_is_mps(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_mps");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_mps());
  END_HANDLE_TH_ERRORS
}

PyObject* THPVariable_is_ort(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_ort");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_ort());
  END_HANDLE_TH_ERRORS
}

PyObject* THPVariable_is_vulkan(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_vulkan");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_vulkan());
  END_HANDLE_TH_ERRORS
}

PyObject* THPVariable_is_quantized(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_quantized");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_quantized());
  END_HANDLE_TH_ERRORS
}

PyObject* THPVariable_is_meta(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_meta");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_meta());
  END_HANDLE_TH_ERRORS
}

PyObject* THPVariable_is_complex(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_complex");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_complex());
  END_HANDLE_TH_ERRORS
}

PyObject* THPVariable_is_nested(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_nested");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_nested());
  END_HANDLE_TH_ERRORS
}

PyObject* THPVariable_has_symbolic_sizes_strides(
    THPVariable* self,
    void* unused) {
  HANDLE_TH_ERRORS
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(
      self_.unsafeGetTensorImpl()->has_symbolic_sizes_strides());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_dtype(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "dtype");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(torch::getTHPDtype(self_.scalar_type()));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_layout(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "layout");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(torch::getTHPLayout(self_.layout()));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_device(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "device");
  }
  return THPDevice_New(THPVariable_Unpack(self).device());
  END_HANDLE_TH_ERRORS
}

int THPVariable_set_real(PyObject* self, PyObject* real, void* unused) {
  HANDLE_TH_ERRORS
  auto& self_ = THPVariable_Unpack(self);
  auto self_real = at::real(self_);
  auto real_ = valueToTensor(self_real.options(), real, self_real.device());
  {
    pybind11::gil_scoped_release no_gil;
    self_real.copy_(real_);
    return 0;
  }
  END_HANDLE_TH_ERRORS_RET(-1)
}

int THPVariable_set_imag(PyObject* self, PyObject* imag, void* unused) {
  HANDLE_TH_ERRORS
  auto& self_ = THPVariable_Unpack(self);
  auto self_imag = at::imag(self_);
  auto imag_ = valueToTensor(self_imag.options(), imag, self_imag.device());
  {
    pybind11::gil_scoped_release no_gil;
    self_imag.copy_(imag_);
    return 0;
  }
  END_HANDLE_TH_ERRORS_RET(-1)
}

// properties are registered here because we are currently only able to bind
// them manually. TODO: make declarable in native_functions
// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static struct PyGetSetDef THPVariable_properties[] = {
    {"_python_dispatch",
     (getter)THPVariable_get_python_dispatch,
     nullptr,
     nullptr,
     nullptr},
    {"T", (getter)PropertyT::getter, nullptr, nullptr, nullptr},
    {"H", (getter)PropertyH::getter, nullptr, nullptr, nullptr},
    {"mT", (getter)PropertymT::getter, nullptr, nullptr, nullptr},
    {"mH", (getter)PropertymH::getter, nullptr, nullptr, nullptr},
    {"_cdata", (getter)THPVariable_get_cdata, nullptr, nullptr, nullptr},
    {"_version", (getter)THPVariable_get_version, nullptr, nullptr, nullptr},
    {"grad_fn", (getter)THPVariable_get_grad_fn, nullptr, nullptr, nullptr},
    {"_grad_fn",
     (getter)THPVariable_get_grad_fn,
     (setter)THPVariable_set_grad_fn,
     nullptr,
     nullptr},
    {"is_leaf", (getter)THPVariable_is_leaf, nullptr, nullptr, nullptr},
    {"retains_grad",
     (getter)THPVariable_retains_grad,
     nullptr,
     nullptr,
     nullptr},
    {"data",
     (getter)PropertyData::getter,
     (setter)THPVariable_set_data,
     nullptr,
     nullptr},
    {"_grad",
     (getter)PropertyGrad::getter,
     (setter)THPVariable_set_grad,
     nullptr,
     nullptr}, // Allows the python class to override .grad
    {"grad",
     (getter)PropertyGrad::getter,
     (setter)THPVariable_set_grad,
     nullptr,
     nullptr},
    {"_base", (getter)THPVariable_get_base, nullptr, nullptr, nullptr},
    {"volatile",
     (getter)THPVariable_get_volatile,
     (setter)THPVariable_set_volatile,
     nullptr,
     nullptr},
    {"output_nr", (getter)THPVariable_get_output_nr, nullptr, nullptr, nullptr},
    {"requires_grad",
     (getter)THPVariable_get_requires_grad,
     (setter)THPVariable_set_requires_grad,
     nullptr,
     nullptr},
    {"_backward_hooks",
     (getter)THPVariable_get_backwards_hooks,
     (setter)THPVariable_set_backwards_hooks,
     nullptr,
     nullptr},
    {"name", (getter)THPVariable_get_name, nullptr, nullptr, nullptr},
    {"shape", (getter)THPVariable_get_shape, nullptr, nullptr, nullptr},
    {"is_cuda", (getter)THPVariable_is_cuda, nullptr, nullptr, nullptr},
    {"is_cpu", (getter)THPVariable_is_cpu, nullptr, nullptr, nullptr},
    {"is_xpu", (getter)THPVariable_is_xpu, nullptr, nullptr, nullptr},
    {"is_ipu", (getter)THPVariable_is_ipu, nullptr, nullptr, nullptr},
    {"is_sparse", (getter)THPVariable_is_sparse, nullptr, nullptr, nullptr},
    {"is_sparse_csr",
     (getter)THPVariable_is_sparse_csr,
     nullptr,
     nullptr,
     nullptr},
    {"is_mkldnn", (getter)THPVariable_is_mkldnn, nullptr, nullptr, nullptr},
    {"is_mps", (getter)THPVariable_is_mps, nullptr, nullptr, nullptr},
    {"is_ort", (getter)THPVariable_is_ort, nullptr, nullptr, nullptr},
    {"is_vulkan", (getter)THPVariable_is_vulkan, nullptr, nullptr, nullptr},
    {"is_complex", (getter)THPVariable_is_complex, nullptr, nullptr, nullptr},
    {"is_quantized",
     (getter)THPVariable_is_quantized,
     nullptr,
     nullptr,
     nullptr},
    {"is_meta", (getter)THPVariable_is_meta, nullptr, nullptr, nullptr},
    {"is_nested", (getter)THPVariable_is_nested, nullptr, nullptr, nullptr},
    {"_has_symbolic_sizes_strides",
     (getter)THPVariable_has_symbolic_sizes_strides,
     nullptr,
     nullptr,
     nullptr},
    {"dtype", (getter)THPVariable_dtype, nullptr, nullptr, nullptr},
    {"layout", (getter)THPVariable_layout, nullptr, nullptr, nullptr},
    {"device", (getter)THPVariable_device, nullptr, nullptr, nullptr},
    {"ndim", (getter)THPVariable_get_ndim, nullptr, nullptr, nullptr},
    {"names",
     (getter)THPVariable_get_names,
     (setter)THPVariable_set_names,
     nullptr,
     nullptr},
    {"real",
     (getter)PropertyReal::getter,
     (setter)THPVariable_set_real,
     nullptr,
     nullptr},
    {"imag",
     (getter)PropertyImag::getter,
     (setter)THPVariable_set_imag,
     nullptr,
     nullptr},
    {nullptr}};

static PyMappingMethods THPVariable_as_mapping = {
    THPVariable_length,
    THPVariable_getitem,
    THPVariable_setitem,
};

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static PyMethodDef extra_methods[] = {
    {"as_subclass",
     castPyCFunctionWithKeywords(THPVariable_as_subclass),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_make_subclass",
     castPyCFunctionWithKeywords(THPVariable_make_subclass),
     METH_STATIC | METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_make_wrapper_subclass",
     castPyCFunctionWithKeywords(THPVariable_make_wrapper_subclass),
     METH_STATIC | METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_fix_weakref", THPVariable_fix_weakref, METH_NOARGS, nullptr},
    {nullptr}};

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

int THPVariableMetaType_init(PyObject* cls, PyObject* args, PyObject* kwargs);

PyTypeObject THPVariableMetaType = {
    PyVarObject_HEAD_INIT(
        DEFERRED_ADDRESS(&PyType_Type),
        0) "torch._C._TensorMeta", /* tp_name */
    sizeof(THPVariableMeta), /* tp_basicsize */
    0, /* tp_itemsize */
    nullptr, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    nullptr, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    DEFERRED_ADDRESS(&PyType_Type), /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    THPVariableMetaType_init, /* tp_init */
    nullptr, /* tp_alloc */
    nullptr, /* tp_new */
};

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
    (traverseproc)THPFunction_traverse, /* tp_traverse */
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

PyObject* THPVariable_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      type != &THPVariableType,
      "Cannot directly construct _TensorBase; subclass it and then construct that");
  jit::tracer::warn("torch.Tensor", jit::tracer::WARN_CONSTRUCTOR);
  auto tensor = torch::utils::base_tensor_ctor(args, kwargs);
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
  mp = type->tp_members;
  for (i = 0; i < n; i++, mp++) {
    if (mp->type == T_OBJECT_EX && !(mp->flags & READONLY)) {
      char* addr = (char*)self + mp->offset;
      PyObject* obj = *(PyObject**)addr;
      if (obj != nullptr) {
        *(PyObject**)addr = nullptr;
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
    if (dictptr != nullptr) {
      PyObject* dict = *dictptr;
      if (dict != nullptr) {
        Py_DECREF(dict);
        *dictptr = nullptr;
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
  // This function overwrite the Tensor's pyobj field without extra checks
  // Make sure it is not set otherwise we would leak memory
  auto mb_obj = _var.unsafeGetTensorImpl()->check_pyobj(self_interpreter.get());
  TORCH_CHECK(
      !mb_obj.has_value() || !mb_obj.value(),
      "Creating a new Tensor subclass ",
      type->tp_name,
      " but the raw Tensor object is already associated to a python object ",
      "of type ",
      mb_obj.value()->ob_type->tp_name);

  // Make sure that the reinterpret into a THPVariable* will be valid
  TORCH_CHECK(
      PyType_IsSubtype(type, &THPVariableType),
      "Creating a Tensor subclass from a class ",
      "that does not inherit from Tensor is not possible. Make sure your class inherits from Tensor.");

  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    auto v = (THPVariable*)obj;
    // TODO: named constructor to avoid default initialization
    new (&v->cdata) MaybeOwned<Variable>();
    if (c10::impl::HermeticPyObjectTLS::get_state()) {
      // Do NOT initialize pyobj field on the tensor, you own the C++
      v->cdata = MaybeOwned<Variable>::owned(std::move(_var));
      TORCH_INTERNAL_ASSERT(
          !check_has_torch_dispatch(obj),
          "While HermeticPyObject was enabled, we attempted to create a tensor "
          "subclass with __torch_dispatch__.  This violates the invariant that "
          "operations in HermeticPyObject have equivalent C++ implementations. "
          "If your operator registered from Python operator registration isn't "
          "doing anything strange, there may be an internal PyTorch bug involving "
          "not appropriately disabling TorchDispatchMode before executing "
          "Python op registration.");
    } else {
      // Normal codepath
      v->cdata = MaybeOwned<Variable>::owned(std::move(_var));
      const auto& var = THPVariable_Unpack(v);
      var.unsafeGetTensorImpl()->init_pyobj(
          self_interpreter.get(), obj, status);
      if (check_has_torch_dispatch(obj)) {
        var.unsafeGetTensorImpl()->set_python_dispatch(true);
      }
    }
  }
  return obj;
}

/// NOTE [ PyObject Traversal ]
///
/// PyObjects that are wrapping c++ objects can lead to non-trivial traverse
/// logic and it can be tricky to know what to traverse and when. This note
/// tries to clarify what is the danger here and a simple algorithm to choose
/// how to write the tp_traverse and tp_clear functions. If you're not already
/// familiar with how the CPython GC works, you should read this in-depth
/// description: https://devguide.python.org/garbage_collector/
///
/// The complexity for us comes from the fact that some c++ shared_ptr objects
/// own references to python objects and are also owned both by other python
/// objects and c++ objects. This means that to allow the GC to collect all
/// cycles, we need to properly implement the traverse/clear methods that take
/// into account these C++ ownership links.
///
/// The main danger here comes from the fact that, while all python-related code
/// is thread safe wrt the GC execution (thanks to the GIL), other threads might
/// be using our C++ objects arbitrarily which can lead to shared_ptr ref count
/// going up or down in between the different traverse/clear invocations. The
/// one constraint we add here that is not explicitly mentioned in the GC
/// description above is that for a given GC run (meaning while the GIL is
/// held), the traverse/clear pair should never report different ownership
/// relations: if traverse visited a given PyObject, then the clear within that
/// same GC run must still be the sole owner and clear that PyObject.
///
/// A more mechanical algorithm to know what to traverse/clear is as follows:
///   - Any field on this PyObject that contains a strong reference to another
///   PyObject
///     must be visited and cleared. An example of that is the "backward_hooks"
///     field of the THPVariable.
///   - Any field that contains a C++ object that is uniquely owned by this
///   PyObject (either
///     a unique_ptr or a shared_ptr with use_count==1) should have all the
///     PyObject it owns visited and cleared. An example would be here the
///     tensor hooks.
///   - If that uniquely owned C++ object also uniquely owns other C++ objects,
///   these should be
///     visited and cleared as well if they contain any PyObject.
///
/// Caveat: to avoid slow runtime, we limit the depth of this exploration of C++
/// objects in practice and we do not, for example, go through the whole
/// autograd graph, even if it is uniquely owned. This is a known place where
/// users can create noncollectable cycles as described in:
/// https://github.com/pytorch/pytorch/issues/7343
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
  mp = type->tp_members;
  for (i = 0; i < n; i++, mp++) {
    if (mp->type == T_OBJECT_EX) {
      char* addr = (char*)self + mp->offset;
      PyObject* obj = *(PyObject**)addr;
      if (obj != nullptr) {
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
  if (isResurrectable(var)) {
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
      // WARNING: The grad_fn traversal logic is very subtle, if you change
      // this, be very careful not to re-introduce this bug:
      // https://gist.github.com/zou3519/7ac92b84dd7d206dcc6eae55fee8372c

      // We ensure that we follow NOTE [ PyObject Traversal ] he by checking
      // that this python object is the sole owner of the underlying Tensor and
      // that this Tensor is the sole owner of its grad_fn. In this case, the
      // only way to get a new reference to the grad_fn is by using this python
      // object, which requires the GIL to be accessed. Note that this is only
      // valid as long as user don't share non-owning references across
      // different threads (which is crazy and should never be done).

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
        if (auto pyhook = dynamic_cast<PyFunctionTensorPreHook*>(hook.get())) {
          Py_VISIT(pyhook->dict);
        }
      }
    }
  }

  return 0;
}

int THPVariableMetaType_init(PyObject* cls, PyObject* args, PyObject* kwargs) {
  if (PyType_Type.tp_init(cls, args, kwargs) < 0) {
    return -1;
  }
  ((PyTypeObject*)cls)->tp_dealloc = (destructor)THPVariable_subclass_dealloc;
  ((PyTypeObject*)cls)->tp_traverse =
      (traverseproc)THPVariable_subclass_traverse;
  return 0;
}

namespace torch {
namespace autograd {

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
extern PyMethodDef variable_methods[];
extern void initTorchFunctions(PyObject* module);

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
} // namespace autograd
} // namespace torch

bool THPVariable_initModule(PyObject* module) {
  THPVariableMetaType.tp_base = &PyType_Type;
  if (PyType_Ready(&THPVariableMetaType) < 0)
    return false;
  Py_INCREF(&THPVariableMetaType);
  PyModule_AddObject(module, "_TensorMeta", (PyObject*)&THPVariableMetaType);

  static std::vector<PyMethodDef> methods;
  THPUtils_addPyMethodDefs(methods, torch::autograd::variable_methods);
  THPUtils_addPyMethodDefs(methods, extra_methods);
  THPVariableType.tp_methods = methods.data();
  if (PyType_Ready(&THPVariableType) < 0)
    return false;
  Py_INCREF(&THPVariableType);
  PyModule_AddObject(module, "_TensorBase", (PyObject*)&THPVariableType);
  torch::autograd::initTorchFunctions(module);
  torch::autograd::initTensorImplConversion(module);
  return true;
}

namespace {

bool isPythonTensor(const Tensor& tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(c10::DispatchKey::Python);
}

py::object torchDispatchFromTensorImpl(
    const c10::TensorImpl* self,
    const char* func_name,
    PyObject* torch_api_function,
    const char* module_name,
    // WARNING: MUST NOT BE TENSOR ARGS
    c10::SmallVector<py::object, 1> extra_args = {}) {
  if (torch_api_function == nullptr) {
    throw python_error();
  }
  TORCH_CHECK(
      PyGILState_Check(),
      "GIL must be held before you call parseIValuesToPyArgsKwargs");

  std::vector<py::handle> overloaded_args;
  // TODO: there should be a shorter way to spell this
  // TODO: fix the constness of target
  Tensor self_t = Tensor(
      c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>::
          unsafe_reclaim_from_nonowning(const_cast<c10::TensorImpl*>(self)));
  auto self_p = py::reinterpret_steal<py::object>(THPVariable_Wrap(self_t));
  // NB: this may not be a python tensor if you got here from a mode!
  // TORCH_INTERNAL_ASSERT(isPythonTensor(self_t));
  append_overloaded_tensor(&overloaded_args, self_p.ptr());
  auto args =
      py::reinterpret_steal<py::object>(PyTuple_New(1 + extra_args.size()));
  PyTuple_SET_ITEM(args.ptr(), 0, self_p.release().ptr());
  int64_t i = 1;
  for (auto& a : extra_args) {
    if (a.ptr() == nullptr)
      throw python_error();
    PyTuple_SET_ITEM(args.ptr(), i, std::move(a).release().ptr());
    i++;
  }

  py::dict kwargs;

  return py::reinterpret_steal<py::object>(
      handle_torch_function_no_python_arg_parser(
          overloaded_args,
          args.ptr(),
          kwargs.ptr(),
          func_name,
          torch_api_function,
          module_name,
          TorchFunctionName::TorchDispatch));
}

py::handle getTorchApiFunction(const c10::OperatorHandle& op) {
  return op.getPythonOp(getPyInterpreter(), [&]() -> PyObject* {
    // Parse the name into namespace and name (no overload_name)
    // TODO: put this into the library
    const auto& schema = op.schema();
    const auto& qualified_name = op.operator_name().name;
    const auto& overload_name = schema.overload_name();
    auto pos = qualified_name.find("::");
    TORCH_INTERNAL_ASSERT(pos != std::string::npos, qualified_name);
    // Make me some null terminated strings
    std::string ns_str = qualified_name.substr(0, pos);
    const char* ns = ns_str.c_str();
    const char* func_name = qualified_name.c_str() + pos + strlen("::");

    py::handle torch_api_function =
        py::module::import("torch").attr("ops").attr(ns).attr(func_name);
    if (overload_name == "") {
      return torch_api_function.attr("default").ptr();
    } else {
      return torch_api_function.attr(overload_name.c_str()).ptr();
    }
  });
}

void ConcretePyInterpreterVTable::dispatch(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) const {
  const auto& schema = op.schema();
  const auto num_arguments = schema.arguments().size();
  auto arguments = torch::jit::pop(*stack, num_arguments);

  // The plan: convert all the arguments back into PyObjects,
  // extracting out the tensor handles, then call
  // handle_torch_function_no_python_arg_parser
  // NB: at the point arguments are pushed to the stack, ALL defaults
  // are already present

  py::gil_scoped_acquire g;

  std::vector<py::handle> overloaded_args;
  py::handle torch_api_function_overload = getTorchApiFunction(op);

  // Find overloaded tensors
  for (const auto idx : c10::irange(arguments.size())) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isTensor()) {
      const auto& tensor = ivalue.toTensor();
      if (isPythonTensor(tensor)) {
        append_overloaded_tensor(&overloaded_args, py::cast(tensor).ptr());
      }
    } else if (ivalue.isList()) {
      const auto& list = ivalue.toListRef();
      for (const auto jdx : c10::irange(list.size())) {
        const auto& nv = list[jdx];
        if (nv.isTensor()) {
          const auto& tensor = nv.toTensor();
          if (isPythonTensor(tensor)) {
            append_overloaded_tensor(&overloaded_args, py::cast(tensor).ptr());
          }
        }
      }
    }
  }

  auto args_kwargs = parseIValuesToPyArgsKwargs(op, arguments);
  auto args = std::move(args_kwargs.first);
  auto kwargs = std::move(args_kwargs.second);

  PyObject* obj = handle_torch_function_no_python_arg_parser(
      overloaded_args,
      args.ptr(),
      kwargs.ptr(),
      nullptr,
      torch_api_function_overload.ptr(),
      nullptr,
      TorchFunctionName::TorchDispatch);
  pushPyOutToStack(
      op, stack, py::reinterpret_steal<py::object>(obj), "__torch_dispatch__");
}

void ConcretePyInterpreterVTable::python_dispatcher(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet ks,
    torch::jit::Stack* stack) const {
  py::gil_scoped_acquire g;
  py::handle torch_api_function_overload = getTorchApiFunction(op);

  c10::DispatchKey k = ks.highestPriorityTypeId();
  auto handler = torch_api_function_overload.attr(toString(k));
  if (handler.ptr() == nullptr) {
    throw python_error();
  }
  if (py::isinstance<c10::DispatchKey>(handler)) {
    // NB: not redispatch, as that will permanently remove the python
    // dispatcher for subsequent redispatches
    op.callBoxedForDispatchKey(py::cast<c10::DispatchKey>(handler), *stack);
    return;
  }

  const auto& schema = op.schema();
  const auto num_arguments = schema.arguments().size();
  auto arguments = torch::jit::pop(*stack, num_arguments);

  auto args_kwargs = parseIValuesToPyArgsKwargs(op, arguments);
  auto args = std::move(args_kwargs.first);
  auto kwargs = std::move(args_kwargs.second);

  py::object obj = py::reinterpret_steal<py::object>(
      PyObject_Call(handler.ptr(), args.ptr(), kwargs.ptr()));

  if (obj.ptr() == nullptr) {
    throw python_error();
  }

  pushPyOutToStack(op, stack, std::move(obj), "Python dispatcher");
}

c10::intrusive_ptr<TensorImpl> ConcretePyInterpreterVTable::detach(
    const c10::TensorImpl* self) const {
  pybind11::gil_scoped_acquire gil;
  at::impl::MaybeSetTLSOnEntryGuard guard;

  auto out = torchDispatchFromTensorImpl(
      self,
      "detach",
      py::module::import("torch")
          .attr("ops")
          .attr("aten")
          .attr("detach")
          .attr("default")
          .ptr(),
      "torch.ops.aten");

  TORCH_CHECK(
      THPVariable_Check(out.ptr()),
      "detach returned invalid type ",
      py::detail::get_fully_qualified_tp_name(Py_TYPE(out.ptr())),
      ", expected Tensor");
  const Tensor& res_t = THPVariable_Unpack(out.ptr());
  return res_t.getIntrusivePtr();
}

bool ConcretePyInterpreterVTable::is_contiguous(
    const c10::TensorImpl* self,
    at::MemoryFormat memory_format) const {
  pybind11::gil_scoped_acquire gil;
  at::impl::MaybeSetTLSOnEntryGuard guard;

  py::object out;
  if (memory_format == at::MemoryFormat::Contiguous) {
    // For backwards compatibility
    out = torchDispatchFromTensorImpl(
        self,
        "is_contiguous",
        py::module::import("torch")
            .attr("ops")
            .attr("aten")
            .attr("is_contiguous")
            .attr("default")
            .ptr(),
        "torch.ops.aten");
  } else {
    out = torchDispatchFromTensorImpl(
        self,
        "is_contiguous",
        py::module::import("torch")
            .attr("ops")
            .attr("aten")
            .attr("is_contiguous")
            .attr("memory_format")
            .ptr(),
        "torch.ops.aten",
        {py::cast(memory_format)});
  }

  if (out.is_none()) {
    return self->is_contiguous_default(memory_format);
  }

  TORCH_CHECK(
      PyBool_Check(out.ptr()),
      "is_contiguous returned invalid type ",
      py::detail::get_fully_qualified_tp_name(Py_TYPE(out.ptr())),
      ", expected bool");

  return PyObject_IsTrue(out.ptr());
}

bool ConcretePyInterpreterVTable::is_strides_like(
    const c10::TensorImpl* self,
    at::MemoryFormat memory_format) const {
  pybind11::gil_scoped_acquire gil;
  at::impl::MaybeSetTLSOnEntryGuard guard;

  auto out = torchDispatchFromTensorImpl(
      self,
      "is_strides_like",
      py::module::import("torch")
          .attr("ops")
          .attr("aten")
          // NB: intentionally suffixed with _format to avoid
          // triggering matches against "_like" suffix
          .attr("is_strides_like_format")
          .attr("default")
          .ptr(),
      "torch.ops.aten",
      {py::cast(memory_format)});

  if (out.is_none()) {
    return self->is_strides_like_default(memory_format);
  }

  TORCH_CHECK(
      PyBool_Check(out.ptr()),
      "is_strides_like_format returned invalid type ",
      py::detail::get_fully_qualified_tp_name(Py_TYPE(out.ptr())),
      ", expected bool");

  return PyObject_IsTrue(out.ptr());
}

bool ConcretePyInterpreterVTable::is_non_overlapping_and_dense(
    const c10::TensorImpl* self) const {
  pybind11::gil_scoped_acquire gil;
  at::impl::MaybeSetTLSOnEntryGuard guard;

  auto out = torchDispatchFromTensorImpl(
      self,
      "is_non_overlapping_and_dense",
      py::module::import("torch")
          .attr("ops")
          .attr("aten")
          .attr("is_non_overlapping_and_dense")
          .attr("default")
          .ptr(),
      "torch.ops.aten");

  if (out.is_none()) {
    return self->is_non_overlapping_and_dense_default();
  }

  TORCH_CHECK(
      PyBool_Check(out.ptr()),
      "is_non_overlapping_and_dense returned invalid type ",
      py::detail::get_fully_qualified_tp_name(Py_TYPE(out.ptr())),
      ", expected bool");

  return PyObject_IsTrue(out.ptr());
}

int64_t ConcretePyInterpreterVTable::dim(const c10::TensorImpl* self) const {
  pybind11::gil_scoped_acquire gil;
  at::impl::MaybeSetTLSOnEntryGuard guard;

  auto out = torchDispatchFromTensorImpl(
      self,
      "dim",
      py::module::import("torch")
          .attr("ops")
          .attr("aten")
          .attr("dim")
          .attr("default")
          .ptr(),
      "torch.ops.aten");

  TORCH_CHECK(
      PyLong_Check(out.ptr()),
      "dim returned invalid type ",
      py::detail::get_fully_qualified_tp_name(Py_TYPE(out.ptr())),
      ", expected int");

  return THPUtils_unpackLong(out.ptr());
}

c10::Device ConcretePyInterpreterVTable::device(
    const c10::TensorImpl* self) const {
  pybind11::gil_scoped_acquire gil;
  at::impl::MaybeSetTLSOnEntryGuard guard;

  auto out = torchDispatchFromTensorImpl(
      self,
      "device",
      py::module::import("torch")
          .attr("ops")
          .attr("prim")
          .attr("device")
          .attr("default")
          .ptr(),
      "torch.ops.prim");

  return toDevice(out.ptr());
}

c10::IntArrayRef ConcretePyInterpreterVTable::strides(
    const c10::TensorImpl* self) const {
  pybind11::gil_scoped_acquire gil;
  at::impl::MaybeSetTLSOnEntryGuard guard;

  auto out = torchDispatchFromTensorImpl(
      self,
      "stride",
      py::module::import("torch")
          .attr("ops")
          .attr("aten")
          .attr("stride")
          .attr("default")
          .ptr(),
      "torch.ops.aten");

  if (out.is_none()) {
    TORCH_CHECK(
        !self->has_symbolic_sizes_strides(),
        "Cannot call strides on a tensor with symbolic shapes/strides");
    return self->strides_default();
  }

  py::object values = py::reinterpret_steal<py::object>(out.ptr());

  c10::optional<PyObject*> mb_obj = self->check_pyobj(getPyInterpreter());
  TORCH_CHECK(
      mb_obj.has_value(), "Tensor subclass's PyInterpreter has no value");
  PyObject* subclass = *mb_obj;
  Py_INCREF(subclass);
  py::object sub = py::reinterpret_steal<py::object>(subclass);

  py::object os = py::module_::import("torch").attr("overrides");
  py::function get_buffer =
      py::reinterpret_borrow<py::function>(os.attr("get_buffer"));
  auto buffer = get_buffer(sub, values, "stride");
  auto result = THPUtils_unpackLongs(buffer.ptr());
  int64_t* start = (int64_t*)result[0];
  int64_t len = result[1];

  return c10::IntArrayRef(start, len);
}

static std::vector<int64_t> values_from_buffer(
    const c10::TensorImpl* self,
    py::handle values) {
  c10::TensorImpl* ptr = const_cast<c10::TensorImpl*>(self);
  c10::optional<PyObject*> mb_obj = ptr->check_pyobj(getPyInterpreter());
  TORCH_CHECK(
      mb_obj.has_value(), "Tensor subclass's PyInterpreter has no value");

  py::object os = py::module_::import("torch").attr("overrides");
  py::function get_buffer =
      py::reinterpret_borrow<py::function>(os.attr("get_buffer"));
  auto buffer = get_buffer(py::handle(*mb_obj), values, "size");
  auto result = THPUtils_unpackLongs(buffer.ptr());
  return result;
}

c10::IntArrayRef ConcretePyInterpreterVTable::sizes(
    const c10::TensorImpl* self) const {
  pybind11::gil_scoped_acquire gil;
  at::impl::MaybeSetTLSOnEntryGuard guard;

  auto out = torchDispatchFromTensorImpl(
      self,
      "size",
      py::module::import("torch")
          .attr("ops")
          .attr("aten")
          .attr("size")
          .attr("default")
          .ptr(),
      "torch.ops.aten");

  if (out.is_none()) {
    TORCH_CHECK(
        !self->has_symbolic_sizes_strides(),
        "Cannot call sizes on a tensor with symbolic shapes/strides");
    return self->sizes_default();
  }

  py::object values = py::reinterpret_steal<py::object>(out.ptr());
  auto result = values_from_buffer(self, values);
  int64_t* start = (int64_t*)result[0];
  int64_t len = result[1];

  return c10::IntArrayRef(start, len);
}

c10::SymIntArrayRef ConcretePyInterpreterVTable::sym_sizes(
    const c10::TensorImpl* self) const {
  pybind11::gil_scoped_acquire gil;
  at::impl::MaybeSetTLSOnEntryGuard guard;
  HANDLE_TH_ERRORS
  auto out = torchDispatchFromTensorImpl(
      self,
      "sym_size",
      py::module::import("torch")
          .attr("ops")
          .attr("aten")
          .attr("sym_size")
          .attr("default")
          .ptr(),
      "torch.ops.aten");

  if (out.is_none()) {
    return self->sym_sizes_default();
  }
  // We need to squeeze SymIntNodes and ints into `SymInts`
  // since it's a format `sym_sizes()` are stored in
  TORCH_CHECK(
      py::isinstance<py::tuple>(out) || py::isinstance<py::list>(out),
      "Symshape must be a list or a tuple");
  py::list symints;
  for (auto it = out.begin(); it != out.end(); it++) {
    auto elm = *it;
    auto si = py::cast<c10::SymInt>(elm);
    // TODO: the buffer will need to be made owning later
    symints.append(si.as_int_unchecked());
  }

  auto result = values_from_buffer(self, symints);
  c10::SymInt* start = (c10::SymInt*)result[0];
  int64_t len = result[1];

  return c10::SymIntArrayRef(start, len);
  END_HANDLE_TH_ERRORS_PYBIND
}

c10::Layout ConcretePyInterpreterVTable::layout(
    const c10::TensorImpl* self) const {
  pybind11::gil_scoped_acquire gil;
  at::impl::MaybeSetTLSOnEntryGuard guard;
  auto out = torchDispatchFromTensorImpl(
      self,
      "layout",
      py::module::import("torch")
          .attr("ops")
          .attr("prim")
          .attr("layout")
          .attr("default")
          .ptr(),
      "torch.ops.prim");

  TORCH_CHECK(
      THPLayout_Check(out.ptr()),
      "layout returned invalid type ",
      py::detail::get_fully_qualified_tp_name(Py_TYPE(out.ptr())),
      ", expected Layout");

  return toLayout(out.ptr());
}

c10::SymInt ConcretePyInterpreterVTable::sym_numel(
    const c10::TensorImpl* self) const {
  pybind11::gil_scoped_acquire gil;
  at::impl::MaybeSetTLSOnEntryGuard guard;
  auto out = torchDispatchFromTensorImpl(
      self,
      "sym_numel",
      py::module::import("torch")
          .attr("ops")
          .attr("aten")
          .attr("sym_numel")
          .attr("default")
          .ptr(),
      "torch.ops.aten");

  if (out.is_none()) {
    TORCH_CHECK(
        !self->has_symbolic_sizes_strides(),
        "Cannot call numel on a tensor with symbolic shapes/strides");
    return self->sym_numel_default();
  }
  return torch::is_symint(out) ? out.cast<c10::SymInt>()
                               : c10::SymInt{py::cast<int64_t>(out)};
}

c10::SymInt ConcretePyInterpreterVTable::sym_storage_offset(
    const c10::TensorImpl* self) const {
  pybind11::gil_scoped_acquire gil;
  at::impl::MaybeSetTLSOnEntryGuard guard;
  auto out = torchDispatchFromTensorImpl(
      self,
      "sym_storage_offset",
      py::module::import("torch")
          .attr("ops")
          .attr("aten")
          .attr("sym_storage_offset")
          .attr("default")
          .ptr(),
      "torch.ops.aten");

  if (out.is_none()) {
    return self->sym_storage_offset_default();
  }
  return torch::is_symint(out) ? out.cast<c10::SymInt>()
                               : c10::SymInt{py::cast<int64_t>(out)};
}

c10::SymIntArrayRef ConcretePyInterpreterVTable::sym_strides(
    const c10::TensorImpl* self) const {
  pybind11::gil_scoped_acquire gil;
  at::impl::MaybeSetTLSOnEntryGuard guard;
  HANDLE_TH_ERRORS
  auto out = torchDispatchFromTensorImpl(
      self,
      "sym_stride",
      py::module::import("torch")
          .attr("ops")
          .attr("aten")
          .attr("sym_stride")
          .attr("default")
          .ptr(),
      "torch.ops.aten");

  if (out.is_none()) {
    return self->sym_strides_default();
  }
  // We need to squeeze SymIntNodes and ints into `SymInts`
  // since it's a format `sym_strides()` are stored in
  TORCH_CHECK(
      py::isinstance<py::tuple>(out) || py::isinstance<py::list>(out),
      "Symshape must be a list or a tuple");
  py::list symints;
  for (auto it = out.begin(); it != out.end(); it++) {
    auto elm = *it;
    auto si = torch::is_symint(elm) ? elm.cast<c10::SymInt>()
                                    : c10::SymInt{py::cast<int64_t>(elm)};
    symints.append(si.as_int_unchecked());
  }

  auto result = values_from_buffer(self, symints);
  c10::SymInt* start = (c10::SymInt*)result[0];
  int64_t len = result[1];

  return c10::SymIntArrayRef(start, len);
  END_HANDLE_TH_ERRORS_PYBIND
}

} // anonymous namespace
