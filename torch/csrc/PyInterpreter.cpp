#include <ATen/core/PythonFallbackKernel.h>
#include <ATen/core/PythonOpRegistrationTrampoline.h>
#include <torch/csrc/PyInterpreter.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_dispatch.h>

#include <string>

using namespace torch;
using namespace at;
using namespace c10;

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

  // TODO: Need to make this work for StorageImpl too. I imagine I'll want to
  // operate upon a PyObjectSlot rather than a TensorImpl
  c10::intrusive_ptr<c10::TensorImpl> detach(
      const c10::TensorImpl* self) const override;

  void dispatch(const c10::OperatorHandle& op, torch::jit::Stack* stack)
      const override;
  void reportErrorCallback(PyObject* callback, DispatchKey key) const override;
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

  bool is_contiguous(const c10::TensorImpl* self, at::MemoryFormat)
      const override;
  bool is_strides_like(const c10::TensorImpl* self, at::MemoryFormat)
      const override;
  bool is_non_overlapping_and_dense(const c10::TensorImpl* self) const override;
  c10::Device device(const c10::TensorImpl* self) const override;
  int64_t dim(const c10::TensorImpl* self) const override;
  c10::IntArrayRef strides(const c10::TensorImpl* self) const override;
  c10::IntArrayRef sizes(const c10::TensorImpl* self) const override;
  c10::SymIntArrayRef sym_sizes(const c10::TensorImpl* self) const override;
  c10::Layout layout(const c10::TensorImpl* self) const override;
  c10::SymInt sym_numel(const c10::TensorImpl* self) const override;
  c10::SymIntArrayRef sym_strides(const c10::TensorImpl* self) const override;
  c10::SymInt sym_storage_offset(const c10::TensorImpl* self) const override;

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

  void reset_backward_hooks(const c10::TensorImpl* self) const override;

  static ConcretePyInterpreterVTable* instance() {
    static ConcretePyInterpreterVTable s;
    return &s;
  }
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

  std::vector<PyObject*> overloaded_args;
  // TODO: there should be a shorter way to spell this
  // TODO: fix the constness of target
  at::Tensor self_t = at::Tensor(
      c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>::
          unsafe_reclaim_from_nonowning(const_cast<c10::TensorImpl*>(self)));
  auto self_p =
      py::reinterpret_steal<py::object>(THPVariable_Wrap(std::move(self_t)));
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
    ((THPVariable*)pyobj)->cdata = c10::MaybeOwned<torch::autograd::Variable>();
  }
  Py_DECREF(pyobj);
};

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
    if (overload_name.empty()) {
      return torch_api_function.attr("default").ptr();
    } else {
      return torch_api_function.attr(overload_name.c_str()).ptr();
    }
  });
}

bool isPythonTensor(const at::Tensor& tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(c10::DispatchKey::Python);
}

void ConcretePyInterpreterVTable::reportErrorCallback(
    PyObject* callback,
    DispatchKey key) const {
  py::gil_scoped_acquire g;
  auto func = py::reinterpret_borrow<py::object>(callback);
  // Not all DispatchKeys are pybind'ed into Python and we do not have infra
  // to ensure this, so just pass a string back to Python.
  func(c10::toString(key));
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

  std::vector<PyObject*> overloaded_args;
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
  // TODO: if necessary, can optimize to cache the cache lookup
  // TODO: if necessary, can optimize OpOverload to have slots
  auto cache = py::dict(torch_api_function_overload.attr("_dispatch_cache"));
  if (cache.ptr() == nullptr) {
    throw python_error();
  }

  c10::DispatchKey k = ks.highestPriorityTypeId();
  // TODO: allow this to be non-owning
  auto handler = py::reinterpret_borrow<py::object>(
      PyDict_GetItem(cache.ptr(), py::cast(k).ptr()));
  if (handler.ptr() == nullptr) {
    // Slow path
    handler = torch_api_function_overload.attr("_get_dispatch")(k);
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

c10::intrusive_ptr<c10::TensorImpl> ConcretePyInterpreterVTable::detach(
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
  const at::Tensor& res_t = THPVariable_Unpack(out.ptr());
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

  c10::optional<PyObject*> mb_obj =
      self->pyobj_slot()->check_pyobj(getPyInterpreter());
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

template <typename T>
static void attach_array_lifetime_to_tensor(
    const c10::TensorImpl* tensor,
    T* array_data,
    const char* attr_name) {
  c10::TensorImpl* t_ptr = const_cast<c10::TensorImpl*>(tensor);
  c10::optional<PyObject*> mb_obj =
    t_ptr->pyobj_slot()->check_pyobj(getPyInterpreter());
  TORCH_CHECK(
      mb_obj.has_value(), "Tensor subclass's PyInterpreter has no value");
  auto capsule = py::capsule(array_data, [](void* p) {
      delete[] reinterpret_cast<T*>(p);
  });
  py::handle(mb_obj.value()).attr(attr_name) = capsule;
}

c10::IntArrayRef ConcretePyInterpreterVTable::sizes(
    const c10::TensorImpl* self) const {
  pybind11::gil_scoped_acquire gil;
  at::impl::MaybeSetTLSOnEntryGuard guard;
  HANDLE_TH_ERRORS
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
  TORCH_CHECK(
      py::isinstance<py::tuple>(out) || py::isinstance<py::list>(out),
      "sizes must be a list or a tuple");
  int64_t len = py::len(out);
  // Cleanup happens in the callback of a py::capsule attached to tensor
  // See attach_array_lifetime_to_tensor.
  int64_t* ptr = new int64_t[len];
  int64_t idx = 0;
  for (auto it = out.begin(); it != out.end(); ++it, ++idx) {
    ptr[idx] = py::cast<int64_t>(*it);
  }
  attach_array_lifetime_to_tensor<int64_t>(self, ptr, "_sizes_capsule");
  return c10::IntArrayRef(ptr, len);
  END_HANDLE_TH_ERRORS_PYBIND
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
  TORCH_CHECK(
      py::isinstance<py::tuple>(out) || py::isinstance<py::list>(out),
      "sym_size must be a list or a tuple");
  int64_t len = py::len(out);
  // Cleanup happens in the callback of a py::capsule attached to tensor
  // See attach_array_lifetime_to_tensor.
  c10::SymInt* ptr = new c10::SymInt[len];
  int64_t idx = 0;
  for (auto it = out.begin(); it != out.end(); ++it, ++idx) {
    ptr[idx] = py::cast<c10::SymInt>(*it);
  }
  attach_array_lifetime_to_tensor<c10::SymInt>(self, ptr, "_sym_sizes_capsule");
  return c10::SymIntArrayRef(ptr, len);
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
      "sym_strides must be a list or a tuple");
  int64_t len = py::len(out);
  // Cleanup happens in the callback of a py::capsule attached to tensor
  // See attach_array_lifetime_to_tensor.
  c10::SymInt* ptr = new c10::SymInt[len];
  int64_t idx = 0;
  for (auto it = out.begin(); it != out.end(); ++it, ++idx) {
    ptr[idx] = py::cast<c10::SymInt>(*it);
  }
  attach_array_lifetime_to_tensor<c10::SymInt>(self, ptr, "_sym_strides_capsule");
  return c10::SymIntArrayRef(ptr, len);
  END_HANDLE_TH_ERRORS_PYBIND
}

PyInterpreterHolder self_interpreter;

void ConcretePyInterpreterVTable::reset_backward_hooks(
    const c10::TensorImpl* self) const {
  pybind11::gil_scoped_acquire gil;
  at::impl::MaybeSetTLSOnEntryGuard guard;
  HANDLE_TH_ERRORS
  Tensor self_t = Tensor(
      c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>::
          unsafe_reclaim_from_nonowning(const_cast<c10::TensorImpl*>(self)));
  auto self_p =
      py::reinterpret_steal<py::object>(THPVariable_Wrap(std::move(self_t)));
  PyObject_SetAttrString(self_p.ptr(), "_backward_hooks", Py_None);
  END_HANDLE_TH_ERRORS_PYBIND
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
