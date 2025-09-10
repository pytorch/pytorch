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

namespace torch::detail {

namespace {

// NB: This is a macro and not a template function (like it was before)
// because passing in constexpr char* as template argument breaks some
// versions of MSVC that are being used internally at Meta.
// MSVC 14.16.27023 (vs2017_15.9)
#define CONCRETE_GPU_TRACE(device_type, func_name, ...)                       \
  at::impl::MaybeSetTLSOnEntryGuard guard;                                    \
  if (Py_IsInitialized()) {                                                   \
    pybind11::gil_scoped_acquire gil;                                         \
    try {                                                                     \
      /* Masquerade hip as cuda because hip uses `torch.cuda` module. */      \
      if (device_type == at::kHIP) {                                          \
        device_type = at::kCUDA;                                              \
      }                                                                       \
      std::string module_name = "torch." + DeviceTypeName(device_type, true); \
      py::module mod = py::module::import(module_name.c_str());               \
      py::object hook =                                                       \
          mod.attr("_gpu_trace").attr(func_name).attr("fire_callbacks");      \
      hook(__VA_ARGS__);                                                      \
    } catch (const std::exception& e) {                                       \
      LOG(ERROR) << device_type                                               \
                 << " trace hook execution failed: " << e.what();             \
    }                                                                         \
  }

struct ConcretePyInterpreterVTable final
    : public c10::impl::PyInterpreterVTable {
  std::string name() const override;

  void incref(PyObject* pyobj) const override;
  void decref(PyObject* pyobj, bool has_pyobj_slot) const override;

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
      c10::DispatchKeySet keyset,
      torch::jit::Stack* stack,
      bool with_keyset,
      bool with_op) const override {
    torch::impl::dispatch::python_op_registration_trampoline_impl(
        op, key, keyset, stack, with_keyset, with_op);
  }
  void throw_abstract_impl_not_imported_error(
      std::string opname,
      const char* pymodule,
      const char* context) const override {
    py::gil_scoped_acquire gil;
    pybind11::module::import("torch._utils_internal")
        .attr("throw_abstract_impl_not_imported_error")(
            opname, pymodule, context);
  }

  bool is_contiguous(const c10::TensorImpl* self, at::MemoryFormat)
      const override;
  c10::SymBool sym_is_contiguous(const c10::TensorImpl* self, at::MemoryFormat)
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
  int64_t numel(const c10::TensorImpl* self) const override;
  c10::SymInt sym_numel(const c10::TensorImpl* self) const override;
  c10::SymIntArrayRef sym_strides(const c10::TensorImpl* self) const override;
  c10::SymInt sym_storage_offset(const c10::TensorImpl* self) const override;

  void trace_gpu_event_creation(at::DeviceType device_type, uintptr_t event)
      const override {
    CONCRETE_GPU_TRACE(device_type, "EventCreationCallbacks", event);
  }
  void trace_gpu_event_deletion(at::DeviceType device_type, uintptr_t event)
      const override {
    CONCRETE_GPU_TRACE(device_type, "EventDeletionCallbacks", event);
  }
  void trace_gpu_event_record(
      at::DeviceType device_type,
      uintptr_t event,
      uintptr_t stream) const override {
    CONCRETE_GPU_TRACE(device_type, "EventRecordCallbacks", event, stream);
  }
  void trace_gpu_event_wait(
      at::DeviceType device_type,
      uintptr_t event,
      uintptr_t stream) const override {
    CONCRETE_GPU_TRACE(device_type, "EventWaitCallbacks", event, stream);
  }
  void trace_gpu_memory_allocation(at::DeviceType device_type, uintptr_t ptr)
      const override {
    CONCRETE_GPU_TRACE(device_type, "MemoryAllocationCallbacks", ptr);
  }
  void trace_gpu_memory_deallocation(at::DeviceType device_type, uintptr_t ptr)
      const override {
    CONCRETE_GPU_TRACE(device_type, "MemoryDeallocationCallbacks", ptr);
  }
  void trace_gpu_stream_creation(at::DeviceType device_type, uintptr_t stream)
      const override {
    CONCRETE_GPU_TRACE(device_type, "StreamCreationCallbacks", stream);
  }
  void trace_gpu_device_synchronization(
      at::DeviceType device_type) const override {
    CONCRETE_GPU_TRACE(device_type, "DeviceSynchronizationCallbacks");
  }
  void trace_gpu_stream_synchronization(
      at::DeviceType device_type,
      uintptr_t stream) const override {
    CONCRETE_GPU_TRACE(device_type, "StreamSynchronizationCallbacks", stream);
  }
  void trace_gpu_event_synchronization(
      at::DeviceType device_type,
      uintptr_t event) const override {
    CONCRETE_GPU_TRACE(device_type, "EventSynchronizationCallbacks", event);
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
            ConcretePyInterpreterVTable::instance())),
        is_main_interpreter_(
            at::impl::PythonOpRegistrationTrampoline::registerInterpreter(
                impl_)) {}
  PyInterpreterHolder(const PyInterpreterHolder&) = delete;
  PyInterpreterHolder(PyInterpreterHolder&&) = delete;
  PyInterpreterHolder& operator=(const PyInterpreterHolder&) = delete;
  PyInterpreterHolder& operator=(PyInterpreterHolder&&) = delete;
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
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      unsafe_reclaim_from_nonowning(const_cast<c10::TensorImpl*>(self)));
  auto self_p = py::reinterpret_steal<py::object>(THPVariable_Wrap(self_t));
  // NB: this may not be a python tensor if you got here from a mode!
  // TORCH_INTERNAL_ASSERT(isPythonTensor(self_t));
  append_overloaded_tensor(&overloaded_args, self_p.ptr());
  auto args = py::reinterpret_steal<py::object>(
      PyTuple_New(static_cast<Py_ssize_t>(1 + extra_args.size())));
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

// NOTE [PyInterpreter::decref takes a `has_pyobj_slot` arg]
// Before calling PyInterpreter::decref, we must statically know if the
// pyobj has a PyObjectSlot or not.
// - If it has a PyObjectSlot, we need to be careful about PyObject resurrection
// - If it does not have a PyObjectSlot, we can freely decref
// One alternative to this is using PyObject_IsInstance
// to get at this information. However, we don't want to risk an incorrect
// `__instancecheck__` changing the semantics here.
void ConcretePyInterpreterVTable::decref(PyObject* pyobj, bool has_pyobj_slot)
    const {
  // Leak the pyobj if not initialized.  This can happen if we are running
  // exit handlers that are destructing tensors with residual (owned)
  // PyObjects stored in them.
  if (!Py_IsInitialized())
    return;

  pybind11::gil_scoped_acquire gil;
  // Two possibilities:
  // 1. We are decref-ing an object that has a PyObjectSlot, like a Tensor or
  // Storage. Then we must be careful about PyObject resurrection (see
  // THPVariable_clear).
  // 2. We are decref-ing some other Python object. We don't do
  // PyObject resurrection on non-Tensors, so we just carry on as usual
  if (has_pyobj_slot && Py_REFCNT(pyobj) > 1) {
    if (THPVariable_Check(pyobj)) {
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
      ((THPVariable*)pyobj)->cdata =
          c10::MaybeOwned<torch::autograd::Variable>();
    } else if (THPStorage_Check(pyobj)) {
      TORCH_WARN(
          "Deallocating UntypedStorage that still has live PyObject references.  "
          "This probably happened because you took out a weak reference to "
          "UntypedStorage and didn't call _fix_weakref() after dereferencing it.  "
          "Subsequent accesses to this storage via the PyObject will now fail.");
      ((THPStorage*)pyobj)->cdata = c10::MaybeOwned<c10::Storage>();
    }
  }
  Py_DECREF(pyobj);
}

void ConcretePyInterpreterVTable::incref(PyObject* pyobj) const {
  if (!Py_IsInitialized())
    return;
  pybind11::gil_scoped_acquire gil;
  Py_INCREF(pyobj);
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
  PyObject* raw_handler = nullptr;
  if (PyDict_GetItemRef(cache.ptr(), py::cast(k).ptr(), &raw_handler) < 0) {
    // There was an error that is not missing key (which would return 0)
    throw python_error();
  }
  auto handler = py::reinterpret_steal<py::object>(raw_handler);
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

c10::SymBool ConcretePyInterpreterVTable::sym_is_contiguous(
    const c10::TensorImpl* self,
    at::MemoryFormat memory_format) const {
  pybind11::gil_scoped_acquire gil;
  at::impl::MaybeSetTLSOnEntryGuard guard;

  py::object out;
  out = torchDispatchFromTensorImpl(
      self,
      "sym_is_contiguous",
      py::module::import("torch")
          .attr("ops")
          .attr("aten")
          .attr("sym_is_contiguous")
          .attr("default")
          .ptr(),
      "torch.ops.aten",
      {py::cast(memory_format)});

  if (out.is_none()) {
    return self->sym_is_contiguous_default(memory_format);
  }

  return torch::is_symbool(out) ? out.cast<c10::SymBool>()
                                : c10::SymBool{py::cast<bool>(out)};
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

static void set_tensor_attr_with_capsule(
    const c10::TensorImpl* tensor,
    py::capsule& capsule,
    const char* attr_name) {
  std::optional<PyObject*> mb_obj = tensor->pyobj_slot()->check_pyobj(
      /*ignore_hermetic_tls=*/false);
  TORCH_CHECK(
      mb_obj.has_value(), "Tensor subclass's PyInterpreter has no value");
  auto obj = mb_obj.value();
  py::handle(obj).attr(attr_name) = capsule;
}

// Note [Tensor Subclass custom size/stride caching strategy]
// Tensor subclasses can use __torch_dispatch__ to override size/stride calls.
// However, this presents a problem:
// (1) When you return a custom (maybe symbolic) size/stride
//     from python, we need to stash this fresh vector of ints/symints
//     somewhere so that it has the same lifetime as the tensor.
// (2) If the subclass experiences a metadata mutation,
//     this stashed vector is no longer valid, so we need to allocate a fresh
//     buffer to store the new sizes the next time someone asks for them.
//
// We handle this in the same way that `TensorImpl::sizes_default()`
// handles its buffer: we simply reallocate the buffer whenever
// the number of dimensions changes due to a resize.
// Notable, we do *not* reallocate the buffer if the values changed,
// but the number of dimensions stayed the same (e.g. `.transpose_()`).
template <typename T>
static c10::ArrayRef<T> get_set_cached_attr(
    const c10::TensorImpl* tensor,
    const char* base_attr_name,
    const py::object& obj) {
  std::optional<PyObject*> mb_obj =
      tensor->pyobj_slot()->check_pyobj(getPyInterpreter());
  TORCH_CHECK(
      mb_obj.has_value(), "Tensor subclass's PyInterpreter has no value");
  auto tensor_obj = mb_obj.value();
  auto buffer_len_attr_name = std::string(base_attr_name) + std::string("_len");

  bool is_buffer_allocated = false;
  size_t curr_size = 0;
  if (PyObject_HasAttrString(tensor_obj, buffer_len_attr_name.c_str())) {
    auto len_pyobj = py::handle(tensor_obj).attr(buffer_len_attr_name.c_str());
    curr_size = py::cast<size_t>(len_pyobj);
    is_buffer_allocated = true;
  }

  size_t new_size = py::len(obj);

  // We do the smallvector optimization here: any time the new_size is <=5,
  // we always allocate our buffer to size 5, so that if the next resize
  // is also to <=5 elements, we don't need to reallocate.
  // Note: I tried removing this optimization and tripped ASAN
  // in a batchnorm kernel here:
  // https://pipelinesghubeus21.actions.githubusercontent.com/mBh68xKhi8LyM7tp3vECvYXNFvuV4gyVGgmYCteuEZP9JH92QN/_apis/pipelines/1/runs/3373307/signedlogcontent/790?urlExpires=2023-09-15T21%3A13%3A51.4327798Z&urlSigningMethod=HMACV1&urlSignature=tDeX7ZqaARVU5NNwyr5yYqqkWq3A2j4z8FFdqYwGr0Q%3D@lint-ignore
  // We should fix this instead.
  bool needs_resize = false;
  // We need to resize if:
  // (1) we haven't allocated our buffer at all yet
  // (2) Our buffer size is different from the new size
  //     (note: we use the small vector optimization, where our buffer
  //     is always allocated to at least size 5, and any resizes
  //     within the <= 5 regime to not require a reallocation).
  auto is_smallvector = curr_size <= 5;
  needs_resize = !is_buffer_allocated || (is_smallvector && new_size > 5) ||
      (!is_smallvector && curr_size != new_size);
  if (needs_resize) {
    // If our current buffer is not the right size (either because we haven't
    // allocated it yet, or there was a metadata mutation that changed the
    // number of dims of the tensor), allocate a fresh buffer. Note that this
    // will trash the previous buffer if there already was one, invalidating any
    // existing SymIntArrayRef's from an old .sym_size() call.
    auto new_buffer_size = new_size;
    if (new_size <= 5) {
      // This is the smallvector optimization
      new_buffer_size = 5;
    }
    T* ptr = new T[new_buffer_size];
    auto capsule =
        py::capsule(ptr, [](void* p) { delete[] reinterpret_cast<T*>(p); });
    int64_t idx = 0;
    for (auto it = obj.begin(); it != obj.end(); ++it, ++idx) {
      ptr[idx] = py::cast<T>(*it);
    }
    // Set the buffer
    set_tensor_attr_with_capsule(tensor, capsule, base_attr_name);
    // Set the len buffer
    py::handle(tensor_obj).attr(buffer_len_attr_name.c_str()) = new_size;
  } else {
    TORCH_INTERNAL_ASSERT(PyObject_HasAttrString(tensor_obj, base_attr_name));
    auto curr_buffer_pyobj = py::handle(tensor_obj).attr(base_attr_name);
    void* buffer_pycapsule =
        PyCapsule_GetPointer(curr_buffer_pyobj.ptr(), nullptr);
    auto curr_buffer = reinterpret_cast<T*>(buffer_pycapsule);

    // Overwrite the buffer with our new values, but only if any of them changed
    // (due to a metadata mutation).
    // This is technically not thread safe, because the update happens lazily.
    // The original metadata mutation call on the tensor might have been thread
    // safe (e.g. a .resize_() call), but we won't actually mutate the size
    // buffer until the first call to .sizes() which the user might not access
    // in a thread-safe way. For now we are not explicitly locking, but maybe we
    // should.
    int64_t idx = 0;
    // Quick sanity assert that our buffer size is large enough
    // to compare against all the elements in the new buffer.
    size_t curr_buffer_size = 5;
    if (curr_buffer_size < curr_size) {
      curr_buffer_size = curr_size;
    }
    TORCH_INTERNAL_ASSERT(curr_buffer_size >= new_size);
    for (auto it = obj.begin(); it != obj.end(); ++it, ++idx) {
      auto actual_val = py::cast<T>(*it);
      if constexpr (std::is_same_v<T, c10::SymInt>) {
        // if our SymInts are symbolic, we are *not* doing an equality check on
        // the symints. we just want to see if the nodes are the same. this is
        // because we don't want to introduce any guards here.
        if (!curr_buffer[idx].is_same(actual_val)) {
          curr_buffer[idx] = actual_val;
        }
      } else {
        if (curr_buffer[idx] != actual_val) {
          curr_buffer[idx] = actual_val;
        }
      }
    }
  }

  // The correct data is now stored at the buffer - read and return it.
  auto curr_buffer_pyobj = py::handle(tensor_obj).attr(base_attr_name);
  void* buffer_pycapsule =
      PyCapsule_GetPointer(curr_buffer_pyobj.ptr(), nullptr);
  auto curr_buffer = reinterpret_cast<T*>(buffer_pycapsule);
  return c10::ArrayRef<T>(curr_buffer, new_size);
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
  TORCH_CHECK(
      py::isinstance<py::tuple>(out) || py::isinstance<py::list>(out),
      "strides must be a list or a tuple");
  auto updated_strides =
      get_set_cached_attr<int64_t>(self, "_strides_capsule", out);
  return updated_strides;
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

  auto updated_sizes =
      get_set_cached_attr<int64_t>(self, "_sizes_capsule", out);
  return updated_sizes;
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

  // See Note [Tensor Subclass custom size/stride caching strategy]
  auto updated_sym_sizes =
      get_set_cached_attr<c10::SymInt>(self, "_sym_sizes_capsule", out);
  return updated_sym_sizes;
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
      THPLayout_Check(out.ptr()) || PyLong_Check(out.ptr()),
      "layout returned invalid type ",
      py::detail::get_fully_qualified_tp_name(Py_TYPE(out.ptr())),
      ", expected Layout");

  if (THPLayout_Check(out.ptr())) {
    return toLayout(out.ptr());
  } else {
    return c10::Layout(py::cast<int64_t>(out));
  }
}

int64_t ConcretePyInterpreterVTable::numel(const c10::TensorImpl* self) const {
  pybind11::gil_scoped_acquire gil;
  at::impl::MaybeSetTLSOnEntryGuard guard;
  auto out = torchDispatchFromTensorImpl(
      self,
      "numel",
      py::module::import("torch")
          .attr("ops")
          .attr("aten")
          .attr("numel")
          .attr("default")
          .ptr(),
      "torch.ops.aten");

  if (out.is_none()) {
    TORCH_CHECK(
        !self->has_symbolic_sizes_strides(),
        "Cannot call sizes on a tensor with symbolic shapes/strides");
    return self->numel_default();
  }
  return py::cast<int64_t>(out);
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

  auto updated_sym_strides =
      get_set_cached_attr<c10::SymInt>(self, "_sym_strides_capsule", out);
  return updated_sym_strides;
  END_HANDLE_TH_ERRORS_PYBIND
}

void ConcretePyInterpreterVTable::reset_backward_hooks(
    const c10::TensorImpl* self) const {
  pybind11::gil_scoped_acquire gil;
  at::impl::MaybeSetTLSOnEntryGuard guard;
  HANDLE_TH_ERRORS
  Tensor self_t =
      Tensor(c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>::
                 // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
             unsafe_reclaim_from_nonowning(const_cast<c10::TensorImpl*>(self)));
  auto self_p = py::reinterpret_steal<py::object>(THPVariable_Wrap(self_t));
  PyObject_SetAttrString(self_p.ptr(), "_backward_hooks", Py_None);
  END_HANDLE_TH_ERRORS_PYBIND
}

std::string ConcretePyInterpreterVTable::name() const {
  std::stringstream ss;
  ss << getPyInterpreter();
  return ss.str();
}

PyInterpreterHolder self_interpreter;

} // anonymous namespace

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

} // namespace torch::detail

c10::impl::PyInterpreter* getPyInterpreter() {
  return torch::detail::self_interpreter.get();
}
