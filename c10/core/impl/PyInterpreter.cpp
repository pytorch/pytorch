#include <c10/core/SymIntArrayRef.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/impl/PyInterpreter.h>
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-parameter")
namespace c10::impl {

struct NoopPyInterpreterVTable final : public PyInterpreterVTable {
  std::string name() const override {
    return "<unloaded interpreter>";
  }

  void incref(PyObject* pyobj) const override {} // do nothing

  void decref(PyObject* pyobj) const override {} // do nothing

  bool try_incref(const c10::impl::PyObjectSlot& pyobj_slot) const override {
    return false;
  }

#define PANIC(m)              \
  TORCH_INTERNAL_ASSERT(      \
      0,                      \
      "attempted to call " #m \
      " on a Tensor with nontrivial PyObject after corresponding interpreter died")

  size_t refcnt(PyObject* pyobj) const override {
    PANIC(refcnt);
  }

  c10::intrusive_ptr<TensorImpl> detach(const TensorImpl* self) const override {
    PANIC(detach);
  }

  void dispatch(const c10::OperatorHandle& op, torch::jit::Stack* stack)
      const override {
    PANIC(dispatch);
  }

  void reportErrorCallback(PyObject* callback, DispatchKey key) const override {
    PANIC(reportErrorCallback);
  }

  void python_op_registration_trampoline(
      const c10::OperatorHandle& op,
      c10::DispatchKey /*unused*/,
      c10::DispatchKeySet keyset,
      torch::jit::Stack* stack,
      bool with_keyset,
      bool with_op) const override {
    PANIC(python_op_registration_trampoline);
  }

  void throw_abstract_impl_not_imported_error(
      std::string opname,
      const char* pymodule,
      const char* context) const override {
    PANIC(throw_abstract_impl_not_imported_error);
  }

  void python_dispatcher(
      const c10::OperatorHandle& op,
      c10::DispatchKeySet /*unused*/,
      torch::jit::Stack* stack) const override {
    PANIC(python_dispatcher);
  }

  bool is_contiguous(const TensorImpl* self, at::MemoryFormat /*unused*/)
      const override {
    PANIC(is_contiguous);
  }
  c10::SymBool sym_is_contiguous(
      const TensorImpl* self,
      at::MemoryFormat /*unused*/) const override {
    PANIC(sym_is_contiguous);
  }
  bool is_strides_like(const TensorImpl* self, at::MemoryFormat /*unused*/)
      const override {
    PANIC(is_strides_like);
  }
  bool is_non_overlapping_and_dense(const TensorImpl* self) const override {
    PANIC(is_non_overlapping_and_dense);
  }
  c10::Device device(const TensorImpl* self) const override {
    PANIC(device);
  }
  int64_t dim(const TensorImpl* self) const override {
    PANIC(dim);
  }
  c10::IntArrayRef strides(const TensorImpl* self) const override {
    PANIC(strides);
  }
  c10::IntArrayRef sizes(const TensorImpl* self) const override {
    PANIC(sizes);
  }
  c10::SymIntArrayRef sym_sizes(const TensorImpl* self) const override {
    PANIC(sym_sizes);
  }
  c10::Layout layout(const TensorImpl* self) const override {
    PANIC(layout);
  }
  int64_t numel(const TensorImpl* self) const override {
    PANIC(numel);
  }
  c10::SymInt sym_numel(const TensorImpl* self) const override {
    PANIC(sym_numel);
  }
  c10::SymIntArrayRef sym_strides(const TensorImpl* self) const override {
    PANIC(sym_strides);
  }
  c10::SymInt sym_storage_offset(const TensorImpl* self) const override {
    PANIC(sym_storage_offset);
  }

  // Just swallow the event, don't do anything
  void trace_gpu_event_creation(c10::DeviceType device_type, uintptr_t event)
      const override {}
  void trace_gpu_event_deletion(c10::DeviceType device_type, uintptr_t event)
      const override {}
  void trace_gpu_event_record(
      c10::DeviceType device_type,
      uintptr_t event,
      uintptr_t stream) const override {}
  void trace_gpu_event_wait(
      c10::DeviceType device_type,
      uintptr_t event,
      uintptr_t stream) const override {}
  void trace_gpu_memory_allocation(c10::DeviceType device_type, uintptr_t ptr)
      const override {}
  void trace_gpu_memory_deallocation(c10::DeviceType device_type, uintptr_t ptr)
      const override {}
  void trace_gpu_stream_creation(c10::DeviceType device_type, uintptr_t stream)
      const override {}
  void trace_gpu_device_synchronization(
      c10::DeviceType device_type) const override {}
  void trace_gpu_stream_synchronization(
      c10::DeviceType device_type,
      uintptr_t stream) const override {}
  void trace_gpu_event_synchronization(
      c10::DeviceType device_type,
      uintptr_t event) const override {}

  void reset_backward_hooks(const TensorImpl* self) const override {
    PANIC(reset_backward_hooks);
  }
};

// Construct this in Global scope instead of within `disarm`
// where it will be only initialized first time `disarm` is called.
// This increases the likelihood `noop_vtable` lives longer than
// any object that refers to it.

// If `noop_vtable` goes out of scope first, other objects will have dangling
// reference to it.
static NoopPyInterpreterVTable noop_vtable;

void PyInterpreter::disarm() noexcept {
  vtable_ = &noop_vtable;
}

} // namespace c10::impl
C10_DIAGNOSTIC_POP()
