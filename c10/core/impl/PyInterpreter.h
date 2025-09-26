#pragma once

#include <c10/core/Device.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/macros/Export.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/python_stub.h>
#include <string>
#include <vector>

// Forward declarations

namespace c10 {
struct IValue;
class OperatorHandle;
struct TensorImpl;
} // namespace c10

namespace torch::jit {
using Stack = std::vector<c10::IValue>;
}

// Actual implementation

namespace c10::impl {

struct C10_API PyInterpreter;

struct C10_API PyInterpreterVTable {
  virtual ~PyInterpreterVTable() = default;

  // Report the name of this interpreter
  virtual std::string name() const = 0;

  // Run Py_INCREF on a PyObject.
  virtual void incref(PyObject* pyobj) const = 0;
  // Run Py_DECREF on a PyObject.  We DO NOT assume the GIL is held on call
  // See NOTE [PyInterpreter::decref takes a `has_pyobj_slot` arg]
  virtual void decref(PyObject* pyobj, bool has_pyobj_slot) const = 0;

  // Perform a detach by deferring to the __torch_dispatch__ implementation of
  // detach, which will also arrange for the PyObject to get copied in this
  // situation
  virtual c10::intrusive_ptr<TensorImpl> detach(
      const TensorImpl* self) const = 0;

  // Invoke the Python boxed fallback dispatch to go back into Python
  virtual void dispatch(const c10::OperatorHandle& op, torch::jit::Stack* stack)
      const = 0;

  virtual void reportErrorCallback(PyObject* callback, DispatchKey key)
      const = 0;

  // This is only invoked in the multipy/torchdeploy // codespell:ignore multipy
  // situation from pythonOpRegistrationTrampoline; this lets us get to the
  // Python interpreter to actually find the appropriate Python op registration
  // entry to call.
  virtual void python_op_registration_trampoline(
      const c10::OperatorHandle& op,
      c10::DispatchKey,
      c10::DispatchKeySet keyset,
      torch::jit::Stack* stack,
      bool with_keyset,
      bool with_op) const = 0;

  virtual void throw_abstract_impl_not_imported_error(
      std::string opname,
      const char* pymodule,
      const char* context) const = 0;

  // Invoke the Python dispatcher to handle this call
  virtual void python_dispatcher(
      const c10::OperatorHandle& op,
      c10::DispatchKeySet,
      torch::jit::Stack* stack) const = 0;

  virtual bool is_contiguous(const TensorImpl* self, at::MemoryFormat)
      const = 0;
  virtual c10::SymBool sym_is_contiguous(
      const TensorImpl* self,
      at::MemoryFormat) const = 0;
  virtual bool is_strides_like(const TensorImpl* self, at::MemoryFormat)
      const = 0;
  virtual bool is_non_overlapping_and_dense(const TensorImpl* self) const = 0;
  virtual c10::Device device(const TensorImpl* self) const = 0;
  virtual int64_t dim(const TensorImpl* self) const = 0;
  virtual c10::IntArrayRef strides(const TensorImpl* self) const = 0;
  virtual c10::IntArrayRef sizes(const TensorImpl* self) const = 0;
  virtual c10::SymIntArrayRef sym_sizes(const TensorImpl* self) const = 0;
  virtual c10::Layout layout(const TensorImpl* self) const = 0;
  virtual int64_t numel(const TensorImpl* self) const = 0;
  virtual c10::SymInt sym_numel(const TensorImpl* self) const = 0;
  virtual c10::SymIntArrayRef sym_strides(const TensorImpl* self) const = 0;
  virtual c10::SymInt sym_storage_offset(const TensorImpl* self) const = 0;

  virtual void trace_gpu_event_creation(
      c10::DeviceType device_type,
      uintptr_t event) const = 0;
  virtual void trace_gpu_event_deletion(
      c10::DeviceType device_type,
      uintptr_t event) const = 0;
  virtual void trace_gpu_event_record(
      c10::DeviceType device_type,
      uintptr_t event,
      uintptr_t stream) const = 0;
  virtual void trace_gpu_event_wait(
      c10::DeviceType device_type,
      uintptr_t event,
      uintptr_t stream) const = 0;
  virtual void trace_gpu_memory_allocation(
      c10::DeviceType device_type,
      uintptr_t ptr) const = 0;
  virtual void trace_gpu_memory_deallocation(
      c10::DeviceType device_type,
      uintptr_t ptr) const = 0;
  virtual void trace_gpu_stream_creation(
      c10::DeviceType device_type,
      uintptr_t stream) const = 0;
  virtual void trace_gpu_device_synchronization(
      c10::DeviceType device_type) const = 0;
  virtual void trace_gpu_stream_synchronization(
      c10::DeviceType device_type,
      uintptr_t stream) const = 0;
  virtual void trace_gpu_event_synchronization(
      c10::DeviceType device_type,
      uintptr_t event) const = 0;

  virtual void reset_backward_hooks(const TensorImpl* self) const = 0;
};

struct C10_API PyInterpreter {
  const PyInterpreterVTable* vtable_;

  PyInterpreter(const PyInterpreterVTable* vtable) : vtable_(vtable) {}

  const PyInterpreterVTable& operator*() const noexcept {
    return *vtable_;
  }
  const PyInterpreterVTable* operator->() const noexcept {
    return vtable_;
  }

  // Disarm this PyInterpreter, making all of its methods noops.
  // The vtable pointer is not an atomic at the moment, which means
  // a disarm() invocation that is concurrent with active destructors
  // is not thread safe and will trigger TSAN.  My hope is that this
  // situations doesn't ever actually happen; tensor destruction should
  // quiesce when a dlclose happens, and any long lived tensors whose
  // destructors would be disarmed here only begin the destruction process
  // on process shutdown (long after the dlclose has occurred).
  void disarm() noexcept;
};

} // namespace c10::impl
