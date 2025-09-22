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

// Note [Python interpreter tag]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Traditionally, PyTorch is layered such that our Python library
// (libtorch_python) references our pure C++ library (libtorch) as the
// natural order of things.  However, sometimes this natural order is
// subverted: C++ objects refer to Python objects (for example, we
// store a PyObject* pointer on TensorImpl so that converting from a
// C++ Tensor to a Python Tensor is just a memory dereference).
//
// These unusual orderings must be treated with care.  To start, you need to
// virtualize the destructor so that the PyObject can be decref'ed on
// destruction (because the C++ object itself doesn't know anything about
// Python--remember, layering!).  This process itself is fraught, since
// acquiring the GIL could lead to deadlocks if someone is blocking on you
// while holding the GIL.  Furthermore, if the C++ objects outlive the
// interpreter (which can happen if you stash them in a static global
// variable defined in libtorch), you may attempt to decref the object when
// the Python interpreter has already been shutdown.
//
// SINGLE INTERPRETER MODE: Since torch/deploy and multipy are deprecated,
// PyTorch now only supports a single Python interpreter per process.
// This simplifies the design significantly.
//
// The PyInterpreter "tag" (object with a vtable) represents the single
// Python interpreter:
//
//  - All objects are associated with the same single Python interpreter.
//    We represent the interpreter tag as a memory address to an instance of
//    a virtual class that is allocated once for the interpreter (this is so
//    that we can request the interpreter to perform operations for us, if
//    necessary).
//
//  - It contains a vtable that can be used to perform various Python
//    operations from ordinary C++ code that ordinarily wouldn't be accessible
//    from libtorch.
//
// A simple use case is when a C++ object must be associated with a PyObject.
// However, for TensorImpl, we lazily allocate a PyObject the first time the
// object passes into Python.  The invariants for this situation are more
// subtle:
//
//  - A given TensorImpl's interpreter tag can only go from uninitialized to
//    tagged; once tagged, this is a quiescent state (once tagged to an
//    interpreter, ALWAYS tagged to that interpreter)
//
//  - A thread may mutate the PyObject field of a TensorImpl if and only if it
//    holds the GIL for the interpreter tagged on the TensorImpl.  (If the
//    TensorImpl is not tagged, it must first atomically claim its tag before it
//    can validly write)
//
// WARNING: This class has to be written very carefully, because it may be
// possible for a Tensor to have a reference an interpreter corresponding to
// a shared library that has ALREADY BEEN UNLOADED.  This makes blindly calling
// virtual methods very dangerous, because the vtable may be garbage at that
// point (on a good day, you might get "pure virtual method called").
//
// The idea to solve this problem is we always leak PyInterpreters (so they
// always stay live even after dlclose), and make sure we can disarm their
// virtual methods by indirecting through a separate PyInterpreterVTable
// object.  This can be replaced with a no-op vtable from libc10.so, which
// is guaranteed to stick around until the bitter end.
//
// NB: In single-interpreter mode, the PyInterpreter tag could theoretically
// be eliminated entirely since there's only one interpreter. However, we
// keep it for:
// 1. Backward compatibility with existing code
// 2. Clean abstraction between Python and C++ layers
// 3. Future flexibility if we need to support multiple interpreters again

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

  // This method bridges Python operator registration with C++ dispatch.
  // It allows the C++ dispatcher to call back into Python for ops
  // that are implemented in Python (e.g., custom operators, torch.compile).
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
