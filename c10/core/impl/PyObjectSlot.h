#pragma once

#include <c10/core/impl/HermeticPyObjectTLS.h>
#include <c10/core/impl/PyInterpreter.h>
#include <c10/core/impl/PyInterpreterHooks.h>
#include <c10/util/python_stub.h>
#include <optional>

#include <atomic>

namespace torch::utils {
class PyObjectPreservation;
}

namespace c10::impl {

struct C10_API PyObjectSlot {
 public:
  PyObjectSlot() : pyobj_interpreter_(nullptr), pyobj_(nullptr) {}

  // Query the PyObject interpreter.  This may return null if there is no
  // interpreter.
  PyInterpreter* pyobj_interpreter() const {
    return pyobj_interpreter_.load(std::memory_order_acquire);
  }

  PyInterpreter& load_pyobj_interpreter() const {
    auto interpreter = pyobj_interpreter_.load(std::memory_order_acquire);
    TORCH_INTERNAL_ASSERT(
        interpreter, "cannot access PyObject for Tensor - no interpreter set");
    return *interpreter;
  }

  PyObject* load_pyobj() const {
    return pyobj_.load(std::memory_order_acquire);
  }

  void store_pyobj(PyObject* obj) {
    pyobj_.store(obj, std::memory_order_release);
  }

  bool has_unique_reference() const {
    PyObject* pyobj = load_pyobj();
    return pyobj != nullptr && load_pyobj_interpreter()->refcnt(pyobj) == 1;
  }

  void clear() {
    pyobj_.store(nullptr, std::memory_order_relaxed);
    pyobj_interpreter_.store(nullptr, std::memory_order_relaxed);
  }

  // Helper methods for incref/decref/try_incref of the stored PyObject.
  // Used by intrusive_ptr_target subclasses (TensorImpl, StorageImpl, Node)
  // to implement their virtual pyobject refcount overrides.
  void incref() const noexcept {
    // Because intrusive_ptr incref uses relaxed memory order, we need to
    // do an acquire fence to ensure that the kHasPyObject bit was
    // observed before the load of the PyObject* below.
    // NB: This is a no-op on x86/x86-64
    std::atomic_thread_fence(std::memory_order_acquire);
    PyObject* obj = load_pyobj();
    load_pyobj_interpreter()->incref(obj);
  }

  void decref() const noexcept {
    PyObject* obj = load_pyobj();
    load_pyobj_interpreter()->decref(obj);
  }

  bool try_incref() const noexcept {
    PyInterpreter* interp = pyobj_interpreter();
    if (C10_UNLIKELY(!interp)) {
      return false;
    }
    return (*interp)->try_incref(*this);
  }

 private:
  // This is now always the global interpreter if the PyObject is set.
  // Maybe we can remove this field some day...
  std::atomic<PyInterpreter*> pyobj_interpreter_;

  // The PyObject representing this Tensor or nullptr. Ownership is managed
  // by intrusive_ptr. By the time the PyObjectSlot is destroyed, this
  // reference is already dead.
  std::atomic<PyObject*> pyobj_;

  friend class torch::utils::PyObjectPreservation;
};

} // namespace c10::impl
