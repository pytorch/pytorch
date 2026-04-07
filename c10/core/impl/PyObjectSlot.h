#pragma once

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
  PyObjectSlot() : pyobj_(nullptr) {}

  PyObject* load_pyobj() const {
    return pyobj_.load(std::memory_order_acquire);
  }

  void store_pyobj(PyObject* obj) {
    pyobj_.store(obj, std::memory_order_release);
  }

  bool has_unique_reference() const {
    PyObject* pyobj = load_pyobj();
    return pyobj != nullptr && (*getGlobalPyInterpreter())->refcnt(pyobj) == 1;
  }

  void clear() {
    pyobj_.store(nullptr, std::memory_order_relaxed);
  }

 private:
  // The PyObject representing this Tensor or nullptr. Ownership is managed
  // by intrusive_ptr. By the time the PyObjectSlot is destroyed, this
  // reference is already dead.
  std::atomic<PyObject*> pyobj_;

  friend class torch::utils::PyObjectPreservation;
};

} // namespace c10::impl
