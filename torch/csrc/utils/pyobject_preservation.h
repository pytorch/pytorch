#pragma once

#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/python_compat.h>

#include <c10/core/impl/PyObjectSlot.h>
#include <c10/util/intrusive_ptr.h>

// This file contains utilities used for handling PyObject preservation

namespace torch::utils {

class PyObjectPreservation {
 public:
  // Store a PyObject wrapper on a fresh c10 wrapper. The caller must hold
  // a unique reference to `target`.
  template <typename T>
  requires requires(T& t) {
    t.pyobj_slot();
  }
  static void init_fresh_nonatomic(T& target, PyObject* pyobj) {
    auto* slot = target.pyobj_slot();
    TORCH_INTERNAL_ASSERT(slot->load_pyobj() == nullptr);
    TORCH_INTERNAL_ASSERT(
        target.combined_refcount_.load(std::memory_order_relaxed) ==
        c10::detail::kUniqueRef);

    // Ensure that PyUnstable_TryIncref calls don't fail spuriously in
    // free-threaded Python.
    PyUnstable_EnableTryIncRef(pyobj);

    slot->pyobj_.store(pyobj, std::memory_order_relaxed);
    target.combined_refcount_.store(
        c10::detail::kHasPyObject | c10::detail::kUniqueRef,
        std::memory_order_relaxed);
  }

  // Thread-safe get-or-create for the PyObject wrapper. Returns a new
  // reference. The factory is called at most once if no wrapper exists yet;
  // if another thread races and wins, the factory's result is destroyed and
  // the winner's wrapper is returned instead.
  template <typename T, typename Factory>
  requires requires(T& t) {
    t.pyobj_slot();
  }
  static PyObject* get_or_init(T& target, Factory&& pyobj_factory) {
    auto* slot = target.pyobj_slot();
    PyObject* obj = slot->load_pyobj();
    if (obj) {
      return Py_NewRef(obj);
    }

    obj = pyobj_factory();

    // Ensure that PyUnstable_TryIncref calls don't fail spuriously in
    // free-threaded Python.
    PyUnstable_EnableTryIncRef(obj);

    // Fast path: if we're the only owner, no other thread can see this
    // object, so we can skip the atomic CAS.
    auto combined = target.combined_refcount_.load(std::memory_order_relaxed);
    if (combined == c10::detail::kUniqueRef) {
      slot->pyobj_.store(obj, std::memory_order_relaxed);
      target.combined_refcount_.store(
          c10::detail::kHasPyObject | c10::detail::kUniqueRef,
          std::memory_order_relaxed);
      return obj;
    }

    // Slow path: atomically store our new wrapper into the slot.
    PyObject* expected = nullptr;
    if (!slot->pyobj_.compare_exchange_strong(
            expected, obj, std::memory_order_acq_rel)) {
      // Another thread won the race — discard ours, use theirs.
      Py_DECREF(obj);
      return Py_NewRef(expected);
    }

    // We won. Set the kHasPyObject bit in the combined refcount.
    bool increfed = false;
    do {
      if (c10::detail::refcount(combined) > 1 && !increfed) {
        // Preserve the invariant that if refcount > 1, the c10 object
        // holds a reference to the PyObject. This must happen before we
        // set the kHasPyObject bit.
        Py_INCREF(obj);
        increfed = true;
      }
    } while (!target.combined_refcount_.compare_exchange_weak(
        combined,
        combined | c10::detail::kHasPyObject,
        std::memory_order_acq_rel,
        std::memory_order_relaxed));

    if (increfed && c10::detail::refcount(combined) == 1) {
      // We incref'd because refcount was > 1 during an earlier CAS attempt,
      // but by the time we succeeded, refcount had dropped to 1. Undo.
      Py_DECREF(obj);
    }

    return obj;
  }
};

} // namespace torch::utils
