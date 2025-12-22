#include <torch/csrc/utils/pyobject_preservation.h>

#include <c10/core/impl/PyObjectSlot.h>
#include <c10/util/intrusive_ptr.h>

namespace torch::utils {

using c10::intrusive_ptr_target;
using c10::impl::PyObjectSlot;

void PyObjectPreservation::init_fresh_nonatomic(
    intrusive_ptr_target* target,
    PyObjectSlot* slot,
    PyObject* pyobj) {
  TORCH_INTERNAL_ASSERT(slot->load_pyobj() == nullptr);
  TORCH_INTERNAL_ASSERT(
      target->combined_refcount_.load(std::memory_order_relaxed) ==
      c10::detail::kUniqueRef);

  slot->pyobj_.store(pyobj, std::memory_order_relaxed);
  slot->pyobj_interpreter_.store(
      c10::impl::getGlobalPyInterpreter(), std::memory_order_relaxed);
  target->combined_refcount_.store(
      c10::detail::kHasPyObject | c10::detail::kUniqueRef,
      std::memory_order_relaxed);
}

PyObject* PyObjectPreservation::init_once(
    intrusive_ptr_target* target,
    PyObjectSlot* slot,
    PyObject* pyobj) {
  PyObject* expected = nullptr;
  if (!slot->pyobj_.compare_exchange_strong(
          expected, pyobj, std::memory_order_acq_rel)) {
    TORCH_INTERNAL_ASSERT(expected != nullptr);
    return expected;
  }

  slot->pyobj_interpreter_.store(
      c10::impl::getGlobalPyInterpreter(), std::memory_order_release);

  bool increfed = false;
  auto combined = target->combined_refcount_.load(std::memory_order_relaxed);
  do {
    TORCH_INTERNAL_ASSERT(!c10::detail::has_pyobject(combined));
    if (c10::detail::refcount(combined) > 1 && !increfed) {
      // We need to incref the object to preserve the invariant that
      // if refcount > 1, the c10 object holds a reference to the PyObject.
      // This must happen before we set the kHasPyObject bit.
      Py_INCREF(pyobj);
      increfed = true;
    }
  } while (!target->combined_refcount_.compare_exchange_weak(
      combined,
      combined | c10::detail::kHasPyObject,
      std::memory_order_acq_rel,
      std::memory_order_relaxed));

  if (increfed && c10::detail::refcount(combined) == 1) {
    // Fix up if refcount if we did the incref in a failed compare-exchange
    Py_DECREF(pyobj);
  }

  return pyobj;
}

} // namespace torch::utils
