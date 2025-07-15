#include <c10/core/impl/PyObjectSlot.h>
// Forward declare to avoid circular dependency
extern "C" c10::impl::PyInterpreter* getPyInterpreter();

namespace c10::impl {

PyObjectSlot::PyObjectSlot() : pyobj_interpreter_(nullptr), pyobj_(nullptr) {}

PyObjectSlot::~PyObjectSlot() {
  maybe_destroy_pyobj();
}

void PyObjectSlot::maybe_destroy_pyobj() {
  if (owns_pyobj()) {
    auto interpreter = pyobj_interpreter_.load(std::memory_order_acquire);
    if (interpreter == nullptr) {
      // Try to get global interpreter if not set
      interpreter = getPyInterpreter();
    }
    TORCH_INTERNAL_ASSERT(interpreter != nullptr);
    TORCH_INTERNAL_ASSERT(pyobj_ != nullptr);
    interpreter->decref(_unchecked_untagged_pyobj(), /*has_pyobj_slot*/ true);
    // NB: this destructor can only be entered when there are no
    // references to this C++ object (obviously), NOR any references
    // to the PyObject (if there are references to the PyObject,
    // then the PyObject holds an owning reference to the tensor).
    // So it is OK to clear pyobj_ here as it is impossible for it to
    // be used again (modulo weak reference races)
    pyobj_ = nullptr; // for safety
  }
}

PyInterpreter* PyObjectSlot::pyobj_interpreter() {
  auto interpreter = pyobj_interpreter_.load(std::memory_order_acquire);
  if (interpreter == nullptr) {
    // Try to initialize with global interpreter
    auto* global_interpreter = getPyInterpreter();
    if (global_interpreter != nullptr) {
      PyInterpreter* expected = nullptr;
      pyobj_interpreter_.compare_exchange_strong(expected, global_interpreter, std::memory_order_acq_rel);
      interpreter = pyobj_interpreter_.load(std::memory_order_acquire);
    }
  }
  return interpreter;
}

PyObject* PyObjectSlot::_unchecked_untagged_pyobj() const {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  return reinterpret_cast<PyObject*>(
      reinterpret_cast<uintptr_t>(pyobj_) & ~0x1ULL);
}

void PyObjectSlot::unchecked_clear_pyobj(PyInterpreter* interpreter) {
  // Ignore passed interpreter, just verify we have the global one
  (void)interpreter; // Suppress unused parameter warning
  auto* global_interpreter = getPyInterpreter();
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(global_interpreter == pyobj_interpreter_.load());
  pyobj_ = nullptr;
}

PyInterpreter& PyObjectSlot::load_pyobj_interpreter() const {
  auto interpreter = pyobj_interpreter_.load(std::memory_order_acquire);
  if (interpreter) {
    return *interpreter;
  }
  TORCH_CHECK(
      false,
      "cannot access PyObject for Tensor - interpreter not initialized");
}

bool PyObjectSlot::owns_pyobj() {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  return reinterpret_cast<uintptr_t>(pyobj_) & 1;
}

void PyObjectSlot::set_owns_pyobj(bool b) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  pyobj_ = reinterpret_cast<PyObject*>(
      reinterpret_cast<uintptr_t>(_unchecked_untagged_pyobj()) | b);
}

} // namespace c10::impl
