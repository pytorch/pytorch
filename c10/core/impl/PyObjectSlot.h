#pragma once

#include <c10/core/impl/HermeticPyObjectTLS.h>
#include <c10/core/impl/PyInterpreter.h>
#include <c10/util/Optional.h>
#include <c10/util/python_stub.h>

#include <atomic>

namespace c10 {
namespace impl {

struct C10_API PyObjectSlot {
 public:
  PyObjectSlot();

  void destroy_pyobj_if_needed();

  // Associate the TensorImpl with the specified PyObject, and, if necessary,
  // also tag the interpreter.
  //
  // NB: This lives in a header so that we can inline away the switch on status
  //
  // NB: THIS FUNCTION CAN RAISE AN EXCEPTION.  Make sure to clean up after
  // PyObject if necessary!
  void init_pyobj(
      PyInterpreter* self_interpreter,
      PyObject* pyobj,
      PyInterpreterStatus status) {
    impl::PyInterpreter* expected = nullptr;
    switch (status) {
      case impl::PyInterpreterStatus::DEFINITELY_UNINITIALIZED:
        // caller guarantees there is no multithreaded access; if there is
        // no data race OK to do a relaxed store
        pyobj_interpreter_.store(self_interpreter, std::memory_order_relaxed);
        break;
      case impl::PyInterpreterStatus::TAGGED_BY_US:
        // no tagging is necessary, the tag is already correct
        break;
      case impl::PyInterpreterStatus::MAYBE_UNINITIALIZED:
        // attempt to claim this TensorImpl with the specified interpreter
        // tag
        if (pyobj_interpreter_.compare_exchange_strong(
                expected, self_interpreter, std::memory_order_acq_rel)) {
          break;
        }
        // test if, actually, it was already tagged by us!  this situation can't
        // be caused by a race, but it could be caused by a situation
        // where someone conservatively tagged the tensor as MAYBE_UNINITIALIZED
        // (because they didn't pre-check the tag) when actually it was
        // owned by the interpreter
        if (expected == self_interpreter) {
          break;
        }
        // fallthrough, we lost the race.  We are guaranteed not to lose the
        // race with ourself, as calls to init_pyobj with the same interpreter
        // ID must be sequentialized by the GIL
        C10_FALLTHROUGH;
      case impl::PyInterpreterStatus::TAGGED_BY_OTHER:
        TORCH_CHECK(
            false,
            "cannot allocate PyObject for Tensor on interpreter ",
            self_interpreter,
            " that has already been used by another torch deploy interpreter ",
            pyobj_interpreter_.load());
    }

    // we are the ONLY thread that can have gotten to this point.  It is not
    // possible to conflict with another zero interpreter as access is protected
    // by GIL
    // NB: owns_pyobj tag is initially false
    pyobj_ = pyobj;
  }

  // Query the PyObject interpreter.  This may return null if there is no
  // interpreter.  This is racy!
  PyInterpreter* pyobj_interpreter();

  PyObject* _unchecked_untagged_pyobj() const;

  // Test the interpreter tag.  If tagged for the current interpreter, return
  // a non-nullopt (but possibly null) PyObject.  If (possibly) untagged,
  // returns a nullopt.  If it is definitely invalid, raises an error.
  //
  // NB: this lives in header so that we can avoid actually creating the
  // c10::optional
  c10::optional<PyObject*> check_pyobj(PyInterpreter* self_interpreter) const {
    // Note [Memory ordering on Python interpreter tag]
    impl::PyInterpreter* interpreter =
        pyobj_interpreter_.load(std::memory_order_acquire);
    if (interpreter == nullptr) {
      // NB: This never returns DEFINITELY_UNINITIALIZED because there is
      // always the possibility that another thread races to initialize
      // after we query here.  The only time when we can conclude a tensor
      // is definitely uninitialized is when we have just allocated it and
      // it cannot have escaped to other threads yet
      return c10::nullopt;
    } else if (interpreter == self_interpreter) {
      // NB: pyobj_ could still be null!
      if (c10::impl::HermeticPyObjectTLS::get_state()) {
        return c10::nullopt;
      } else {
        return c10::make_optional(_unchecked_untagged_pyobj());
      }
    } else {
      TORCH_CHECK(
          false,
          "cannot access PyObject for Tensor on interpreter ",
          (*self_interpreter)->name(),
          " that has already been used by another torch deploy interpreter ",
          (*pyobj_interpreter_.load())->name());
    }
  }

  // Clear the PyObject field for an interpreter, in situations where we
  // statically know the tensor is tagged with our interpreter.
  void unchecked_clear_pyobj(PyInterpreter* interpreter);

  PyInterpreter& load_pyobj_interpreter() const;

  bool owns_pyobj();

  void set_owns_pyobj(bool b);

 private:
  // This field contains the interpreter tag for this object.  See
  // Note [Python interpreter tag] for general context
  //
  // Note [Memory ordering on Python interpreter tag]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // What memory_order do we need when accessing this atomic?  We don't
  // need a single total modification order (as provided by
  // memory_order_seq_cst) as pyobj_interpreter_ is monotonic: it can only
  // transition from -1 to some positive integer and never changes afterwards.
  // Because there is only one modification, it trivially already has a total
  // modification order (e.g., we don't need fences or locked instructions on
  // x86)
  //
  // In fact, one could make a reasonable argument that relaxed reads are OK,
  // due to the presence of external locking (GIL) to ensure that interactions
  // with other data structures are still correctly synchronized, so that
  // we fall in the "Single-Location Data Structures" case as described in
  // http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p2055r0.pdf
  // However, on x86, it doesn't matter if I use acquire or relaxed on the load
  // as I get the same assembly in both cases.  So I just use the more
  // conservative acquire (which will impede compiler optimizations but I don't
  // care)
  std::atomic<PyInterpreter*> pyobj_interpreter_;

  // This field contains a reference to a PyObject representing this Tensor.
  // If pyobj is nullptr, when we transfer Tensor to Python, we allocate a new
  // PyObject for it and set this field.  This field does not have to be
  // protected by an atomic as it is only allowed to be accessed when you hold
  // the GIL, or during destruction of the tensor.
  //
  // When a PyObject dies, you are obligated to clear this field
  // (otherwise, you will try to use-after-free the pyobj); this currently
  // occurs in THPVariable_clear in torch/csrc/autograd/python_variable.cpp
  //
  // NB: Ordinarily, this should not be a strong reference, as if the
  // PyObject owns the Tensor, this would create a reference cycle.
  // However, sometimes this ownership flips.  To track who owns
  // who, this has a single pointer tag indicating whether or not the
  // C++ object owns the PyObject (the common case, zero, means PyObject
  // owns the C++ object); see _unchecked_untagged_pyobj for raw access
  // or check_pyobj for checked access.  See references to PyObject
  // resurrection in torch/csrc/autograd/python_variable.cpp
  PyObject* pyobj_;
};

} // namespace impl
} // namespace c10
