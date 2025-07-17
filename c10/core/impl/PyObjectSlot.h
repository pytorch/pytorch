#pragma once

#include <c10/core/impl/HermeticPyObjectTLS.h>
#include <c10/core/impl/PyInterpreter.h>
#include <c10/core/impl/PyInterpreterHooks.h>
#include <c10/util/python_stub.h>
#include <optional>

#include <atomic>

namespace c10::impl {

struct C10_API PyObjectSlot {
 public:
  PyObjectSlot();

  ~PyObjectSlot();

  void maybe_destroy_pyobj();

  // Associate the TensorImpl with the specified PyObject, and, if necessary,
  // also tag the interpreter.
  //
  // NB: This lives in a header so that we can inline away the switch on status
  //
  // NB: THIS FUNCTION CAN RAISE AN EXCEPTION.  Make sure to clean up after
  // PyObject if necessary!
  void init_pyobj(PyObject* pyobj) {
    pyobj_interpreter_.store(
        getGlobalPyInterpreter(), std::memory_order_relaxed);
    pyobj_ = pyobj;
  }

  // Query the PyObject interpreter.  This may return null if there is no
  // interpreter.  This is racy!
  PyInterpreter* pyobj_interpreter();

  PyObject* _unchecked_untagged_pyobj() const;

  PyObject* get_pyobj() const {
    // Note that PyObject* can be a nullptr, so please check before
    // using it.
    return _unchecked_untagged_pyobj();
  }

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
  // owns the C++ object); see get_pyobj() for access.  See references to
  // PyObject resurrection in torch/csrc/autograd/python_variable.cpp
  PyObject* pyobj_;
};

} // namespace c10::impl
