#pragma once

#include <c10/core/impl/HermeticPyObjectTLS.h>
#include <c10/core/impl/PyInterpreter.h>
#include <c10/core/impl/PyInterpreterHooks.h>
#include <c10/util/python_stub.h>
#include <optional>

namespace c10::impl {

// Function pointer type for getting the global interpreter
using GetPyInterpreterFn = PyInterpreter* (*)();

// Global function pointer (set by csrc initialization)
C10_API extern GetPyInterpreterFn g_get_pyinterpreter_fn;

// Helper function to get the global interpreter
C10_API PyInterpreter* getGlobalPyInterpreter();

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
  // If `ignore_hermetic_tls` is false and this function is called from a
  // hermetic context (ie, `HermeticPyObjectTLS::get_state()` is true), then
  // nullopt is returned. If `ignore_hermetic_tls` is true, then the hermetic
  // context is ignored, allowing you to check the interpreter tag of a
  // nonhermetic PyObject from within a hermetic context. This is necessary
  // because there are some cases where the deallocator function of a
  // nonhermetic PyObject is called from within a hermetic context, so it must
  // be properly treated as a nonhermetic PyObject.
  //
  // NB: this lives in header so that we can avoid actually creating the
  // std::optional

  // @todo alban: I'm not too sure what's going on here, we can probably delete
  // it but it's worthwhile making sure
  std::optional<PyObject*> check_pyobj() const {
    impl::PyInterpreter* interpreter = getGlobalPyInterpreter();
    if (interpreter == nullptr) {
      return std::nullopt;
    }

    return _unchecked_untagged_pyobj();
  }

  PyInterpreter& load_pyobj_interpreter() const;

  bool owns_pyobj();

  void set_owns_pyobj(bool b);

 private:
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

} // namespace c10::impl
