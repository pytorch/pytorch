#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/python_stub.h>
#include <string>
#include <vector>

// Forward declarations

namespace c10 {
struct IValue;
class OperatorHandle;
struct TensorImpl;
struct SafePyObject;
} // namespace c10

namespace torch {
namespace jit {
using Stack = std::vector<c10::IValue>;
}
} // namespace torch

// Actual implementation

namespace c10 {
namespace impl {

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
// BUT WAIT, IT GETS WORSE.  With torchdeploy, there may be multiple Python
// interpreters in a single process. If a C++ object is accessible from
// multiple interpreters, we must take care not to accidentally pass a
// PyObject from one interpreter with another interpreter.
//
// To prevent these mixups, we introduce a PyInterpreter "tag" (object with
// a vtable), which specifies a specific Python interpreter.
//
//  - Any given object can be associated with AT MOST one Python interpreter.
//    We represent the interpreter tag as a memory address to an instance of
//    a virtual class that is allocated once per interpreter (this is so that
//    we can request the interpreter to perform operations for us, if
//    necessary).
//
//  - It can be recorded with a PyObject (PyInterpreterObject) so that
//    we know what interpreter the object is associated with, and we can
//    raise an error if you try to use the PyObject from the wrong
//    interpreter context.
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
// always stay live even after dlclose), and disarm the "virtual methods" by
// replacing them with function pointers that just no-op.  This can't be done
// with a traditional C++ vtable, so we have to roll our own.
//
// NB: The downside with representing PyInterpreter tags as full objects is that
// it takes an extra word on TensorImpl.  If tags were instead just integer
// indices, on 64-bit architectures we could pack the tag and PyObject together
// into a single atomic word.  On 32-bit architectures we could simply say that
// only one Python interpreter is supported (erroring if a nontrivial
// interpreter tag is attempted to be set).
//
// The difficulty with this scheme is we need to maintain an out-of-line table
// to get at the PyInterpreters so that we can do virtual method calls on them,
// and registration/deregistration to this table must be done in a thread safe
// manner.  This can be easily done if the number of possible PyInterpreters is
// small enough (e.g., 8-bit integer) by simply preallocating an array of
// sufficient size to hold all possible interpreters.  Surely 128 threads is
// more than enough for anyone!
//
// I didn't decide to do this technique at the moment, because the extra word
// added by the PyInterpreter tag takes us to 24 words, which means that we
// still fit inside three eight word cache lines.  If you need to penny pinch
// another word consider doing this!

struct C10_API PyInterpreter {
  // Feel free to add as much random crap here as you need; each of these
  // can be thought of as a "C++ to Python" hook.
  using name_sig = std::string(const PyInterpreter*);
  using decref_sig = void(const PyInterpreter*, PyObject*, bool);
  using detach_sig =
      c10::intrusive_ptr<TensorImpl>(const PyInterpreter*, const TensorImpl*);
  using dispatch_sig = void(
      const PyInterpreter*,
      const c10::OperatorHandle&,
      torch::jit::Stack* stack,
      // This is a Tensor subclass type object
      const std::shared_ptr<SafePyObject>& type);
  using is_contiguous_sig = bool(const PyInterpreter*, const TensorImpl*);

  PyInterpreter(
      name_sig* name_fn,
      decref_sig* decref_fn,
      detach_sig* detach,
      dispatch_sig* dispatch,
      is_contiguous_sig* is_contiguous)
      : name_fn_(name_fn),
        decref_fn_(decref_fn),
        detach_fn_(detach),
        dispatch_fn_(dispatch),
        is_contiguous_fn_(is_contiguous) {}

  name_sig* name_fn_;
  decref_sig* decref_fn_;
  detach_sig* detach_fn_;
  dispatch_sig* dispatch_fn_;
  is_contiguous_sig* is_contiguous_fn_;

  // UBSAN suppression fixes: "call to function
  // (anonymous namespace)::concrete_decref_fn(c10::impl::PyInterpreter const*,
  // _object*) through pointer to incorrect function type 'void (*)(const
  // c10::impl::PyInterpreter *, _object *)'" See
  // https://github.com/google/sanitizers/issues/911

  // Report the name of this interpreter
  __ubsan_ignore_function__ std::string name() const {
    return (*name_fn_)(this);
  }

  // Run Py_DECREF on a PyObject.  We DO NOT assume the GIL is held on call
  // See NOTE [PyInterpreter::decref takes an `is_tensor` arg]
  __ubsan_ignore_function__ void decref(PyObject* pyobj, bool is_tensor) const {
    return (*decref_fn_)(this, pyobj, is_tensor);
  }

  // Perform a detach by deferring to the __torch_dispatch__ implementation of
  // detach, which will also arrange for the PyObject to get copied in this
  // situation
  __ubsan_ignore_function__ c10::intrusive_ptr<TensorImpl> detach(
      const TensorImpl* self) const {
    return (*detach_fn_)(this, self);
  }

  // Invoke the Python boxed fallback dispatch to go back into Python
  __ubsan_ignore_function__ void dispatch(
      const c10::OperatorHandle& op,
      torch::jit::Stack* stack,
      const std::shared_ptr<SafePyObject>& type) const {
    return (*dispatch_fn_)(this, op, stack, type);
  }

  __ubsan_ignore_function__ bool is_contiguous(const TensorImpl* self) const {
    return (*is_contiguous_fn_)(this, self);
  }

  // Disarm this PyInterpreter, making all of its methods noops.
  // Because the function pointers are raw pointers (not atomics),
  // a disarm() invocation that is concurrent with active destructors
  // is not thread safe and will trigger TSAN.  My hope is that this
  // situations doesn't ever actually happen; tensor destruction should
  // quiesce when a dlclose happens, and any long lived tensors whose
  // destructors would be disarmed here only begin the destruction process
  // on process shutdown (long after the dlclose has occurred).
  void disarm() noexcept;
};

} // namespace impl
} // namespace c10
