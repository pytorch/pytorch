#include <c10/core/impl/HermeticPyObjectTLS.h>
#include <ATen/core/PythonOpRegistrationTrampoline.h>

namespace at {
namespace impl {

std::atomic<c10::impl::PyInterpreter*> PythonOpRegistrationTrampoline::interpreter_{nullptr};

void PythonOpRegistrationTrampoline::registerInterpreter(c10::impl::PyInterpreter* interp) {
  c10::impl::PyInterpreter* expected = nullptr;
  interpreter_.compare_exchange_strong(expected, interp);
  if (expected != nullptr) {
    // This is the second (or later) Python interpreter, which means we need
    // non-trivial hermetic PyObject TLS
    c10::impl::HermeticPyObjectTLS::init_state();
  }
}

void PythonOpRegistrationTrampoline::operator()(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
  // The simple implementation of Python op registration involves directly
  // registering a Python callback to the dispatcher.  However, this doesn't
  // work with torchdeploy/multipy, where there are multiple Python
  // interpreters and each of these registrations would clobber each other.
  //
  // First, we assume that all Python interpreters will make identical Python
  // op registrations (e.g., by loading all the same libraries).  What we
  // do is register a single trampoline function from libtorch (non-Python)
  // that doesn't vary from interpreter to interpreter, and figures out how
  // to get to the appropriate interpreter to actually do the call.  This
  // trampoline is special cased to not emit warnings when it clobbers itself
  // in the dispatcher.
  //
  // Which interpreter should we call?  Python operator registrations operate
  // in "hermetic mode", which means that tensors passing to Python get fresh
  // Python objects (not associated with the previous interpreter), and Python
  // objects aren't preserved when the Python objects go out of scope.  This
  // means that, in fact, these calls can be serviced by ANY interpreter.  So
  // we just arbitrarily pick one and jump to it.
  //
  // We only trigger hermetic mode even without torchdeploy/multipy, to avoid
  // paying for a TLS hit.  This means operator behavior is slightly different
  // between deploy and multipy.  I'm willing to reconsider this; to have
  // hermetic mode trigger unconditionally, we only need to remove the
  // hasState_ field test.

  auto* interp = interpreter_.load(std::memory_order_relaxed);
  TORCH_INTERNAL_ASSERT(interp != nullptr);
  c10::impl::EnableHermeticPyObject g;
  (*interp)->python_op_registration_trampoline(
    op, dispatch_key_, stack
  );
}

} // namespace impl
} // namespace at
