#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <c10/core/impl/PythonDispatcherTLS.h>
#include <ATen/core/PythonFallbackKernel.h>
#include <c10/core/SafePyObject.h>

namespace {

// This TLS is used to track the state of the dispatcher to be able to restore
// it when calling back into python.
// It has the following invariant:
//  - It must be empty while python code is executed.
//  - It should only be set once even for multiple dispatcher calls that do not come
//    back to python.
// To achieve this, we ensure that the tls is empty by default and emptied again both when
// we call into user torch_dispatch or returning back to python after this call.

thread_local c10::optional<c10::impl::LocalDispatchKeySet> tls_on_entry;

c10::impl::LocalDispatchKeySet safe_get_tls_on_entry() {
  TORCH_CHECK(tls_on_entry.has_value(), "Accessing torch dispatch state outside of '__torch_dispatch__' "
              "is not allowed.");
  return tls_on_entry.value();
}

// All the keys below the Python key
constexpr c10::DispatchKeySet after_Python_keyset = c10::DispatchKeySet(c10::DispatchKeySet::FULL) ^
  (c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Python) |
   c10::DispatchKeySet(c10::DispatchKey::Python));


// This guard assumes that tls_on_entry has a value.
struct StashTLSOnEntryGuard {
public:
  StashTLSOnEntryGuard(): saved_(tls_on_entry.value()) {
    tls_on_entry = c10::nullopt;
  }

  ~StashTLSOnEntryGuard() {
    TORCH_INTERNAL_ASSERT(!tls_on_entry.has_value());
    tls_on_entry = saved_;
  }

private:
  c10::impl::LocalDispatchKeySet saved_;
};

void pythonFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  TORCH_INTERNAL_ASSERT(tls_on_entry.has_value());
  // c10::impl::ForceDispatchKeyGuard dispatcher_guard(tls_on_entry.value());
  // StashTLSOnEntryGuard stash_guard;
  c10::impl::ExcludeDispatchKeyGuard guard(after_Python_keyset);


  // If Torch Dispatch Mode is active, use its PyInterpreter for dispatch
  const auto mode_stack_len = c10::impl::TorchDispatchModeTLS::stack_len();
  if (mode_stack_len > 0) {
    const auto& cur_torch_dispatch_mode_state = c10::impl::TorchDispatchModeTLS::get_stack_at(mode_stack_len - 1);
    cur_torch_dispatch_mode_state->pyinterpreter()->dispatch(op, stack);
    return;
  }

  // Otherwise, find a PyInterpreter on a Tensor
  const auto& schema = op.schema();
  const auto num_arguments = schema.arguments().size();
  // It is safe to dispatch on the very first Tensor with a pyobj_interpreter
  // without checking the interpreters of any of the arguments, because when
  // we actually run dispatch(), we will take out PyObjects in the context
  // of that interpreter, and this will ensure that everyone is on the same
  // interpreter.
  for (const auto& ivalue : torch::jit::last(*stack, num_arguments)) {
    if (ivalue.isTensor()) {
      auto* interpreter = ivalue.unsafeToTensorImpl()->pyobj_slot()->pyobj_interpreter();
      if (interpreter) {
        (*interpreter)->dispatch(op, stack);
        return;
      }
    } else if (ivalue.isTensorList() || ivalue.isOptionalTensorList()) {
      // NB: use toListRef as it doesn't induce refcount bumps (toTensorListRef
      // is not a thing)
      for (const auto& nv : ivalue.toListRef()) {
        if (nv.isNone()) {
          continue;
        }
        auto* interpreter = nv.unsafeToTensorImpl()->pyobj_slot()->pyobj_interpreter();
        if (interpreter) {
          (*interpreter)->dispatch(op, stack);
          return;
        }
      }
    }
  }
  TORCH_INTERNAL_ASSERT(0, "Hit Python dispatch key but no arguments had PyInterpreter (no tensor args?)");
}

void pythonDispatcherFallback(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
  auto* state = c10::impl::PythonDispatcherTLS::get_state();
  TORCH_INTERNAL_ASSERT(state, "Hit PythonDispatcher dispatch key but PythonDispatcherTLS was not set");
  (*state)->python_dispatcher(op, dispatch_keys.remove(c10::DispatchKey::PythonDispatcher), stack);
}

void pythonTLSSnapshotFallback(const c10::OperatorHandle &op, c10::DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
  // It is ok for the tls to be already set here.
  // It means that there are multiple calls into the dispatcher not originating from python code.
  // The guard below will properly ignore such calls.
  at::impl::MaybeSetTLSOnEntryGuard guard;

  op.redispatchBoxed(dispatch_keys & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::PythonTLSSnapshot), stack);
}

// The PreDispatch key gets a no-op fallback that just redispatches.
// The main way this key is used is that we can register a mode to it from python (e.g. TorchProxyDispatchMode, for pre_dispatch tracing)
// Can't this be a fallthrough kernel, instead of a fallback that just no-ops and redispatches?
// Unfortunately, no: we need a real kernel that is not a fallthrough, in order for the PythonDispatcher to interpose on it.
// Alternatively, we could have hardcoded this kernel (in C++) to directly call in TorchProxyDispatchMode.
// Doing that in C++ is a pain though, so it's done in python using the PythonDispatcher for convenience.
void preDispatchFallback(const c10::OperatorHandle& op, c10::DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
  op.redispatchBoxed(dispatch_keys & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::PreDispatch), stack);
}

} // anonymous namespace

namespace at {
namespace impl {

RestorePythonTLSSnapshot::RestorePythonTLSSnapshot() : saved_(safe_get_tls_on_entry()), guard_(safe_get_tls_on_entry()) {
  tls_on_entry = c10::nullopt;
}

RestorePythonTLSSnapshot::~RestorePythonTLSSnapshot() {
  TORCH_INTERNAL_ASSERT(!tls_on_entry.has_value());
  tls_on_entry = saved_;
}

MaybeSetTLSOnEntryGuard::MaybeSetTLSOnEntryGuard() {
  if (tls_on_entry.has_value()) {
    value_set_ = false;
  } else {
    value_set_ = true;
    tls_on_entry = c10::impl::tls_local_dispatch_key_set();
  }
}
MaybeSetTLSOnEntryGuard::~MaybeSetTLSOnEntryGuard() {
  if (value_set_) {
    TORCH_INTERNAL_ASSERT(tls_on_entry.has_value());
    tls_on_entry = c10::nullopt;
  }
}


} // namespace impl
} // namespace at

TORCH_LIBRARY_IMPL(_, Python, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&pythonFallback>());
}

TORCH_LIBRARY_IMPL(_, PythonDispatcher, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&pythonDispatcherFallback>());
}

TORCH_LIBRARY_IMPL(_, PythonTLSSnapshot, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&pythonTLSSnapshotFallback>());
}

TORCH_LIBRARY_IMPL(_, PreDispatch, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&preDispatchFallback>());
}
