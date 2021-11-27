#include <c10/core/PythonDispatcher.h>

#include <cstdint>

namespace c10 {

PythonDispatcher::PythonDispatcher(PyObject* type,
                                   c10::impl::PyInterpreter* interpreter) noexcept
    : type_(type), interpreter_(interpreter) {}

PythonDispatcher::~PythonDispatcher() {
  interpreter_->decref(type_, /*is_tensor=*/false);
}

void PythonDispatcher::dispatch(const OperatorHandle& op, torch::jit::Stack* s) const {
  interpreter_->dispatch(op, s, type_);
}

intrusive_ptr<TensorImpl> PythonDispatcher::detach(const c10::TensorImpl* self) const {
  return interpreter_->detach(self, type_);
}

PyObject* PythonDispatcher::type() const noexcept {
  return type_;
}

c10::impl::PyInterpreter* PythonDispatcher::interpreter() const noexcept {
  return interpreter_;
}

namespace {

const std::shared_ptr<PythonDispatcher> null_dispatcher_{};

thread_local std::shared_ptr<PythonDispatcher> dispatcher_{};
thread_local std::size_t disabled_{};

} // namespace

const std::shared_ptr<PythonDispatcher>& getPythonDispatcher() noexcept {
  if (C10_UNLIKELY(disabled_ > 0)) {
    return null_dispatcher_;
  } else {
    return dispatcher_;
  }
}

void setPythonDispatcher(const std::shared_ptr<PythonDispatcher>& dispatcher) noexcept {
  if (C10_UNLIKELY(disabled_ > 0)) {
    return;
  }

  dispatcher_ = dispatcher;

  c10::impl::tls_set_dispatch_key_included(
      DispatchKey::Python, dispatcher != nullptr ? true : false);
}

void popPythonDispatcher() noexcept {
  if (C10_UNLIKELY(disabled_ > 0)) {
    return;
  }

  dispatcher_ = {};

  c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, false);
}

bool hasPythonDispatcher() noexcept {
  if (C10_UNLIKELY(disabled_ > 0)) {
    return false;
  } else {
    return dispatcher_ != nullptr;
  }
}

DisablePythonDispatcherGuard::DisablePythonDispatcherGuard() noexcept {
  disabled_ += 1;
}

DisablePythonDispatcherGuard::~DisablePythonDispatcherGuard() {
  TORCH_INTERNAL_ASSERT(disabled_ >= 1);

  disabled_ -= 1;
}

} // namespace c10
