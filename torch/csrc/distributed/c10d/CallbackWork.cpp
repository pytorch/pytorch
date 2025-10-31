#include <c10/core/TensorOptions.h>
#include <torch/csrc/distributed/c10d/CallbackWork.hpp>

namespace c10d {

CallbackWork::CallbackWork(py::function callback)
    : callback_(std::move(callback)) {
  // Create a future that will be marked as complete when wait() is called
  future_ = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()));
}

// NOLINTNEXTLINE(bugprone-exception-escape)
CallbackWork::~CallbackWork() {
  py::gil_scoped_acquire ag;
  callback_.dec_ref();
  // Explicitly set callback_ to nullptr to prevent py::object's dtor
  // to decref on the PyObject again.
  // See Note [Destructing py::object] in python_ivalue.h
  callback_.ptr() = nullptr;
}

bool CallbackWork::wait(std::chrono::milliseconds timeout) {
  py::gil_scoped_acquire ag;

  try {
    // Call the Python callback with timeout
    py::object result = callback_(timeout);

    // Extract the boolean result
    bool success = result.cast<bool>();

    // Mark the work as completed if successful
    if (success) {
      finish();
      // Mark the future as complete with an empty list
      if (!future_->completed()) {
        future_->markCompleted(c10::IValue(c10::List<at::Tensor>()));
      }
    }

    return success;
  } catch (py::error_already_set& e) {
    // Capture the Python exception and store it
    finish(std::current_exception());
    if (!future_->completed()) {
      future_->setErrorIfNeeded(std::current_exception());
    }
    throw;
  } catch (const std::exception& e) {
    // Capture any C++ exception and store it
    finish(std::current_exception());
    if (!future_->completed()) {
      future_->setErrorIfNeeded(std::current_exception());
    }
    throw;
  }
}

c10::intrusive_ptr<c10::ivalue::Future> CallbackWork::getFuture() {
  return future_;
}

} // namespace c10d
