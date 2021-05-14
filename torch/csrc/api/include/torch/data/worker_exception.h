#pragma once

#include <exception>
#include <string>
#include <utility>

namespace torch {
namespace data {

/// An exception thrown when a DataLoader's worker thread throws an exception,
/// which is caught. A `WorkerException` stores an `exception_ptr` to the
/// original exception thrown in the worker thread.
struct WorkerException : public std::exception {
  /// Constructs a `WorkerException` from an `exception_ptr`.
  explicit WorkerException(std::exception_ptr original)
      // NOLINTNEXTLINE(performance-move-const-arg)
      : original_exception(std::move(original)),
        message("Caught exception in DataLoader worker thread.") {
    try {
      std::rethrow_exception(original_exception);
    } catch (std::exception& e) {
      message += " Original message: ";
      message += e.what();
    }
  }

  const char* what() const noexcept override {
    return message.c_str();
  }

  /// The original exception thrown in the worker thread.
  std::exception_ptr original_exception;

  /// This exception's message (not the original exception's message).
  std::string message;
};

} // namespace data
} // namespace torch
