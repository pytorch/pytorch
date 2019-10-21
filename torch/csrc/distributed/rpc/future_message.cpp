#include <torch/csrc/distributed/rpc/future_message.h>

namespace torch {
namespace distributed {
namespace rpc {

FutureMessage::FutureMessage() {
  startTimer();
}

const Message& FutureMessage::wait() {
  std::unique_lock<std::mutex> lock(mutex_);
  finished_cv_.wait(lock, [this] { return completed_.load(); });

  // Throw an exception if we encounter one.
  if (message_.type() == MessageType::EXCEPTION) {
    std::string err(message_.payload().begin(), message_.payload().end());
    throw std::runtime_error(err);
  }
  return message_;
}

void FutureMessage::markCompleted(Message message) {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    TORCH_CHECK(!completed());
    completed_ = true;
    message_ = std::move(message);

    fireCallbacks();
  }
  finished_cv_.notify_all();
}

void FutureMessage::markCompleted() {
  markCompleted(Message());
}

bool FutureMessage::completed() const {
  return completed_;
}

void FutureMessage::addCallback(const FutureMessage::Callback& callback) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (completed()) {
    lock.unlock();
    callback(message_);
    return;
  }
  callbacks_.push_back(callback);
}

void FutureMessage::fireCallbacks() {
  TORCH_CHECK(completed(), "Firing callbacks on incomplete FutureMessage.");
  // There is no need to protect callbacks_ with the lock.
  // Once completed_ is set to true, no one can add new callback to the list.
  for (auto& callback : callbacks_) {
    callback(message_);
  }
  callbacks_.clear();
}

void FutureMessage::startTimer() {
  futureStartTime_ = std::chrono::high_resolution_clock::now();
}

bool FutureMessage::checkTimeElapsed(
    const std::chrono::seconds& timeoutSeconds) {
  std::unique_lock<std::mutex> lock(mutex_);

  if (completed_) {
    return false;
  }
  const auto now = std::chrono::high_resolution_clock::now();
  const auto elapsed = now - futureStartTime_;
  const auto elapsedSeconds =
      std::chrono::duration_cast<std::chrono::seconds>(elapsed);
  return elapsedSeconds > timeoutSeconds;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
