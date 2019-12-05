#include <torch/csrc/distributed/rpc/future_message.h>

namespace torch {
namespace distributed {
namespace rpc {

FutureMessage::FutureMessage(Message message)
    : completed_(true), message_(std::move(message)) {}

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

const Message& FutureMessage::waitNoThrow() {
  std::unique_lock<std::mutex> lock(mutex_);
  finished_cv_.wait(lock, [this] { return completed_.load(); });
  return message_;
}

Message&& FutureMessage::moveMessage() && {
  std::unique_lock<std::mutex> lock(mutex_);
  return std::move(message_);
}

void FutureMessage::markCompleted(Message message) {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    TORCH_CHECK(!completed());
    message_ = std::move(message);
    completed_ = true;
    std::vector<Callback> cbs;
    cbs.swap(callbacks_);
    lock.unlock();
    for (auto& callback : cbs) {
      callback(message_);
    }
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

} // namespace rpc
} // namespace distributed
} // namespace torch
