#include <torch/csrc/distributed/rpc/FutureMessage.h>

namespace torch {
namespace distributed {
namespace rpc {

void FutureMessage::wait() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (!completed_) {
    finished_cv_.wait(lock);
  }
}

void FutureMessage::markCompleted(Message message) {
  std::unique_lock<std::mutex> lock(mutex_);
  TORCH_CHECK(!completed());
  completed_ = true;
  message_ = std::move(message);

  fireCallbacks();
  finished_cv_.notify_all();
}

void FutureMessage::markCompleted() {
  markCompleted(Message());
}

Message& FutureMessage::message() {
  std::unique_lock<std::mutex> lock(mutex_);
  TORCH_CHECK(completed());

  return message_;
}

bool FutureMessage::completed() {
  return completed_;
}

void FutureMessage::addCallback(std::function<void(Message)> callback) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (completed()) {
    lock.unlock();
    callback(message_);
    return;
  }
  callbacks.push_back(callback);
}

void FutureMessage::fireCallbacks() {
  AT_ASSERT(completed());
  // There is no need to protect callbacks with the lock.
  // Once completed_ is set to true, no one can add new callback to the list.
  for (auto& callback : callbacks) {
    callback(message_);
  }
  callbacks.clear();
}

}
}
}
