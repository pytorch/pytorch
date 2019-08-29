#include <torch/csrc/distributed/rpc/FutureMessage.h>

namespace torch {
namespace distributed {
namespace rpc {

const Message& FutureMessage::wait() {
  std::unique_lock<std::mutex> lock(mutex_);
  finished_cv_.wait(lock, [this]{return completed_.load();});
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

const Message& FutureMessage::message() {
  std::unique_lock<std::mutex> lock(mutex_);
  TORCH_CHECK(completed(), "Cannot retrieve message before completed.");

  return message_;
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
  callbacks.push_back(callback);
}

void FutureMessage::fireCallbacks() {
  TORCH_CHECK(completed(), "Firing callbacks on incomplete FutureMessage.");
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
