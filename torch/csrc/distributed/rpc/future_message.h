#pragma once

#include <torch/csrc/distributed/rpc/message.h>

namespace torch {
namespace distributed {
namespace rpc {

// This class holds a message that will be ready in the future.
//
// TODO: consider using ivalue::Future.
struct TORCH_API FutureMessage final {
 public:
  using Callback = std::function<void(const Message&)>;

  // TODO: add a get() API that returns immediately with an optional Message
  // object.
  const Message& wait();
  void markCompleted(Message message);
  void markCompleted();
  bool completed() const;

  // If completed() the callback will be invoked in-place.
  void addCallback(const Callback& callback);

 private:
  void fireCallbacks();

  mutable std::mutex mutex_;
  std::atomic_bool completed_{false}; // is this future complete
  std::condition_variable finished_cv_;
  std::vector<Callback> callbacks_;
  // TODO: make message_ an optional field, and get rid of UNKNOWN message type
  Message message_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
