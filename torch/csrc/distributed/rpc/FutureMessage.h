#pragma once

#include <torch/csrc/distributed/rpc/Message.h>

namespace torch {
namespace distributed {
namespace rpc {


// This class holds a message that will be ready in the future.
//
// TODO: consider using ivalue::Future.
struct TORCH_API FutureMessage final {

 public:
  // Keep cb the same as ivalue::Future to make it easy for future merge after
  // we made Message an IValue type.
  using Callback = std::function<void(void)>;

  const Message& wait();
  void markCompleted(Message message);
  void markCompleted();
  const Message& message();
  bool completed() const;

  // If completed() the callback will be invoked in-place.
  void addCallback(const Callback& callback);

 private:

  void fireCallbacks();

  std::mutex mutex_;
  std::atomic_bool completed_ {false}; // is this future complete
  std::condition_variable finished_cv_;
  std::vector<Callback> callbacks;
  // TODO: make message_ an optional field, and get rid of UNKNOWN message type
  Message message_;
};

}
}
}
