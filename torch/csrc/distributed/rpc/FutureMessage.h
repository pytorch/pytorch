#pragma once

#include <c10/util/Optional.h>
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

  // TODO: add a get() API that returns immediately with an optional Message
  // object.
  const Message& wait();
  void markCompleted(Message message);
  void markCompleted();
  const c10::optional<Message>& message();
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
  c10::optional<Message> message_;
};

}
}
}
