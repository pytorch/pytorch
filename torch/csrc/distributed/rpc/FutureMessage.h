#pragma once

#include <torch/csrc/distributed/rpc/Message.h>

namespace torch {
namespace distributed {
namespace rpc {


// This class holds a message that will be ready in the future.
struct TORCH_API FutureMessage final {

 public:
  using Callback = std::function<void(const Message&)>;

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
  Message message_;
};

}
}
}
