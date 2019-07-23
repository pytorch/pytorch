#pragma once

#include <torch/csrc/distributed/rpc/Message.h>
#include <torch/csrc/distributed/rpc/rpc_headers.h>


namespace torch {
namespace distributed {
namespace rpc {

struct TORCH_API FutureMessage final {

 public:
  void wait();
  void markCompleted(Message message);
  void markCompleted();
  Message& message();
  bool completed();
  void addCallback(std::function<void(Message)> callback);

 private:
  void fireCallbacks();

  std::mutex mutex_;
  std::atomic_bool completed_ = {false}; // is this future complete
  std::condition_variable finished_cv_;
  std::vector<std::function<void(Message)>> callbacks;
  Message message_;
};

}
}
}
