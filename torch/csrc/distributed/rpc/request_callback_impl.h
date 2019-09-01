#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/request_callback.h>

namespace torch {
namespace distributed {
namespace rpc {

class RequestCallbackImpl : public RequestCallback {
 public:
  virtual Message processMessage(Message&& request) override;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
