#pragma once

#include <torch/csrc/distributed/rpc/message.h>

namespace torch::distributed::rpc {

// Functor which is invoked to process an RPC message. This is an abstract class
// with some common functionality across all request handlers. Users need to
// implement this interface to perform the actual business logic.
class TORCH_API RequestCallback {
 public:
  // Invoke the callback.
  c10::intrusive_ptr<JitFuture> operator()(
      Message& request,
      std::vector<c10::Stream> streams) const;

  virtual ~RequestCallback() = default;

 protected:
  // RpcAgent implementation should invoke ``RequestCallback`` to process
  // received requests. There is no restriction on the implementation's
  // threading model. This function takes an rvalue reference of the Message
  // object. It is expected to return the future to a response message or
  // message containing an exception. Different rpc agent implementations are
  // expected to ensure delivery of the response/exception based on their
  // implementation specific mechanisms.
  virtual c10::intrusive_ptr<JitFuture> processMessage(
      Message& request,
      std::vector<c10::Stream> streams) const = 0;
};

} // namespace torch::distributed::rpc
