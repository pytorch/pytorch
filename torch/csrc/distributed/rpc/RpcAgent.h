#pragma once

#include <torch/csrc/distributed/rpc/FutureMessage.h>
#include <torch/csrc/distributed/rpc/Message.h>
#include <torch/csrc/distributed/rpc/rpc_headers.h>

namespace torch {
namespace distributed {
namespace rpc {

class RpcAgent;

// RpcAgent implementation should invoke ``RequestCallback`` to process received
// requests. It takes the name of the request sender, the Message object,
// and a reference to the RpcAgent itself. It means that, the implementation of
// ``RequestCallback`` can be either blocking (finish processing request in this
// method, and send the response out), or non-blocking (e.g., just enqueue the
// message and the RpcAgent reference, and use a different set of threads to
// process them). The current implementation is blocking.
using RequestCallback = std::function<void(std::string, Message, RpcAgent&)>;

class RpcAgent {
 public:
  // It is up to the RpcAgent implement to determine how to resolve names.
  // ProcessGroupAgent just use map from name to rank. ThriftAgent could use
  // a separate kv store or sth for that.
  RpcAgent(std::string workerName, RequestCallback cb);

  virtual ~RpcAgent() noexcept(false);

  // Send a message to the worker with name ``to`` and returns a FutureMessage.
  // If message.isOp() is true, the FutureMessage will be completed when the
  // response arrives. For other message types, the Future should be ignored by
  // the caller.
  //
  // The Message object contains 4 fields:
  //    meta (std::vector<char>): a binary chunk of data.
  //    tensors (std::vector<torch::Tensor>): all tensors.
  //    type (MessageType): type of the message.
  //    id (int64_t): message id, this is used by ProcessGroupAgent to match
  //                  request and response. Other implementation can ignore it
  //                  if they have their own ways to do matching.
  //
  // Layers above ``RpcAgent`` only converts BuiltinOp, BuiltinRet, PythonUdfOp,
  // and PythonUdfRet into a Message, and it is up to the RpcAgent
  // implementation to determine how to serialize and commute a message. This
  // should make future streaming serialization possible.
  virtual std::shared_ptr<FutureMessage> send(
      std::string to, Message message) = 0;

  // This is a temporary solution to gracefully stop the listening loop.
  // ProcessGroupAgent does this by sending a SHUTDOWN message to the
  // (rank + 1) % world_size peer, which means we cannot create
  // ProcessGroupAgent with world_size == 1. We can drop this in the future when
  // we find a way to gracefully exit the blocking recvAnysource call.
  //
  // NB: this cannot be put into the destructor because we should not call
  // virtual methods in destructor.
  virtual void shutdown() = 0;

 protected:
  const std::string workerName_;
  const RequestCallback cb_;
};

}
}
}
