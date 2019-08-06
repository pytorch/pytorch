#pragma once

#include <torch/csrc/distributed/rpc/FutureMessage.h>
#include <torch/csrc/distributed/rpc/Message.h>

namespace torch {
namespace distributed {
namespace rpc {


// RpcAgent is the base class for sending and receiving RPC messages. It
// provides a unified ``send`` API for both request and response messages, and
// will invoke the given ``RequestCallback`` to process received requests. It
// should immediately become ready to serve request and accept response after
// construction.
class RpcAgent;

// RpcAgent implementation should invoke ``RequestCallback`` to process received
// requests. There is no restriction on the implementation's threading model.
// This function takes the name of the request sender, the an rvalue reference
// of the Message object, and a reference to the RpcAgent itself. Having a
// reference to the RpcAgent allows the ``RequestCallback`` implementation to
// be both stateless and non-blocking. It may enqueue the message and the
// RpcAgent reference, and use a different set of threads to process them later.
using RequestCallback = std::function<void(std::string, Message&&, RpcAgent&)>;

class RpcAgent {
 public:
  // The ``workerName`` is the globally unique name for this RpcAgent. It is up
  // to the RpcAgent implementation to determine how to resolve names.
  // The ``RequestCallback`` will be invoked to handle received requests. This
  // RpcAgent base class makes no assumption on the thread-safeness of the
  // ``RequestCallback``. RpcAgent implementations need to make sure that its
  // threading model conform to ``RequestCallback``'s requirement.
  RpcAgent(std::string workerName, RequestCallback cb);

  virtual ~RpcAgent();

  // Send a message to the ``RpcAgent`` of name ``to`` and returns a
  // ``FutureMessage`` ptr. The implementation must be asynchronous, i.e., it
  // cannot block until it receives the response.
  //
  // If ``message.isRequest()`` is true, the ``FutureMessage`` will be completed
  // when the response arrives. For other message types, the Future should be
  // ignored by the caller.
  //
  // TODO: avoid passing strings all the time, e.g., by using symbols as a
  // faster alternative.
  virtual std::shared_ptr<FutureMessage> send(
      const std::string& to, Message&& message) = 0;

  // Call sync and join all internal threads. This method should be called
  // before every RPC process exits.
  virtual void join() = 0;

  // Synchronize the this process with other RpcAgent processes. Block until all
  // RpcAgents reach this method and send all pending messages.
  virtual void sync() = 0;

 protected:
  const std::string workerName_;
  const RequestCallback cb_;
};

}
}
}
