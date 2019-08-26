#pragma once

#include <torch/csrc/distributed/rpc/future_message.h>
#include <torch/csrc/distributed/rpc/message.h>
//#include <torch/csrc/distributed/rpc/types.h>

namespace torch {
namespace distributed {
namespace rpc {

using worker_id_t = int16_t;


// ``RpcAgent`` is the base class for sending and receiving RPC messages. It
// provides a unified ``send`` API for both request and response messages, and
// will invoke the given ``RequestCallback`` to process received requests. It
// should immediately become ready to serve request and accept response after
// construction.
class RpcAgent;

// ``RpcAgent`` implementation should invoke ``RequestCallback`` to process
// received requests. There is no restriction on the implementation's threading
// model. This function takes the id of the request sender, the an rvalue
// reference of the Message object, and a reference to the ``RpcAgent`` itself.
// Having a reference to the ``RpcAgent`` allows the ``RequestCallback``
// implementation to be both stateless and non-blocking. For example, t may
// enqueue the message and the ``RpcAgent`` reference, and use a different set
// of threads to process them later.
using RequestCallback = std::function<void(worker_id_t, Message&&, RpcAgent&)>;

class RpcAgent {
 public:
  // ``workerName`` is the globally unique name for this ``RpcAgent``. It is up
  // to the ``RpcAgent`` implementation to determine how to resolve names.
  // ``id`` is the globally unique ID for this ``RpcAgent``. This should be
  // determined by the ``RpcAgent`` implementation.
  // The ``RequestCallback`` will be invoked to handle received requests. This
  // ``RpcAgent`` base class makes no assumption on the thread-safeness of the
  // ``RequestCallback``. ``RpcAgent`` implementations need to make sure that
  // its threading model conform to ``RequestCallback``'s requirement.
  RpcAgent(std::string workerName, worker_id_t id, RequestCallback cb);

  virtual ~RpcAgent();

  // Send a message to the ``RpcAgent`` of id ``to`` and returns a
  // ``FutureMessage`` ptr. The implementation must be asynchronous, i.e., it
  // cannot block until it receives the response.
  //
  // If ``message.isRequest()`` is true, the ``FutureMessage`` will be completed
  // when the response arrives. For other message types, the Future should be
  // ignored by the caller.
  virtual std::shared_ptr<FutureMessage> send(
      worker_id_t to, Message&& message) = 0;

  // Return the id of this RpcAgent
  virtual worker_id_t getId() = 0;

  // Return the id of the given ``workerName``.
  virtual worker_id_t getWorkerId(const std::string& workerName) = 0;

  // Call sync and join all internal threads. This method should be called
  // before every RPC process exits.
  virtual void join() = 0;

  // Synchronize the this process with other ``RpcAgent`` processes. Block until
  // all ``RpcAgent``s reach this method and send all pending messages.
  virtual void sync() = 0;

 protected:
  const std::string workerName_;
  const worker_id_t id_;
  const RequestCallback cb_;
};

}
}
}
