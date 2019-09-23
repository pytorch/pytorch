#pragma once

#include <torch/csrc/distributed/rpc/future_message.h>
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/types.h>

#include <algorithm>

namespace torch {
namespace distributed {
namespace rpc {

// A globally unique ID to identify an RpcAgent
struct WorkerInfo {
  WorkerInfo(std::string name, int id)
      : WorkerInfo(std::move(name), (worker_id_t)id) {
    TORCH_CHECK(
        id <= std::numeric_limits<worker_id_t>::max(),
        "RPC worker id ",
        id,
        " out of bound of int16_t.");
  }

  WorkerInfo(std::string name, worker_id_t id)
      : name_(std::move(name)), id_(id) {
    bool validSize = name_.length() < MAX_NAME_LEN && name_.length() > 0;
    bool validChar =
        std::find_if(name_.begin(), name_.end(), [](char c) {
          return !(std::isalnum(c) || c == '-' || c == '_' || c == ':');
        }) == name_.end();
    TORCH_CHECK(
        validSize && validChar,
        "Worker name must match ^[A-Za-z0-9-_:]*$, "
        "and must be non-empty and shorter than ",
        MAX_NAME_LEN,
        " chars, "
        "but got ",
        name_);
  }

  static constexpr size_t MAX_NAME_LEN = 128;

  const std::string name_;
  const worker_id_t id_;
};

// ``RpcAgent`` is the base class for sending and receiving RPC messages. It
// provides a unified ``send`` API for both request and response messages, and
// will invoke the given ``RequestCallback`` to process received requests. It
// should immediately become ready to serve request and accept response after
// construction.
class RpcAgent;

// RpcAgent implementation should invoke ``RequestCallback`` to process received
// requests. There is no restriction on the implementation's threading model.
// This function takes an rvalue reference of the Message object.
// It is expected to return the response message or message containing an
// exception. Different rpc agent implementations are expected to ensure
// delivery of the response/exception based on their implementation specific
// mechanisms.
using RequestCallback = std::function<Message(Message&&)>;

class RpcAgent {
 public:
  // `WorkerInfo` is the globally unique identifier for this RpcAgent instance.
  // It contains a ``name_`` field and an ``id_`` field. ``name_`` is the
  // globally unique name for this ``RpcAgent``. It is up to the ``RpcAgent``
  // implementation to determine how to resolve names. ``id_`` is the globally
  // unique ID for this ``RpcAgent``. This should be determined by the
  // ``RpcAgent`` implementation.
  // The ``RequestCallback`` will be invoked to handle received requests. This
  // ``RpcAgent`` base class makes no assumption on the thread-safeness of the
  // ``RequestCallback``. ``RpcAgent`` implementations need to make sure that
  // its threading model conform to ``RequestCallback``'s requirement.
  RpcAgent(WorkerInfo id, RequestCallback cb);

  virtual ~RpcAgent();

  // Send a message to the ``RpcAgent`` of id ``to`` and returns a
  // ``FutureMessage`` ptr. The implementation must be asynchronous, i.e., it
  // cannot block until it receives the response.
  //
  // If ``message.isRequest()`` is true, the ``FutureMessage`` will be completed
  // when the response arrives. For other message types, the Future should be
  // ignored by the caller.
  std::shared_ptr<FutureMessage> send(const WorkerInfo& to, Message&& message);

  // Return a reference to the ``WorkerInfo`` of this RpcAgent.
  // NB: not using ``c10::optional<const std::string&>`` here because we might
  // need to create a separate RPC API lib and avoid forcing all ``RpcAgent``
  // implementations to depend on libtorch.
  const WorkerInfo& getWorkerInfo() const;

  // Return a reference to the ``WorkerInfo`` of the given ``workerName``.
  virtual const WorkerInfo& getWorkerInfo(
      const std::string& workerName) const = 0;

  virtual const WorkerInfo& getWorkerInfo(worker_id_t id) const = 0;

  // Call sync and join all internal threads. This method should be called
  // before every RPC process exits.
  virtual void join() = 0;

  // Synchronize the this process with other ``RpcAgent`` processes. Block until
  // all ``RpcAgent``s reach this method and send all pending messages.
  virtual void sync() = 0;

 protected:
  const WorkerInfo workerInfo_;

  // Method that needs to be overridden by all implementations of this
  // interface. The public send() method is responsible for common
  // pre-processing shared across all implementations.
  virtual std::shared_ptr<FutureMessage> sendImpl(
      const WorkerInfo& to,
      Message&& message) = 0;
  const std::string workerName_;
  const RequestCallback cb_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
