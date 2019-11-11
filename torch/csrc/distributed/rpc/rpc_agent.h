#pragma once

#include <torch/csrc/distributed/rpc/future_message.h>
#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/request_callback.h>
#include <torch/csrc/distributed/rpc/types.h>
#include <torch/csrc/utils/future.h>

#include <algorithm>
#include <cctype>

namespace torch {
namespace distributed {
namespace rpc {

// A globally unique ID to identify an RpcAgent
struct TORCH_API WorkerInfo {
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
class TORCH_API RpcAgent {
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
  // NB: RpcAgent implementations should not start serving requests until
  // ``start()`` is called, as there could be other contexts that have not been
  // initialized yet at this time.
  RpcAgent(
      WorkerInfo id,
      std::unique_ptr<RequestCallback> cb,
      std::chrono::milliseconds rpcTimeout);

  virtual ~RpcAgent();

  // Send a message to the ``RpcAgent`` of id ``to`` and returns a
  // ``Future<Message>`` ptr. The implementation must be asynchronous, i.e., it
  // cannot block until it receives the response.
  //
  // If ``message.isRequest()`` is true, the ``Future<Message>`` will be
  // completed when the response arrives. For other message types, the Future
  // should be ignored by the caller.
  virtual std::shared_ptr<torch::utils::Future<Message>> send(
      const WorkerInfo& to,
      Message&& message) = 0;

  // Return a reference to the ``WorkerInfo`` of this RpcAgent.
  // NB: not using ``c10::optional<const std::string&>`` here because we might
  // need to create a separate RPC API lib and avoid forcing all ``RpcAgent``
  // implementations to depend on libtorch.
  const WorkerInfo& getWorkerInfo() const;

  // Return a reference to the ``WorkerInfo`` of the given ``workerName``.
  virtual const WorkerInfo& getWorkerInfo(
      const std::string& workerName) const = 0;

  virtual const WorkerInfo& getWorkerInfo(worker_id_t id) const = 0;

  // Retrieve the timeout for all RPCs.
  inline const std::chrono::milliseconds& getRpcTimeout() const {
    return rpcTimeout_;
  }

  // Call sync and join all internal threads. This method should be called
  // before every RPC process exits.
  virtual void join() = 0;

  // Synchronize the this process with other ``RpcAgent`` processes. Block until
  // all ``RpcAgent``s reach this method and send all pending messages.
  virtual void sync() = 0;

  // start accepting requests
  virtual void start() {}

  // Set the default rpc agent.
  static void setDefaultRpcAgent(std::shared_ptr<RpcAgent> defaultRpcAgent);

  // Retrieve the default rpc agent.
  static std::shared_ptr<RpcAgent> getDefaultRpcAgent();

 protected:
  const WorkerInfo workerInfo_;
  const std::string workerName_;
  const std::unique_ptr<RequestCallback> cb_;
  const std::chrono::milliseconds rpcTimeout_;

 private:
  static std::shared_ptr<RpcAgent> defaultRpcAgent_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
