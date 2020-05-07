#pragma once

#include <c10/core/thread_pool.h>
#include <c10d/Store.hpp>
#include <tensorpipe/core/context.h>
#include <tensorpipe/core/listener.h>
#include <tensorpipe/core/pipe.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>

#include <atomic>
#include <thread>

namespace torch {
namespace distributed {
namespace rpc {

struct TensorPipeRpcBackendOptions : public RpcBackendOptions {
  TensorPipeRpcBackendOptions(
      std::map<std::string, worker_id_t> worker_name_to_id,
      float rpc_timeout,
      std::string init_method)
      : RpcBackendOptions(rpc_timeout, init_method),
        workerNameToId(std::move(worker_name_to_id)) {}

  std::map<std::string, worker_id_t> workerNameToId;
};

// TensorPipeAgent leverages tensorpipe (https://github.com/pytorch/tensorpipe)
// to move tensors and payload through fatested transport and channel
// transparently. We can see it as a hybrid RPC transport, providing
// shared memory (linux) and tcp (linux & mac). CUDA will be supported next.
class TensorPipeAgent : public RpcAgent {
 public:
  TensorPipeAgent(
      worker_id_t selfId,
      std::string selfName,
      std::shared_ptr<::c10d::Store> addressStore,
      TensorPipeRpcBackendOptions opts);

  TensorPipeAgent(const TensorPipeAgent&) = delete;
  TensorPipeAgent& operator=(const TensorPipeAgent&) = delete;

  std::shared_ptr<FutureMessage> send(
      const WorkerInfo& to,
      Message&& message,
      const float rpcTimeoutSeconds = kUnsetRpcTimeout) override;

  // join() and sync() would be deprecated -
  // https://github.com/pytorch/pytorch/issues/27647
  void join() override;
  void sync() override;
  void startImpl() override;
  void shutdownImpl() override;

  ~TensorPipeAgent() override;

  const WorkerInfo& getWorkerInfo(const std::string& workerName) const override;
  const WorkerInfo& getWorkerInfo(worker_id_t workerId) const override;
  std::vector<WorkerInfo> getWorkerInfos() const override;

  std::unordered_map<std::string, std::string> getMetrics() override {
    std::unordered_map<std::string, std::string> metrics;
    return metrics;
  }

  void addGilWaitTime(const std::chrono::microseconds /* unused */) override {}

 private:
  const std::string& findWorkerURL(const WorkerInfo& worker) const;

#ifdef TP_ENABLE_SHM
  std::string createUniqueShmAddr();
#endif

  // TensorPipe read function that could be used to read response messages
  // by client, and read request messages by server.
  void pipeRead(
      const std::shared_ptr<tensorpipe::Pipe>&,
      std::function<void(const tensorpipe::Error&, Message&&)>);

  // TensorPipe write function that could be used to write response
  // messages by server, and write request messages by client.
  void pipeWrite(
      const std::shared_ptr<tensorpipe::Pipe>&,
      Message&& message,
      std::function<void(const tensorpipe::Error&)>);

  // Callback of listener accept()
  void onListenerAccepted(
      const tensorpipe::Error& error,
      std::shared_ptr<tensorpipe::Pipe>& pipe);

  // Respond to a call from a peer
  void respond(std::shared_ptr<tensorpipe::Pipe>& pipe);

  void sendCompletedResponseMessage(
      std::shared_ptr<tensorpipe::Pipe>& pipe,
      std::shared_ptr<FutureMessage>& futureResponseMessage,
      uint64_t messageId);

  // State per client pipe to keep tracking of pending response message
  // and error sate. pendingResponseMessage_ should be protected by
  // mutex since it can be raced with user send() call.
  // TODO: To achieve better performance we can have a pipe pool per
  // client and work together with RpcBackendOptions to configure.
  struct ClientPipe {
    explicit ClientPipe(std::shared_ptr<tensorpipe::Pipe> pipe) : pipe_(pipe) {}
    std::shared_ptr<tensorpipe::Pipe> pipe_;
    bool readError_{false};
    std::unordered_map<uint64_t, std::shared_ptr<FutureMessage>>
        pendingResponseMessage_;
  };

  // TODO: configure thread pool size through RpcBackendOptions.
  ThreadPool threadPool_{16};
  std::shared_ptr<tensorpipe::Context> context_;
  std::shared_ptr<tensorpipe::Listener> listener_;
  std::unordered_map<worker_id_t, ClientPipe> connectedPipes_;

  // We need map one keyed on name and one on id for easy lookup.
  std::unordered_map<worker_id_t, WorkerInfo> workerIdToInfo_;
  std::unordered_map<std::string, WorkerInfo> workerNameToInfo_;
  std::unordered_map<std::string, std::string> workerNameToURL_;

  const std::shared_ptr<::c10d::Store> addressStore_;
  const TensorPipeRpcBackendOptions opts_;

  mutable std::mutex mutex_;
  uint64_t nextMessageID_{0};
};

} // namespace rpc
} // namespace distributed
} // namespace torch
