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

using steady_clock_time_point =
    std::chrono::time_point<std::chrono::steady_clock>;

struct TensorPipeRpcBackendOptions : public RpcBackendOptions {
  TensorPipeRpcBackendOptions(float rpc_timeout, std::string init_method)
      : RpcBackendOptions(rpc_timeout, init_method) {}
};

// Struct to track the network source metrics
struct NetworkSourceInfo {
  worker_id_t srcRank;
  std::vector<uint8_t> srcMachineAddr;
};

// Struct to track aggregated network metrics
struct AggregatedNetworkData {
  uint64_t numCalls{0};
  uint64_t totalSentBytes{0};
  uint64_t totalRecvBytes{0};
  uint64_t totalErrors{0};
};

// TensorPipeAgent leverages tensorpipe (https://github.com/pytorch/tensorpipe)
// to move tensors and payload through fatested transport and channel
// transparently. We can see it as a hybrid RPC transport, providing
// shared memory (linux) and tcp (linux & mac). CUDA will be supported next.
class TensorPipeAgent : public RpcAgent {
 public:
  TensorPipeAgent(
      std::shared_ptr<::c10d::Store> addressStore,
      std::string selfName,
      worker_id_t selfId,
      int worldSize,
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

  std::unordered_map<std::string, std::string> getMetrics() override;

  void addGilWaitTime(const std::chrono::microseconds gilWaitTime) override;

  using NetworkDataDict =
      std::unordered_map<std::string, AggregatedNetworkData>;

  NetworkDataDict getNetworkData();
  NetworkSourceInfo getNetworkSourceInfo();

 private:
  void collectNames();

  const std::string& findWorkerURL(const WorkerInfo& worker) const;

#ifdef TP_ENABLE_SHM
  std::string createUniqueShmAddr();
#endif

  // Retrieve IP address for a given network device for corss-hosts
  // to set up tensorpipe connection. For now we default the device
  // name eth0.
  static std::string getDefaultIPAddress();

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

  // Collects metrics from successful RPC calls
  void trackNetworkData(
      uint64_t requestSize,
      uint64_t responseSize,
      const std::string& destWorkerName);

  // Collects metrics from failed RPC calls
  void trackNetworkError(
      uint64_t requestSize,
      const std::string& destWorkerName);

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
  const int worldSize_;
  const TensorPipeRpcBackendOptions opts_;

  mutable std::mutex mutex_;
  uint64_t nextMessageID_{0};

  // Map to store the expiration times for each message.
  std::map<steady_clock_time_point, std::vector<std::shared_ptr<FutureMessage>>>
      timeoutMap_;

  // Thread that will poll the timeoutMap_ for timed out messages and mark them
  // with an error accordingly
  std::thread timeoutThread_;

  // Function run by the timeoutThread_ to check for timed out RPCs
  void pollTimeoutRpcs();

  // Mutex to guard the timeoutMap_
  std::mutex timeoutMapMutex_;

  // Condition Variable to signal population of the timeoutMap_
  std::condition_variable timeoutThreadCV_;

  // Returns the expiration time for an RPC by adding the current time to the
  // passed in timeout.
  inline steady_clock_time_point computeRpcMessageExpiryTime(
      std::chrono::milliseconds timeout) const {
    return std::chrono::time_point_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() + timeout);
  }

  // This is a generic struct for capturing Time-Series Metrics. It keeps a
  // running sum and count of data points (observations), and can return an
  // average of the data points seen so far. This is currently only used for
  // tracking the GIL Wait Time in RPC Agents, but can be used for other metrics
  // as well.
  struct TimeSeriesMetricsTracker {
    // Running sum of the data points seen so far
    uint64_t currentSum_;
    // Running count of the data points seen so far
    uint64_t currentCount_;

    explicit TimeSeriesMetricsTracker(
        uint64_t currentSum = 0,
        uint64_t currentCount = 0);

    // Adds a data point (which is basically one observation for the metric
    // being tracked) to the running sum and count.
    void addData(uint64_t dataPoint);
    // Returns the average of all the data points seen so far.
    float computeAverage() const;
  };

  // Map of Time-Series metrics tracked by the RPC Agent
  std::unordered_map<std::string, std::unique_ptr<TimeSeriesMetricsTracker>>
      timeSeriesMetrics_;
  // Mutex to guard timeSeriesMetrics_
  std::mutex metricsMutex_;

  // Map to Track Network Data
  NetworkDataDict networkData_;
  // Mutex to guarg networkData_
  std::mutex networkDataMutex_;

  // Running total of un-processed, un-errored RPC calls sent
  std::atomic<int32_t> clientActiveCalls_{0};
  // Running total of un-processed RPC requests received
  std::atomic<int32_t> serverActiveCalls_{0};
  // Running total of RPC requests that will be completed asynchronously
  std::atomic<int32_t> serverActiveAsyncCalls_{0};
};

} // namespace rpc
} // namespace distributed
} // namespace torch
