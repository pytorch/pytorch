#pragma once

#ifdef USE_TENSORPIPE

#include <atomic>
#include <thread>

#include <c10/core/thread_pool.h>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <utility>

// Forward-declare the TensorPipe classes we need, to avoid including its
// headers in PyTorch's ones and thus have it become a public dependency.

namespace tensorpipe {

class Context;
class Error;
class Listener;
class Message;
class Pipe;

namespace transport {
class Context;
} // namespace transport

namespace channel {
class Context;
} // namespace channel

} // namespace tensorpipe

namespace torch::distributed::rpc {

// These priorities instruct TensorPipe on which transport/channel to pick
// during handshake. Higher priorities will take precedence over lower ones.
// The transport with lowest priority will be the one used to bootstrap pipes.

constexpr int64_t kShmTransportPriority = 200;
constexpr int64_t kIbvTransportPriority = 100;
// The UV transport just uses TCP and should work everywhere, thus keep it last.
constexpr int64_t kUvTransportPriority = 0;

constexpr int64_t kCmaChannelPriority = 1200;
constexpr int64_t kMultiplexedUvChannelPriority = 1100;
// The basic channel reuses a transport as a channel, and is thus our fallback.
constexpr int64_t kBasicChannelPriority = 1000;

// CPU channel have higher priority than CUDA channels, since the latter might
// handle CPU-to-CPU transfers, but will always be less efficient than their
// CPU-only counterparts.
constexpr int64_t kCudaIpcChannelPriority = 300;
constexpr int64_t kCudaGdrChannelPriority = 200;
constexpr int64_t kCudaXthChannelPriority = 400;
constexpr int64_t kCudaBasicChannelPriority = 0;

using steady_clock_time_point =
    std::chrono::time_point<std::chrono::steady_clock>;

struct TORCH_API TransportRegistration {
  std::shared_ptr<tensorpipe::transport::Context> transport;
  int64_t priority;
  std::string address;
};

TORCH_DECLARE_REGISTRY(TensorPipeTransportRegistry, TransportRegistration);

struct TORCH_API ChannelRegistration {
  std::shared_ptr<tensorpipe::channel::Context> channel;
  int64_t priority;
};

TORCH_DECLARE_REGISTRY(TensorPipeChannelRegistry, ChannelRegistration);

constexpr auto kDefaultNumWorkerThreads = 16;

struct TORCH_API TensorPipeRpcBackendOptions : public RpcBackendOptions {
  TensorPipeRpcBackendOptions(
      int numWorkerThreads,
      std::optional<std::vector<std::string>> transports,
      std::optional<std::vector<std::string>> channels,
      float rpc_timeout,
      std::string init_method,
      std::unordered_map<std::string, DeviceMap> device_maps = {},
      std::vector<c10::Device> devices = {})
      : RpcBackendOptions(rpc_timeout, std::move(init_method)),
        numWorkerThreads(numWorkerThreads),
        transports(std::move(transports)),
        channels(std::move(channels)),
        deviceMaps(std::move(device_maps)),
        devices(std::move(devices)) {
    TORCH_CHECK(
        numWorkerThreads > 0,
        "num_worker_threads must be positive, got ",
        numWorkerThreads);

    if (this->transports.has_value()) {
      for (const std::string& transportName : this->transports.value()) {
        TORCH_CHECK(
            TensorPipeTransportRegistry()->Has(transportName),
            "Unknown transport: ",
            transportName);
      }
    }

    if (this->channels.has_value()) {
      for (const std::string& channelName : this->channels.value()) {
        TORCH_CHECK(
            TensorPipeChannelRegistry()->Has(channelName),
            "Unknown channel: ",
            channelName);
      }
    }
  }

  void setDeviceMap(const std::string& workerName, const DeviceMap& deviceMap) {
    auto iter = deviceMaps.find(workerName);
    if (iter == deviceMaps.end()) {
      deviceMaps[workerName] = deviceMap;
    } else {
      for (auto& entry : deviceMap) {
        // c10::Device has no default constructor, hence map[device] doesn't
        // work In C++-17 we can use insert_or_assign.
        auto entryIter = iter->second.find(entry.first);
        if (entryIter == iter->second.end()) {
          iter->second.emplace(entry.first, entry.second);
        } else {
          entryIter->second = entry.second;
        }
      }
    }
  }

  int numWorkerThreads;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const std::optional<std::vector<std::string>> transports;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const std::optional<std::vector<std::string>> channels;
  std::unordered_map<std::string, DeviceMap> deviceMaps;
  std::vector<c10::Device> devices;
};

// Struct to track the network source metrics
struct TORCH_API NetworkSourceInfo {
  worker_id_t srcRank;
  std::vector<uint8_t> srcMachineAddr;
};

// Struct to track aggregated network metrics
struct TORCH_API AggregatedNetworkData {
  uint64_t numCalls{0};
  uint64_t totalSentBytes{0};
  uint64_t totalRecvBytes{0};
  uint64_t totalErrors{0};
};

// TensorPipeAgent leverages TensorPipe (https://github.com/pytorch/tensorpipe)
// to transparently move tensors and payloads through the fastest available
// transport or channel. It acts like a hybrid RPC transport, providing shared
// memory (linux) and TCP (linux & mac) support. CUDA support is in progress.
class TORCH_API TensorPipeAgent : public RpcAgent {
 public:
  TensorPipeAgent(
      const c10::intrusive_ptr<::c10d::Store>& store,
      std::string selfName,
      worker_id_t selfId,
      std::optional<int> worldSize,
      TensorPipeRpcBackendOptions opts,
      std::unordered_map<std::string, DeviceMap> reverseDeviceMaps,
      std::vector<c10::Device> devices,
      std::unique_ptr<RequestCallback> cb);

  TensorPipeAgent(const TensorPipeAgent&) = delete;
  TensorPipeAgent& operator=(const TensorPipeAgent&) = delete;

  c10::intrusive_ptr<JitFuture> send(
      const WorkerInfo& to,
      c10::intrusive_ptr<Message> message,
      const float rpcTimeoutSeconds = kUnsetRpcTimeout,
      const DeviceMap& deviceMap = {}) override;

  // join() and sync() would be deprecated -
  // https://github.com/pytorch/pytorch/issues/27647
  void join(bool shutdown = false, float timeout = 0) override;
  void sync() override {}
  void startImpl() override;
  void shutdownImpl() override;

  ~TensorPipeAgent() override;

  const WorkerInfo& getWorkerInfo(const std::string& workerName) const override;
  const WorkerInfo& getWorkerInfo(worker_id_t workerId) const override;
  std::vector<WorkerInfo> getWorkerInfos() const override;
  void updateGroupMembership(
      const WorkerInfo& workerInfo,
      const std::vector<c10::Device>& devices,
      const std::unordered_map<std::string, DeviceMap>& reverseDeviceMaps,
      bool isJoin);

  std::unordered_map<std::string, std::string> getMetrics() override;

  void addGilWaitTime(const std::chrono::microseconds gilWaitTime) override;

  TensorPipeRpcBackendOptions getBackendOptions() const;

  const c10::intrusive_ptr<::c10d::Store> getStore() const;

  DeviceMap getDeviceMap(const WorkerInfo& dest) const override;

  const std::vector<c10::Device>& getDevices() const override;

  using NetworkDataDict =
      std::unordered_map<std::string, AggregatedNetworkData>;

  // Returns metrics tracked by the NetworkDataDict
  NetworkDataDict getNetworkData();
  // Returns NetworkSourceInfo struct
  NetworkSourceInfo getNetworkSourceInfo();

  static const std::string& guessAddress();

  // For testing purposes.
  size_t timeoutMapSize();
  size_t numPendingResponses();
  size_t messageIdToTimeoutMapSize();

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const bool isStaticGroup_;

 protected:
  // TensorPipe write function that could be used to write response
  // messages by server, and write request messages by client. This
  // is a protected method since it is overwritten by FaultyTensorPipeAgent
  virtual void pipeWrite(
      const std::shared_ptr<tensorpipe::Pipe>&,
      const c10::intrusive_ptr<Message>& message,
      std::vector<c10::Device>&& devices,
      std::vector<c10::Stream> streams,
      std::function<void(const tensorpipe::Error&)>) noexcept;

 private:
  // Removes the given messageId with the given expirationTime from the
  // timeoutMap_.
  void removeFromTimeoutMap(uint64_t messageId);

  // Populates workerIdToInfo_ and workerNameToInfo_ using addressStore_
  void prepareNames(bool isStaticGroup);

  // Check the static group attribute with the value set in store
  void checkAndSetStaticGroup(const c10::intrusive_ptr<::c10d::Store>& store);

  const std::string& findWorkerURL(const WorkerInfo& worker) const;

  // Only use for Dynamic RPC groups, method to have worker leave group
  void leaveGroup();

  // TensorPipe read function that could be used to read response messages
  // by client, and read request messages by server.
  void pipeRead(
      const std::shared_ptr<tensorpipe::Pipe>&,
      std::function<void(
          const tensorpipe::Error&,
          c10::intrusive_ptr<Message>,
          std::vector<c10::Stream>)>) noexcept;

  // Callback of listener accept()
  void onListenerAccepted(
      const tensorpipe::Error& error,
      std::shared_ptr<tensorpipe::Pipe>& pipe);

  // Respond to a call from a peer
  void respond(std::shared_ptr<tensorpipe::Pipe>& pipe);

  void sendCompletedResponseMessage(
      std::shared_ptr<tensorpipe::Pipe>& pipe,
      JitFuture& futureResponseMessage,
      uint64_t messageId,
      std::vector<c10::Stream> stream);

  // Collects metrics from successful RPC calls
  void trackNetworkData(
      uint64_t requestSize,
      uint64_t responseSize,
      const std::string& destWorkerName);

  // Collects metrics from failed RPC calls
  void trackNetworkError(
      uint64_t requestSize,
      const std::string& destWorkerName);

  inline std::vector<c10::Device> getDevicesForRemote(
      const std::string& remoteName,
      const Message& message) const;

  // When a request+response completes, we need to mark the future message as
  // complete. However, if its timeout has already expired, it already has an
  // error set. There is no atomic "test-and-set" way to mark a future complete
  // only if it isn't yet. It does exist for errors (setErrorIfNeeded) but, even
  // then, it ends up printing a log message, which may worry the user. To solve
  // both issues we use a separate atomic flag to know the status of the future.
  struct AtomicJitFuture {
    explicit AtomicJitFuture(const std::vector<c10::Device>& devices) {
      jitFuture = c10::make_intrusive<at::ivalue::Future>(
          at::AnyClassType::get(), devices);
    }

    std::atomic_flag isComplete = ATOMIC_FLAG_INIT;
    c10::intrusive_ptr<JitFuture> jitFuture;
  };

  // Maintains state per client pipe to track pending response messages and
  // error states. pendingResponseMessage_ should be protected by a mutex since
  // it can be raced with user send() call.
  // TODO: To achieve better performance we can have a pipe pool per
  // client that can be configured using RpcBackendOptions.
  struct ClientPipe {
    explicit ClientPipe(std::shared_ptr<tensorpipe::Pipe> pipe)
        : pipe_(std::move(pipe)) {}
    std::shared_ptr<tensorpipe::Pipe> pipe_;
    mutable std::mutex mutex_;
    bool inError_{false};
    // Map from Message Request ID's to corresponding futures.
    std::unordered_map<uint64_t, std::shared_ptr<AtomicJitFuture>>
        pendingResponseMessage_;
  };

  const c10::intrusive_ptr<::c10d::Store> store_;

  const TensorPipeRpcBackendOptions opts_;
  // For dynamic RPC, the reverse device maps are updated whenever a new rank
  // joins or leaves the group
  std::unordered_map<std::string, DeviceMap> reverseDeviceMaps_;
  // Local devices used by this agent. If application didn't specify this
  // field, it will be initialized using corresponding local devices in
  // opts_.deviceMaps and reverseDeviceMaps_;
  std::vector<c10::Device> devices_;

  ThreadPool threadPool_;
  std::shared_ptr<tensorpipe::Context> context_;
  std::shared_ptr<tensorpipe::Listener> listener_;

  mutable std::mutex connectedPipesMutex_;
  std::unordered_map<worker_id_t, ClientPipe> connectedPipes_;

  // Maps keyed on name and id for easy WorkerInfo lookup.
  std::unordered_map<worker_id_t, WorkerInfo> workerIdToInfo_;
  std::unordered_map<std::string, WorkerInfo> workerNameToInfo_;
  std::unordered_map<std::string, std::string> workerNameToURL_;

  ::c10d::PrefixStore rankToNameStore_;
  ::c10d::PrefixStore nameToAddressStore_;
  // Store keys that will used to count joined processes and active calls during
  // the shutdown process
  ::c10d::PrefixStore shutdownStore_;
  int worldSize_ = 0;
  std::atomic<uint64_t> nextMessageID_{0};

  // Metadata used for tracking of whether certain RPCs have timed out or not.
  struct TimeoutMessageMetadata {
    TimeoutMessageMetadata(
        uint64_t messageId_,
        std::shared_ptr<AtomicJitFuture> responseFuture_,
        std::chrono::milliseconds timeout_)
        : messageId(messageId_),
          responseFuture(std::move(responseFuture_)),
          timeout(timeout_) {}
    uint64_t messageId;
    std::shared_ptr<AtomicJitFuture> responseFuture;
    std::chrono::milliseconds timeout;
  };

  // Map to store the expiration times for each message.
  std::map<steady_clock_time_point, std::vector<TimeoutMessageMetadata>>
      timeoutMap_;

  // Map to store the messageId to expiry time.
  std::unordered_map<uint64_t, steady_clock_time_point> messageIdToTimeout_;

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

  // Handle error on an outgoing pipe
  void handleClientError(
      ClientPipe& clientPipe,
      const tensorpipe::Error& error);

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
  std::unordered_map<std::string, TimeSeriesMetricsTracker> timeSeriesMetrics_;
  // Mutex to guard timeSeriesMetrics_
  std::mutex metricsMutex_;

  // Custom lock guard used to check if the RPC group is dynamic and lock the
  // mutex if so
  struct GroupMembershipLockGuard {
    GroupMembershipLockGuard(std::mutex& mutex, bool isStaticGroup)
        : ref_(mutex), isStaticGroup_(isStaticGroup) {
      if (isStaticGroup_) {
        ref_.lock();
      }
    }

    ~GroupMembershipLockGuard() {
      if (isStaticGroup_) {
        ref_.unlock();
      }
    }

    GroupMembershipLockGuard(const GroupMembershipLockGuard&) = delete;

   private:
    std::mutex& ref_;
    bool isStaticGroup_;
  };
  // Mutex to guard access to group membership data
  // e.g. updates to (workerIdToInfo_, workerNameToInfo_, workerNameToURL_)
  mutable std::mutex groupMembershipMutex_;

  // Map to Track Network Data
  NetworkDataDict networkData_;
  // Mutex to guard networkData_
  std::mutex networkDataMutex_;

  // A mutex and a cv to guard access to the call counts and watch for changes.
  std::mutex callCountMutex_;
  std::condition_variable callCountCV_;
  // Running total of un-processed, un-errored RPC calls sent
  int32_t clientActiveCalls_{0};
  // Running total of un-processed RPC requests received
  int32_t serverActiveCalls_{0};
  // Running total of RPC requests that will be completed asynchronously
  int32_t serverActiveAsyncCalls_{0};

  // Whether a global graceful shutdown has begun, in which case we'll silence
  // error messages due to remote workers closing their pipes.
  std::atomic<bool> shuttingDown_{false};

  // Helpers to modify the counts while correctly dealing with the mutex and cv.
  void increaseCallCount(int32_t& count);
  void decreaseCallCount(int32_t& count);

  // Helpers to set the state of the requests.
  void markFutureAsComplete(
      std::shared_ptr<AtomicJitFuture> atomicFuture,
      c10::intrusive_ptr<Message> message,
      std::vector<c10::Stream> streams);
  void markFutureWithError(
      std::shared_ptr<AtomicJitFuture> atomicFuture,
      std::string errorMsg);
};

} // namespace torch::distributed::rpc

#endif // USE_TENSORPIPE
