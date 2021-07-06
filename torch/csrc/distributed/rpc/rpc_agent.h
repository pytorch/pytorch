#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/request_callback.h>
#include <torch/csrc/distributed/rpc/types.h>

#include <algorithm>
#include <cctype>

namespace torch {
namespace distributed {
namespace rpc {

using DeviceMap = std::unordered_map<c10::Device, c10::Device>;

// Default RPC timeout
constexpr float kDefaultRpcTimeoutSeconds = 60;
// Unset RPC timeout. This is the value agent::send() will have if user does not
// pass in a specific timeout, and indicates that we must use the default
// timeout for RPCs.
constexpr float kUnsetRpcTimeout = -1;
constexpr auto kDefaultInitMethod = "env://";
constexpr float kSecToMsConversion = 1000;
constexpr auto kRpcTimeoutErrorStr =
    "RPC ran for more than set timeout ({} ms) and will now be marked with an error";

using steady_clock_time_point =
    std::chrono::time_point<std::chrono::steady_clock>;
// Input is qualified name string, output is JIT StrongTypePtr
// Same as jit::TypeResolver, did not import jit::TypeResolver to here
// because it could instroduce cyclic dependencies.
using TypeResolver =
    std::function<c10::StrongTypePtr(const c10::QualifiedName&)>;

struct RpcBackendOptions {
  RpcBackendOptions()
      : RpcBackendOptions(kDefaultRpcTimeoutSeconds, kDefaultInitMethod) {}

  RpcBackendOptions(float rpcTimeoutSeconds, std::string initMethod)
      : rpcTimeoutSeconds(rpcTimeoutSeconds),
        initMethod(std::move(initMethod)) {
    TORCH_CHECK(rpcTimeoutSeconds >= 0, "RPC Timeout must be non-negative");
  }

  float rpcTimeoutSeconds;
  std::string initMethod;
};

// A globally unique ID to identify an RpcAgent
struct TORCH_API WorkerInfo : torch::CustomClassHolder {
  WorkerInfo(std::string name, int64_t id)
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

  bool operator==(const WorkerInfo& rhs) {
    return (id_ == rhs.id_) && (name_ == rhs.name_);
  }

  static constexpr size_t MAX_NAME_LEN = 128;

  const std::string name_;
  const worker_id_t id_;
};

TORCH_API std::ostream& operator<<(
    std::ostream& os,
    const WorkerInfo& workerInfo);

// Struct for options to configure the RPC Retry protocol.
struct TORCH_API RpcRetryOptions {
  // Using a default constructor like all other Options structs in the RPC
  // codebase. TORCH_CHECKs for input validation are done in the
  // sendWithRetries function.
  RpcRetryOptions() = default;
  // Maximum number of times we will retry the RPC
  int maxRetries{5};
  // Initial duration between consecutive RPC send attempts
  std::chrono::milliseconds rpcRetryDuration{std::chrono::milliseconds(1000)};
  // Constant for exponential backoff used while calculating future wait
  // durations
  float retryBackoff{1.5};
};

// Struct that stores all the metadata needed to retry a given RPC.
struct TORCH_API RpcRetryInfo {
  RpcRetryInfo(
      const WorkerInfo& to,
      c10::intrusive_ptr<Message> message,
      c10::intrusive_ptr<JitFuture> originalFuture,
      int retryCount,
      RpcRetryOptions options)
      : to_(to),
        message_(std::move(message)),
        originalFuture_(std::move(originalFuture)),
        retryCount_(retryCount),
        options_(options) {}

  const WorkerInfo& to_;
  c10::intrusive_ptr<Message> message_;
  // Future that is returned to the caller of sendWithRetries().
  c10::intrusive_ptr<JitFuture> originalFuture_;
  // Number of send attempts completed so far.
  int retryCount_;
  RpcRetryOptions options_;
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
  // ``JitFuture`` ptr. The implementation must be asynchronous, i.e., it
  // cannot block until it receives the response.
  //
  // If ``message.isRequest()`` is true, the ``JitFuture`` will be
  // completed when the response arrives. For other message types, the Future
  // should be ignored by the caller.
  virtual c10::intrusive_ptr<JitFuture> send(
      const WorkerInfo& to,
      c10::intrusive_ptr<Message> message,
      const float rpcTimeoutSeconds = kUnsetRpcTimeout,
      const std::unordered_map<c10::Device, c10::Device>& deviceMap = {}) = 0;

  // Retries sending the message up to maxRetries times until an ACK is
  // receieved. The duration between consecutive sends is increased over
  // time using an exponential backoff algorithm.
  //
  // Sends ``message`` to the ``RpcAgent`` of id ``to`` and returns a
  // ``JitFuture`` ptr, just like send(). Caller can specify the maximum
  // number of retries for this RPC (default is 5), initial duration between
  // sends (default is 1000ms), and backoff constant (default is 1.5) by
  // passing in the RpcRetryOptions struct. This API might end up
  // executing a method twice on the remote end (it does not guarantee
  // exactly-once semantics). Therefore, the user must ensure their requests
  // are idempotent.
  c10::intrusive_ptr<JitFuture> sendWithRetries(
      const WorkerInfo& to,
      c10::intrusive_ptr<Message> message,
      RpcRetryOptions retryOptions = RpcRetryOptions());

  // Return a reference to the ``WorkerInfo`` of this RpcAgent.
  // NB: not using ``c10::optional<const std::string&>`` here because we might
  // need to create a separate RPC API lib and avoid forcing all ``RpcAgent``
  // implementations to depend on libtorch.
  const WorkerInfo& getWorkerInfo() const;

  // Return a reference to the ``WorkerInfo`` of the given ``workerName``.
  virtual const WorkerInfo& getWorkerInfo(
      const std::string& workerName) const = 0;

  virtual const WorkerInfo& getWorkerInfo(worker_id_t id) const = 0;

  virtual std::vector<WorkerInfo> getWorkerInfos() const = 0;

  // Retrieve the timeout for all RPCs.
  inline std::chrono::milliseconds getRpcTimeout() const {
    return rpcTimeout_.load();
  }

  // Set the timeout for all RPCs
  inline void setRpcTimeout(const std::chrono::milliseconds& rpcTimeout) {
    rpcTimeout_.store(rpcTimeout);
  }

  // Call sync and join all internal threads. This method should be called
  // before every RPC process exits.
  virtual void join(bool shutdown = false) = 0;

  // Synchronize the this process with other ``RpcAgent`` processes. Block until
  // all ``RpcAgent``s reach this method and send all pending messages.
  virtual void sync() = 0;

  // Sets up backend-agnostic state for accepting requests. Currently, this
  // entails setting rpcAgentRunning_ to true, creating the retry thread, and
  // calling the backend's startImpl.
  void start();

  // Derived classes must override this function to start accepting requests.
  // This is used to initialize any backend-specific state. Users must call
  // start, not startImpl, to initialize the RPC Agent.
  virtual void startImpl() = 0;

  // Stop accepting requests and shutdown the RPC framework as soon as possible
  // by terminating all RPC threads.
  void shutdown();

  // Derived classes must override this function to start accepting requests.
  // THis is used to clean up any backend-specific state. Users must call
  // shutdown, not shutdownImpl, to shutdown the RPC Agent.
  virtual void shutdownImpl() = 0;

  // Check if current RPC agent is set.
  static bool isCurrentRpcAgentSet();

  // Retrieve the valid current RPC agent.
  static std::shared_ptr<RpcAgent> getCurrentRpcAgent();

  // Set the current RPC agent.
  static void setCurrentRpcAgent(std::shared_ptr<RpcAgent> rpcAgent);

  // Retrieve metrics as KV map
  virtual std::unordered_map<std::string, std::string> getMetrics() = 0;

  // Retrive debug info in addition to metrics as KV map
  virtual std::unordered_map<std::string, std::string> getDebugInfo();

  // Flag to control whether GIL wait times
  // should be profiled or not.
  void enableGILProfiling(bool flag);

  // Retrieve wheher we should profile GIL wait times or not.
  bool isGILProfilingEnabled();

  // Set type resolver that will be passed to JIT pickler to resolver type Ptr
  // based on type str.
  void setTypeResolver(std::shared_ptr<TypeResolver> typeResolver);

  // Get the type resolver
  std::shared_ptr<TypeResolver> getTypeResolver();

  // Retrieves the device map for the provided destination worker.
  virtual DeviceMap getDeviceMap(const WorkerInfo& dst) const;

  // Retrieve the (non-CPU) devices that are supported by the agent.
  virtual const std::vector<c10::Device>& getDevices() const;

 protected:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const WorkerInfo workerInfo_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const std::unique_ptr<RequestCallback> cb_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::atomic<std::chrono::milliseconds> rpcTimeout_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::atomic<bool> profilingEnabled_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::shared_ptr<TypeResolver> typeResolver_;
  // Atomic boolean indicating whether this agent is running. It controls
  // whether several background threads should be running. It is set in
  // RpcAgent::start() and unset in the derived class shutdown().
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::atomic<bool> rpcAgentRunning_;

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static std::shared_ptr<RpcAgent> currentRpcAgent_;
  // Add GIL wait time data point to metrics
  virtual void addGilWaitTime(const std::chrono::microseconds gilWaitTime) = 0;
  friend class PythonRpcHandler;

  // Map that stores metadata for RPC's that may need to be re-tried as well as
  // the timepoint at which we should re-try them.
  std::map<
      steady_clock_time_point,
      std::unordered_set<std::shared_ptr<RpcRetryInfo>>>
      rpcRetryMap_;

  // Thread that checks for retryable RPC's in the rpcRetryMap_ and sleeps until
  // the next unACKed RPC's timeout has expired.
  std::thread rpcRetryThread_;

  // Function that rpcRetryThread_ calls in a loop as long as RpcAgent is
  // running.
  void retryExpiredRpcs();

  // This is the callback attached to futures corresponding to send retries.
  // This handles 3 cases: 1). send was completed, 2). send failed with an
  // error and we've done maxRetries failed send attempts, and 3). send
  // failed with an error and we have more retries to go. In case 1, we mark
  // the original future as complete. In case 2, we mark the future with an
  // error and do not retry again. In case 3, we move the RpcRetryInfo struct
  // to another time point in the map to schedule the RPC for a future send.
  void rpcRetryCallback(
      JitFuture& message,
      steady_clock_time_point newTime,
      std::shared_ptr<RpcRetryInfo> earliestRpc);

  // Function that uses the exponential backoff algorithm to compute the next
  // time point to retry a given RPC.
  inline steady_clock_time_point computeNewRpcRetryTime(
      RpcRetryOptions& options,
      int retryCount) {
    // The exponential backoff algorithm being used here is:
    // newTime = timeNow + (retryDuration * (backoffConstant ^ retryCount)).
    std::chrono::milliseconds timedelta =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            options.rpcRetryDuration * pow(options.retryBackoff, retryCount));
    return std::chrono::time_point_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() + timedelta);
  }

  // Condition Variable to signal when the rpcRetryMap_ has been populated.
  std::condition_variable rpcRetryMapCV_;

  // Mutex to protect RpcRetryMap_.
  std::mutex rpcRetryMutex_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch

namespace std {
template <>
struct hash<torch::distributed::rpc::WorkerInfo> {
  std::size_t operator()(
      const torch::distributed::rpc::WorkerInfo& worker_info) const noexcept {
    return worker_info.id_;
  }
};
} // namespace std
