#include <torch/csrc/distributed/rpc/tensorpipe_agent.h>

#ifdef USE_TENSORPIPE

#include <limits>

#include <fmt/format.h>
#include <tensorpipe/tensorpipe.h>

#include <torch/csrc/distributed/rpc/tensorpipe_utils.h>
#include <torch/csrc/distributed/rpc/utils.h>

#ifdef USE_CUDA_NOT_ROCM
#include <ATen/cuda/CUDAMultiStreamGuard.h>
#endif

#if TENSORPIPE_HAS_SHM_TRANSPORT
// Needed for ::getpid(), which is used to create a unique address.
#include <sys/types.h>
#include <unistd.h>
#endif

namespace torch {
namespace distributed {
namespace rpc {

namespace {

// An environment variable along the lines of GLOO_ and NCCL_SOCKET_IFNAME that
// allows the user to specify a device to bind to, instead of binding to the
// address that the hostname resolves to.
const std::string kSocketIfnameEnvVar = "TP_SOCKET_IFNAME";
const std::string kDefaultUvAddress = "127.0.0.1";

const std::string kGilAverageWaitTime = "agent.gil_average_wait_time_us";
const std::string kThreadPoolSize = "agent.thread_pool_size";
const std::string kNumIdleThreads = "agent.num_idle_threads";
const std::string kClientActiveCalls = "agent.client_active_calls";
const std::string kServerActiveCalls = "agent.server_active_calls";
const std::string kServerActiveAsyncCalls = "agent.server_active_async_calls";

std::vector<c10::DeviceIndex> getDevicesForTensors(
    const std::vector<torch::Tensor>& tensors,
    const tensorpipe::DeviceMap& deviceMap,
    const std::string& remoteName) {
  // If the deviceMap is overridden, use that instead.
  const auto errStr = c10::str(
      "TensorPipe RPC backend only supports CPU tensors by default, please "
      "move your tensors to CPU before sending them over RPC, or call "
      "`set_device_map` on `TensorPipeRpcBackendOptions` to explicitly "
      "configure device mapping. ",
      "Request device mapping is not available for destination ",
      remoteName);
  std::vector<c10::DeviceIndex> deviceIndices;
  deviceIndices.reserve(tensors.size());
  bool hasCudaTensor = false;
  for (const auto& t : tensors) {
    if (t.device().is_cpu()) {
      deviceIndices.push_back(-1);
    } else {
      const auto deviceIter = deviceMap.find(t.device().index());
      TORCH_CHECK(
          deviceIter != deviceMap.end(),
          errStr,
          " for device ",
          t.device(),
          " but received a tensor on that device.");
      deviceIndices.push_back(deviceIter->second);
      hasCudaTensor = true;
    }
  }
  if (!hasCudaTensor) {
    deviceIndices.clear();
  }
  return deviceIndices;
}

} // namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_REGISTRY(TensorPipeTransportRegistry, TransportRegistration);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_REGISTRY(TensorPipeCpuChannelRegistry, CpuChannelRegistration);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_REGISTRY(TensorPipeCudaChannelRegistry, CudaChannelRegistration);

std::string TensorPipeAgent::guessUvAddress() {
  tensorpipe::Error error;
  std::string uvAddress;
  char* ifnameEnv = std::getenv(kSocketIfnameEnvVar.c_str());
  if (ifnameEnv != nullptr) {
    std::tie(error, uvAddress) =
        tensorpipe::transport::uv::lookupAddrForIface(ifnameEnv);
    if (error) {
      LOG(WARNING) << "Failed to look up the IP address for interface "
                   << ifnameEnv << " (" << error.what() << "), defaulting to "
                   << kDefaultUvAddress;
      uvAddress = kDefaultUvAddress;
    }
  } else {
    std::tie(error, uvAddress) =
        tensorpipe::transport::uv::lookupAddrForHostname();
    if (error) {
      LOG(WARNING) << "Failed to look up the IP address for the hostname ("
                   << error.what() << "), defaulting to " << kDefaultUvAddress;
      uvAddress = kDefaultUvAddress;
    }
  }
  return uvAddress;
}

namespace {

// These priorities instruct TensorPipe on which transport/channel to pick
// during handshake. Higher priorities will take precedence over lower ones.
// The transport with lowest priority will be the one used to bootstrap pipes.

constexpr int64_t kShmTransportPriority = 200;
constexpr int64_t kIbvTransportPriority = 100;
// The UV transport just uses TCP and should work everywhere, thus keep it last.
constexpr int64_t kUvTransportPriority = 0;

constexpr int64_t kCmaChannelPriority = 200;
constexpr int64_t kMultiplexedUvChannelPriority = 100;
// The basic channel reuses a transport as a channel, and is thus our fallback.
constexpr int64_t kBasicChannelPriority = 0;

#if TENSORPIPE_HAS_CUDA_IPC_CHANNEL && defined(USE_CUDA_NOT_ROCM)
constexpr int64_t kCudaIpcChannelPriority = 300;
#endif

#if TENSORPIPE_HAS_CUDA_GDR_CHANNEL && defined(USE_CUDA_NOT_ROCM)
constexpr int64_t kCudaGdrChannelPriority = 200;
#endif

#ifdef USE_CUDA_NOT_ROCM
constexpr int64_t kCudaXthChannelPriority = 400;
constexpr int64_t kCudaBasicChannelPriority = 100;
#endif

std::unique_ptr<TransportRegistration> makeUvTransport() {
  auto context = tensorpipe::transport::uv::create();
  std::string address = TensorPipeAgent::guessUvAddress();
  return std::make_unique<TransportRegistration>(TransportRegistration{
      std::move(context), kUvTransportPriority, std::move(address)});
}

// The UV transport is implemented using standard TCP connections. It leverages
// libuv (https://github.com/libuv/libuv) in order to be cross-platform.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_REGISTER_CREATOR(TensorPipeTransportRegistry, uv, makeUvTransport);

#if TENSORPIPE_HAS_SHM_TRANSPORT

std::string createUniqueShmAddr() {
  thread_local uint32_t threadLocalId = 0;
  return c10::str(
      "shm://tensorpipe_rpc_agent_",
      std::this_thread::get_id(),
      "_",
      ::getpid(),
      "_",
      threadLocalId++);
}

std::unique_ptr<TransportRegistration> makeShmTransport() {
  auto context = tensorpipe::transport::shm::create();
  std::string address = createUniqueShmAddr();
  return std::make_unique<TransportRegistration>(TransportRegistration{
      std::move(context), kShmTransportPriority, std::move(address)});
}

// The SHM implements connections using ringbuffers residing in anonymous shared
// memory (plus UNIX domain sockets to bootstrap the connection and exchange
// file descriptors). It is Linux-only due to some advanced features (O_TMPFILE,
// eventfd, ...).
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_REGISTER_CREATOR(TensorPipeTransportRegistry, shm, makeShmTransport);

#endif // TENSORPIPE_HAS_SHM_TRANSPORT

#if TENSORPIPE_HAS_IBV_TRANSPORT

std::string guessIbvAddress() {
  tensorpipe::Error error;
  std::string ibvAddress;
  char* ifnameEnv = std::getenv(kSocketIfnameEnvVar.c_str());
  if (ifnameEnv != nullptr) {
    std::tie(error, ibvAddress) =
        tensorpipe::transport::ibv::lookupAddrForIface(ifnameEnv);
    if (error) {
      LOG(WARNING) << "Failed to look up the IP address for interface "
                   << ifnameEnv << " (" << error.what() << "), defaulting to "
                   << kDefaultUvAddress;
      ibvAddress = kDefaultUvAddress;
    }
  } else {
    std::tie(error, ibvAddress) =
        tensorpipe::transport::ibv::lookupAddrForHostname();
    if (error) {
      LOG(WARNING) << "Failed to look up the IP address for the hostname ("
                   << error.what() << "), defaulting to " << kDefaultUvAddress;
      ibvAddress = kDefaultUvAddress;
    }
  }
  return ibvAddress;
}

std::unique_ptr<TransportRegistration> makeIbvTransport() {
  auto context = tensorpipe::transport::ibv::create();
  std::string address = guessIbvAddress();
  return std::make_unique<TransportRegistration>(TransportRegistration{
      std::move(context), kIbvTransportPriority, std::move(address)});
}

// The IBV transport sends data across using an InfiniBand queue pair, locally
// copying data to and from a staging buffer (registered with libibverbs) and
// issuing a RDMA write for transferring data across machines (plus a send for
// acknowledging it). It bootstraps using a standard TCP connection to exchange
// setup information. It is Linux-only.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_REGISTER_CREATOR(TensorPipeTransportRegistry, ibv, makeIbvTransport);

#endif // TENSORPIPE_HAS_IBV_TRANSPORT

std::unique_ptr<CpuChannelRegistration> makeBasicChannel() {
  auto context = tensorpipe::channel::basic::create();
  return std::make_unique<CpuChannelRegistration>(
      CpuChannelRegistration{std::move(context), kBasicChannelPriority});
}

// The basic channel is just a straightforward adapter wrapper that allows any
// transport to be used as a channel.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_REGISTER_CREATOR(TensorPipeCpuChannelRegistry, basic, makeBasicChannel);

#if TENSORPIPE_HAS_CMA_CHANNEL

std::unique_ptr<CpuChannelRegistration> makeCmaChannel() {
  auto context = tensorpipe::channel::cma::create();
  return std::make_unique<CpuChannelRegistration>(
      CpuChannelRegistration{std::move(context), kCmaChannelPriority});
}

// The CMA channel uses the Linux cross-memory attach syscalls (process_vm_readv
// and _writev), which allow one process to access the private memory of another
// process (as long as they belong to the same user and other security
// constraints are satisfied). It does, more or less, what GDB does when it's
// attached to a running process.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_REGISTER_CREATOR(TensorPipeCpuChannelRegistry, cma, makeCmaChannel);

#endif // TENSORPIPE_HAS_CMA_CHANNEL

constexpr static int kNumUvThreads = 16;

std::unique_ptr<CpuChannelRegistration> makeMultiplexedUvChannel() {
  std::vector<std::shared_ptr<tensorpipe::transport::Context>> contexts;
  std::vector<std::shared_ptr<tensorpipe::transport::Listener>> listeners;
  for (int laneIdx = 0; laneIdx < kNumUvThreads; ++laneIdx) {
    auto context = tensorpipe::transport::uv::create();
    std::string address = TensorPipeAgent::guessUvAddress();
    contexts.push_back(std::move(context));
    listeners.push_back(contexts.back()->listen(address));
  }
  auto context = tensorpipe::channel::mpt::create(
      std::move(contexts), std::move(listeners));
  return std::make_unique<CpuChannelRegistration>(CpuChannelRegistration{
      std::move(context), kMultiplexedUvChannelPriority});
}

// The multiplexed UV channel encapsulates multiple UV transports (each with its
// own event loop thread). Each channel will, in turn, contain multiple UV
// connections, one for each of those contexts. When sending a tensor, its data
// is split in equal chunks and each chunks is sent on a different connection
// and thus driven by a different thread. This is needed to reach very high
// bandwidths.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_REGISTER_CREATOR(
    TensorPipeCpuChannelRegistry,
    mpt_uv,
    makeMultiplexedUvChannel);

#if TENSORPIPE_HAS_CUDA_IPC_CHANNEL && defined(USE_CUDA_NOT_ROCM)

std::unique_ptr<CudaChannelRegistration> makeCudaIpcChannel() {
  auto context = tensorpipe::channel::cuda_ipc::create();
  return std::make_unique<CudaChannelRegistration>(
      CudaChannelRegistration{std::move(context), kCudaIpcChannelPriority});
}

// The cuda_ipc channels use cudaMemcpy to transmit CUDA tensor across processes
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_REGISTER_CREATOR(
    TensorPipeCudaChannelRegistry,
    cuda_ipc,
    makeCudaIpcChannel);

#endif

#if TENSORPIPE_HAS_CUDA_GDR_CHANNEL && defined(USE_CUDA_NOT_ROCM)

std::unique_ptr<CudaChannelRegistration> makeCudaGdrChannel() {
  auto context = tensorpipe::channel::cuda_gdr::create();
  return std::make_unique<CudaChannelRegistration>(
      CudaChannelRegistration{std::move(context), kCudaGdrChannelPriority});
}

// The cuda_gdr channel sends CUDA memory over InfiniBand using GPUDirect RDMA.
// It directly registers the user-provided tensor with libibverbs, an operation
// which is expensive the first time, but it then caches the registration in
// order to amortize the cost and get low latency for subsequent transfers. A
// ready-to-send/ready-to-receive handshake is still needed before the transfer
// in order to ensure readiness and to agree on the device indices and thus the
// queue pair to use. It automatically pairs each GPU to the "closest" NIC if
// there are multiple of them (closest = longest prefix match in PCI tree).
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_REGISTER_CREATOR(
    TensorPipeCudaChannelRegistry,
    cuda_gdr,
    makeCudaGdrChannel);

#endif

#ifdef USE_CUDA_NOT_ROCM

std::unique_ptr<CudaChannelRegistration> makeCudaXthChannel() {
  auto context = tensorpipe::channel::cuda_xth::create();
  return std::make_unique<CudaChannelRegistration>(
      CudaChannelRegistration{std::move(context), kCudaXthChannelPriority});
}

// The cuda_xth channel supports same-process GPU-to-GPU comm
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_REGISTER_CREATOR(
    TensorPipeCudaChannelRegistry,
    cuda_xth,
    makeCudaXthChannel);

std::unique_ptr<CudaChannelRegistration> makeCudaBasicChannel() {
  auto context = tensorpipe::channel::cuda_basic::create(
      tensorpipe::channel::basic::create());
  return std::make_unique<CudaChannelRegistration>(
      CudaChannelRegistration{std::move(context), kCudaBasicChannelPriority});
}

// The cuda_basic is the fallback channel for GPU-to-GPU comm
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_REGISTER_CREATOR(
    TensorPipeCudaChannelRegistry,
    cuda_basic,
    makeCudaBasicChannel);

#endif

} // namespace

namespace {

// This is a wrapper of CUDAMultiStreamGuard to run in both CUDA-enabled and
// CPU-only environments. When CUDA is not available, all methods are no-ops.
struct MultiStreamGuard {
  MultiStreamGuard(const MultiStreamGuard& other) = delete;
  MultiStreamGuard(MultiStreamGuard&& other) = delete;
  MultiStreamGuard& operator=(const MultiStreamGuard& rhs) = delete;
  MultiStreamGuard& operator=(MultiStreamGuard&& rhs) = delete;

#ifndef USE_CUDA_NOT_ROCM
  explicit MultiStreamGuard(
      const std::shared_ptr<LazyStreamContext>& /* unused */) {}
#else
  explicit MultiStreamGuard(const std::shared_ptr<LazyStreamContext>& ctx)
      : guard(ctx->getReservedStreams()) {}

 private:
  at::cuda::CUDAMultiStreamGuard guard;
#endif
};

} // namespace

//////////////////////////  MetricsTracker  /////////////////////////////////

TensorPipeAgent::TimeSeriesMetricsTracker::TimeSeriesMetricsTracker(
    uint64_t currentSum,
    uint64_t currentCount)
    : currentSum_(currentSum), currentCount_(currentCount) {}

void TensorPipeAgent::TimeSeriesMetricsTracker::addData(uint64_t dataPoint) {
  currentSum_ += dataPoint;
  ++currentCount_;
}

float TensorPipeAgent::TimeSeriesMetricsTracker::computeAverage() const {
  return currentCount_ == 0 ? 0 : currentSum_ / (float)currentCount_;
}

////////////////////////  TensorpipeRpcAgent  /////////////////////////////////

void TensorPipeAgent::removeFromTimeoutMap(
    uint64_t messageId,
    steady_clock_time_point expirationTime) {
  // Remove entry from timeoutMap_.
  {
    std::unique_lock<std::mutex> lock(timeoutMapMutex_);
    auto& timedOutFuturesVector = timeoutMap_[expirationTime];
    for (auto it = timedOutFuturesVector.begin();
         it != timedOutFuturesVector.end();
         it++) {
      if (it->messageId == messageId) {
        it = timedOutFuturesVector.erase(it);
        break;
      }
    }

    if (timedOutFuturesVector.empty()) {
      timeoutMap_.erase(expirationTime);
    }
  }
}

void TensorPipeAgent::collectNames() {
  const worker_id_t selfId = workerInfo_.id_;
  const std::string& selfName = workerInfo_.name_;

  std::vector<uint8_t> selfNameVector(
      (uint8_t*)selfName.c_str(),
      (uint8_t*)selfName.c_str() + selfName.length());
  rankToNameStore_.set(c10::to_string(selfId), selfNameVector);

  workerIdToInfo_.emplace(selfId, WorkerInfo(selfName, selfId));
  workerNameToInfo_.emplace(selfName, WorkerInfo(selfName, selfId));
  for (worker_id_t workerId = 0; workerId < worldSize_; ++workerId) {
    if (workerId == selfId) {
      continue;
    }
    std::vector<uint8_t> workerNameVector =
        rankToNameStore_.get(c10::to_string(workerId));
    std::string workerName(
        (char*)workerNameVector.data(), workerNameVector.size());

    TORCH_CHECK(
        workerNameToInfo_.find(workerName) == workerNameToInfo_.end(),
        "RPC worker name ",
        workerName,
        " is not unique.");

    workerIdToInfo_.emplace(workerId, WorkerInfo(workerName, workerId));
    workerNameToInfo_.emplace(workerName, WorkerInfo(workerName, workerId));
  }
}

TensorPipeAgent::TensorPipeAgent(
    const c10::intrusive_ptr<::c10d::Store>& store,
    std::string selfName,
    worker_id_t selfId,
    int worldSize,
    c10::intrusive_ptr<::c10d::ProcessGroup> processGroup,
    TensorPipeRpcBackendOptions opts,
    std::unique_ptr<RequestCallback> cb)
    : RpcAgent(
          WorkerInfo(std::move(selfName), selfId),
          std::move(cb),
          std::chrono::milliseconds(
              (long)(opts.rpcTimeoutSeconds * kSecToMsConversion))),
      opts_(std::move(opts)),
      threadPool_(opts_.numWorkerThreads),
      context_(std::make_shared<tensorpipe::Context>(
          tensorpipe::ContextOptions().name(workerInfo_.name_))),
      rankToNameStore_("names", store),
      nameToAddressStore_("addrs", store),
      worldSize_(worldSize),
      processGroup_(std::move(processGroup)) {
  collectNames();

  // Initialize the time-series metrics tracking map
  timeSeriesMetrics_.emplace(kGilAverageWaitTime, TimeSeriesMetricsTracker());
}

TensorPipeAgent::~TensorPipeAgent() {
  VLOG(1) << "RPC agent for " << workerInfo_.name_ << " is being destroyed";
  shutdown();
}

void TensorPipeAgent::startImpl() {
  VLOG(1) << "RPC agent for " << workerInfo_.name_ << " is starting";

  std::vector<std::string> addresses;
  int lowestPriority = std::numeric_limits<int>::max();
  std::string lowestPriorityTransport;

  for (auto& key : TensorPipeTransportRegistry()->Keys()) {
    int64_t priority = -1;
    if (opts_.transports.has_value()) {
      auto iter =
          std::find(opts_.transports->begin(), opts_.transports->end(), key);
      if (iter == opts_.transports->end()) {
        continue;
      }
      // Assign priorities in reverse order of occurrence in the vector, so that
      // a transport that comes before another receives a higher priority.
      priority =
          opts_.transports->size() - 1 - (iter - opts_.transports->begin());
    }
    std::unique_ptr<TransportRegistration> reg =
        TensorPipeTransportRegistry()->Create(key);
    if (!reg->transport->isViable()) {
      continue;
    }
    if (priority == -1) {
      priority = reg->priority;
    }
    if (priority < lowestPriority) {
      lowestPriority = priority;
      lowestPriorityTransport = key;
    }
    addresses.push_back(c10::str(key, "://", reg->address));
    context_->registerTransport(
        priority, std::move(key), std::move(reg->transport));
  }

  auto registerChannel = [this](const auto& registry) {
    // The registry argument is either TensorPipeCpuChannelRegistry or
    // TensorPipeCudaChannelRegistry.
    for (auto& key : registry->Keys()) {
      int64_t priority = -1;
      if (opts_.channels.has_value()) {
        auto iter =
            std::find(opts_.channels->begin(), opts_.channels->end(), key);
        if (iter == opts_.channels->end()) {
          continue;
        }
        // Assign priorities in reverse order of occurrence in the vector, so
        // that a channel that comes before another receives a higher priority.
        priority =
            opts_.channels->size() - 1 - (iter - opts_.channels->begin());
      }

      // The reg var is either a std::unique_ptr<CpuChannelRegistration> or a
      // std::unique_ptr<CudaChannelRegistration>, depending on the type of the
      // registry.
      auto reg = registry->Create(key);
      if (!reg->channel->isViable()) {
        continue;
      }
      if (priority == -1) {
        priority = reg->priority;
      }
      context_->registerChannel(
          priority, std::move(key), std::move(reg->channel));
    }
  };

  registerChannel(TensorPipeCpuChannelRegistry());

#ifdef USE_CUDA_NOT_ROCM

  registerChannel(TensorPipeCudaChannelRegistry());

#endif

  listener_ = context_->listen(addresses);

  // Store our own url.
  const auto address = listener_->url(lowestPriorityTransport);
  const std::vector<uint8_t> selfAddrData(address.begin(), address.end());
  nameToAddressStore_.set(workerInfo_.name_, selfAddrData);

  VLOG(1) << "RPC agent for " << workerInfo_.name_ << " is using address "
          << address;

  for (const auto& p : workerNameToInfo_) {
    const auto& name = p.first;
    auto nodeAddrData = nameToAddressStore_.get(name);
    auto nodeAddrStr =
        std::string((const char*)nodeAddrData.data(), nodeAddrData.size());
    workerNameToURL_.insert({name, nodeAddrStr});
  }

  // Start the Timeout Thread
  timeoutThread_ = std::thread(&TensorPipeAgent::pollTimeoutRpcs, this);

  listener_->accept([this](
                        const tensorpipe::Error& error,
                        std::shared_ptr<tensorpipe::Pipe> pipe) {
    onListenerAccepted(error, pipe);
  });
}

void TensorPipeAgent::onListenerAccepted(
    const tensorpipe::Error& error,
    std::shared_ptr<tensorpipe::Pipe>& pipe) {
  if (error) {
    if (error.isOfType<tensorpipe::ListenerClosedError>() &&
        !rpcAgentRunning_.load()) {
      // This is expected.
    } else {
      LOG(WARNING) << "RPC agent for " << workerInfo_.name_
                   << " encountered error when accepting incoming pipe: "
                   << error.what();
    }
    return;
  }

  // Accept the next connection request
  listener_->accept([this](
                        const tensorpipe::Error& error,
                        std::shared_ptr<tensorpipe::Pipe> pipe) {
    onListenerAccepted(error, pipe);
  });

  VLOG(1) << "RPC agent for " << workerInfo_.name_
          << " accepted incoming pipe from " << pipe->getRemoteName();

  // Arm for server read
  respond(pipe);
}

void TensorPipeAgent::pipeRead(
    const std::shared_ptr<tensorpipe::Pipe>& pipe,
    std::function<void(
        const tensorpipe::Error&,
        Message&&,
        std::shared_ptr<LazyStreamContext>)> fn) noexcept {
  pipe->readDescriptor([fn{std::move(fn)}, pipe](
                           const tensorpipe::Error& error,
                           tensorpipe::Message tpMessage) mutable {
    if (error) {
      fn(error, Message(), nullptr);
      return;
    }

    auto ctx = createLazyStreamContext();
    TensorpipeReadBuffers tpBuffers = tensorpipeAllocate(tpMessage, ctx);

    pipe->read(
        std::move(tpMessage),
        [tpBuffers{
             std::make_shared<TensorpipeReadBuffers>(std::move(tpBuffers))},
         fn{std::move(fn)},
         ctx{std::move(ctx)}](
            const tensorpipe::Error& error,
            tensorpipe::Message tpMessage) mutable {
          if (error) {
            fn(error, Message(), nullptr);
            return;
          }

          // FIXME This does some unpickling, which could be a bit expensive:
          // perhaps it would be best to perform it inside the worker threads?
          Message rpcMessage = tensorpipeDeserialize(
              std::move(tpMessage), std::move(*tpBuffers));

          fn(error, std::move(rpcMessage), std::move(ctx));
        });
  });
}

void TensorPipeAgent::pipeWrite(
    const std::shared_ptr<tensorpipe::Pipe>& pipe,
    Message&& rpcMessage,
    std::vector<c10::DeviceIndex>&& devices,
    std::shared_ptr<LazyStreamContext> ctx,
    std::function<void(const tensorpipe::Error&)> fn,
    const std::unordered_map<c10::DeviceIndex, c10::DeviceIndex>&
        deviceMap) noexcept {
  tensorpipe::Message tpMessage;
  TensorpipeWriteBuffers tpBuffers;

  std::tie(tpMessage, tpBuffers) =
      tensorpipeSerialize(std::move(rpcMessage), std::move(devices), ctx);

  pipe->write(
      std::move(tpMessage),
      [tpBuffers{
           std::make_shared<TensorpipeWriteBuffers>(std::move(tpBuffers))},
       fn{std::move(fn)},
       ctx{std::move(ctx)}](
          const tensorpipe::Error& error, tensorpipe::Message /* unused */) {
        fn(error);
      });
}

void TensorPipeAgent::sendCompletedResponseMessage(
    std::shared_ptr<tensorpipe::Pipe>& pipe,
    std::shared_ptr<JitFuture>& futureResponseMessage,
    uint64_t messageId,
    std::shared_ptr<LazyStreamContext> ctx) {
  if (!rpcAgentRunning_.load()) {
    LOG(WARNING) << "RPC agent for " << workerInfo_.name_
                 << " won't send response to request #" << messageId << " to "
                 << pipe->getRemoteName() << ", as the agent is shutting down";
    return;
  }

  VLOG(1) << "RPC agent for " << workerInfo_.name_
          << " is sending response to request #" << messageId << " to "
          << pipe->getRemoteName();

  if (!futureResponseMessage->hasError()) {
    Message&& responseMessage =
        std::move(*futureResponseMessage->value().toCustomClass<Message>());
    responseMessage.setId(messageId);
    std::vector<c10::DeviceIndex> devices;
    try {
      devices = getDevicesForRemote(pipe->getRemoteName(), responseMessage);
    } catch (const std::exception& e) {
      responseMessage = createExceptionResponse(e.what(), messageId);
    }

    pipeWrite(
        pipe,
        std::move(responseMessage),
        std::move(devices),
        std::move(ctx),
        [this, pipe, messageId](const tensorpipe::Error& error) {
          if (error) {
            LOG(WARNING)
                << "RPC agent for " << workerInfo_.name_
                << " encountered error when sending response to request #"
                << messageId << " to " << pipe->getRemoteName() << ": "
                << error.what();
            return;
          }

          VLOG(1) << "RPC agent for " << workerInfo_.name_
                  << " done sending response to request #" << messageId
                  << " to " << pipe->getRemoteName();
        });
  } else {
    pipeWrite(
        pipe,
        createExceptionResponse(
            futureResponseMessage->tryRetrieveErrorMessage(), messageId),
        /* devices */ {},
        std::move(ctx),
        [this, pipe, messageId](const tensorpipe::Error& error) {
          if (error) {
            LOG(WARNING)
                << "RPC agent for " << workerInfo_.name_
                << " encountered error when sending response to request #"
                << messageId << " to " << pipe->getRemoteName() << ": "
                << error.what();
            return;
          }

          VLOG(1) << "RPC agent for " << workerInfo_.name_
                  << " done sending response to request #" << messageId
                  << " to " << pipe->getRemoteName();
        });
  }
}

void TensorPipeAgent::respond(std::shared_ptr<tensorpipe::Pipe>& pipe) {
  pipeRead(
      pipe,
      [this, pipe](
          const tensorpipe::Error& error,
          Message&& requestMessage,
          std::shared_ptr<LazyStreamContext> ctx) mutable {
        if (error) {
          // FIXME This is not a correct way to check whether this error was
          // "intentionally" caused by the remote end shutting down. We should
          // find a better way, Perhaps sending an empty message?
          if ((error.isOfType<tensorpipe::PipeClosedError>() &&
               !rpcAgentRunning_.load()) ||
              error.isOfType<tensorpipe::EOFError>()) {
            // This is expected.
          } else {
            LOG(WARNING)
                << "RPC agent for " << workerInfo_.name_
                << " encountered error when reading incoming request from "
                << pipe->getRemoteName() << ": " << error.what()
                << " (this is expected to happen during shutdown)";
          }
          return;
        }

        // Arm for next read
        respond(pipe);

        uint64_t messageId = requestMessage.id();
        increaseCallCount(serverActiveCalls_);

        VLOG(1) << "RPC agent for " << workerInfo_.name_
                << " received request #" << messageId << " from "
                << pipe->getRemoteName();

        // Defer user RPC UDF run to thread pool
        threadPool_.run([this,
                         pipe,
                         messageId,
                         requestMessage{std::move(requestMessage)},
                         ctx{std::move(ctx)}]() mutable {
          // create guards again as this function runs on a different thread
          MultiStreamGuard guard(ctx);
          VLOG(1) << "RPC agent for " << workerInfo_.name_
                  << " is running request #" << messageId << " from "
                  << pipe->getRemoteName() << " in thread pool";

          std::shared_ptr<JitFuture> futureResponseMessage;
          try {
            futureResponseMessage = cb_->operator()(requestMessage);
          } catch (const std::exception& /* unused */) {
            futureResponseMessage =
                std::make_shared<JitFuture>(at::AnyClassType::get());
            futureResponseMessage->setError(std::current_exception());
          }

          // Shortcut if immediately done
          if (futureResponseMessage->completed()) {
            decreaseCallCount(serverActiveCalls_);
            sendCompletedResponseMessage(
                pipe, futureResponseMessage, messageId, std::move(ctx));
          } else {
            // Not complete yet
            increaseCallCount(serverActiveAsyncCalls_);
            futureResponseMessage->addCallback([this,
                                                pipe,
                                                futureResponseMessage,
                                                messageId,
                                                ctx{std::move(ctx)}]() mutable {
              decreaseCallCount(serverActiveCalls_);
              decreaseCallCount(serverActiveAsyncCalls_);
              sendCompletedResponseMessage(
                  pipe, futureResponseMessage, messageId, std::move(ctx));
            });
          }

          VLOG(1) << "RPC agent for " << workerInfo_.name_
                  << " done running request #" << messageId << " from "
                  << pipe->getRemoteName() << " in thread pool";
        });
      });
}

std::shared_ptr<JitFuture> TensorPipeAgent::send(
    const WorkerInfo& toWorkerInfo,
    Message&& requestMessage,
    const float rpcTimeoutSeconds,
    const std::unordered_map<c10::DeviceIndex, c10::DeviceIndex>& deviceMap) {
  TORCH_CHECK(
      requestMessage.isRequest(),
      "TensorPipeAgent::send(..) is only for sending requests.");

  if (!rpcAgentRunning_.load()) {
    auto err = c10::str(
        "Node ",
        RpcAgent::getWorkerInfo().id_,
        "tried to send() a message of type ",
        requestMessage.type(),
        " but RPC is no longer running on this node.");
    throw std::runtime_error(err);
  }

  const auto& url = findWorkerURL(toWorkerInfo);

  std::unique_lock<std::mutex> lock(mutex_);

  // See if we already have a connection to this address or not
  auto it = connectedPipes_.find(toWorkerInfo.id_);
  if (it == connectedPipes_.end()) {
    std::tie(it, std::ignore) = connectedPipes_.emplace(
        toWorkerInfo.id_,
        ClientPipe(context_->connect(
            url, tensorpipe::PipeOptions().remoteName(toWorkerInfo.name_))));
  }
  ClientPipe& clientPipe = it->second;
  auto& pendingResponseMessage = clientPipe.pendingResponseMessage_;

  auto futureResponseMessage = std::make_shared<AtomicJitFuture>(
      reverseDeviceMaps_.empty() && opts_.deviceMaps.empty());
  uint64_t messageId = nextMessageID_++;
  requestMessage.setId(messageId);
  pendingResponseMessage[messageId] = futureResponseMessage;

  lock.unlock();

  // Get devices for tensors in the request message. This can throw if device
  // maps are not configured properly for this request.
  std::vector<c10::DeviceIndex> devices;
  if (deviceMap.empty()) {
    devices =
        getDevicesForRemote(clientPipe.pipe_->getRemoteName(), requestMessage);
  } else {
    // If deviceMap is specified, use that instead.
    devices = getDevicesForTensors(
        requestMessage.tensors(), deviceMap, clientPipe.pipe_->getRemoteName());
  }

  futureResponseMessage->jitFuture->addCallback([this]() {
    TORCH_INTERNAL_ASSERT(
        this->threadPool_.inThreadPool(),
        "Future marked complete from outside the thread pool");
  });

  increaseCallCount(clientActiveCalls_);
  // Use the default RPC timeout if no timeout is specified for this send call
  auto timeout = rpcTimeoutSeconds == kUnsetRpcTimeout
      ? getRpcTimeout()
      : std::chrono::milliseconds(
            static_cast<int>(rpcTimeoutSeconds * kSecToMsConversion));

  // We only add to the timeoutMap_ if the timeout is not 0. Per our
  // documentation, a user-provided timeout of 0 indicates the RPC should never
  // expire (infinite timeout), so there is no need to track it in the
  // timeoutMap_.
  steady_clock_time_point expirationTime;
  if (timeout.count() != 0) {
    // Compute the expiration time for this message based on the timeout
    expirationTime = computeRpcMessageExpiryTime(timeout);

    // Add the Future to the right vector in the timeoutMap_
    {
      std::unique_lock<std::mutex> lock(timeoutMapMutex_);
      auto& timeoutFuturesVector = timeoutMap_[expirationTime];
      timeoutFuturesVector.emplace_back(
          messageId, futureResponseMessage, timeout);
    }
    timeoutThreadCV_.notify_one();
  }

  VLOG(1) << "RPC agent for " << workerInfo_.name_ << " is sending request #"
          << messageId << " to " << clientPipe.pipe_->getRemoteName();

  auto ctx = createLazyStreamContext();
  ctx->waitForCurrentStreams(requestMessage.tensors());
  pipeWrite(
      clientPipe.pipe_,
      std::move(requestMessage),
      std::move(devices),
      std::move(ctx),
      [this, &clientPipe, messageId, expirationTime](
          const tensorpipe::Error& error) mutable {
        if (error) {
          if (error.isOfType<tensorpipe::PipeClosedError>() &&
              !rpcAgentRunning_.load()) {
            // This is expected.
          } else {
            LOG(WARNING) << "RPC agent for " << workerInfo_.name_
                         << " encountered error when sending outgoing request #"
                         << messageId << " to "
                         << clientPipe.pipe_->getRemoteName() << ": "
                         << error.what();
          }
          auto pendingFutIt =
              clientPipe.pendingResponseMessage_.find(messageId);
          if (pendingFutIt != clientPipe.pendingResponseMessage_.end()) {
            markFutureWithError(pendingFutIt->second, error.what());
          }
          return;
        }

        VLOG(1) << "RPC agent for " << workerInfo_.name_ << " sent request #"
                << messageId << " to " << clientPipe.pipe_->getRemoteName();

        pipeRead(
            clientPipe.pipe_,
            [this, &clientPipe, expirationTime](
                const tensorpipe::Error& error,
                Message&& responseMessage,
                std::shared_ptr<LazyStreamContext> ctx) {
              if (error) {
                if (error.isOfType<tensorpipe::PipeClosedError>() &&
                    !rpcAgentRunning_.load()) {
                  // This is expected.
                } else {
                  LOG(WARNING)
                      << "RPC agent for " << workerInfo_.name_
                      << " encountered error when reading incoming response from "
                      << clientPipe.pipe_->getRemoteName() << ": "
                      << error.what();
                }
                // We may get garbage content in responseMessage upon error.
                // Flushing all future messages belonging to this pipe due to
                // error state.
                decltype(clientPipe.pendingResponseMessage_) pendingMsgs;
                {
                  std::lock_guard<std::mutex> lock(mutex_);
                  std::swap(clientPipe.pendingResponseMessage_, pendingMsgs);
                  clientPipe.readError_ = true;
                }
                std::string errorMsg = error.what();
                for (auto& p : pendingMsgs) {
                  markFutureWithError(std::move(p.second), errorMsg);
                }

                // Remove entry from timeoutMap_.
                removeFromTimeoutMap(responseMessage.id(), expirationTime);
                return;
              }

              // Identify future response message by message ID
              uint64_t messageId = responseMessage.id();

              VLOG(1) << "RPC agent for " << workerInfo_.name_
                      << " received response #" << messageId << " from "
                      << clientPipe.pipe_->getRemoteName();

              std::shared_ptr<AtomicJitFuture> futureResponseMessage;
              {
                std::lock_guard<std::mutex> lock(mutex_);
                // A read error will lead all following callbacks to be
                // invoked with error, and shouldn't reach here.
                TORCH_INTERNAL_ASSERT(
                    !clientPipe.readError_, "Shouldn't in error state");
                auto it = clientPipe.pendingResponseMessage_.find(messageId);
                TORCH_INTERNAL_ASSERT(
                    it != clientPipe.pendingResponseMessage_.end(),
                    "message ID ",
                    messageId,
                    " is not recognized");
                futureResponseMessage = std::move(it->second);
                clientPipe.pendingResponseMessage_.erase(it);
              }

              // Remove entry from timeoutMap_.
              removeFromTimeoutMap(messageId, expirationTime);

              if (responseMessage.type() == MessageType::EXCEPTION) {
                markFutureWithError(
                    std::move(futureResponseMessage),
                    std::string(
                        responseMessage.payload().begin(),
                        responseMessage.payload().end()));
              } else {
                markFutureAsComplete(
                    std::move(futureResponseMessage),
                    std::move(responseMessage),
                    std::move(ctx));
              }
            });
      },
      deviceMap);

  return futureResponseMessage->jitFuture;
}

void TensorPipeAgent::pollTimeoutRpcs() {
  while (rpcAgentRunning_.load()) {
    std::unique_lock<std::mutex> lock(timeoutMapMutex_);

    // We sleep until the earliest expiring RPC in the timeoutMap_. We must
    // also ensure that we sleep while the map is empty, and we exit sleeping
    // if the RPC Agent has been shutdown.
    for (;;) {
      if (!rpcAgentRunning_.load()) {
        return;
      }

      if (!timeoutMap_.empty()) {
        steady_clock_time_point earliestTimeout = timeoutMap_.begin()->first;
        if (std::chrono::steady_clock::now() >= earliestTimeout) {
          break;
        }
        timeoutThreadCV_.wait_until(lock, earliestTimeout);
      } else {
        timeoutThreadCV_.wait(lock);
      }
    }

    // Move all these futures to a separate vector so we can process them
    // outside the lock.
    std::vector<TimeoutMessageMetadata> timedOutFutures =
        std::move(timeoutMap_.begin()->second);
    // We can safely remove this key from the timeoutMap_ since all these
    // futures will be processed.
    timeoutMap_.erase(timeoutMap_.begin());

    lock.unlock();

    // Set an error on futures added to the timedOutFutures vector. We do this
    // outside the lock to prevent potential lock-order-inversions by callbacks
    // triggered by the setError call.
    for (auto& timeoutMetadata : timedOutFutures) {
      std::string errorMsg =
          fmt::format(kRpcTimeoutErrorStr, timeoutMetadata.timeout.count());
      auto err = makeRPCError(errorMsg, RPCErrorType::TIMEOUT);
      markFutureWithError(
          std::move(timeoutMetadata.responseFuture), std::move(err));
    }
  }
}

// TODO: Remove sync()
void TensorPipeAgent::sync() {
  VLOG(1) << "RPC agent for " << workerInfo_.name_ << " is syncing (no-op)";
}

// TODO: Remove join()
void TensorPipeAgent::join() {
  VLOG(1) << "RPC agent for " << workerInfo_.name_ << " is joining";
  // This method behaves like a barrier, as it can only return once all workers
  // have no more requests pending, including "nested" requests (triggered from
  // within the remote code of another call) and "follow-up" requests (triggered
  // from the callback of a future).
  while (true) {
    {
      std::unique_lock<std::mutex> lock(callCountMutex_);
      // It is enough to wait for there to be no more active client calls, since
      // each server call corresponds to a client call for some other worker.
      callCountCV_.wait(lock, [this] { return clientActiveCalls_ == 0; });
      // We'd like to immediately proceed with the allreduce, but it's a call
      // that may block for some time, as it waits for other workers to also
      // complete all their active client calls. While we call allreduce we must
      // hold the mutex, or else the count we send to other workers may get
      // stale (e.g., if some nested call happens in the meantime). But we can't
      // hold the lock for an indeterminately long time, as that would block
      // other operations (e.g., send). Thus we must release the lock and only
      // re-acquire it when all workers are ready to proceed with the allreduce.
      // We perform this synchronization using a barrier.
    }
    VLOG(1) << "RPC agent for " << workerInfo_.name_
            << " completed all client calls and is entering a barrier";
    processGroup_->barrier()->wait();
    {
      std::unique_lock<std::mutex> lock(callCountMutex_);
      // At this point, the count may have become non-zero again. We can't wait
      // for those calls to complete as other workers are waiting for us in the
      // allreduce and we would block them. Thus we send our count even if it is
      // non-zero and if anyone (be it us or another worker) has a non-zero
      // count we'll just do another round.
      VLOG(1) << "RPC agent for " << workerInfo_.name_
              << " exited the barrier and found " << clientActiveCalls_
              << " active client calls";
      std::vector<at::Tensor> totalClientActiveCalls = {
          at::zeros({}, at::kLong)};
      *totalClientActiveCalls[0].data_ptr<int64_t>() = clientActiveCalls_;
      processGroup_->allreduce(totalClientActiveCalls)->wait();
      VLOG(1) << "RPC agent for " << workerInfo_.name_
              << " completed the allreduce and got a total of "
              << (*totalClientActiveCalls[0].data_ptr<int64_t>())
              << " active client calls across all workers";
      if (*totalClientActiveCalls[0].data_ptr<int64_t>() == 0) {
        break;
      }
    }
  }
  VLOG(1) << "RPC agent for " << workerInfo_.name_ << " done joining";
}

void TensorPipeAgent::shutdownImpl() {
  // FIXME Isn't it too verbose for a library to print logs in normal operation?
  LOG(INFO) << "RPC agent for " << workerInfo_.name_ << " is shutting down";

  // Join the Timeout Thread
  timeoutThreadCV_.notify_one();
  if (timeoutThread_.joinable()) {
    timeoutThread_.join();
  }
  VLOG(1) << "RPC agent for " << workerInfo_.name_
          << " done waiting for timeout thread to join";

  // This will close all the pipes and listeners, invoke all callbacks with
  // errors, turn down the I/O event loops and wait for everything to terminate.
  context_->join();
  VLOG(1) << "RPC agent for " << workerInfo_.name_
          << " done waiting for TensorPipe context to join";

  // NOTE: We need to call waitWorkComplete in the end after we have shutdown
  // all listeners for Tensorpipe. This is to drain any already accepted work
  // in the ThreadPool. If this is done before we shutdown the listeners,
  // additional work could be added after this call and before we shutdown
  // listeners. This work would continue executing in the threadpool and might
  // cause issues during shutdown of the system.
  threadPool_.waitWorkComplete();
  VLOG(1) << "RPC agent for " << workerInfo_.name_
          << " done waiting for thread pool to complete work";
}

const WorkerInfo& TensorPipeAgent::getWorkerInfo(
    const std::string& workerName) const {
  const auto& it = workerNameToInfo_.find(workerName);
  TORCH_CHECK(
      it != workerNameToInfo_.end(), "Unknown destination worker ", workerName);
  return it->second;
}

const WorkerInfo& TensorPipeAgent::getWorkerInfo(worker_id_t workerId) const {
  const auto& it = workerIdToInfo_.find(workerId);
  TORCH_CHECK(
      it != workerIdToInfo_.end(), "Unknown destination worker ", workerId);
  return it->second;
}

std::vector<WorkerInfo> TensorPipeAgent::getWorkerInfos() const {
  std::vector<WorkerInfo> workerInfos;
  for (auto& item : workerNameToInfo_) {
    workerInfos.emplace_back(item.second);
  }
  return workerInfos;
}

const std::string& TensorPipeAgent::findWorkerURL(
    const WorkerInfo& worker) const {
  const auto it = workerNameToURL_.find(worker.name_);
  TORCH_CHECK(
      it != workerNameToURL_.end(), "Unknown worker name: ", worker.name_);
  return it->second;
}

std::unordered_map<std::string, std::string> TensorPipeAgent::getMetrics() {
  std::unordered_map<std::string, std::string> metrics;
  metrics[kThreadPoolSize] = c10::to_string(threadPool_.size());
  metrics[kNumIdleThreads] = c10::to_string(threadPool_.numAvailable());
  {
    std::unique_lock<std::mutex> lock(callCountMutex_);
    metrics[kClientActiveCalls] = c10::to_string(clientActiveCalls_);
    metrics[kServerActiveCalls] = c10::to_string(serverActiveCalls_);
    metrics[kServerActiveAsyncCalls] = c10::to_string(serverActiveAsyncCalls_);
  }
  if (isGILProfilingEnabled()) {
    {
      std::unique_lock<std::mutex> lock(metricsMutex_);
      // Include the averages for each time series metric. This is just the GIL
      // Wait Time for now.
      auto averageGilWaitTime =
          timeSeriesMetrics_[kGilAverageWaitTime].computeAverage();
      lock.unlock();
      metrics[kGilAverageWaitTime] = c10::to_string(averageGilWaitTime);
    }
  }

  return metrics;
}

void TensorPipeAgent::addGilWaitTime(
    const std::chrono::microseconds gilWaitTime) {
  std::lock_guard<std::mutex> lock(metricsMutex_);
  timeSeriesMetrics_[kGilAverageWaitTime].addData(gilWaitTime.count());
}

TensorPipeAgent::NetworkDataDict TensorPipeAgent::getNetworkData() {
  std::lock_guard<std::mutex> lock(networkDataMutex_);
  return networkData_;
}

NetworkSourceInfo TensorPipeAgent::getNetworkSourceInfo() {
  NetworkSourceInfo info = {
      RpcAgent::getWorkerInfo().id_,
      nameToAddressStore_.get(RpcAgent::getWorkerInfo().name_)};

  return info;
}

void TensorPipeAgent::trackNetworkData(
    uint64_t requestSize,
    uint64_t responseSize,
    const std::string& destWorkerName) {
  std::lock_guard<std::mutex> lock(networkDataMutex_);
  networkData_[destWorkerName].numCalls++;
  networkData_[destWorkerName].totalSentBytes += requestSize;
  networkData_[destWorkerName].totalRecvBytes += responseSize;
}

void TensorPipeAgent::trackNetworkError(
    uint64_t requestSize,
    const std::string& destWorkerName) {
  std::lock_guard<std::mutex> lock(networkDataMutex_);
  networkData_[destWorkerName].numCalls++;
  networkData_[destWorkerName].totalSentBytes += requestSize;
  networkData_[destWorkerName].totalErrors++;
}

void TensorPipeAgent::increaseCallCount(int32_t& count) {
  {
    std::unique_lock<std::mutex> lock(callCountMutex_);
    ++count;
  }
  callCountCV_.notify_all();
}

void TensorPipeAgent::decreaseCallCount(int32_t& count) {
  {
    std::unique_lock<std::mutex> lock(callCountMutex_);
    --count;
  }
  callCountCV_.notify_all();
}

void TensorPipeAgent::markFutureAsComplete(
    std::shared_ptr<AtomicJitFuture> atomicFuture,
    Message message,
    std::shared_ptr<LazyStreamContext> ctx) {
  if (!atomicFuture->isComplete.test_and_set()) {
    // Completing the future will run its callbacks, which could execute
    // arbitrary user code. To prevent blocking or stalling the TensorPipe event
    // loops, we defer this to a worker thread.
    threadPool_.run([this,
                     atomicFuture{std::move(atomicFuture)},
                     message{std::move(message)},
                     ctx{std::move(ctx)}]() mutable {
      MultiStreamGuard guard(ctx);
      atomicFuture->jitFuture->markCompleted(
          IValue(c10::make_intrusive<Message>(std::move(message))));
      // The future's callbacks may schedule further RPCs, increasing the count.
      // Thus we must decrease it after completing the future, otherwise it may
      // briefly dip to zero and trick join into thinking all work is done.
      decreaseCallCount(clientActiveCalls_);
    });
  }
}

void TensorPipeAgent::markFutureWithError(
    std::shared_ptr<AtomicJitFuture> atomicFuture,
    std::string errorMsg) {
  if (!atomicFuture->isComplete.test_and_set()) {
    // Completing the future will run its callbacks, which could execute
    // arbitrary user code. To prevent blocking or stalling the TensorPipe event
    // loops, we defer this to a worker thread.
    threadPool_.run([this,
                     atomicFuture{std::move(atomicFuture)},
                     errorMsg{std::move(errorMsg)}]() mutable {
      atomicFuture->jitFuture->setError(
          std::make_exception_ptr(std::runtime_error(errorMsg)));
      // The future's callbacks may schedule further RPCs, increasing the count.
      // Thus we must decrease it after completing the future, otherwise it may
      // briefly dip to zero and trick join into thinking all work is done.
      decreaseCallCount(clientActiveCalls_);
    });
  }
}

std::vector<c10::DeviceIndex> TensorPipeAgent::getDevicesForRemote(
    const std::string& remoteName,
    const Message& message) const {
  const auto& deviceMaps =
      message.isRequest() ? opts_.deviceMaps : reverseDeviceMaps_;

  const auto errStr = c10::str(
      "TensorPipe RPC backend only supports CPU tensors by default, please "
      "move your tensors to CPU before sending them over RPC, or call "
      "`set_device_map` on `TensorPipeRpcBackendOptions` to explicitly "
      "configure device mapping. ",
      message.isRequest() ? "Request" : "Response",
      " device mapping is not available for destination ",
      remoteName);

  const auto& iter = deviceMaps.find(remoteName);
  if (iter == deviceMaps.end()) {
    for (const auto& t : message.tensors()) {
      TORCH_CHECK(
          t.device().is_cpu(),
          errStr,
          ", but found tensor on device: ",
          t.device());
    }
    return {};
  } else {
    return getDevicesForTensors(message.tensors(), iter->second, errStr);
  }
}

tensorpipe::DeviceMap TensorPipeAgent::getDeviceMap(const WorkerInfo& dest) {
  auto it = opts_.deviceMaps.find(dest.name_);
  if (it == opts_.deviceMaps.end()) {
    return {};
  }
  return it->second;
}

size_t TensorPipeAgent::timeoutMapSize() {
  std::unique_lock<std::mutex> lock(timeoutMapMutex_);
  return timeoutMap_.size();
}

size_t TensorPipeAgent::numPendingResponses() {
  size_t totalPending = 0;
  std::unique_lock<std::mutex> lock(mutex_);
  for (const auto& entry : connectedPipes_) {
    totalPending += entry.second.pendingResponseMessage_.size();
  }
  return totalPending;
}

} // namespace rpc
} // namespace distributed
} // namespace torch

#endif // USE_TENSORPIPE
