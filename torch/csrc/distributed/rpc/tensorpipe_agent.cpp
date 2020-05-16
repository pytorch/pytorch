#include <torch/csrc/distributed/rpc/tensorpipe_agent.h>

#include <torch/csrc/distributed/rpc/request_callback_impl.h>
#include <torch/csrc/distributed/rpc/utils.h>

#include <tensorpipe/channel/basic/context.h>
#ifdef TP_ENABLE_CMA
#include <tensorpipe/channel/cma/context.h>
#endif
#ifdef TP_ENABLE_SHM
#include <tensorpipe/transport/shm/context.h>
#endif
#include <tensorpipe/transport/uv/context.h>

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

namespace torch {
namespace distributed {
namespace rpc {

constexpr long kToMilliseconds = 1000;
// This large time duration is for the timeoutMapCV_. We cannot use
// std::chrono::time_point::max() due to a known overflow-related bug. Here is
// an explanation of the bug:
// https://stackoverflow.com/questions/42638847/what-is-the-maximum-value-i-can-pass-to-stdthreadsleep-for-and-sleep-until
constexpr auto kLargeTimeDuration = std::chrono::hours(10000);

const std::string kGilAverageWaitTime = "agent.gil_average_wait_time_us";
const std::string kThreadPoolSize = "agent.thread_pool_size";
const std::string kNumIdleThreads = "agent.num_idle_threads";
const std::string kClientActiveCalls = "agent.client_active_calls";
const std::string kServerActiveCalls = "agent.server_active_calls";
const std::string kServerActiveAsyncCalls = "agent.server_active_async_calls";

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

void TensorPipeAgent::collectNames() {
  const worker_id_t selfId = workerInfo_.id_;
  const std::string& selfName = workerInfo_.name_;

  std::vector<uint8_t> selfNameVector(
      (uint8_t*)selfName.c_str(),
      (uint8_t*)selfName.c_str() + selfName.length());
  addressStore_->set("names/" + c10::to_string(selfId), selfNameVector);

  workerIdToInfo_.emplace(selfId, WorkerInfo(selfName, selfId));
  workerNameToInfo_.emplace(selfName, WorkerInfo(selfName, selfId));
  for (worker_id_t workerId = 0; workerId < worldSize_; ++workerId) {
    if (workerId == selfId) {
      continue;
    }
    std::vector<uint8_t> workerNameVector =
        addressStore_->get("names/" + c10::to_string(workerId));
    std::string workerName(
        (char*)workerNameVector.data(), workerNameVector.size());
    workerIdToInfo_.emplace(workerId, WorkerInfo(workerName, workerId));
    workerNameToInfo_.emplace(workerName, WorkerInfo(workerName, workerId));
  }
}

TensorPipeAgent::TensorPipeAgent(
    std::shared_ptr<::c10d::Store> addressStore,
    std::string selfName,
    worker_id_t selfId,
    int worldSize,
    TensorPipeRpcBackendOptions opts)
    : RpcAgent(
          WorkerInfo(std::move(selfName), selfId),
          std::make_unique<RequestCallbackImpl>(),
          std::chrono::milliseconds(
              (long)(opts.rpcTimeoutSeconds * kToMilliseconds))),
      context_(std::make_shared<tensorpipe::Context>()),
      addressStore_(std::move(addressStore)),
      worldSize_(worldSize),
      opts_(std::move(opts)) {
  collectNames();

  // Initialize the time-series metrics tracking map
  timeSeriesMetrics_[kGilAverageWaitTime] =
      std::make_unique<TimeSeriesMetricsTracker>();
}

TensorPipeAgent::~TensorPipeAgent() {
  shutdown();
}

void TensorPipeAgent::startImpl() {
  context_->registerTransport(
      1, "tcp", std::make_shared<tensorpipe::transport::uv::Context>());
#ifdef TP_ENABLE_SHM
  context_->registerTransport(
      0, "shm", std::make_shared<tensorpipe::transport::shm::Context>());
#endif
  context_->registerChannel(
      1, "basic", std::make_shared<tensorpipe::channel::basic::Context>());
#ifdef TP_ENABLE_CMA
  context_->registerChannel(
      0, "cma", std::make_shared<tensorpipe::channel::cma::Context>());
#endif

  // TODO: We currently hardcoded localhost as pipes handshake IP address.
  // Ideally tensorpipe could provide a helper to get IP address for given
  // device interface or host names, or return the IP address of the default
  // host name. https://github.com/pytorch/pytorch/issues/36715
  std::vector<std::string> addresses = {"tcp://" + getDefaultIPAddress()};
#ifdef TP_ENABLE_SHM
  addresses.push_back(createUniqueShmAddr());
#endif

  listener_ = context_->listen(addresses);

  // Store our own url.
  const auto address = listener_->url("tcp");
  const std::vector<uint8_t> selfAddrData(address.begin(), address.end());
  addressStore_->set(workerInfo_.name_, selfAddrData);

  for (const auto& p : workerNameToInfo_) {
    const auto& name = p.first;
    auto nodeAddrData = addressStore_->get(name);
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
    LOG(WARNING) << "got error from listener: " << error.what();
    return;
  }

  // Accept the next connection request
  listener_->accept([this](
                        const tensorpipe::Error& error,
                        std::shared_ptr<tensorpipe::Pipe> pipe) {
    onListenerAccepted(error, pipe);
  });

  // Arm for server read
  respond(pipe);
}

void TensorPipeAgent::pipeRead(
    const std::shared_ptr<tensorpipe::Pipe>& pipe,
    std::function<void(const tensorpipe::Error&, Message&&)> fn) {
  pipe->readDescriptor([fn{std::move(fn)}, pipe](
                           const tensorpipe::Error& error,
                           tensorpipe::Message&& tpMessage) mutable {
    if (error) {
      fn(error, Message());
      return;
    }

    // Allocate memory and fill in pointers
    Message rpcMessage = tensorpipeAllocateMessage(tpMessage);

    pipe->read(
        std::move(tpMessage),
        [fn{std::move(fn)}, rpcMessage{std::move(rpcMessage)}](
            const tensorpipe::Error& error,
            tensorpipe::Message&& /* unused */) mutable {
          fn(error, std::move(rpcMessage));
        });
  });
}

void TensorPipeAgent::pipeWrite(
    const std::shared_ptr<tensorpipe::Pipe>& pipe,
    Message&& rpcMessage,
    std::function<void(const tensorpipe::Error&)> fn) {
  TensorPipeEntry tpEntry = tensorpipeSerialize(rpcMessage);
  tensorpipe::Message tpMessage = std::move(tpEntry.message);
  pipe->write(
      std::move(tpMessage),
      // Note: keep payload and tensors of rpcMessage alive.
      [rpcMessage{std::move(rpcMessage)},
       reservedTensors{std::move(tpEntry.reservedTensors)},
       copiedTensors{std::move(tpEntry.copiedTensors)},
       fn{std::move(fn)}](
          const tensorpipe::Error& error,
          tensorpipe::Message&& /* unused */) mutable { fn(error); });
}

void TensorPipeAgent::sendCompletedResponseMessage(
    std::shared_ptr<tensorpipe::Pipe>& pipe,
    std::shared_ptr<FutureMessage>& futureResponseMessage,
    uint64_t messageId) {
  if (!rpcAgentRunning_.load()) {
    LOG(WARNING) << "RPC agent is being closed. Skip sending rpc response";
    return;
  }

  const c10::optional<utils::FutureError> error =
      futureResponseMessage->error();
  Message&& responseMessage = std::move(*futureResponseMessage).moveValue();
  responseMessage.setId(messageId);
  if (!error) {
    pipeWrite(
        pipe, std::move(responseMessage), [](const tensorpipe::Error& error) {
          if (error) {
            LOG(WARNING) << "sending response failed: " << error.what();
            return;
          }
        });
  } else {
    pipeWrite(
        pipe,
        createExceptionResponse(error->what(), responseMessage.id()),
        [](const tensorpipe::Error& error) {
          if (error) {
            LOG(WARNING) << "sending error response failed: " << error.what();
            return;
          }
        });
  }
}

void TensorPipeAgent::respond(std::shared_ptr<tensorpipe::Pipe>& pipe) {
  pipeRead(
      pipe,
      [this, pipe](
          const tensorpipe::Error& error, Message&& requestMessage) mutable {
        // TODO: Handle server pipe read error
        if (error) {
          LOG(WARNING) << "Server read message: " << error.what();
          return;
        }

        // Arm for next read
        respond(pipe);

        uint64_t messageId = requestMessage.id();
        ++serverActiveCalls_;

        // Defer user RPC UDF run to thread pool
        threadPool_.run([this,
                         pipe,
                         messageId,
                         requestMessage{std::move(requestMessage)}]() mutable {
          std::shared_ptr<FutureMessage> futureResponseMessage;
          try {
            futureResponseMessage = cb_->operator()(requestMessage);
          } catch (const std::exception& e) {
            futureResponseMessage = std::make_shared<FutureMessage>();
            futureResponseMessage->setError(e.what());
          }

          // Shortcut if immediately done
          if (futureResponseMessage->completed()) {
            --serverActiveCalls_;
            sendCompletedResponseMessage(
                pipe, futureResponseMessage, messageId);
          } else {
            // Not complete yet
            ++serverActiveAsyncCalls_;
            futureResponseMessage->addCallback(
                [this, pipe, futureResponseMessage, messageId]() mutable {
                  --serverActiveCalls_;
                  --serverActiveAsyncCalls_;
                  sendCompletedResponseMessage(
                      pipe, futureResponseMessage, messageId);
                });
          }
        });
      });
}

std::shared_ptr<FutureMessage> TensorPipeAgent::send(
    const WorkerInfo& toWorkerInfo,
    Message&& requestMessage,
    const float rpcTimeoutSeconds) {
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
        toWorkerInfo.id_, ClientPipe(context_->connect(url)));
  }
  ClientPipe& clientPipe = it->second;
  auto& pendingResponseMessage = clientPipe.pendingResponseMessage_;

  std::shared_ptr<FutureMessage> futureResponseMessage =
      std::make_shared<FutureMessage>();
  requestMessage.setId(nextMessageID_++);
  pendingResponseMessage[requestMessage.id()] = futureResponseMessage;

  ++clientActiveCalls_;
  // Use the default RPC timeout if no timeout is specified for this send call
  auto timeout = rpcTimeoutSeconds == kUnsetRpcTimeout
      ? getRpcTimeout()
      : std::chrono::milliseconds(
            static_cast<int>(rpcTimeoutSeconds * kToMilliseconds));

  // We only add to the timeoutMap_ if the timeout is not 0. Per our
  // documentation, a user-provided timeout of 0 indicates the RPC should never
  // expire (infinite timeout), so there is no need to track it in the
  // timeoutMap_.
  if (timeout.count() != 0) {
    // Compute the expiration time for this message based on the timeout
    auto expirationTime = computeRpcMessageExpiryTime(timeout);

    // Add the Future to the right vector in the timeoutMap_
    auto& timeoutFuturesVector = timeoutMap_[expirationTime];
    timeoutFuturesVector.emplace_back(futureResponseMessage);
    timeoutThreadCV_.notify_one();
  }

  // Don't need to hold lock while calling tensorpipe API.
  lock.unlock();

  pipeWrite(
      clientPipe.pipe_,
      std::move(requestMessage),
      [this, &clientPipe, futureResponseMessage](
          const tensorpipe::Error& error) {
        if (error) {
          LOG(WARNING) << "client write error: " << error.what();
          --clientActiveCalls_;
          futureResponseMessage->setError(error.what());
          return;
        }

        pipeRead(
            clientPipe.pipe_,
            [this, &clientPipe](
                const tensorpipe::Error& error, Message&& responseMessage) {
              if (error) {
                LOG(WARNING) << "Read response error: " << error.what();
                std::lock_guard<std::mutex> lock(mutex_);
                // We may get garbage content in responseMessage upon error.
                // Flushing all future messages belonging to this pipe due to
                // error state.
                for (auto& p : clientPipe.pendingResponseMessage_) {
                  --clientActiveCalls_;
                  std::shared_ptr<FutureMessage>& futureMessage = p.second;
                  futureMessage->setError(error.what());
                }
                clientPipe.pendingResponseMessage_.clear();
                clientPipe.readError_ = true;
                return;
              }

              // Identify future response message by message ID
              uint64_t messageId = responseMessage.id();
              std::shared_ptr<FutureMessage> futureResponseMessage;
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

              threadPool_.run(
                  [this,
                   futureResponseMessage,
                   responseMessage{std::move(responseMessage)}]() mutable {
                    --clientActiveCalls_;
                    if (responseMessage.type() == MessageType::EXCEPTION) {
                      futureResponseMessage->setError(std::string(
                          responseMessage.payload().begin(),
                          responseMessage.payload().end()));
                    } else {
                      futureResponseMessage->markCompleted(
                          std::move(responseMessage));
                    }
                  });
            });
      });

  return futureResponseMessage;
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

      steady_clock_time_point earliestTimeout =
          std::chrono::steady_clock::now() + kLargeTimeDuration;

      if (std::chrono::steady_clock::now() >= earliestTimeout) {
        break;
      }
      if (!timeoutMap_.empty()) {
        earliestTimeout = timeoutMap_.begin()->first;
      }
      timeoutThreadCV_.wait_until(lock, earliestTimeout);
    }

    // Move all these futures to a separate vector so we can process them
    // outside the lock.
    std::vector<std::shared_ptr<FutureMessage>> timedOutFutures =
        std::move(timeoutMap_.begin()->second);
    // We can safely remove this key from the timeoutMap_ since all these
    // futures will be processed.
    timeoutMap_.erase(timeoutMap_.begin());

    lock.unlock();

    // Set an error on futures added to the timedOutFutures vector. We do this
    // outside the lock to prevent potential lock-order-inversions by callbacks
    // triggered by the serError call.
    for (const auto& future : timedOutFutures) {
      std::string errorMsg = c10::str(
          "RPC ran for more than set timeout and will now be marked with an error");
      // Using setErrorIfNeeded so completed futures are ignored.
      future->setErrorIfNeeded(errorMsg);
    }
  }
}

// TODO: Remove sync()
void TensorPipeAgent::sync() {}

// TODO: Remove join()
void TensorPipeAgent::join() {
  shutdown();
}

void TensorPipeAgent::shutdownImpl() {
  threadPool_.waitWorkComplete();

  // Join the Timeout Thread
  timeoutThreadCV_.notify_one();
  if (timeoutThread_.joinable()) {
    timeoutThread_.join();
  }
  // TODO: context_->join() is not absolutely ready yet.
  // NOTE: context_->join() will wait for available RPC message to be
  //       read or written, and wait for the remaining unavailable ones
  //       to be called with error by invoking callbacks.
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

#ifdef TP_ENABLE_SHM
std::string TensorPipeAgent::createUniqueShmAddr() {
  thread_local uint32_t threadLocalId = 0;
  return c10::str(
      "shm://tensorpipe_rpc_agent_",
      std::this_thread::get_id(),
      "_",
      ::getpid(),
      "_",
      threadLocalId++);
}
#endif

std::unordered_map<std::string, std::string> TensorPipeAgent::getMetrics() {
  std::unordered_map<std::string, std::string> metrics;
  metrics[kThreadPoolSize] = c10::to_string(threadPool_.size());
  metrics[kNumIdleThreads] = c10::to_string(threadPool_.numAvailable());
  metrics[kClientActiveCalls] = c10::to_string(clientActiveCalls_.load());
  metrics[kServerActiveCalls] = c10::to_string(serverActiveCalls_.load());
  metrics[kServerActiveAsyncCalls] =
      c10::to_string(serverActiveAsyncCalls_.load());
  if (isGILProfilingEnabled()) {
    {
      std::unique_lock<std::mutex> lock(metricsMutex_);
      // Include the averages for each time series metric. This is just the GIL
      // Wait Time for now.
      auto averageGilWaitTime =
          timeSeriesMetrics_[kGilAverageWaitTime]->computeAverage();
      lock.unlock();
      metrics[kGilAverageWaitTime] = c10::to_string(averageGilWaitTime);
    }
  }

  return metrics;
}

void TensorPipeAgent::addGilWaitTime(
    const std::chrono::microseconds gilWaitTime) {
  std::lock_guard<std::mutex> lock(metricsMutex_);
  timeSeriesMetrics_[kGilAverageWaitTime]->addData(gilWaitTime.count());
}

TensorPipeAgent::NetworkDataDict TensorPipeAgent::getNetworkData() {
  std::lock_guard<std::mutex> lock(networkDataMutex_);
  return networkData_;
}

NetworkSourceInfo TensorPipeAgent::getNetworkSourceInfo() {
  NetworkSourceInfo info = {
      RpcAgent::getWorkerInfo().id_,
      addressStore_->get(RpcAgent::getWorkerInfo().name_)};

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

std::string TensorPipeAgent::getDefaultIPAddress() {
  std::string defaultIP = "127.0.0.1";

  std::array<char, NI_MAXHOST> hostname{};
  int rv = gethostname(hostname.data(), NI_MAXHOST);
  if (rv != 0) {
    LOG(WARNING) << "Unable to get local hostname. Falling back to "
                 << "bind with " << defaultIP;
    return defaultIP;
  }

  struct addrinfo hints {};
  memset(&hints, 0, sizeof hints);
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;
  struct addrinfo* servinfo;
  rv = getaddrinfo(hostname.data(), nullptr, &hints, &servinfo);
  if (rv != 0) {
    LOG(WARNING) << "Get address info error: " << gai_strerror(rv)
                 << ". Falling back to bind with " << defaultIP;
    return defaultIP;
  }

  // Loop through all the results and pick up the first we can bind.
  for (struct addrinfo* p = servinfo; p != nullptr; p = p->ai_next) {
    int fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (fd == -1) {
      continue;
    }
    int bind_rv = bind(fd, p->ai_addr, p->ai_addrlen);
    if (bind_rv == -1) {
      close(fd);
      continue;
    }
    close(fd);

    if (p->ai_family == AF_INET6) {
      std::string ipv6(INET6_ADDRSTRLEN, '\0');
      struct sockaddr_in6* h = (struct sockaddr_in6*)p->ai_addr;
      inet_ntop(AF_INET6, &h->sin6_addr, (char*)ipv6.data(), INET6_ADDRSTRLEN);
      freeaddrinfo(servinfo);
      return ipv6;
    } else if (p->ai_family == AF_INET) {
      std::string ipv4(INET_ADDRSTRLEN, '\0');
      struct sockaddr_in* h = (struct sockaddr_in*)p->ai_addr;
      inet_ntop(AF_INET, &h->sin_addr, (char*)ipv4.data(), INET_ADDRSTRLEN);
      freeaddrinfo(servinfo);
      return ipv4;
    }
  }

  freeaddrinfo(servinfo);

  LOG(WARNING) << "TensorPipe agent didn't find associated IP address with "
               << hostname.data() << ". Using " << defaultIP << " to bind";
  return defaultIP;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
