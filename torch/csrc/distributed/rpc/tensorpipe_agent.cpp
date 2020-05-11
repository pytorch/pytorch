#include <torch/csrc/distributed/rpc/tensorpipe_agent.h>

#include <torch/csrc/distributed/rpc/request_callback_impl.h>
#include <torch/csrc/distributed/rpc/utils.h>

#include <tensorpipe/channel/basic/context.h>
#ifdef TP_ENABLE_CMA
#include <tensorpipe/channel/cma/context.h>
#endif
#ifdef TP_ENABLE_SHM
#include <tensorpipe/transport/shm/context.h>
#include <unistd.h>
#endif
#include <tensorpipe/transport/uv/context.h>

namespace torch {
namespace distributed {
namespace rpc {

constexpr long kToMilliseconds = 1000;

const std::string kGilAverageWaitTime = "agent.gil_average_wait_time_us";
const std::string kThreadPoolSize = "agent.thread_pool_size";
const std::string kNumIdleThreads = "agent.num_idle_threads";

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
  std::vector<std::string> addresses = {"tcp://127.0.0.1"};
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
    const float /* unused */) {
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

  // Don't need to hold lock while calling tensorpipe API.
  lock.unlock();

  pipeWrite(
      clientPipe.pipe_,
      std::move(requestMessage),
      [this, &clientPipe, futureResponseMessage](
          const tensorpipe::Error& error) {
        if (error) {
          LOG(WARNING) << "client write error: " << error.what();
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
                  [futureResponseMessage,
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

// TODO: Remove sync()
void TensorPipeAgent::sync() {}

// TODO: Remove join()
void TensorPipeAgent::join() {
  shutdown();
}

void TensorPipeAgent::shutdownImpl() {
  threadPool_.waitWorkComplete();
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

} // namespace rpc
} // namespace distributed
} // namespace torch
