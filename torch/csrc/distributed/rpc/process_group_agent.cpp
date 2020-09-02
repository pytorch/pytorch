#include <torch/csrc/distributed/rpc/process_group_agent.h>

#include <c10/util/C++17.h>
#include <c10d/ProcessGroup.hpp>
#include <fmt/format.h>
#include <torch/csrc/distributed/rpc/utils.h>

namespace torch {
namespace distributed {
namespace rpc {
const std::string kRPCTimeoutErrorStr =
    "RPC ran for more than {} milliseconds and timed out.";

namespace {
constexpr auto kSecToMsConversion = 1000;
}

//////////////////////////  MessageCounter  /////////////////////////////////

ProcessGroupAgent::MessageCounter::MessageCounter(int worldSize)
    : counters_(worldSize) {}

void ProcessGroupAgent::MessageCounter::increment(int dst) {
  std::lock_guard<std::mutex> guard(mutex_);
  ++counters_[dst];
}

std::vector<int64_t> ProcessGroupAgent::MessageCounter::snapshot() {
  std::lock_guard<std::mutex> guard(mutex_);
  return counters_;
}

//////////////////////////  MetricsTracker  /////////////////////////////////

ProcessGroupAgent::AverageMetricsTracker::AverageMetricsTracker(
    std::string key,
    uint64_t currentSum,
    uint64_t currentCount)
    : key_(std::move(key)),
      currentSum_(currentSum),
      currentCount_(currentCount) {}

void ProcessGroupAgent::AverageMetricsTracker::addData(uint64_t dataPoint) {
  currentSum_ += dataPoint;
  ++currentCount_;
}

double ProcessGroupAgent::AverageMetricsTracker::computeAverage() {
  return currentCount_ == 0 ? 0 : currentSum_ / (double)currentCount_;
}

////////////////////////  ProcessGroupAgent  /////////////////////////////////

using steady_clock_time_point =
    std::chrono::time_point<std::chrono::steady_clock>;
const steady_clock_time_point kInfiniteTimeoutTimePoint =
    std::chrono::time_point<std::chrono::steady_clock>::max();
const std::string kNumPendingRequests = "agent.num_pending_requests";
const std::string kThreadPoolSize = "agent.thread_pool_size";
const std::string kNumIdleThreads = "agent.num_idle_threads";
const std::string kGilAverageWaitTime = "agent.gil_average_wait_time_us";
const std::string kClientActiveCalls = "agent.client_active_calls";
const std::string kServerActiveCalls = "agent.server_active_calls";
const std::string kServerActiveAsyncCalls = "agent.server_active_async_calls";

void ProcessGroupAgent::collectNames() {
  const std::string& workerName = workerInfo_.name_;
  const auto worldSize = pg_->getSize();

  // use c10d allgather to collect names
  torch::Tensor nameTensor =
      torch::zeros({WorkerInfo::MAX_NAME_LEN}, torch::kChar);
  memcpy(nameTensor.storage().data(), workerName.c_str(), workerName.length());
  std::vector<torch::Tensor> inputName = {nameTensor};
  std::vector<std::vector<torch::Tensor>> outputNames(1);
  for (int i = 0; i < worldSize; ++i) {
    outputNames[0].emplace_back(
        torch::empty({WorkerInfo::MAX_NAME_LEN}, {torch::kChar}));
  }
  pg_->allgather(outputNames, inputName)->wait();

  // convert collected name tensors into string names
  for (worker_id_t i = 0; i < worldSize; ++i) {
    torch::Tensor& tensor = outputNames[0][i];
    std::string peerName((const char*)tensor.storage().data<signed char>());

    TORCH_CHECK(
        nameMap_.find(peerName) == nameMap_.end(),
        "RpcAgent name ",
        peerName,
        " is not unique.");

    nameMap_[std::move(peerName)] = i;
  }
}

ProcessGroupAgent::ProcessGroupAgent(
    std::string workerName,
    std::shared_ptr<c10d::ProcessGroup> pg,
    int numSendRecvThreads,
    std::chrono::milliseconds rpcTimeout,
    std::unique_ptr<RequestCallback> cb)
    : RpcAgent(
          WorkerInfo(std::move(workerName), (int64_t)pg->getRank()),
          std::move(cb),
          rpcTimeout),
      pg_(std::move(pg)),
      sendCounts_(pg_->getSize()),
      recvCounts_(pg_->getSize()),
      nextId_(0),
      sendMutexes_(pg_->getSize()),
      threadPool_(numSendRecvThreads),
      timeoutThreadEnabled_{false} {
  // initialize metric info counters
  metrics_.resize(ProcessGroupAgentMetrics::N_METRICS);
  metrics_[ProcessGroupAgentMetrics::GIL_WAIT_TIME] =
      std::make_unique<AverageMetricsTracker>(kGilAverageWaitTime);
  collectNames();
  auto workerRankIter = nameMap_.find(workerInfo_.name_);
  TORCH_CHECK(
      workerRankIter != nameMap_.end(),
      "Failed to resolve worker "
      "name ",
      workerInfo_.name_,
      " to a ProcessGroup rank.");
  TORCH_CHECK(
      pg_->getRank() == workerRankIter->second,
      "Resolved worker rank ",
      workerRankIter->second,
      " does not match ProcessGroup rank ",
      pg_->getRank());

  // tmp vector to sort names in rank's order
  const auto worldSize = pg_->getSize();
  std::vector<std::string> tmpWorkerIds(worldSize);
  for (auto& entry : nameMap_) {
    tmpWorkerIds[entry.second] = entry.first;
  }

  allWorkerInfo_.reserve(worldSize);
  for (worker_id_t rank = 0; rank < worldSize; ++rank) {
    allWorkerInfo_.emplace_back(std::move(tmpWorkerIds[rank]), rank);
  }
}

ProcessGroupAgent::~ProcessGroupAgent() {
  if (rpcAgentRunning_) {
    shutdown();
  }
}

const WorkerInfo& ProcessGroupAgent::getWorkerInfo(
    const std::string& workerName) const {
  const auto idIter = nameMap_.find(workerName);
  TORCH_CHECK(
      idIter != nameMap_.end(), "Unknown destination worker ", workerName);

  return allWorkerInfo_[idIter->second];
}

const WorkerInfo& ProcessGroupAgent::getWorkerInfo(worker_id_t id) const {
  return allWorkerInfo_[id];
}

std::vector<WorkerInfo> ProcessGroupAgent::getWorkerInfos() const {
  return allWorkerInfo_;
}

void ProcessGroupAgent::join() {
  sync();
  std::unique_lock<std::mutex> lock(futureMutex_);
  futureCV_.wait(
      lock, [this] { return futures_.empty() && futureTimeouts_.empty(); });
  lock.unlock();
  pg_->barrier()->wait();
}

bool ProcessGroupAgent::hasPendingMessage() {
  const auto worldSize = pg_->getSize();
  auto snapshot = std::make_unique<std::vector<int64_t>>();
  snapshot->reserve(2 * worldSize);
  auto recvSnapshot = recvCounts_.snapshot();
  auto sendSnapshot = sendCounts_.snapshot();
  snapshot->insert(
      snapshot->end(),
      std::make_move_iterator(recvSnapshot.begin()),
      std::make_move_iterator(recvSnapshot.end()));
  snapshot->insert(
      snapshot->end(),
      std::make_move_iterator(sendSnapshot.begin()),
      std::make_move_iterator(sendSnapshot.end()));

  auto snapshotData = snapshot->data();
  auto deleteWhenDone = snapshot.release();
  std::vector<torch::Tensor> inputSnapshot = {torch::from_blob(
      snapshotData,
      {2, worldSize},
      [deleteWhenDone](void*) { delete deleteWhenDone; },
      {torch::kInt64})};
  // allgather both send and recv messages in one shot
  std::vector<std::vector<torch::Tensor>> outputSnapshots(1);

  for (int i = 0; i < worldSize; ++i) {
    outputSnapshots[0].emplace_back(
        torch::zeros({2, worldSize}, {torch::kInt64}));
  }

  pg_->allgather(outputSnapshots, inputSnapshot)->wait();

  // loop through all send/recv pairs to make sure that all sent messages are
  // processed.
  const auto& peerCounts = outputSnapshots[0];
  for (int from = 0; from < worldSize; ++from) {
    for (int to = 0; to < worldSize; ++to) {
      // peerCounts[x][0] is recv counts, and peerCounts[x][1] is send counts

      const auto& sentCnt = peerCounts[from][1][to].data_ptr<int64_t>()[0];
      const auto& recvCnt = peerCounts[to][0][from].data_ptr<int64_t>()[0];
      // NB: we cannot throw an error when sentCnt < recvCnt here. Because, send
      // and recv counts on different workers are read in a distributed manner.
      // It is possible that the sender reads its send count before sending, but
      // the receive reads its recv count after receiving. Hence, both > and <
      // are valid states.
      if (sentCnt != recvCnt) {
        return true;
      }
    }
  }
  return false;
}

void ProcessGroupAgent::sync() {
  // Block until all processes wants to sync.
  pg_->barrier()->wait();
  // block until all peers agree that all sent messages have been processed.
  do {
    // Finish all send/recv tasks in the thread pool
    threadPool_.waitWorkComplete();
    // As there could be nested RPC calls, or response callback could also
    // trigger more messages to be sent, we need to wait for the thread pool
    // again.
  } while (hasPendingMessage());
}

void ProcessGroupAgent::startImpl() {
  timeoutThreadEnabled_.store(true);
  listenerThread_ = std::thread(&ProcessGroupAgent::listenLoop, this);
  futureTimeoutThread_ =
      std::thread(&ProcessGroupAgent::pollTimedOutRPCs, this);
}

void ProcessGroupAgent::shutdownImpl() {
  LOG(INFO) << "Shutting down ProcessGroupAgent on rank " << pg_->getRank()
            << ".";
  {
    std::unique_lock<std::mutex> lock(futureMutex_);
    timeoutThreadEnabled_.store(false);
  }
  futureTimeoutCV_.notify_one();
  futureTimeoutThread_.join();
  // Abort listener thread to stop accepting new work. We need to interrupt the
  // recvWork->wait() call the listener loop may be blocked in before joining
  // the thread.
  {
    std::unique_lock<std::mutex> lock(recvWorkMutex_);
    if (recvWork_) {
      recvWork_->abort();
    }
  }
  listenerThread_.join();
  // Abort any pending sends to any destination rank that have not been
  // completed.
  {
    std::lock_guard<std::mutex> lock(pendingSendMutex_);
    for (auto& it : currentPendingSends_) {
      const auto& pendingSends = it.second;
      const auto dst = it.first;
      for (const auto& send : pendingSends) {
        if (!send->isCompleted()) {
          LOG(INFO) << "Worker " << RpcAgent::getWorkerInfo().id_
                    << " aborting pending send to destination rank " << dst;

          send->abort();
        }
      }
    }
  }
  // Note: calling threadPool_.waitWorkComplete() after listenerThread.join() so
  // that we can finish any possible work enqueued into the thread pool, before
  // python RPC handler is shutdown (see shutdown in rpc/api.py).
  threadPool_.waitWorkComplete();
}

std::shared_ptr<FutureMessage> ProcessGroupAgent::send(
    const WorkerInfo& to,
    Message&& message,
    const float rpcTimeoutSeconds) {
  // Throw if we previously encountered an exception in ::listenLoop.
  {
    std::unique_lock<std::mutex> guard(listenLoopExceptionMutex_);
    if (listenLoopException_) {
      std::rethrow_exception(listenLoopException_);
    }
  }

  if (!rpcAgentRunning_.load()) {
    // We are trying to send but RPC has been shut down on this node. This can
    // happen if we are in a shutdown sequence but background threads are still
    // processing messages that result in send()s. Throw a descriptive error.
    auto err = c10::str(
        "Node ",
        RpcAgent::getWorkerInfo().id_,
        "tried to send() a message of type ",
        message.type(),
        " but RPC is no longer running on this node.");
    throw std::runtime_error(err);
  }
  TORCH_CHECK(
      to.id_ < (worker_id_t)pg_->getSize(),
      "Destination rank is out of bound, got ",
      to.id_,
      ", but world size is ",
      pg_->getRank());

  auto requestId = nextId();
  auto future = std::make_shared<FutureMessage>();
  if (message.isRequest()) {
    // millisecond level precision of when request started.
    auto futureStartTime = std::chrono::steady_clock::now();
    // if passed in timeout is unset, then use the currently set default timeout
    // for all RPCs.
    auto timeout = rpcTimeoutSeconds == kUnsetRpcTimeout
        ? getRpcTimeout()
        : std::chrono::milliseconds(
              static_cast<int>(rpcTimeoutSeconds * kSecToMsConversion));

    // Prepare endTime from timeout. Set infinite timeout if
    // specified.
    steady_clock_time_point endTime = timeout.count() == 0
        ? kInfiniteTimeoutTimePoint
        : futureStartTime + timeout;
    bool notifyThread = false;
    {
      std::lock_guard<std::mutex> lock{futureMutex_};
      // Insert future into future map.
      futures_.emplace(
          std::piecewise_construct,
          std::forward_as_tuple(requestId),
          std::forward_as_tuple(FutureInfo(future, endTime, to.id_, timeout)));
      // insert future into timeouts map to keep track of its timeout
      auto& requestIds = futureTimeouts_[endTime];
      requestIds.insert(requestId);
      // Signal the watchdog to monitor future timeouts if this is the first
      // future created or it has earlier end time than other futures in the
      // map.
      if (futureTimeouts_.begin()->first == endTime &&
          (requestIds.size() == 1)) {
        notifyThread = true;
      }
    }
    if (notifyThread) {
      // Notify the watchdog thread only after releasing the lock,
      // so watchdog can acquire lock on waking up.
      futureTimeoutCV_.notify_one();
    }
    message.setId(requestId);
    ++clientActiveCalls_;
  } else {
    future->markCompleted(Message());
  }

  // Sending to ourselves: bypass the send logic and enqueue directly
  // to our receiving queue.
  if (to.id_ == (worker_id_t)pg_->getRank()) {
    sendToSelf(std::move(message));
    return future;
  }

  // NB: cannot directly pass ``to`` to the ``SendWork``, because it might no
  // longer be alive when the ``SendWork`` is executed. For example, the
  // application could query the ``WorkerInfo`` using name through the
  // ``RpcAgent::getWorkerInfo`` API, and pass the ``WorkerInfo`` back here, so
  // we have C++ -> Python -> C++. For an asynchronous RPC, the ``WorkerInfo``
  // reference on Python side could die before ``SendWork`` uses it, and Pybind
  // will not keep the Python reference alive even if it originally comes from
  // the C++ land. Hence, we have to explicitly use the ``WorkerInfo`` in the
  // C++ land.
  enqueueSend(SendWork(allWorkerInfo_[to.id_], std::move(message)));
  return future;
}

void ProcessGroupAgent::handleSend(const SendWork& work) {
  auto serializedPayload = std::make_unique<std::string>(std::move(
      wireSerialize(work.message_.payload(), work.message_.tensors())));

  std::vector<torch::Tensor> preamble = {torch::tensor(
      {(int64_t)pg_->getRank(),
       (int64_t)serializedPayload->length(),
       (int64_t)work.message_.type(),
       (int64_t)work.message_.id()},
      {torch::kInt64})};

  // ProcessGroup is not thread-safe when sending with the same tag,
  // hence the lock
  std::vector<c10::intrusive_ptr<c10d::ProcessGroup::Work>> pendingSends;
  const auto dst = work.to_.id_;

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  auto serializedPayloadData = const_cast<char*>(serializedPayload->data());
  auto serializedPayloadSize = serializedPayload->size();
  std::string* deleteWhenDone = serializedPayload.release();
  std::vector<torch::Tensor> payload = {torch::from_blob(
      reinterpret_cast<void*>(serializedPayloadData),
      serializedPayloadSize,
      [deleteWhenDone](void*) { delete deleteWhenDone; },
      {torch::kChar})};
  pendingSends.reserve(2);

  sendCounts_.increment(dst);

  {
    std::lock_guard<std::mutex> guard(sendMutexes_[dst]);
    pendingSends.emplace_back(pg_->send(preamble, dst, dst /* channelTag */));
    pendingSends.emplace_back(pg_->send(payload, dst, dst /* channelTag */));
  }
  // Write pendingSends to a global map so that they can be interrupted by
  // ::shutdown().
  {
    std::lock_guard<std::mutex> pendingSendGuard(pendingSendMutex_);
    for (auto& p : pendingSends) {
      currentPendingSends_[dst].insert(p);
    }
  }

  for (auto& pendingSend : pendingSends) {
    if (!rpcAgentRunning_.load() || !pendingSend->wait()) {
      // Send was interrupted or RPC is not running.
      return;
    }
  }

  // Erase the pending sends that we added since we have returned from wait.
  {
    std::lock_guard<std::mutex> pendingSendGuard(pendingSendMutex_);
    // NB: We cannot just erase all of currentPendingSends[dst], since this
    // might preemptively remove sends from other threads.
    auto& set = currentPendingSends_[dst];
    for (auto& p : pendingSends) {
      set.erase(p);
    }
  }
}

void ProcessGroupAgent::sendToSelf(Message&& message) {
  threadPool_.run(std::bind(
      [this](const Message& message) {
        // Unlike the other cases, need to add a tensor deleter, since the
        // data outlives the scope of this function. It's shared_ptr<> due
        // to c++11 lambda capture limitations with unique_ptr<>.
        std::unique_ptr<std::string> payload;
        try {
          payload = std::make_unique<std::string>(
              wireSerialize(message.payload(), message.tensors()));
          // only increment sendCounts when the message is indeed added into
          // local recv.
          sendCounts_.increment(pg_->getRank());
        } catch (std::exception& e) {
          markFutureWithError(message.id(), e.what());
          return;
        }
        const char* data = payload->data();
        size_t len = payload->length();
        std::string* delete_when_done = payload.release();
        enqueueRecv(RecvWork(
            getWorkerInfo(pg_->getRank()),
            message.type(),
            message.id(),
            torch::from_blob(
                (void*)data,
                len,
                [delete_when_done](void*) { delete delete_when_done; },
                {torch::kChar})));
      },
      std::move(message)));
}

void ProcessGroupAgent::enqueueSend(SendWork work) {
  // NB: this can be changed to use a native move capture when moved to C++14
  threadPool_.run(std::bind(
      [this](const SendWork& work) {
        try {
          handleSend(work);
        } catch (std::exception& e) {
          auto errorStr = c10::str(
              "Encountered exception in ProcessGroupAgent::enqueueSend: ",
              e.what(),
              " on node: ",
              RpcAgent::getWorkerInfo().id_);
          auto exceptionMsg =
              rpc::createExceptionResponse(errorStr, work.message_.id());
          if (work.message_.isRequest()) {
            // Mark the future with corresponding to this request with an error.
            markFutureWithError(exceptionMsg);
          } else if (work.message_.isResponse()) {
            // Try sending the error along.
            handleSend(SendWork(work.to_, std::move(exceptionMsg)));
          }
        }
      },
      std::move(work)));
}

bool ProcessGroupAgent::handleRecv(RecvWork& work) {
  torch::Tensor& payload = work.payload_;
  auto data = wireDeserialize(payload.storage().data(), payload.numel());
  Message message(
      std::move(data.first), std::move(data.second), work.type_, work.id_);
  if (message.isRequest()) {
    ++serverActiveCalls_;
    std::shared_ptr<FutureMessage> futureResponse;
    try {
      futureResponse = cb_->operator()(message);
    } catch (const std::exception& e) {
      futureResponse = std::make_shared<FutureMessage>();
      futureResponse->setError(e.what());
    }
    if (futureResponse->completed()) {
      --serverActiveCalls_;
      if (!futureResponse->hasError()) {
        send(work.from_, std::move(*futureResponse).moveValue());
      } else {
        send(
            work.from_,
            createExceptionResponse(
                futureResponse->error()->what(), message.id()));
      }
    } else {
      ++serverActiveAsyncCalls_;
      // Callback processing returned an incomplete future. Add sending the
      // response as a callback which fires when the future completes.
      // Use a weak_ptr, so we can std::move the future's value.
      auto fromId = work.from_.id_;
      auto requestId = work.id_;
      futureResponse->addCallback([this,
                                   fromId,
                                   requestId,
                                   weak = std::weak_ptr<FutureMessage>(
                                       futureResponse)]() {
        auto futureResponse = weak.lock();
        TORCH_INTERNAL_ASSERT(futureResponse);
        --serverActiveCalls_;
        --serverActiveAsyncCalls_;
        if (!futureResponse->hasError()) {
          send(getWorkerInfo(fromId), std::move(*futureResponse).moveValue());
        } else {
          send(
              getWorkerInfo(fromId),
              createExceptionResponse(
                  futureResponse->error()->what(), requestId));
        }
      });
    }
  } else if (message.isResponse()) {
    auto id = message.id();
    std::shared_ptr<FutureMessage> fm = nullptr;
    {
      std::lock_guard<std::mutex> lock{futureMutex_};
      const auto& futureInfo = futures_.find(id);
      if (futureInfo == futures_.end()) {
        // Received a completion for an already-processed future (such as one
        // that timed out), drop the recv. By returning false, recvCounts will
        // not be incremented, it will be incremented by the thread that
        // determined that the future timed out.
        return false;
      }
      // Use futureInfo before destructing it.
      fm = futureInfo->second.future_;
      auto endTime = futureInfo->second.endTime_;
      futures_.erase(id);
      // look up the corresponding future by its time out and request
      // ID, and remove it from the timeouts map
      auto& futuresAtTime = futureTimeouts_[endTime];
      auto it = futuresAtTime.find(id);
      TORCH_INTERNAL_ASSERT(
          it != futuresAtTime.end(),
          "Error: could not find future in futureTimeouts map, race condition.");
      futuresAtTime.erase(it);
      if (futuresAtTime.empty()) {
        // remove the key from futureTimeouts_
        futureTimeouts_.erase(endTime);
      }
    }
    futureCV_.notify_all();
    --clientActiveCalls_;
    if (message.type() == MessageType::EXCEPTION) {
      fm->setError(
          std::string(message.payload().begin(), message.payload().end()));
    } else {
      fm->markCompleted(std::move(message));
    }
  } else {
    // TODO: pass the error back to the caller instead of crashing here.
    TORCH_INTERNAL_ASSERT(false, "unrecognized message type ", message.type());
  }
  return true;
}

void ProcessGroupAgent::enqueueRecv(RecvWork work) {
  threadPool_.run(std::bind(
      [&](RecvWork& work) {
        try {
          // Only increment recvCounts if handleRecv() tells us to. We may not,
          // i.e. if we process work corresponding to a future that has already
          // been processed.
          if (handleRecv(work)) {
            recvCounts_.increment(work.from_.id_);
          }
        } catch (const std::exception& e) {
          // Processing for this request/response failed. Log the details of the
          // request.
          auto fromId = work.from_.id_;
          auto err = c10::str(
              "Internal error while processing request of type ",
              work.type_,
              " on node ",
              RpcAgent::getWorkerInfo().id_,
              ", from node ",
              fromId,
              " : ",
              e.what());
          LOG(INFO) << err;
          // Still increment so that this recv is recognized as non-oustanding
          // during graceful shutdown.
          recvCounts_.increment(work.from_.id_);
        }
      },
      std::move(work)));
}

void ProcessGroupAgent::markFutureWithError(Message& message) {
  TORCH_INTERNAL_ASSERT(
      message.type() == MessageType::EXCEPTION,
      "markFutureWithError should be only called with Message that has type Exception.");
  markFutureWithError(
      message.id(),
      std::string(message.payload().begin(), message.payload().end()));
}

void ProcessGroupAgent::markFutureWithError(int64_t id, std::string errorMsg) {
  std::shared_ptr<FutureMessage> fm = nullptr;
  {
    std::lock_guard<std::mutex> lock{futureMutex_};
    const auto& futureInfo = futures_.find(id);

    if (futureInfo == futures_.end()) {
      // Did not find future in map - this can occur when the future has timed
      // out and been processed accordingly.
      return;
    }
    fm = futureInfo->second.future_;
    auto rpcEndTime = futureInfo->second.endTime_;
    futures_.erase(id);
    // look up the corresponding future by its time out and request ID,
    // and remove it from the timeouts map
    auto& futuresAtTime = futureTimeouts_[rpcEndTime];
    auto it = futuresAtTime.find(id);
    TORCH_INTERNAL_ASSERT(
        it != futuresAtTime.end(),
        "Error: could not find future in futureTimeouts map, race condition.");
    futuresAtTime.erase(it);
    if (futuresAtTime.empty()) {
      // remove the key from futureTimeouts_
      futureTimeouts_.erase(rpcEndTime);
    }
  }

  --clientActiveCalls_;
  fm->setError(std::move(errorMsg));
  futureCV_.notify_all();
}

void ProcessGroupAgent::listenLoop() {
  try {
    listenLoopInternal();
  } catch (const std::exception& e) {
    // Error occured in listenLoop(). Stop receiving thread and store
    // exception to indicate that the RPC agent is in an unhealthy state and
    // we should shutdown.
    auto err = c10::str(
        "Encountered exception in ProcessGroupAgent::listenLoop(): ",
        e.what(),
        " on worker ",
        RpcAgent::getWorkerInfo().id_,
        ". This means that the RPC agent is in an unhealthy state and unusable.");
    LOG(ERROR) << err;
    {
      // Lock write to listenLoopException_ since ::send() reads from it.
      std::lock_guard<std::mutex> guard(listenLoopExceptionMutex_);
      listenLoopException_ = std::current_exception();
    }
  } catch (...) {
    std::string unknownErrorMsg =
        "Unknown exception occured in "
        "ProcessGroupAgent::listenLoop. RPC Agent is in an unhealthy state and "
        "unusable.";
    LOG(ERROR) << unknownErrorMsg;
    {
      // Lock write to listenLoopException_ since ::send() reads from it.
      std::lock_guard<std::mutex> guard(listenLoopExceptionMutex_);
      listenLoopException_ =
          std::make_exception_ptr(std::runtime_error(unknownErrorMsg));
    }
  }
}

void ProcessGroupAgent::listenLoopInternal() {
  while (rpcAgentRunning_.load()) {
    // rank, tensor size, message type
    std::vector<torch::Tensor> preamble = {torch::empty({4}, {torch::kInt64})};
    auto work = pg_->recvAnysource(preamble, pg_->getRank());
    {
      // Write class variable so it can be aborted by shutdown()
      std::lock_guard<std::mutex> guard(recvWorkMutex_);
      recvWork_ = work;
    }

    if (!rpcAgentRunning_.load() || !work->wait() /* aborted */) {
      return;
    }

    int64_t* preamble_items = preamble.front().storage().data<int64_t>();

    auto srcRank = preamble_items[0];
    auto size = preamble_items[1];
    MessageType type = MessageType(preamble_items[2]);
    int64_t id = preamble_items[3];

    std::vector<torch::Tensor> tensors = {torch::empty({size}, {torch::kChar})};
    work = pg_->recv(tensors, srcRank, pg_->getRank());
    {
      // Write class variable so it can be aborted by shutdown()
      std::lock_guard<std::mutex> guard(recvWorkMutex_);
      recvWork_ = work;
    }

    if (!rpcAgentRunning_.load() || !work->wait() /* aborted */) {
      return;
    }

    enqueueRecv(
        RecvWork(allWorkerInfo_[srcRank], type, id, std::move(tensors[0])));
  }
}

void ProcessGroupAgent::pollTimedOutRPCs() {
  while (timeoutThreadEnabled_.load()) {
    std::unique_lock<std::mutex> lock{futureMutex_};
    steady_clock_time_point minEndTime;
    // Estimate amount of time the first future will time out in, and sleep
    // for that long.
    // if there are no futures or the first future's RPC timeout is set to 0
    // (meaning no timeout), then sleep for a set "infinity" time.
    if (futureTimeouts_.empty()) {
      minEndTime = kInfiniteTimeoutTimePoint;
    } else {
      minEndTime = futureTimeouts_.begin()->first;
    }

    auto shouldUpdateMinEndTimePredicate = [&, this]() -> bool {
      // Notice, whoever modifies `timeoutThreadEnabled_`
      // must acquire a lock on `futureMutex_`.
      // Otherwise, this predicate could deadlock.
      // If during evaluating the predicate, `::shutdown()` is called, then
      // the predicate missed the notification before it started waiting
      // on the cond var.
      if (!timeoutThreadEnabled_.load()) {
        return true;
      }
      steady_clock_time_point minEndTimeInMap = kInfiniteTimeoutTimePoint;
      if (futureTimeouts_.empty()) {
        minEndTimeInMap = kInfiniteTimeoutTimePoint;
      } else {
        minEndTimeInMap = futureTimeouts_.begin()->first;
      }
      return minEndTimeInMap < minEndTime;
    };

    bool shouldUpdateMinEndTime = true;
    if (minEndTime == kInfiniteTimeoutTimePoint) {
      futureTimeoutCV_.wait(lock, shouldUpdateMinEndTimePredicate);
    } else {
      shouldUpdateMinEndTime = futureTimeoutCV_.wait_until(
          lock, minEndTime, shouldUpdateMinEndTimePredicate);
    }
    if (shouldUpdateMinEndTime) {
      continue;
    }

    const auto timedOutFutures = processTimedOutFutures();
    lock.unlock();
    futureCV_.notify_all();

    for (const auto& timedOutFuture : timedOutFutures) {
      auto errStr =
          fmt::format(kRPCTimeoutErrorStr, timedOutFuture.timeout_.count());
      auto err = makeRPCError(errStr, RPCErrorType::TIMEOUT);

      if (!timedOutFuture.future_->hasError()) {
        --clientActiveCalls_;
        timedOutFuture.future_->setError(std::move(err));
        // The future timed out and will not be processed by handleRecv(), even
        // if we eventually get a response. In order to keep track of all
        // send/recv pairs, we increment the count here.
        const int dst = timedOutFuture.dstRank_;
        recvCounts_.increment(dst);
      }
    }
  }
}

const std::vector<ProcessGroupAgent::FutureInfo> ProcessGroupAgent::
    processTimedOutFutures() {
  std::vector<FutureInfo> timedOutFutures;
  for (auto it = futureTimeouts_.begin(); it != futureTimeouts_.end();
       /* intentional no increment */) {
    const auto& endTime = it->first;
    if (std::chrono::steady_clock::now() < endTime) {
      // Since the futureTimeouts_ map is ordered by timeout, we don't need
      // to check the remaining futures.
      break;
    } else {
      const auto& futureIDs = it->second;
      for (const auto& futureID : futureIDs) {
        auto futureIt = futures_.find(futureID);
        TORCH_INTERNAL_ASSERT(
            futureIt != futures_.end(),
            "Race Condition - Expected future does not exist in map");
        const auto futInfo = futureIt->second;
        timedOutFutures.push_back(futInfo);
        futures_.erase(futureID);
      }
      it = futureTimeouts_.erase(it);
    }
  }
  return timedOutFutures;
}

std::unordered_map<std::string, std::string> ProcessGroupAgent::getMetrics() {
  std::unordered_map<std::string, std::string> metrics;
  {
    std::unique_lock<std::mutex> lock(futureMutex_);
    auto futuresSize = futures_.size();
    lock.unlock();
    metrics[kNumPendingRequests] = c10::to_string(futuresSize);
  }
  metrics[kThreadPoolSize] = c10::to_string(threadPool_.size());
  metrics[kNumIdleThreads] = c10::to_string(threadPool_.numAvailable());
  metrics[kClientActiveCalls] = c10::to_string(clientActiveCalls_.load());
  metrics[kServerActiveCalls] = c10::to_string(serverActiveCalls_.load());
  metrics[kServerActiveAsyncCalls] =
      c10::to_string(serverActiveAsyncCalls_.load());
  if (isGILProfilingEnabled()) {
    // Add time-series based metrics, just GIL wait times for now.
    {
      std::unique_lock<std::mutex> lock(metricsMutex_);
      auto avgGilWaitTime = metrics_[GIL_WAIT_TIME]->computeAverage();
      lock.unlock();
      metrics[kGilAverageWaitTime] = c10::to_string(avgGilWaitTime);
    }
  }
  return metrics;
}

void ProcessGroupAgent::addGilWaitTime(
    const std::chrono::microseconds gilWaitTime) {
  std::lock_guard<std::mutex> lock(metricsMutex_);
  metrics_[ProcessGroupAgentMetrics::GIL_WAIT_TIME]->addData(
      gilWaitTime.count());
}

} // namespace rpc
} // namespace distributed
} // namespace torch
