#include <torch/csrc/distributed/rpc/process_group_agent.h>

#include <c10/util/C++17.h>
#include <c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/rpc/request_callback_impl.h>
#include <torch/csrc/distributed/rpc/utils.h>

#include <Python.h>

namespace torch {
namespace distributed {
namespace rpc {

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

const ProcessGroupAgent::steady_clock_time_point
    ProcessGroupAgent::kInfiniteTimeoutTimePoint =
        std::chrono::time_point<std::chrono::steady_clock>::max();
const std::string kNumPendingRequests = "agent.num_pending_requests";
const std::string kThreadPoolSize = "agent.thread_pool_size";
const std::string kNumIdleThreads = "agent.num_idle_threads";
const std::string kGilAverageWaitTime = "agent.gil_average_wait_time_us";

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
  for (int i = 0; i < worldSize; ++i) {
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
    std::chrono::milliseconds rpcTimeout)
    : RpcAgent(
          WorkerInfo(std::move(workerName), pg->getRank()),
          std::make_unique<RequestCallbackImpl>(),
          rpcTimeout),
      pg_(std::move(pg)),
      sendCounts_(pg_->getSize()),
      recvCounts_(pg_->getSize()),
      nextId_(0),
      sendMutexes_(pg_->getSize()),
      threadPool_(numSendRecvThreads) {
  // initialize metric info counters
  metrics_.resize(ProcessGroupAgentMetrics::N_METRICS);
  metrics_[ProcessGroupAgentMetrics::GIL_WAIT_TIME] =
      std::make_unique<AverageMetricsTracker>(kGilAverageWaitTime);
  collectNames();
  TORCH_CHECK(
      nameMap_.size() > 1,
      "ProcessGroupAgent requires world_size to "
      "be at least 2, but got ",
      nameMap_.size());
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
  std::vector<std::string> tmpWorkerIds(pg_->getSize());
  for (auto& entry : nameMap_) {
    tmpWorkerIds[entry.second] = entry.first;
  }

  allWorkerInfo_.reserve(pg_->getSize());
  for (int rank = 0; rank < (int)tmpWorkerIds.size(); ++rank) {
    allWorkerInfo_.emplace_back(std::move(tmpWorkerIds[rank]), rank);
  }
}

ProcessGroupAgent::~ProcessGroupAgent() {
  if (rpcRunning_) {
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
  std::vector<int64_t> snapshot;
  snapshot.reserve(2 * worldSize);
  auto recvSnapshot = recvCounts_.snapshot();
  auto sendSnapshot = sendCounts_.snapshot();
  snapshot.insert(
      snapshot.end(),
      std::make_move_iterator(recvSnapshot.begin()),
      std::make_move_iterator(recvSnapshot.end()));
  snapshot.insert(
      snapshot.end(),
      std::make_move_iterator(sendSnapshot.begin()),
      std::make_move_iterator(sendSnapshot.end()));

  std::vector<torch::Tensor> inputSnapshot = {
      torch::from_blob(snapshot.data(), {2, worldSize}, {torch::kInt64})};
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

void ProcessGroupAgent::start() {
  {
    std::lock_guard<std::mutex> futureLock{futureMutex_};
    rpcRunning_.store(true);
  }
  listenerThread_ = std::thread(&ProcessGroupAgent::listenLoop, this);
  futureTimeoutThread_ =
      std::thread(&ProcessGroupAgent::pollTimedOutRPCs, this);
}

void ProcessGroupAgent::shutdown() {
  LOG(INFO) << "Shutting down ProcessGroupAgent.";
  std::unique_lock<std::mutex> lock{futureMutex_};
  if (!rpcRunning_.exchange(false)) {
    return;
  }
  lock.unlock();
  futureTimeoutCV_.notify_one();
  futureTimeoutThread_.join();
  {
    std::unique_lock<std::mutex> lock(recvWorkMutex_);
    if (recvWork_) {
      recvWork_->abort();
    }
  }
  threadPool_.waitWorkComplete();
  listenerThread_.join();
}

std::shared_ptr<FutureMessage> ProcessGroupAgent::send(
    const WorkerInfo& to,
    Message&& message) {
  TORCH_CHECK(rpcRunning_.load(), "ProcessGroupAgent hasn't started.")
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
    // Prepare endTime from timeout.
    auto timeout = rpcTimeout_.load();
    // Set infinite timeout if specified.
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
  } else {
    future->markCompleted(Message());
  }

  // Sending to ourselves: bypass the send logic and enqueue directly
  // to our receiving queue.
  if (to.id_ == (worker_id_t)pg_->getRank()) {
    threadPool_.run(std::bind(
        [this](const Message& message) {
          sendCounts_.increment(pg_->getRank());
          // Unlike the other cases, need to add a tensor deleter, since the
          // data outlives the scope of this function. It's shared_ptr<> due
          // to c++11 lambda capture limitations with unique_ptr<>.
          auto payload = std::make_unique<std::string>(
              wireSerialize(message.payload(), message.tensors()));
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
  std::string serializedPayload =
      wireSerialize(work.message_.payload(), work.message_.tensors());

  std::vector<torch::Tensor> preamble = {torch::tensor(
      {(int64_t)pg_->getRank(),
       (int64_t)serializedPayload.length(),
       (int64_t)work.message_.type(),
       (int64_t)work.message_.id()},
      {torch::kInt64})};

  // ProcessGroup is not thread-safe when sending with the same tag,
  // hence the lock
  std::vector<std::shared_ptr<c10d::ProcessGroup::Work>> pendingSends;
  const auto dst = work.to_.id_;
  std::vector<torch::Tensor> payload = {torch::from_blob(
      (void*)serializedPayload.c_str(),
      serializedPayload.length(),
      {torch::kChar})};
  pendingSends.reserve(2);

  sendCounts_.increment(dst);

  {
    std::lock_guard<std::mutex> guard(sendMutexes_[dst]);
    pendingSends.emplace_back(pg_->send(preamble, dst, dst /* channelTag */));
    pendingSends.emplace_back(pg_->send(payload, dst, dst /* channelTag */));
  }
  for (auto& pendingSend : pendingSends) {
    pendingSend->wait();
  }
}

void ProcessGroupAgent::enqueueSend(SendWork work) {
  // NB: this can be changed to use a native move capture when moved to C++14
  threadPool_.run(std::bind(
      [this](const SendWork& work) {
        try {
          handleSend(work);
        } catch (std::exception& e) {
          if (work.message_.isRequest()) {
            std::ostringstream ss;
            ss << "Encountered exception in ProcessGroupAgent::enqueueSend: "
               << e.what();
            auto exceptionMsg =
                rpc::createExceptionResponse(work.message_, ss.str());
            markFutureWithError(exceptionMsg);
          }
        }
      },
      std::move(work)));
}

void ProcessGroupAgent::enqueueRecv(RecvWork work) {
  threadPool_.run(std::bind(
      [&](RecvWork& work) {
        torch::Tensor& payload = work.payload_;
        auto data = wireDeserialize(payload.storage().data(), payload.numel());
        Message message(
            std::move(data.first),
            std::move(data.second),
            work.type_,
            work.id_);
        if (message.isRequest()) {
          auto futureResponse = cb_->operator()(message);
          if (futureResponse->completed()) {
            if (!futureResponse->hasError()) {
              send(work.from_, std::move(*futureResponse).moveValue());
            } else {
              send(
                  work.from_,
                  createExceptionResponse(
                      message, futureResponse->error()->what()));
            }
          } else {
            auto fromId = work.from_.id_;
            auto requestId = work.id_;
            futureResponse->addCallback(
                [this, fromId, requestId, futureResponse](
                    const Message& /* unused */,
                    const c10::optional<utils::FutureError>& err) {
                  if (!err) {
                    send(
                        getWorkerInfo(fromId),
                        std::move(*futureResponse).moveValue());
                  } else {
                    std::string errStr = err->what();
                    std::vector<char> payload(errStr.begin(), errStr.end());
                    Message m(
                        std::move(payload),
                        {},
                        MessageType::EXCEPTION,
                        requestId);
                    send(getWorkerInfo(fromId), std::move(m));
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
              // Received a completion for a timed out future, drop the recv.
              // RecvCounts will not be incremented here, it will be incremented
              // by the sender who has determined the future has timed out.
              return;
            }
            // Use futureInfo before destructing it.
            fm = futureInfo->second.future_;
            auto endTime = futureInfo->second.endTime_;
            futures_.erase(id);
            // look up the corresponding future by its time out and request ID,
            // and remove it from the timeouts map
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
          if (message.type() == MessageType::EXCEPTION) {
            fm->setError(std::string(
                message.payload().begin(), message.payload().end()));
          } else {
            fm->markCompleted(std::move(message));
          }
        } else {
          // TODO: pass the error back to the caller instead of crashing here.
          TORCH_INTERNAL_ASSERT(
              false, "unrecognized message type ", message.type());
        }

        recvCounts_.increment(work.from_.id_);
      },
      std::move(work)));
}

void ProcessGroupAgent::markFutureWithError(Message& message) {
  TORCH_INTERNAL_ASSERT(
      message.type() == MessageType::EXCEPTION,
      "markFutureWithError should be only called with Message that has type Exception.");
  auto id = message.id();
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

  fm->setError(std::string(message.payload().begin(), message.payload().end()));
  futureCV_.notify_all();
}

void ProcessGroupAgent::listenLoop() {
  while (rpcRunning_.load()) {
    // rank, tensor size, message type
    std::vector<torch::Tensor> preamble = {torch::empty({4}, {torch::kInt64})};
    auto work = pg_->recvAnysource(preamble, pg_->getRank());
    {
      std::lock_guard<std::mutex> guard(recvWorkMutex_);
      recvWork_ = work;
    }

    if (!rpcRunning_.load() || !work->wait() /* aborted */) {
      return;
    }

    int64_t* preamble_items = preamble.front().storage().data<int64_t>();

    auto srcRank = preamble_items[0];
    auto size = preamble_items[1];
    MessageType type = MessageType(preamble_items[2]);
    int64_t id = preamble_items[3];

    std::vector<torch::Tensor> tensors = {torch::empty({size}, {torch::kChar})};
    pg_->recv(tensors, srcRank, pg_->getRank())->wait();

    enqueueRecv(
        RecvWork(allWorkerInfo_[srcRank], type, id, std::move(tensors[0])));
  }
}

void ProcessGroupAgent::pollTimedOutRPCs() {
  while (rpcRunning_.load()) {
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
      // Notice, whoever modifying `rpcRunning_`
      // must acquire lock on `futureMutex_`.
      // Otherwise, this predicate could deadlock.
      // If during evaluating the predicate, `::shutdown()` is called, then
      // the predicate missed the notification before it started waiting
      // on the cond var.
      if (!rpcRunning_.load()) {
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
      std::ostringstream ss;
      ss << "RPC ran for more than " << timedOutFuture.timeout_.count()
         << " milliseconds and timed out.";
      const auto exceptionMsg = createExceptionResponse(
          Message({}, {}, MessageType::EXCEPTION), ss.str());
      timedOutFuture.future_->setError(std::string(
          exceptionMsg.payload().begin(), exceptionMsg.payload().end()));

      const int dst = timedOutFuture.dstRank_;
      recvCounts_.increment(dst);
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
