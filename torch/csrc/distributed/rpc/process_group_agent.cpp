#include <torch/csrc/distributed/rpc/process_group_agent.h>

#include <c10/util/C++17.h>
#include <c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/rpc/request_callback_impl.h>

#include <Python.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

// Write the message payload/tensors into the given std::string.
// We send the id/type separately to avoid creating an extra tensor to
// serialize.
std::string serialize(const Message& message) {
  // We cast const void* to void* here because we need to create a tensor using
  // that memory space. If is fine as that tensor stays function-local, and will
  // not be modified during its lifetime.
  auto payload = const_cast<void*>( // NOLINT
      static_cast<const void*>(message.payload().data()));
  auto payload_size = message.payload().size();

  // getting tensor table from the message
  std::vector<torch::Tensor> tensors = message.tensors();
  // append payload as a tensor
  tensors.push_back(torch::from_blob(payload, payload_size, {torch::kChar}));

  // optional: estimate output size, to avoid some unnecessary resizing.
  static constexpr size_t kBaseOverhead = 2048;
  static constexpr size_t kPerTensor = 128;
  size_t estimate = kBaseOverhead;
  for (const auto& t : tensors) {
    estimate += t.nbytes() + kPerTensor;
  }

  std::string out;
  out.reserve(estimate);
  torch::save(tensors, [&](const void* buf, size_t n) -> size_t {
    out.append(static_cast<const char*>(buf), n);
    return n;
  });
  return out;
}

enum {
  // Using serialize() above.
  SERIALIZATION_NORMAL = 0,
  // For payload-only case, we can avoid torch::save()/load() copying overhead.
  SERIALIZATION_PAYLOAD_ONLY = 1,
};

Message deserialize(
    MessageType type,
    int64_t id,
    int serialization,
    const void* buf,
    size_t size) {
  if (serialization == SERIALIZATION_NORMAL) {
    std::vector<torch::Tensor> tensors;
    torch::load(tensors, static_cast<const char*>(buf), size);

    TORCH_CHECK(!tensors.empty(), "Failed to deserialize a message.");
    auto payloadTensor = std::move(tensors.back());
    tensors.pop_back();

    const char* data = static_cast<const char*>(payloadTensor.storage().data());
    std::vector<char> payload(data, data + payloadTensor.numel());

    return Message(std::move(payload), std::move(tensors), type, id);
  } else if (serialization == SERIALIZATION_PAYLOAD_ONLY) {
    const char* data = static_cast<const char*>(buf);
    std::vector<char> payload(data, data + size);
    return Message(std::move(payload), {}, type, id);
  } else {
    TORCH_INTERNAL_ASSERT(false, "unrecognized serialization ", serialization);
  }
}

} // namespace

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

////////////////////////  ProcessGroupAgent  /////////////////////////////////

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
          c10::guts::make_unique<RequestCallbackImpl>(),
          rpcTimeout),
      pg_(std::move(pg)),
      sendCounts_(pg_->getSize()),
      recvCounts_(pg_->getSize()),
      nextId_(0),
      sendMutexes_(pg_->getSize()),
      threadPool_(numSendRecvThreads) {
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

void ProcessGroupAgent::join() {
  // Every process i sends a SHUTDOWN message to process i + 1. This is
  // necessary for now because:
  // 1. There is no abort API for ProcessGroup::recvAnysource yet. We have to
  //    feed it a message or kill the thread.
  // 2. A GLOO process cannot send message to itself. (there is an ongoing
  //    effort to fix this problem).
  sync();
  std::unique_lock<std::mutex> lock(futureMutex_);
  futureCV_.wait(
      lock, [this] { return futures_.empty() && futureTimeouts_.empty(); });
  lock.unlock();
  pg_->barrier()->wait();
  int dst = (pg_->getRank() + 1) % pg_->getSize();
  enqueueSend(
      SendWork(allWorkerInfo_[dst], Message({}, {}, MessageType::SHUTDOWN)));
  threadPool_.waitWorkComplete();
  listenerThread_.join();
  PythonRpcHandler::getInstance().cleanup();
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
  listenerThread_ = std::thread(&ProcessGroupAgent::listenLoop, this);
}

std::shared_ptr<FutureMessage> ProcessGroupAgent::send(
    const WorkerInfo& to,
    Message&& message) {
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
    auto futureStartTime =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch());
    {
      std::lock_guard<std::mutex> lock{futureMutex_};
      futures_[requestId] = std::make_pair(future, futureStartTime);
      // insert future into timeouts map to keep track of its timeout
      futureTimeouts_[futureStartTime].push_back(requestId);
    }
    message.setId(requestId);
  } else {
    future->markCompleted();
  }

  // Sending to ourselves: bypass the send logic and enqueue directly
  // to our receving queue.
  if (to.id_ == (worker_id_t)pg_->getRank()) {
    TORCH_CHECK(!message.isShutdown(), "Shutting down self not supported");
    threadPool_.run(std::bind(
        [this](const Message& message) {
          sendCounts_.increment(pg_->getRank());
          // Unlike the other cases, need to add a tensor deleter, since the
          // data outlives the scope of this function. It's shared_ptr<> due
          // to c++11 lambda capture limitations with unique_ptr<>.
          auto payload =
              c10::guts::make_unique<std::string>(serialize(message));
          const char* data = payload->data();
          size_t len = payload->length();
          std::string* delete_when_done = payload.release();
          enqueueRecv(RecvWork(
              getWorkerInfo(pg_->getRank()),
              message.type(),
              message.id(),
              SERIALIZATION_NORMAL,
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

void ProcessGroupAgent::enqueueSend(SendWork work) {
  // NB: this can be changed to use a native move capture when moved to C++14
  threadPool_.run(std::bind(
      [this](const SendWork& work) {
        std::string serializedPayload; // keep in scope.

        int serialization;
        std::pair<const char*, size_t> data;
        if (work.message_.tensors().empty()) {
          data = {work.message_.payload().data(),
                  work.message_.payload().size()};
          serialization = SERIALIZATION_PAYLOAD_ONLY;
        } else {
          serializedPayload = serialize(work.message_);
          data = {serializedPayload.data(), serializedPayload.size()};
          serialization = SERIALIZATION_NORMAL;
        }

        static constexpr int kShift32 = 32;
        std::vector<torch::Tensor> preamble = {torch::tensor(
            {static_cast<int64_t>(pg_->getRank()),
             static_cast<int64_t>(data.second),
             static_cast<int64_t>(work.message_.type()) |
                 (static_cast<int64_t>(serialization) << kShift32),
             work.message_.id()},
            {torch::kInt64})};

        // ProcessGroup is not thread-safe when sending with the same tag, hence
        // the lock
        std::vector<std::shared_ptr<c10d::ProcessGroup::Work>> pendingSends;
        const auto dst = work.to_.id_;
        if (work.message_.isShutdown()) {
          pendingSends.reserve(1);
          {
            std::lock_guard<std::mutex> guard(sendMutexes_[dst]);
            pendingSends.emplace_back(
                pg_->send(preamble, dst, dst /* channelTag */));
          }
        } else {
          std::vector<torch::Tensor> payload = {
              torch::from_blob((void*)data.first, data.second, {torch::kChar})};
          pendingSends.reserve(2);

          sendCounts_.increment(dst);

          {
            std::lock_guard<std::mutex> guard(sendMutexes_[dst]);
            pendingSends.emplace_back(
                pg_->send(preamble, dst, dst /* channelTag */));
            pendingSends.emplace_back(
                pg_->send(payload, dst, dst /* channelTag */));
          }
        }
        for (auto& pendingSend : pendingSends) {
          pendingSend->wait();
        }
      },
      std::move(work)));
}

void ProcessGroupAgent::enqueueRecv(RecvWork work) {
  threadPool_.run(std::bind(
      [&](RecvWork& work) {
        torch::Tensor& payload = work.payload_;
        Message message = deserialize(
            work.type_,
            work.id_,
            work.serialization_,
            payload.storage().data(),
            payload.numel());
        if (message.isRequest()) {
          send(work.from_, cb_->operator()(message));
        } else if (message.isResponse()) {
          auto id = message.id();
          std::shared_ptr<FutureMessage> fm = nullptr;
          std::chrono::milliseconds futureStartTime;
          {
            std::lock_guard<std::mutex> lock{futureMutex_};
            std::tie(fm, futureStartTime) = futures_[id];
          }
          // Not holding lock on markCompleted as this could run callbacks that
          // call agent_->send
          fm->markCompleted(std::move(message));
          {
            std::lock_guard<std::mutex> lock{futureMutex_};
            futures_.erase(id);
            // look up the corresponding future by its time out and request ID,
            // and remove it from the timeouts map
            auto& futuresAtTime = futureTimeouts_[futureStartTime];
            futuresAtTime.erase(
                std::find(futuresAtTime.begin(), futuresAtTime.end(), id));
            if (futuresAtTime.size() == 0) {
              // remove the key from futureTimeouts_
              futureTimeouts_.erase(futureStartTime);
            }
          }
          futureCV_.notify_all();
        } else {
          // TODO: pass the error back to the caller instead of crashing here.
          TORCH_INTERNAL_ASSERT(
              false, "unrecognized message type ", message.type());
        }

        recvCounts_.increment(work.from_.id_);
      },
      std::move(work)));
}

void ProcessGroupAgent::listenLoop() {
  while (true) {
    // rank, tensor size, message type
    std::vector<torch::Tensor> preamble = {torch::empty({4}, {torch::kInt64})};
    pg_->recvAnysource(preamble, pg_->getRank())->wait();
    int64_t* preamble_items = preamble.front().storage().data<int64_t>();

    auto srcRank = preamble_items[0];
    auto size = preamble_items[1];
    static constexpr int64_t kLower32 = 0xffffffff;
    static constexpr int kShift32 = 32;
    MessageType type = MessageType(preamble_items[2] & kLower32);
    int serialization = preamble_items[2] >> kShift32;
    int64_t id = preamble_items[3];

    if (type == MessageType::SHUTDOWN) {
      // FIXME: This LOG also prints warnings no InitGoogleLogging() was invoked
      // before logging, but it is not appropriate to call InitGoogleLogging()
      // here either.
      LOG(INFO) << "Shutting down ProcessGroupAgent " << workerInfo_.name_
                << std::endl;
      return;
    }

    std::vector<torch::Tensor> tensors = {torch::empty({size}, {torch::kChar})};
    pg_->recv(tensors, srcRank, pg_->getRank())->wait();

    enqueueRecv(RecvWork(
        allWorkerInfo_[srcRank],
        type,
        id,
        serialization,
        std::move(tensors[0])));
  }
}

} // namespace rpc
} // namespace distributed
} // namespace torch
