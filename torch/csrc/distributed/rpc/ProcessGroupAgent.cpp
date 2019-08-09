#include <torch/csrc/distributed/rpc/ProcessGroupAgent.h>
#include <Python.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

// Write the message into the given ostream
void serialize(const Message& message, std::ostream& os) {
  // We cast const void* to void* here because we need to create a tensor using
  // that memory space. If is fine as that tensor stays function-local, and will
  // not be modified during its lifetime.
  auto payload = const_cast<void*>(  // NOLINT
      static_cast<const void*>(message.payload().data()));
  auto payload_size = message.payload().size();

  // getting tensor table from the message
  std::vector<torch::Tensor> tensors = message.tensors();
  // append payload as a tensor
  tensors.push_back(torch::from_blob(payload, payload_size, {torch::kChar}));
  // append id and type as a tensor
  tensors.push_back(torch::tensor(
      {message.id(), (int64_t) message.type()}, {torch::kInt64}
  ));

  torch::save(tensors, os);
}

Message deserialize(std::istream& is) {
  std::vector<torch::Tensor> tensors;

  torch::load(tensors, is);

  TORCH_CHECK(tensors.size() >= 2, "Failed to deserialize a message.");
  auto miscTensor = std::move(tensors.back());
  tensors.pop_back();
  auto payloadTensor = std::move(tensors.back());
  tensors.pop_back();

  int64_t* miscItems = miscTensor.storage().data<int64_t>();
  int64_t id = miscItems[0];
  MessageType type = MessageType(miscItems[1]);

  std::vector<char> payload(payloadTensor.numel());

  if (payloadTensor.numel() > 0) {
    std::memcpy(payload.data(),
                payloadTensor.storage().data(),
                payloadTensor.numel());
  }

  return Message(std::move(payload), std::move(tensors), type, id);
}

} // namespace

ProcessGroupAgent::ProcessGroupAgent(
    std::string workerName,
    std::unordered_map<std::string, int> nameMap,
    std::shared_ptr<c10d::ProcessGroup> pg,
    int numSendRecvThreads)
    : RpcAgent(std::move(workerName), processRequestBlocking),
      nameMap_(std::move(nameMap)),
      stop_(false),
      pg_(std::move(pg)),
      nextId_(0),
      sendMutexes_(new std::mutex[pg_->getSize()]),
      threadPool_(numSendRecvThreads) {
  TORCH_CHECK(nameMap_.size() > 1, "ProcessGroupAgent requires world_size to "
      "be at least 2, but got ", nameMap_.size());
  auto workerRankIter = nameMap_.find(workerName_);
  TORCH_CHECK(workerRankIter != nameMap_.end(), "Failed to resolve worker "
      "name ", workerName_, " to a ProcessGroup rank.");
  TORCH_CHECK(pg_->getRank() == workerRankIter -> second,
      "Resolved worker rank ", workerRankIter -> second,
      " does not match ProcessGroup rank ", pg_->getRank());

  names_.resize(nameMap_.size());
  for (auto& entry : nameMap_) {
    names_[entry.second] = entry.first;
  }
  PythonRpcHandler::init();
  listenerThread_ = std::thread(&ProcessGroupAgent::listenLoop, this);
}

void ProcessGroupAgent::join() {
  // Every process i sends a SHUTDOWN message to process i + 1. This is
  // necessary for now because:
  // 1. There is no abort API for ProcessGroup::recvAnysource yet. We have to
  //    feed it a message or kill the thread.
  // 2. A GLOO process cannot send message to itself. (there is an ongoing
  //    effort to fix this problem).
  waitAll();
  int dst = (pg_->getRank() + 1) % pg_->getSize();
  enqueueSend(RpcWork(dst, Message({}, {}, MessageType::SHUTDOWN)));
  threadPool_.waitWorkComplete();
  listenerThread_.join();

  PythonRpcHandler::cleanUp();
}

void ProcessGroupAgent::waitAll() {
  // Intuitively, waitAll is like all workers in the same communication world
  // reaching a consensus that they all want to stop talking to others,
  // so that, on exiting waitAll, the communication world will be
  // like brand new without any pending Rpc work.

  // A message can be in one of these states,
  //   1. A worker calls send, putting a req message queue A.
  //   2. The req message is popped and sent to wire.
  //   3. The req message is received from wire and put into queue B.
  //   4. The req message is popped and processed, a rep message is put
  //      into queue B.
  //   6. The rep message is popped and sent to wire.
  //   7. The rep message is received from wire and put into queue A.
  //   8. The rep is popped, marking the request future as complete.

  // The assumptions of this solution are
  //   1. Only one thread calls send, waitAll, join.
  //   2. The state of a message only transfers in one-way.
  //   3. Every worker has only limited work, so there is always a time that
  //      on worker add a message in state 1.

  // waitAll achieves this goal by waiting for the state space reducing,
  // {12345678}, to {2345678}, to {345678}, to {45678}, ..., and eventually
  // to an empty state space, {}.

  // Assuming the state of a message can keep tranfering in best effort, this
  // is to wait for the state space to collapse to an empty state space, {}.
  // Note that the lifetime of a request Future span from state 1 to state 8,
  // so exsitence of any Future in futures_ map is equivalent to exsitence
  // of any message in the communication world, except for the TERMINATION
  // message, which is not inserted into the futures_ map and does not
  // ask for ACK or response.
  waitSelf();

  // This is to ensure no worker will add send work to their own thread pool
  // queue before the shared goal is achived, elliminating the possibility of
  // adding a message in state 1. Otherwise, a quick peer could exit waitAll
  // and send again, disrruping the assumption when others are calling wait.
  pg_->barrier()->wait();
}

void ProcessGroupAgent::waitSelf() {
    std::unique_lock<std::mutex> lock{futureMutex_};
    futureDecreaseCV_.wait(lock, [&](){return futures_.empty();});
}

std::shared_ptr<FutureMessage> ProcessGroupAgent::send(
    const std::string& to, Message&& message) {

  auto dstRankIter = nameMap_.find(to);
  TORCH_CHECK(dstRankIter != nameMap_.end(), "Unknown destination worker ", to);

  const int dstRank = dstRankIter -> second;
  TORCH_CHECK(dstRank != pg_->getRank(), "ProcessGroupAgent does not support "
    "making RPC calls to self.")

  auto requestId = nextId();
  auto future = std::make_shared<FutureMessage>();
  if (message.isRequest()) {
    {
      std::lock_guard<std::mutex> lock{futureMutex_};
      futures_[requestId] = future;
    }
    message.setId(requestId);
  } else {
    future->markCompleted();
  }

  enqueueSend(RpcWork(dstRank, std::move(message)));
  return future;
}

void ProcessGroupAgent::enqueueSend(RpcWork work) {
  threadPool_.run(std::bind(
    [&](const RpcWork& work) {
      std::stringstream ss;
      serialize(work.message_, ss);
      std::string str = ss.str();

      std::vector<torch::Tensor> preamble = {
        torch::tensor(
          {
            (int64_t)pg_->getRank(),
            (int64_t)str.length(),
          }, {torch::kLong})
      };

      // ProcessGroup is not thread-safe when sending with the same tag, hence
      // the lock
      std::unique_lock<std::mutex> lock(sendMutexes_.get()[work.rank_]);
      pg_->send(preamble, work.rank_, work.rank_ /* channelTag */)->wait();
      std::vector<torch::Tensor> payload =
          {torch::from_blob((void *)str.c_str(), str.length(), {torch::kChar})};
      pg_->send(payload, work.rank_, work.rank_ /* channelTag */)->wait();
      lock.unlock();
    },
    std::move(work)
  ));
}

void ProcessGroupAgent::enqueueRecv(RpcWork work) {
  threadPool_.run(std::bind(
    [&](RpcWork& work) {
      if (work.message_.isRequest()) {
        cb_(names_[work.rank_], std::move(work.message_), *this);
      } else if (work.message_.isResponse()) {
        auto id = work.message_.id();
        {
          std::lock_guard<std::mutex> lock{futureMutex_};
          futures_[id]->markCompleted(std::move(work.message_));
          futures_.erase(id);
        }
        futureDecreaseCV_.notify_one();
      } else {
        AT_ERROR("unrecognized message type ", work.message_.type());
      }
    },
    std::move(work)
  ));
}

void ProcessGroupAgent::listenLoop() {
  while (true) {
    // rank, tensor size
    std::vector<torch::Tensor> preamble = {torch::empty({2}, {torch::kInt64})};
    pg_->recvAnysource(preamble, pg_->getRank())->wait();
    int64_t* preamble_items = preamble.front().storage().data<int64_t>();

    auto srcRank = preamble_items[0];
    auto size = preamble_items[1];

    std::vector<torch::Tensor> tensors = {torch::empty({size}, {torch::kChar})};
    pg_->recv(tensors, srcRank, pg_->getRank())->wait();

    std::stringstream ss(std::string(
      (char*)tensors[0].storage().data<signed char>(), tensors[0].numel()));

    Message message = deserialize(ss);

    if (message.isShutdown()) {
      return;
    }

    enqueueRecv(RpcWork(srcRank, std::move(message)));
  }
}

}
}
}
