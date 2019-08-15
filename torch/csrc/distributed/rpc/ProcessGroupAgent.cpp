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
  // append id as a tensor
  tensors.push_back(torch::tensor({message.id()}, {torch::kInt64}
  ));

  torch::save(tensors, os);
}

Message deserialize(MessageType type, std::istream& is) {
  std::vector<torch::Tensor> tensors;

  torch::load(tensors, is);

  TORCH_CHECK(tensors.size() >= 2, "Failed to deserialize a message.");
  auto idTensor = std::move(tensors.back());
  tensors.pop_back();
  auto payloadTensor = std::move(tensors.back());
  tensors.pop_back();

  int64_t id = idTensor.storage().data<int64_t>()[0];

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
      sendMutexes_(pg_->getSize()),
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
  sync();
  int dst = (pg_->getRank() + 1) % pg_->getSize();
  enqueueSend(SendWork(dst, Message({}, {}, MessageType::SHUTDOWN)));
  threadPool_.waitWorkComplete();
  listenerThread_.join();
}

void ProcessGroupAgent::sync() {
  // Block until all processes wants to sync. This is necessary before acquiring
  // the lock below, because other processes might not enter sync() until it
  // gets some response from this RpcAgent.
  pg_->barrier()->wait();
  // Wait until the all send works are done.
  // NB: There might be additional send works inserted while waiting.
  threadPool_.waitWorkComplete();
  // Use another barrier in case different RpcAgent handles different amounts of
  // workloads.
  pg_->barrier()->wait();
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

  enqueueSend(SendWork(dstRank, std::move(message)));
  return future;
}

void ProcessGroupAgent::enqueueSend(SendWork work) {
  // NB: this can be changed to use a native move capture when moved to C++14
  threadPool_.run(std::bind(
    [&](const SendWork& work) {
      std::stringstream ss;
      serialize(work.message_, ss);
      std::string str = ss.str();

      std::vector<torch::Tensor> preamble = {
        torch::tensor(
          {
            (int64_t)pg_->getRank(),
            (int64_t)str.length(),
            (int64_t)work.message_.type()
          }, {torch::kLong})
      };

      // ProcessGroup is not thread-safe when sending with the same tag, hence
      // the lock
      std::lock_guard<std::mutex> lock(sendMutexes_[work.to_]);
      pg_->send(preamble, work.to_, work.to_ /* channelTag */)->wait();
      if (work.message_.type() != MessageType::SHUTDOWN) {
        std::vector<torch::Tensor> payload ={
            torch::from_blob((void *)str.c_str(), str.length(), {torch::kChar})
        };
        pg_->send(payload, work.to_, work.to_ /* channelTag */)->wait();
      }
    },
    std::move(work)
  ));
}

void ProcessGroupAgent::enqueueRecv(RecvWork work) {
  threadPool_.run(std::bind(
    [&](RecvWork& work) {

      torch::Tensor& payload = work.payload_;
      std::stringstream ss(std::string(
        (char*)payload.storage().data<signed char>(), payload.numel()));

      Message message = deserialize(work.type_, ss);

      if (message.isRequest()) {
        cb_(names_[work.from_], std::move(message), *this);
      } else if (message.isResponse()) {
        auto id = message.id();
        {
          std::lock_guard<std::mutex> lock{futureMutex_};
          futures_[id]->markCompleted(std::move(message));
          futures_.erase(id);
        }
      } else {
        // TODO: pass the error back to the caller instead of crashing here.
        AT_ERROR("unrecognized message type ", message.type());
      }
    },
    std::move(work)
  ));
}

void ProcessGroupAgent::listenLoop() {
  while (true) {
    // rank, tensor size, message type
    std::vector<torch::Tensor> preamble = {torch::empty({3}, {torch::kInt64})};
    pg_->recvAnysource(preamble, pg_->getRank())->wait();
    int64_t* preamble_items = preamble.front().storage().data<int64_t>();

    auto srcRank = preamble_items[0];
    auto size = preamble_items[1];
    MessageType type = MessageType(preamble_items[2]);

    if (type == MessageType::SHUTDOWN) {
      return;
    }

    std::vector<torch::Tensor> tensors = {torch::empty({size}, {torch::kChar})};
    pg_->recv(tensors, srcRank, pg_->getRank())->wait();

    enqueueRecv(RecvWork(srcRank, type, std::move(tensors[0])));
  }
}

}
}
}
