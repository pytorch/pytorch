#include <torch/csrc/distributed/rpc/ProcessGroupAgent.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

// Write the message into the given ostream
void serialize(Message& message, std::ostream& os) {
  // We cast const void* to void* here because we need to create a tensor using
  // that memory space. If is fine as that tensor stays function-local, and will
  // not be modified during its lifetime.
  auto meta =
      const_cast<void*>(static_cast<const void*>(message.meta().data()));
  auto meta_size = message.meta().size();

  // getting tensor table from the message
  std::vector<torch::Tensor> tensors = message.tensors();
  // append meta as a tensor
  tensors.push_back(torch::from_blob(meta, meta_size, {torch::kChar}));
  // append id and type as a tensor
  tensors.push_back(torch::tensor(
      {message.id(), (int64_t) message.type()}, {torch::kInt64}
  ));

  torch::save(tensors, os);
}

Message deserialize(std::istream& is) {
  std::vector<torch::Tensor> tensors;

  torch::load(tensors, is);

  auto miscTensor = std::move(tensors.back());
  tensors.pop_back();
  auto metaTensor = std::move(tensors.back());
  tensors.pop_back();

  int64_t* miscItems = miscTensor.storage().data<int64_t>();
  int64_t id = miscItems[0];
  MessageType type = MessageType(miscItems[1]);

  std::vector<char> meta(metaTensor.numel());
  std::memcpy(
      meta.data(), metaTensor.storage().data(), metaTensor.numel());

  return Message(std::move(meta), std::move(tensors), type, id);
}

} // namespace

ProcessGroupAgent::ProcessGroupAgent(
    std::string workerName,
    std::unordered_map<std::string, int> nameMap,
    std::shared_ptr<c10d::ProcessGroup> pg)
    : RpcAgent(std::move(workerName), processRequestBlocking),
      nameMap_(std::move(nameMap)),
      stop_(false),
      pg_(std::move(pg)),
      nextId_(0) {

  auto workerRankIter = nameMap_.find(workerName_);
  TORCH_CHECK(workerRankIter != nameMap_.end(),
      "Failed to resolve worker name ", workerName_, " to a ProcessGroup rank.");
  TORCH_CHECK(pg_->getRank() == workerRankIter -> second,
      "Resolved worker rank ", workerRankIter -> second,
      " does not match ProcessGroup rank ", pg_->getRank());

  names_.resize(nameMap_.size());
  for (auto& entry : nameMap_) {
    names_[entry.second] = entry.first;
  }
  sendThread_ = std::thread(&ProcessGroupAgent::sendLoop, this);
  listenerThread_ = std::thread(&ProcessGroupAgent::listenLoop, this);
}

ProcessGroupAgent::~ProcessGroupAgent() noexcept(false) {
  TORCH_CHECK(stop_, "ProcessGroupAgent cannot be destroyed before shutdown.");
}

void ProcessGroupAgent::shutdown() {
  // cannot put this into the destructor, as it is not safe to call virtual
  // functions in constructor and destructor.

  // Every process i sends a SHUTDOWN message to process i + 1. This is
  // necessary for now because:
  // 1. There is no abort API for ProcessGrouprecv::Anysource yet. We have to
  //    feed it a message or kill the thread.
  // 2. A GLOO process cannot send message to itself. (there is an ongoing
  //    effort to fix this problem).
  int dst = (pg_->getRank() + 1) % pg_->getSize();
  enqueue(SendWork(dst, Message({}, {}, MessageType::SHUTDOWN)));
  std::unique_lock<std::mutex> lock(sendQueueMutex_);
  workConsumeCV_.wait(lock, [&] { return sendQueue_.empty(); });
  stop_ = true;
  lock.unlock();

  workProduceCV_.notify_all();
  sendThread_.join();
  listenerThread_.join();
}

std::shared_ptr<FutureMessage> ProcessGroupAgent::send(
    std::string to, Message message) {

  auto dstRankIter = nameMap_.find(to);
  TORCH_CHECK(dstRankIter != nameMap_.end(), "Unknown destination worker ", to);

  const int dstRank = dstRankIter -> second;
  TORCH_CHECK(dstRank != pg_->getRank(), "ProcessGroupAgent does not support "
    "making RPC calls to self.")

  auto requestId = nextId();
  auto future = std::make_shared<FutureMessage>();
  if (message.isOp()) {
    {
      std::lock_guard<std::mutex> lock{futureMutex_};
      futures_[requestId] = future;
    }
    message.setId(requestId);
  } else {
    future->markCompleted();
  }

  SendWork work(dstRank, std::move(message));
  enqueue(std::move(work));
  return future;
}

void ProcessGroupAgent::enqueue(SendWork work) {
//void ProcessGroupAgent::enqueue(const int dst, torch::Tensor tensor) {
  std::unique_lock<std::mutex> lock(sendQueueMutex_);
  sendQueue_.emplace_back(std::move(work));
  lock.unlock();

  workProduceCV_.notify_one();
}

// making sure tensors are not deleted before send finishes
void ProcessGroupAgent::sendLoop() {
  std::unique_lock<std::mutex> lock(sendQueueMutex_);

  while (!stop_) {
    if (sendQueue_.empty()) {
      workProduceCV_.wait(lock);
      continue;
    }

    auto work = std::move(sendQueue_.front());
    sendQueue_.pop_front();
    lock.unlock();

    workConsumeCV_.notify_one();


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
    pg_->send(preamble, work.dstRank_, work.dstRank_ /* channelTag */)->wait();
    std::vector<torch::Tensor> payload =
        {torch::from_blob((void *)str.c_str(), str.length(), {torch::kChar})};
    pg_->send(payload, work.dstRank_, work.dstRank_ /* channelTag */)->wait();

    lock.lock();
  }
}

void ProcessGroupAgent::listenLoop() {
  while (!stop_) {
    // rank, tensor size
    std::vector<torch::Tensor> preamble = {torch::empty({2}, {torch::kInt64})};
    pg_->recvAnysource(preamble, pg_->getRank())->wait();
    int64_t* header_items = preamble.front().storage().data<int64_t>();

    auto srcRank = header_items[0];
    auto size = header_items[1];

    std::vector<torch::Tensor> tensors = {torch::empty({size}, {torch::kChar})};
    pg_->recv(tensors, srcRank, pg_->getRank())->wait();

    std::stringstream ss(std::string(
      (char*)tensors[0].storage().data<signed char>(), tensors[0].numel()));

    Message message = deserialize(ss);

    if (message.isOp()) {
      cb_(names_[srcRank], message, *this);
    } else if (message.isRet()) {
      auto id = message.id();
      {
        std::lock_guard<std::mutex> lock{futureMutex_};
        futures_[id]->markCompleted(std::move(message));
        futures_.erase(id);
      }
    } else if (message.isShutdown()) {
      break;
    } else {
      AT_ERROR("unrecognized message type ", message.type());
    }
  }
}

}
}
}
