#include <torch/csrc/distributed/rpc/ProcessGroupAgent.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

// Write the message into the given ostream
void serialize(Message message, std::ostream& os) {
  auto data = static_cast<void*>(message.meta().data());
  auto size = message.meta().size();

  // getting tensor table from the message
  auto& tensors = message.tensors();
  // append meta as a tensor
  tensors.push_back(torch::from_blob(data, size, {torch::kChar}));
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
    c10d::ProcessGroup& pg)
    : RpcAgent(std::move(workerName), processRequestBlocking),
      nameMap_(std::move(nameMap)),
      stop_(false),
      pg_(pg),
      nextId_(0) {
  if (nameMap_.find(workerName_) == nameMap_.end()
      || pg_.getRank() != nameMap_[workerName_]) {
    throw std::runtime_error("resolved worker name does not match rank");
  }
  for (auto entry : nameMap_) {
    reversedNameMap_[entry.second] = entry.first;
  }
  sendThread_ = std::thread(&ProcessGroupAgent::sendLoop, this);
  listenerThread_ = std::thread(&ProcessGroupAgent::listen, this);
}

ProcessGroupAgent::~ProcessGroupAgent() noexcept(false) {
  if (!stop_) {
    throw std::runtime_error("ProcessGroupAgent cannot be destroyed before"
      "calling shutdown");
  }
}

void ProcessGroupAgent::shutdown() {
  // cannot put this into the destructor, as it is not safe to call virtual
  // functions in constructor and destructor. We can drop this when we can
  // gracefully abort a recvAnysource.
  int dst = (pg_.getRank() + 1) % pg_.getSize();
  std::unique_ptr<std::stringstream> stream(new std::stringstream);
  *stream << 0;
  enqueue(SendWork(dst, std::move(Message({}, {}, MessageType::SHUTDOWN))));
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
  if (nameMap_.find(to) == nameMap_.end()) {
    throw std::runtime_error("unrecoganized destination in _send");
  }

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

  SendWork work(nameMap_[to], std::move(message));
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
    serialize(std::move(work.message_), ss);
    std::string str = ss.str();

    std::vector<torch::Tensor> header = {
      torch::tensor(
        {
          (int64_t)pg_.getRank(),
          (int64_t)str.length(),
        }, {torch::kLong})
    };
    pg_.send(header, work.dstRank_, work.dstRank_ /* channelTag */)->wait();
    std::vector<torch::Tensor> payload =
        {torch::from_blob((void *)str.c_str(), str.length(), {torch::kChar})};
    pg_.send(payload, work.dstRank_, work.dstRank_ /* channelTag */)->wait();

    lock.lock();
  }
}

void ProcessGroupAgent::listen() {
  while (!stop_) {
    // rank, tensor size
    std::vector<torch::Tensor> header = {torch::empty({2}, {torch::kInt64})};
    pg_.recvAnysource(header, pg_.getRank())->wait();
    int64_t* header_items = header.front().storage().data<int64_t>();

    auto srcRank = header_items[0];
    auto size = header_items[1];

    std::vector<torch::Tensor> tensors = {torch::empty({size}, {torch::kChar})};
    pg_.recv(tensors, srcRank, pg_.getRank())->wait();
    std::stringstream ss(std::string(
      (char*)tensors[0].storage().data<signed char>(), tensors[0].numel()));

    Message message = deserialize(ss);

    if (message.isOp()) {
      cb_(reversedNameMap_[srcRank], message, *this);
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
      throw std::runtime_error("unrecognized message type.");
    }
  }
}

}
}
}
