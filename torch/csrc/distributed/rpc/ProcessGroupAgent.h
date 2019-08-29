#pragma once

#include <c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/rpc/FutureMessage.h>
#include <torch/csrc/distributed/rpc/RpcAgent.h>
#include <torch/csrc/distributed/rpc/functions.h>

#include <deque>
#include <thread>

namespace torch {
namespace distributed {
namespace rpc {

struct SendWork {
  SendWork(const int dstRank,
           Message&& message)
      : dstRank_(dstRank),
        message_(message) {}

  const int dstRank_;
  Message message_;

};

class ProcessGroupAgent : public RpcAgent {
 public:

  ProcessGroupAgent(std::string workerName,
                    std::unordered_map<std::string, int> nameMap,
                    std::shared_ptr<c10d::ProcessGroup> pg);

  // This method wraps the destination information and the message into a
  // SendWork object, and put the SendWork into a queue. Another thread will
  // consume SendWork from the queue and send it out.
  std::shared_ptr<FutureMessage> send(
      const std::string& to, Message&& message) override;

  void join() override;

  void sync() override;

 private:
  // put SendWork into a queue and notify the sendLoop thread
  void enqueue(SendWork work);
  // sending out the message
  void sendLoop();
  // receiving messages
  void listenLoop();

  int64_t nextId() {
    return nextId_++;
  }

  // worker name -> rank
  std::unordered_map<std::string, int> nameMap_;
  bool stop_;
  std::shared_ptr<c10d::ProcessGroup> pg_;
  std::atomic<int64_t> nextId_;
  // names_[rank] stores the name of the corresponding worker, use this vector
  // to get worker name from rank and pass it to the RequestCallback.
  std::vector<std::string> names_;
  std::deque<SendWork> sendQueue_;
  std::mutex sendQueueMutex_;
  std::condition_variable workProduceCV_;
  std::condition_variable workConsumeCV_;
  std::thread sendThread_;
  std::thread listenerThread_;
  std::unordered_map<int64_t, std::shared_ptr<FutureMessage>> futures_;
  std::mutex futureMutex_;
};

}
}
}
