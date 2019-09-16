#pragma once

#include <c10/core/thread_pool.h>
#include <c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/rpc/future_message.h>
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>

#include <thread>

namespace torch {
namespace distributed {
namespace rpc {

// SendWork and RecvWork will be put into a task queue, and later picked up by
// worker threads from the same ThreadPool.
struct SendWork {
  SendWork(const WorkerId& to, Message&& message)
      : to_(to), message_(message) {}

  const WorkerId& to_;
  Message message_;
};

// SendWork wraps a Message and RecvWork wraps a Tensor. The difference here is
// to allow us to run serialization/deserialization in the worker threads.
struct RecvWork {
  RecvWork(const WorkerId& from, MessageType type, torch::Tensor&& payload)
      : from_(from), type_(type), payload_(payload) {}

  const WorkerId& from_;
  const MessageType type_;
  torch::Tensor payload_;
};

class ProcessGroupAgent : public RpcAgent {
 public:
  ProcessGroupAgent(
      std::string workerName,
      std::shared_ptr<c10d::ProcessGroup> pg,
      int numSendRecvThreads = 4);

  const WorkerId& getWorkerId(const std::string& workerName) const override;

  const WorkerId& getWorkerId(worker_id_t id) const override;

  void join() override;

  void sync() override;

 protected:
  // This method wraps the destination information and the message into a
  // SendWork object, and put the SendWork into a queue. Another thread will
  // consume SendWork from the queue and send it out.
  std::shared_ptr<FutureMessage> send(const WorkerId& to, Message&& message)
      override;

 private:
  void collectNames();
  // put SendWork into a queue and notify the worker thread
  void enqueueSend(SendWork work);
  // put RecvWork into a queue and notify the worker thread
  void enqueueRecv(RecvWork work);
  // receiving messages
  void listenLoop();

  int64_t nextId() {
    return nextId_++;
  }

  std::shared_ptr<c10d::ProcessGroup> pg_;
  // worker name -> rank
  std::unordered_map<std::string, int> nameMap_;
  std::vector<WorkerId> workerIds_;
  std::atomic<int64_t> nextId_;
  // one mutex per ProcessGroup rank, as ProcessGroup::send is not thread-safe
  // when using the same tag.
  std::vector<std::mutex> sendMutexes_;
  std::thread listenerThread_;
  // A threadPool that processing both SendWork and RecvWork. There are two
  // motivations for adding a ThreadPool:
  // (1) RPC serialization/deserialization and processing can be expensive,
  //     hence using multiple threads to speed it up.
  // (2) The current RPC API does not support asynchronous UDFs, e.g., UDFs can
  //     not yield in the middle of execution to wait for IO, and resume the IO
  //     is done. This would result in deadlocks when we have nested RPC calls.
  //     NB: Ideally, this should be addressed by supporting asynchronous UDF.
  //         This is just a temporary solution for (2).
  ThreadPool threadPool_;
  std::unordered_map<int64_t, std::shared_ptr<FutureMessage>> futures_;
  std::mutex futureMutex_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
