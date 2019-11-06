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
  SendWork(const WorkerInfo& to, Message&& message)
      : to_(to), message_(message) {}

  const WorkerInfo& to_;
  Message message_;
};

// SendWork wraps a Message and RecvWork wraps a Tensor. The difference here is
// to allow us to run serialization/deserialization in the worker threads.
struct RecvWork {
  RecvWork(
      const WorkerInfo& from,
      MessageType type,
      int64_t id,
      int32_t serialization,
      torch::Tensor&& payload)
      : from_(from),
        type_(type),
        id_(id),
        serialization_(serialization),
        payload_(payload) {}

  const WorkerInfo& from_;
  const MessageType type_;
  const int64_t id_;
  const int32_t serialization_;
  torch::Tensor payload_;
};

class ProcessGroupAgent : public RpcAgent {
 public:
  ProcessGroupAgent(
      std::string workerName,
      std::shared_ptr<c10d::ProcessGroup> pg,
      int numSendRecvThreads,
      std::chrono::milliseconds rpcTimeout);

  const WorkerInfo& getWorkerInfo(const std::string& workerName) const override;

  const WorkerInfo& getWorkerInfo(worker_id_t id) const override;

  void join() override;

  void sync() override;

  void start() override;

  // retrieves the timeout for all RPCs
  const std::chrono::milliseconds& getRpcTimeout() const;

 protected:
  // This method wraps the destination information and the message into a
  // SendWork object, and put the SendWork into a queue. Another thread will
  // consume SendWork from the queue and send it out.
  std::shared_ptr<FutureMessage> send(const WorkerInfo& to, Message&& message)
      override;

 private:
  class MessageCounter {
   public:
    explicit MessageCounter(int worldSize);
    void increment(int dst);
    std::vector<int64_t> snapshot();

   private:
    std::vector<int64_t> counters_;
    std::mutex mutex_;
  };

  void collectNames();
  // put SendWork into a queue and notify the worker thread
  void enqueueSend(SendWork work);
  // put RecvWork into a queue and notify the worker thread
  void enqueueRecv(RecvWork work);
  // receiving messages
  void listenLoop();

  // Note [Termination Detection]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //
  // RpcAgent implementations must properly detect termination. Otherwise, it
  // would result in message loss, RRef leak, or process hang. It is not
  // sufficient to just wait for the thread pool to finish processing all tasks
  // after all processes hit the join function. There could be nested rpc/remote
  // calls, meaning that an empty task queue in the thread pool does not mean
  // there will be no tasks added in the future. Moreover, in the listenLoop,
  // there is a period of time when the message has been received but not yet
  // inserted into the thread pool, which also suggests that the empty task
  // queue is not a good indicator for termination.
  //
  // To detect termination, each ProcessGroupAgent maintains a sent message
  // counter and a received message counter. The sent message counter is
  // incremented whenever a message is sent, and the receive message counter is
  // only incremented when a message has been processed. During termination, all
  // ProcessGroupAgent instances run an allgather to collect counters from all
  // peers, which means that all agents will have a consistent view on the
  // message count snapshot. They would only terminate if all sent/received
  // message counters match.
  bool hasPendingMessage();

  int64_t nextId() {
    return ++nextId_;
  }

  std::shared_ptr<c10d::ProcessGroup> pg_;
  // worker name -> rank
  std::unordered_map<std::string, int> nameMap_;
  std::vector<WorkerInfo> allWorkerInfo_;
  // record the number of messages sent to and received from each peer. The recv
  // counter is only marked after the message is processed. Join uses allgather
  // to collect all counts from all peers, uses these counters to detect global
  // termination and only exit when all sent messages are processed.
  MessageCounter sendCounts_;
  MessageCounter recvCounts_;

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
  // Mapping of request id to (future, future timeout) pair. We store the future
  // timeout for efficient lookups into the futureTimeouts_ map.
  std::unordered_map<
      int64_t,
      std::pair<std::shared_ptr<FutureMessage>, std::chrono::milliseconds>>
      futures_;
  // A map to keep track of when futures time out. The map is keyed by the time
  // (millisecond level precision) the future started, and the values correspond
  // to a vector of futures that started at that time. When futures time out,
  // the entry in this map is cleared and the corresponding future in the
  // futures_ map is deleted.
  std::map<std::chrono::milliseconds, std::vector<int64_t>> futureTimeouts_;
  mutable std::mutex futureMutex_;
  mutable std::condition_variable futureCV_;
  std::chrono::milliseconds rpcTimeout_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
