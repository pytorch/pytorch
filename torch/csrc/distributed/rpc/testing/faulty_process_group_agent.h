#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/process_group_agent.h>

namespace torch {
namespace distributed {
namespace rpc {

struct FaultyProcessGroupRpcBackendOptions
    : public ProcessGroupRpcBackendOptions {
  FaultyProcessGroupRpcBackendOptions(
      int num_send_recv_threads,
      std::chrono::milliseconds rpc_timeout,
      std::string init_method,
      std::vector<std::string> messages_to_fail,
      int num_fail_sends = 0)
      : ProcessGroupRpcBackendOptions(
            num_send_recv_threads,
            rpc_timeout,
            std::move(init_method)),
        messagesToFail(std::move(messages_to_fail)),
        numFailSends(num_fail_sends) {
    TORCH_CHECK(numFailSends >= 0, "numFailSends should be non-negative");
  }

  std::vector<std::string> messagesToFail;
  int numFailSends;
};

class FaultyProcessGroupAgent : public ProcessGroupAgent {
 public:
  FaultyProcessGroupAgent(
      std::string workerName,
      std::shared_ptr<c10d::ProcessGroup> pg,
      int numSendRecvThreads,
      std::chrono::milliseconds rpcTimeout,
      const std::vector<std::string>& messagesToFail,
      int failNumSends = 0);

  // Faulty send function for this class.
  std::shared_ptr<FutureMessage> send(const WorkerInfo& to, Message&& message)
      override;

 protected:
  // This function checks the messageTypesToFail_ to determine whether to use
  // the faulty send or not.
  virtual bool shouldFailMessage(MessageType type) const;

 private:
  // This function parses the list of strings passed in by the python tests and
  // resolves the Message Types that must use the faulty send.
  std::vector<MessageType> parseMessagesToFailInput(
      const std::vector<std::string>& messagesToFail) const;

  // Number of sends to intentionally fail before allowing one to succeed.
  const int failNumSends_;

  // Vector of the MessageTypes that we must use the faulty send for. This is
  // parsed based on a list of strings passed in by the python tests.
  const std::vector<MessageType> messageTypesToFail_;

  // Map to track the number of sends we've failed for each RPC.
  std::unordered_map<std::string, int> failMessageCountMap_;

  // Mutex to guard failMessageCountMap_
  std::mutex failMapMutex_;
};
} // namespace rpc
} // namespace distributed
} // namespace torch
