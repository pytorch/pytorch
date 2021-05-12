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
      float rpc_timeout,
      std::string init_method,
      std::vector<std::string> messages_to_fail,
      std::unordered_map<std::string, float> messages_to_delay,
      int num_fail_sends = 0)
      : ProcessGroupRpcBackendOptions(
            num_send_recv_threads,
            rpc_timeout,
            std::move(init_method)),
        messagesToFail(std::move(messages_to_fail)),
        messagesToDelay(std::move(messages_to_delay)),
        numFailSends(num_fail_sends) {
    TORCH_CHECK(numFailSends >= 0, "numFailSends should be non-negative");
  }

  std::vector<std::string> messagesToFail;
  std::unordered_map<std::string, float> messagesToDelay;
  int numFailSends;
};

class FaultyProcessGroupAgent : public ProcessGroupAgent {
 public:
  FaultyProcessGroupAgent(
      const c10::intrusive_ptr<::c10d::Store>& store,
      std::string workerName,
      c10::intrusive_ptr<c10d::ProcessGroup> pg,
      int numSendRecvThreads,
      std::chrono::milliseconds rpcTimeout,
      const std::vector<std::string>& messagesToFail,
      const std::unordered_map<std::string, float>& messageTypesToDelay,
      int failNumSends = 0);

  // Faulty send function for this class.
  c10::intrusive_ptr<JitFuture> send(
      const WorkerInfo& to,
      Message&& message,
      const float rpcTimeoutSeconds = torch::distributed::rpc::kUnsetRpcTimeout,
      const std::unordered_map<c10::Device, c10::Device>& deviceMap = {})
      override;

 protected:
  // This function checks the messageTypesToFail_ to determine whether to use
  // the faulty send or not.
  virtual bool shouldFailMessage(MessageType type) const;

 private:
  // Overrides ProcessGroupAgent's enqueueSend to inject delays.
  void enqueueSend(SendWork work) override;
  // Override ProcessGroupAgent's sendToSelf to inject delays.
  void sendToSelf(Message&& message) override;
  // This function parses the list of strings passed in by the python tests and
  // resolves the Message Types that must use the faulty send.
  std::vector<MessageType> parseMessagesToFailInput(
      const std::vector<std::string>& messagesToFail) const;

  // Returns amount of time in seconds to delay sending of the given message
  // type.
  float getDelayForMessage(MessageType type) const;

  // Parse message types that we should inject arbitrary delays for.
  std::unordered_map<MessageType, float, std::hash<int>> parseMessagesToDelay(
      const std::unordered_map<std::string, float>& messageTypesToDelay) const;

  // Number of sends to intentionally fail before allowing one to succeed.
  const int failNumSends_;

  // Vector of the MessageTypes that we must use the faulty send for. This is
  // parsed based on a list of strings passed in by the python tests.
  const std::vector<MessageType> messageTypesToFail_;

  // Mapping of message types to amount we should delay send for in the ::send()
  // function.
  std::unordered_map<MessageType, float, std::hash<int>> messageTypesToDelay_;

  // Map to track the number of sends we've failed for each RPC.
  std::unordered_map<std::string, int> failMessageCountMap_;

  // Mutex to guard failMessageCountMap_
  std::mutex failMapMutex_;

  MessageType messageStringToType(const std::string& messageString) const;
};
} // namespace rpc
} // namespace distributed
} // namespace torch
