#include <torch/csrc/distributed/rpc/testing/faulty_process_group_agent.h>

namespace torch {
namespace distributed {
namespace rpc {

std::string fromVec(const std::vector<char>& vec) {
  return std::string(vec.begin(), vec.end());
}

FaultyProcessGroupAgent::FaultyProcessGroupAgent(
    std::string workerName,
    std::shared_ptr<c10d::ProcessGroup> pg,
    int numSendRecvThreads,
    std::chrono::milliseconds rpcTimeout,
    const std::vector<std::string>& messagesToFail,
    std::unordered_map<std::string, float> messagesToDelay,
    int failNumSends)
    : ProcessGroupAgent(
          std::move(workerName),
          std::move(pg),
          numSendRecvThreads,
          rpcTimeout),
      failNumSends_(failNumSends),
      messageTypesToFail_(parseMessagesToFailInput(messagesToFail)),
      messagesToDelay_(parseMessagesToDelay(messagesToDelay)) {}

std::vector<MessageType> FaultyProcessGroupAgent::parseMessagesToFailInput(
    const std::vector<std::string>& messagesToFail) const {
  // Since we can only pass strings corresponding to the Message Types from the
  // python tests, we must parse the list of strings and resolve the actual
  // types. We will then check this list of types in the send function to
  // determine whether we should fail or not.
  std::vector<MessageType> messageTypesToFail;
  for (const auto& msgString : messagesToFail) {
    messageTypesToFail.emplace_back(
        messageStringToType().find(msgString)->second);
  }
  return messageTypesToFail;
}

std::unordered_map<MessageType, float, std::hash<int>> FaultyProcessGroupAgent::
    parseMessagesToDelay(
        const std::unordered_map<std::string, float>& messagesToDelay) const {
  std::unordered_map<MessageType, float, std::hash<int>> delayMessages;
  for (const auto& messagePair : messagesToDelay) {
    delayMessages.insert({messageStringToType().find(messagePair.first)->second,
                          messagePair.second});
  }
  return delayMessages;
}

std::shared_ptr<FutureMessage> FaultyProcessGroupAgent::send(
    const WorkerInfo& to,
    Message&& message,
    const float rpcTimeoutSeconds) {
  // We only fail control messages that have been specified by the test case.
  // For all other messages, we just send them without any failures.
  if (!shouldFailMessage(message.type())) {
    return ProcessGroupAgent::send(to, std::move(message), rpcTimeoutSeconds);
  }
  // This send function checks the failMessageCountMap_ to check whether
  // we must fail the next send. If the send must be failed, we set an error
  // on the returned future immediately and increment the counter in the map,
  // otherwise we just call the ProcessGroupAgent send.
  const auto key = fromVec(message.payload());
  std::unique_lock<std::mutex> lock(failMapMutex_);
  auto it = failMessageCountMap_.find(key);
  if (it == failMessageCountMap_.end()) {
    failMessageCountMap_[key] = 0;
  }
  if (failMessageCountMap_[key] < failNumSends_) {
    failMessageCountMap_[key]++;
    lock.unlock();
    auto fm = std::make_shared<FutureMessage>();
    fm->setError(c10::str("Send attempt failed intentionally for ", key));
    return fm;
  } else {
    lock.unlock();
    return ProcessGroupAgent::send(to, std::move(message), rpcTimeoutSeconds);
  }
}

void FaultyProcessGroupAgent::enqueueSend(SendWork work) {
  float msgDelay = getDelayForMessage(work.message_.type());
  if (msgDelay != 0) {
    // Sleep for the specified delay for the message.
    std::this_thread::sleep_for(
        std::chrono::milliseconds(static_cast<int>(msgDelay * 1000)));
  }
  ProcessGroupAgent::enqueueSend(std::move(work));
}

bool FaultyProcessGroupAgent::shouldFailMessage(MessageType type) const {
  // Return true if the input message type is in the messageTypesToFail_ list
  return (
      std::find(messageTypesToFail_.begin(), messageTypesToFail_.end(), type) !=
      messageTypesToFail_.end());
}

float FaultyProcessGroupAgent::getDelayForMessage(MessageType type) const {
  return messagesToDelay_.find(type) == messagesToDelay_.end()
      ? 0
      : messagesToDelay_.find(type)->second;
}

// Lazily constructed map that returns string to message type mapping
const std::unordered_map<std::string, MessageType> FaultyProcessGroupAgent::
    messageStringToType() const {
  static std::unordered_map<std::string, MessageType> msgMap = {
      {"RREF_FORK_REQUEST", MessageType::RREF_FORK_REQUEST},
      {"RREF_CHILD_ACCEPT", MessageType::RREF_CHILD_ACCEPT},
      {"RREF_USER_DELETE", MessageType::RREF_USER_DELETE},
      {"CLEANUP_AUTOGRAD_CONTEXT_REQ",
       MessageType::CLEANUP_AUTOGRAD_CONTEXT_REQ},
      {"PYTHON_REMOTE_CALL", MessageType::PYTHON_REMOTE_CALL},
      {"PYTHON_CALL", MessageType::PYTHON_CALL},
      {"SCRIPT_CALL", MessageType::SCRIPT_CALL},
  };
  return msgMap;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
