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
    int failNumSends)
    : ProcessGroupAgent(
          std::move(workerName),
          std::move(pg),
          numSendRecvThreads,
          rpcTimeout),
      failNumSends_(failNumSends),
      messageTypesToFail_(parseMessagesToFailInput(messagesToFail)) {}

std::vector<MessageType> FaultyProcessGroupAgent::parseMessagesToFailInput(
    const std::vector<std::string>& messagesToFail) const {
  // Since we can only pass strings corresponding to the Message Types from the
  // python tests, we must parse the list of strings and resolve the actual
  // types. We will then check this list of types in the send function to
  // determine whether we should fail or not.
  std::vector<MessageType> messageTypesToFail;
  for (const auto& msgString : messagesToFail) {
    if (msgString == "RREF_FORK_REQUEST") {
      messageTypesToFail.emplace_back(MessageType::RREF_FORK_REQUEST);
    } else if (msgString == "RREF_CHILD_ACCEPT") {
      messageTypesToFail.emplace_back(MessageType::RREF_CHILD_ACCEPT);
    } else if (msgString == "RREF_USER_DELETE") {
      messageTypesToFail.emplace_back(MessageType::RREF_USER_DELETE);
    } else if (msgString == "CLEANUP_AUTOGRAD_CONTEXT_REQ") {
      messageTypesToFail.emplace_back(
          MessageType::CLEANUP_AUTOGRAD_CONTEXT_REQ);
    }
  }
  return messageTypesToFail;
}

std::shared_ptr<FutureMessage> FaultyProcessGroupAgent::send(
    const WorkerInfo& to,
    Message&& message) {
  // We only fail control messages that have been specified by the test case.
  // For all other messages, we just send them without any failures.
  if (!shouldFailMessage(message.type())) {
    return ProcessGroupAgent::send(to, std::move(message));
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
    return ProcessGroupAgent::send(to, std::move(message));
  }
}

bool FaultyProcessGroupAgent::shouldFailMessage(MessageType type) const {
  // Return true if the input message type is in the messageTypesToFail_ list
  return (
      std::find(messageTypesToFail_.begin(), messageTypesToFail_.end(), type) !=
      messageTypesToFail_.end());
}

} // namespace rpc
} // namespace distributed
} // namespace torch
