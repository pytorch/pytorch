#include <torch/csrc/distributed/rpc/request_callback_impl.h>
#include <torch/csrc/distributed/rpc/testing/faulty_process_group_agent.h>
#include <torch/csrc/distributed/rpc/utils.h>

namespace torch {
namespace distributed {
namespace rpc {

std::string fromVec(const std::vector<char>& vec) {
  return std::string(vec.begin(), vec.end());
}

FaultyProcessGroupAgent::FaultyProcessGroupAgent(
    const c10::intrusive_ptr<::c10d::Store>& store,
    std::string workerName,
    c10::intrusive_ptr<::c10d::ProcessGroup> pg,
    int numSendRecvThreads,
    std::chrono::milliseconds rpcTimeout,
    const std::vector<std::string>& messagesToFail,
    const std::unordered_map<std::string, float>& messageTypesToDelay,
    int failNumSends)
    : ProcessGroupAgent(
          store,
          std::move(workerName),
          std::move(pg),
          numSendRecvThreads,
          rpcTimeout,
          std::make_unique<RequestCallbackImpl>()),
      failNumSends_(failNumSends),
      messageTypesToFail_(parseMessagesToFailInput(messagesToFail)),
      messageTypesToDelay_(parseMessagesToDelay(messageTypesToDelay)) {}

std::vector<MessageType> FaultyProcessGroupAgent::parseMessagesToFailInput(
    const std::vector<std::string>& messagesToFail) const {
  // Since we can only pass strings corresponding to the Message Types from the
  // python tests, we must parse the list of strings and resolve the actual
  // types. We will then check this list of types in the send function to
  // determine whether we should fail or not.
  std::vector<MessageType> messageTypesToFail;
  messageTypesToFail.reserve(messagesToFail.size());
  for (const auto& msgString : messagesToFail) {
    messageTypesToFail.push_back(messageStringToType(msgString));
  }
  return messageTypesToFail;
}

std::unordered_map<MessageType, float, std::hash<int>> FaultyProcessGroupAgent::
    parseMessagesToDelay(const std::unordered_map<std::string, float>&
                             messageTypesToDelay) const {
  std::unordered_map<MessageType, float, std::hash<int>> delayMessages;
  for (const auto& messagePair : messageTypesToDelay) {
    float delay = messagePair.second;
    TORCH_CHECK(
        delay >= 0,
        "Delays passed to FaultyProcessGroupAgent must be non-negative.")
    delayMessages.insert({messageStringToType(messagePair.first), delay});
  }
  return delayMessages;
}

c10::intrusive_ptr<JitFuture> FaultyProcessGroupAgent::send(
    const WorkerInfo& to,
    Message&& message,
    const float rpcTimeoutSeconds,
    const std::unordered_map<c10::Device, c10::Device>& /* unused */) {
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
    auto jitFuture = c10::make_intrusive<JitFuture>(at::AnyClassType::get());
    jitFuture->setError(std::make_exception_ptr(std::runtime_error(makeRPCError(
        c10::str("Send attempt failed intentionally for ", key),
        RPCErrorType::INTENTIONAL_FAILURE))));
    return jitFuture;
  } else {
    lock.unlock();
    return ProcessGroupAgent::send(to, std::move(message), rpcTimeoutSeconds);
  }
}

void FaultyProcessGroupAgent::enqueueSend(SendWork work) {
  float msgDelay = getDelayForMessage(work.message_.type());
  if (msgDelay != 0) {
    // Sleep for the specified delay for the message.
    std::this_thread::sleep_for(std::chrono::milliseconds(
        static_cast<int>(msgDelay * kSecToMsConversion)));
  }
  ProcessGroupAgent::enqueueSend(std::move(work));
}

void FaultyProcessGroupAgent::sendToSelf(Message&& message) {
  float msgDelay = getDelayForMessage(message.type());
  if (msgDelay != 0) {
    // Sleep for the specified delay for the message.
    std::this_thread::sleep_for(std::chrono::milliseconds(
        static_cast<int>(msgDelay * kSecToMsConversion)));
  }
  ProcessGroupAgent::sendToSelf(std::move(message));
}

bool FaultyProcessGroupAgent::shouldFailMessage(MessageType type) const {
  // Return true if the input message type is in the messageTypesToFail_ list
  return (
      std::find(messageTypesToFail_.begin(), messageTypesToFail_.end(), type) !=
      messageTypesToFail_.end());
}

float FaultyProcessGroupAgent::getDelayForMessage(MessageType type) const {
  const auto& it = messageTypesToDelay_.find(type);
  return it == messageTypesToDelay_.end() ? 0 : it->second;
}

MessageType FaultyProcessGroupAgent::messageStringToType(
    const std::string& messageString) const {
  // Lazily constructed map that returns string to message type mapping
  static std::unordered_map<std::string, MessageType> msgMap = {
      {"RREF_FORK_REQUEST", MessageType::RREF_FORK_REQUEST},
      {"RREF_CHILD_ACCEPT", MessageType::RREF_CHILD_ACCEPT},
      {"RREF_USER_DELETE", MessageType::RREF_USER_DELETE},
      {"CLEANUP_AUTOGRAD_CONTEXT_REQ",
       MessageType::CLEANUP_AUTOGRAD_CONTEXT_REQ},
      {"PYTHON_REMOTE_CALL", MessageType::PYTHON_REMOTE_CALL},
      {"SCRIPT_REMOTE_CALL", MessageType::SCRIPT_REMOTE_CALL},
      {"PYTHON_CALL", MessageType::PYTHON_CALL},
      {"SCRIPT_CALL", MessageType::SCRIPT_CALL},
      {"PYTHON_RREF_FETCH_CALL", MessageType::PYTHON_RREF_FETCH_CALL},
      {"SCRIPT_RREF_FETCH_CALL", MessageType::SCRIPT_RREF_FETCH_CALL}};
  const auto& it = msgMap.find(messageString);
  TORCH_CHECK(
      it != msgMap.end(),
      "No mapping to rpc::MessageType exists for ",
      messageString);
  return it->second;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
