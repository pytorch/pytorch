#ifdef USE_TENSORPIPE

#include <torch/csrc/distributed/rpc/testing/faulty_tensorpipe_agent.h>
#include <torch/csrc/distributed/rpc/utils.h>

namespace torch {
namespace distributed {
namespace rpc {

static std::string fromVecToString(const std::vector<char>& vec) {
  return std::string(vec.begin(), vec.end());
}

FaultyTensorPipeAgent::FaultyTensorPipeAgent(
    const c10::intrusive_ptr<::c10d::Store>& store,
    std::string selfName,
    worker_id_t selfId,
    int worldSize,
    FaultyTensorPipeRpcBackendOptions opts,
    std::unordered_map<std::string, DeviceMap> reverseDeviceMaps,
    std::vector<c10::Device> devices,
    std::unique_ptr<RequestCallback> callback)
    : TensorPipeAgent(
          store,
          std::move(selfName),
          selfId,
          worldSize,
          std::move(opts),
          std::move(reverseDeviceMaps),
          std::move(devices),
          std::move(callback)),
      numFailSends_(opts.numFailSends),
      messageTypesToFail_(parseMessagesToFailInput(opts.messagesToFail)),
      messageTypesToDelay_(parseMessagesToDelay(opts.messagesToDelay)) {}

std::vector<MessageType> FaultyTensorPipeAgent::parseMessagesToFailInput(
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

std::unordered_map<MessageType, float, std::hash<int>> FaultyTensorPipeAgent::
    parseMessagesToDelay(const std::unordered_map<std::string, float>&
                             messageTypesToDelay) const {
  std::unordered_map<MessageType, float, std::hash<int>> delayMessages;
  for (const auto& messagePair : messageTypesToDelay) {
    float delay = messagePair.second;
    TORCH_CHECK(
        delay >= 0,
        "Delays passed to FaultyTensorPipeAgent must be non-negative.")
    delayMessages.insert({messageStringToType(messagePair.first), delay});
  }
  return delayMessages;
}

c10::intrusive_ptr<JitFuture> FaultyTensorPipeAgent::send(
    const WorkerInfo& to,
    c10::intrusive_ptr<Message> message,
    const float rpcTimeoutSeconds,
    const DeviceMap& /* unused */) {
  // We only fail control messages that have been specified by the test case.
  // For all other messages, we just send them without any failures.
  if (!shouldFailMessage(message->type())) {
    return TensorPipeAgent::send(to, std::move(message), rpcTimeoutSeconds);
  }

  // This send function checks the failMessageCountMap_ to check whether
  // we must fail the next send. If the send must be failed, we set an error
  // on the returned future immediately and increment the counter in the map,
  // otherwise we just call the TensorPipeAgent send.
  const auto key = fromVecToString(message->payload());
  std::unique_lock<std::mutex> lock(failMapMutex_);
  auto it = failMessageCountMap_.find(key);
  if (it == failMessageCountMap_.end()) {
    failMessageCountMap_[key] = 0;
  }
  if (failMessageCountMap_[key] < numFailSends_) {
    failMessageCountMap_[key]++;
    lock.unlock();
    auto jitFuture = c10::make_intrusive<JitFuture>(at::AnyClassType::get());
    jitFuture->setError(std::make_exception_ptr(std::runtime_error(makeRPCError(
        c10::str("Send attempt failed intentionally for ", key),
        RPCErrorType::INTENTIONAL_FAILURE))));
    return jitFuture;
  } else {
    lock.unlock();
    return TensorPipeAgent::send(to, std::move(message), rpcTimeoutSeconds);
  }
}

void FaultyTensorPipeAgent::pipeWrite(
    const std::shared_ptr<tensorpipe::Pipe>& pipe,
    c10::intrusive_ptr<Message> rpcMessage,
    std::vector<c10::Device>&& devices,
    std::vector<c10::Stream> streams,
    std::function<void(const tensorpipe::Error&)> fn) noexcept {
  float msgDelay = getDelayForMessage(rpcMessage->type());
  if (msgDelay != 0) {
    // Sleep for the specified delay for the message.
    std::this_thread::sleep_for(std::chrono::milliseconds(
        static_cast<int>(msgDelay * kSecToMsConversion)));
  }
  TensorPipeAgent::pipeWrite(pipe, rpcMessage, std::move(devices), streams, fn);
}

bool FaultyTensorPipeAgent::shouldFailMessage(MessageType type) const {
  // Return true if the input message type is in the messageTypesToFail_ list
  return (
      std::find(messageTypesToFail_.begin(), messageTypesToFail_.end(), type) !=
      messageTypesToFail_.end());
}

float FaultyTensorPipeAgent::getDelayForMessage(MessageType type) const {
  const auto& it = messageTypesToDelay_.find(type);
  return it == messageTypesToDelay_.end() ? 0 : it->second;
}

MessageType FaultyTensorPipeAgent::messageStringToType(
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

#endif // USE_TENSORPIPE
