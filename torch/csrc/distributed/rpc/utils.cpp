#include <torch/csrc/distributed/rpc/utils.h>

#include <fmt/format.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_resp.h>
#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_resp.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_resp.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/profiler/remote_profiler_manager.h>
#include <torch/csrc/distributed/rpc/python_call.h>
#include <torch/csrc/distributed/rpc/python_remote_call.h>
#include <torch/csrc/distributed/rpc/python_resp.h>
#include <torch/csrc/distributed/rpc/rref_proto.h>
#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/distributed/rpc/script_remote_call.h>
#include <torch/csrc/distributed/rpc/script_resp.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/jit/serialization/unpickler.h>

#ifdef USE_TENSORPIPE
#include <tensorpipe/core/message.h>
#endif

namespace torch {
namespace distributed {
namespace rpc {
namespace {
void processRemoteProfiledEvents(
    autograd::RpcWithProfilingResp& rpcWithProfilingResp) {
  // Check if the profiler is enabled
  auto enabled = torch::autograd::profiler::profilerEnabled();
  TORCH_CHECK(
      enabled,
      "Profiler was expected to be enabled. This can happen in callback "
      " continutations that run in different threads, and the TLS of the "
      " profiler was not propagated.");
  std::vector<torch::autograd::profiler::Event> events =
      rpcWithProfilingResp.getProfiledEvents();
  const auto& profilingId = rpcWithProfilingResp.getProfilingId();
  auto& remoteProfilerManager = RemoteProfilerManager::getInstance();
  auto key = remoteProfilerManager.retrieveRPCProfilingKey(profilingId);
  remoteProfilerManager.eraseKey(profilingId);
  auto keyPrefixStr = key + rpc::REMOTE_PROFILING_KEY_PREFIX;
  std::for_each(
      events.begin(),
      events.end(),
      [&keyPrefixStr](torch::autograd::profiler::Event& event) {
        std::string name = keyPrefixStr + std::string(event.name());
        event.setName(at::StringView(name));
      });
  // Add event list to the thread local profiler.
  torch::autograd::profiler::addEventList(std::move(events));
}

inline c10::Device indexToDevice(c10::DeviceIndex index) {
  if (index == -1) {
    return c10::Device(at::kCPU);
  } else {
    return c10::Device(at::kCUDA, index);
  }
}

} // namespace

const std::string kRPCErrorPrefix = std::string("RPCErr");

RPCErrorType getRPCErrorType(const FutureMessage& fm) {
  TORCH_INTERNAL_ASSERT(
      fm.hasError(),
      "FutureMessage passed to getRPCErrorType does not have an error.");

  // Attempt to parse for error string given by makeRPCError, otherwise return
  // unknown error.
  // Note that this function expects errors formatted with makeRPCError().
  auto err = std::string(fm.error()->what());
  size_t pos = err.find(kRPCErrorPrefix);
  if (pos != std::string::npos) {
    // Parse the RPCErrorType.
    auto errStartIdx =
        pos + torch::distributed::rpc::kRPCErrorPrefix.size() + 1;
    auto errEndIdx = err.find(':', errStartIdx);
    if (errEndIdx == std::string::npos) {
      // Indicates error was not formatted correctly.
      return RPCErrorType::UNKNOWN_ERROR;
    }
    auto errStr = err.substr(errStartIdx, errEndIdx - errStartIdx);
    auto errType = static_cast<RPCErrorType>(std::stoi(errStr));
    return errType;
  } else {
    return RPCErrorType::UNKNOWN_ERROR;
  }
}

std::string makeRPCError(
    const std::string& rpcErrorStr,
    RPCErrorType errorType) {
  return fmt::format(
      "{}:{}:{}",
      torch::distributed::rpc::kRPCErrorPrefix,
      errorType,
      rpcErrorStr);
}

std::unique_ptr<RpcCommandBase> deserializeRequest(const Message& request) {
  switch (request.type()) {
    case MessageType::SCRIPT_CALL: {
      return ScriptCall::fromMessage(request);
    }
    case MessageType::PYTHON_CALL: {
      return PythonCall::fromMessage(request);
    }
    case MessageType::SCRIPT_REMOTE_CALL: {
      return ScriptRemoteCall::fromMessage(request);
    }
    case MessageType::PYTHON_REMOTE_CALL: {
      return PythonRemoteCall::fromMessage(request);
    }
    case MessageType::SCRIPT_RREF_FETCH_CALL: {
      return ScriptRRefFetchCall::fromMessage(request);
    }
    case MessageType::PYTHON_RREF_FETCH_CALL: {
      return PythonRRefFetchCall::fromMessage(request);
    }
    case MessageType::RREF_USER_DELETE: {
      return RRefUserDelete::fromMessage(request);
    }
    case MessageType::RREF_CHILD_ACCEPT: {
      return RRefChildAccept::fromMessage(request);
    }
    case MessageType::RREF_FORK_REQUEST: {
      return RRefForkRequest::fromMessage(request);
    }
    case MessageType::FORWARD_AUTOGRAD_REQ: {
      return autograd::RpcWithAutograd::fromMessage(request);
    }
    case MessageType::BACKWARD_AUTOGRAD_REQ: {
      return autograd::PropagateGradientsReq::fromMessage(request);
    }
    case MessageType::CLEANUP_AUTOGRAD_CONTEXT_REQ: {
      return autograd::CleanupAutogradContextReq::fromMessage(request);
    }
    case MessageType::RUN_WITH_PROFILING_REQ: {
      return autograd::RpcWithProfilingReq::fromMessage(request);
    }
    default: {
      TORCH_INTERNAL_ASSERT(
          false, "Request type ", request.type(), " not supported.");
    }
  }
}

std::unique_ptr<RpcCommandBase> deserializeResponse(
    const Message& response,
    MessageType& wrappedMsgType) {
  switch (response.type()) {
    case MessageType::SCRIPT_RET: {
      return ScriptResp::fromMessage(response);
    }
    case MessageType::PYTHON_RET: {
      return PythonResp::fromMessage(response);
    }
    case MessageType::REMOTE_RET: {
      return RemoteRet::fromMessage(response);
    }
    case MessageType::SCRIPT_RREF_FETCH_RET: {
      return ScriptRRefFetchRet::fromMessage(response);
    }
    case MessageType::PYTHON_RREF_FETCH_RET: {
      return PythonRRefFetchRet::fromMessage(response);
    }
    case MessageType::RREF_ACK: {
      return RRefAck::fromMessage(response);
    }
    case MessageType::FORWARD_AUTOGRAD_RESP: {
      std::unique_ptr<RpcCommandBase> rpcPtr =
          autograd::RpcWithAutograd::fromMessage(response);
      RpcCommandBase& rpc = *rpcPtr;
      auto& rpcWithAutograd = static_cast<autograd::RpcWithAutograd&>(rpc);

      // Attach 'recv' autograd function.
      addRecvRpcBackward(
          rpcWithAutograd.autogradMetadata(),
          rpcWithAutograd.tensors(),
          rpcWithAutograd.fromWorkerId());

      wrappedMsgType = rpcWithAutograd.wrappedMessageType();

      return std::move(rpcWithAutograd).moveWrappedRpc();
    }
    case MessageType::BACKWARD_AUTOGRAD_RESP: {
      return autograd::PropagateGradientsResp::fromMessage(response);
    }
    case MessageType::CLEANUP_AUTOGRAD_CONTEXT_RESP: {
      return autograd::CleanupAutogradContextResp::fromMessage(response);
    }
    case MessageType::RUN_WITH_PROFILING_RESP: {
      std::unique_ptr<RpcCommandBase> rpcPtr =
          autograd::RpcWithProfilingResp::fromMessage(response);
      RpcCommandBase& rpc = *rpcPtr;
      auto& rpcWithProfilingResp =
          static_cast<autograd::RpcWithProfilingResp&>(rpc);
      // Process remotely profiled events.
      processRemoteProfiledEvents(rpcWithProfilingResp);

      wrappedMsgType = rpcWithProfilingResp.wrappedMessageType();
      auto wrappedRPC = std::move(rpcWithProfilingResp).moveWrappedRpc();
      return wrappedRPC;
    }
    default: {
      TORCH_INTERNAL_ASSERT(
          false, "Response type ", response.type(), " not supported.");
    }
  }
}

IValue deserializeResptoIValueInternal(
    RpcCommandBase& rpc,
    MessageType messageType) {
  switch (messageType) {
    case MessageType::SCRIPT_RET: {
      auto& ret = static_cast<ScriptResp&>(rpc);
      return ret.value();
    }
    default: {
      TORCH_INTERNAL_ASSERT(
          false,
          "Response type ",
          messageType,
          " is not supported to be deserialized to IValue.");
    }
  }
}

IValue deserializeRespToIValue(const Message& message) {
  MessageType msgType = message.type();
  auto response = deserializeResponse(message, msgType);
  return deserializeResptoIValueInternal(*response, msgType);
}

namespace {

// Helper for wireDeserialize() below.
//
// The format we use below looks like:
//    section_name_1 size_1\n
//    section_name_2 size_2\n
//    ..
//    \n
//    [sections in order]
//
// Sections themselves include:
//    - "payload" - the payload bits
//    - "meta"    - metadata for the unpickler
//    - "0" ...   - tensor sections for the unpickler
//
// Note that per the header comments, the format is subject to change,
// and is best used for rpcs, rather than persistent disk storage.
std::unordered_map<std::string, std::pair<const char*, size_t>>
parseWireSections(const void* data, size_t data_size) {
  const char* ptr = static_cast<const char*>(data);
  const char* endp = ptr + data_size;

  std::vector<std::pair<std::string, size_t>> headerEnts;
  bool ok = false;
  while (ptr != endp) {
    if (*ptr == '\n') {
      ok = true; // The only "correct" exit point.
      ++ptr;
      break;
    }
    // Parse name
    const char* namePtr = ptr;
    while (ptr != endp && *ptr != ' ') {
      ptr++;
    }
    if (ptr == endp) {
      break;
    }
    std::string name(namePtr, ptr - namePtr);
    if (++ptr == endp) {
      break; // past the ' '
    }
    // Parse size
    const char* sizePtr = ptr;
    while (ptr != endp && *ptr != '\n') {
      ptr++;
    }
    if (ptr == endp) {
      break;
    }
    size_t sz = c10::stoll(std::string(sizePtr, ptr - sizePtr));
    headerEnts.emplace_back(std::make_pair(name, sz));
    ++ptr; // past the '\n'
  }
  if (!ok) {
    throw std::runtime_error("failed parse");
  }

  std::unordered_map<std::string, std::pair<const char*, size_t>> out;
  for (const auto& headerEnt : headerEnts) {
    out[headerEnt.first] = {ptr, headerEnt.second};
    ptr += headerEnt.second;
  }
  if (ptr != endp) {
    throw std::runtime_error("failed bounds");
  }
  return out;
}

static const char* kMeta = "meta";
static const char* kPayload = "payload";
}; // namespace

c10::List<at::Tensor> cloneSparseTensors(
    const std::vector<at::Tensor>& tensors) {
  // Sanity-check: If the majority of bits don't need to go over the wire,
  // force a clone(). Some Tensors are effectively small views, only using
  // ~1% of the underlying Storage.
  auto worthRecopying = [](const at::Tensor& t) -> bool {
    if (!t.has_storage()) {
      return false; // avoid throwing below.
    }
    auto storageSize = t.storage().nbytes();
    auto usefulSize = t.element_size() * t.numel();
    constexpr size_t kMinMultiple = 2;
    constexpr size_t kMinRecopyBytes = 8 * 1024;
    return storageSize >= kMinRecopyBytes &&
        storageSize >= usefulSize * kMinMultiple;
  };
  c10::List<at::Tensor> pTensors;
  pTensors.reserve(tensors.size());
  for (const auto& t : tensors) {
    pTensors.push_back(worthRecopying(t) ? t.clone() : t);
  }
  return pTensors;
}

std::string wireSerialize(
    const std::vector<char>& payload,
    const std::vector<at::Tensor>& tensors) {
  for (const auto& tensor : tensors) {
    TORCH_CHECK(
        tensor.device().is_cpu(),
        "ProcessGroup RPC backend only supports",
        " CPU tensors, please move your tensors to CPU before sending ",
        "them over RPC. Found tensor on device: ",
        tensor.device());
  }

  struct Ent {
    std::string name;
    const char* data;
    size_t size;
  };
  std::vector<Ent> entries;
  std::string metaEntry;
  std::vector<at::Tensor> tensorData;

  if (!payload.empty()) {
    entries.push_back({kPayload, payload.data(), payload.size()});
  }

  if (!tensors.empty()) {
    torch::jit::Pickler pickler([&](const void* buf, size_t sz) -> size_t {
      metaEntry.append(static_cast<const char*>(buf), sz);
      return sz;
    });
    pickler.protocol();
    pickler.pushIValue(cloneSparseTensors(tensors));
    pickler.stop();
    tensorData = pickler.tensorData();
    entries.push_back({kMeta, metaEntry.data(), metaEntry.size()});
    for (size_t i = 0; i < tensorData.size(); i++) {
      // Construct WritableTensorData for each tensor in the pickler tensorData
      // Since tensorData is in function scope, and getWritableTensorData just
      // record the tensors, the data() pointers stay valid for CPU tensors
      // Note that RPC serde doesn't support CUDA tensors yet, if we should
      // support CUDA tensor, we need to be careful since getWritableTensorData
      // converts CUDA tensor to cpu and data() might get destructed as we go
      // out of scope of this loop.
      auto writeableTensorData = jit::getWriteableTensorData(tensorData[i]);
      entries.push_back({c10::to_string(i),
                         writeableTensorData.data(),
                         writeableTensorData.sizeInBytes()});
    }
  }

  std::string header;
  size_t tot = 0;
  for (const auto& e : entries) {
    tot += e.size;
    header.append(e.name)
        .append(" ")
        .append(c10::to_string(e.size))
        .append("\n");
  }
  header.push_back('\n');

  std::string out;
  out.reserve(header.size() + tot);
  out.append(header);
  for (const auto& e : entries) {
    out.append(e.data, e.size);
  }
  return out;
}

std::pair<std::vector<char>, std::vector<at::Tensor>> wireDeserialize(
    const void* data,
    size_t data_size) {
  auto sections = parseWireSections(data, data_size);

  std::vector<char> payload;
  auto payloadIt = sections.find(kPayload);
  if (payloadIt != sections.end() && payloadIt->second.second != 0) {
    payload.assign(
        payloadIt->second.first,
        payloadIt->second.first + payloadIt->second.second);
  }

  std::vector<at::Tensor> tensors;
  auto metaIt = sections.find(kMeta);
  if (metaIt != sections.end()) {
    const auto& metaData = metaIt->second;
    size_t metaDataPos = 0;
    auto metaDataReadFunc = [&](char* buf, size_t n) -> size_t {
      if (metaDataPos >= metaData.second || n == 0) {
        return 0;
      }
      size_t toCopy = std::min(metaDataPos + n, metaData.second) - metaDataPos;
      memcpy(buf, metaData.first + metaDataPos, toCopy);
      metaDataPos += toCopy;
      return toCopy;
    };
    auto sectionReadFunc = [&](const std::string& ename) -> at::DataPtr {
      auto it = sections.find(ename);
      if (it == sections.end()) {
        throw std::runtime_error("Couldn't find entity " + ename);
      }
      const auto& idat = it->second;
      auto dptr = at::getCPUAllocator()->allocate(idat.second);
      if (idat.second != 0) {
        memcpy(dptr.get(), idat.first, idat.second);
      }
      return dptr;
    };

    // No need to pass typeResolver here, as it always processes string and
    // tensors only
    torch::jit::Unpickler unpickler(
        metaDataReadFunc, nullptr, nullptr, sectionReadFunc, {});
    auto ival = unpickler.parse_ivalue();
    for (auto&& t : ival.toTensorList()) {
      tensors.emplace_back(std::move(t));
    }
  }
  return {std::move(payload), std::move(tensors)};
}

#ifdef USE_TENSORPIPE
namespace {

// The TensorPipe agent splits the RPC message's information across multiple
// payloads. This allows the agent to provide the data to TensorPipe without
// performing a copy into a single contiguous buffer, and without storing it as
// metadata, which is less efficient.

// First come the rpc::Message::type() and ::id().
constexpr int kTpMessageTypeIdx = 0;
constexpr int kTpMessageIdIdx = 1;
// Then comes the rpc::Message::payload();
constexpr int kTpMessagePayloadIdx = 2;
// Last comes the pickle of rpc::Message::tensors() (with the tensors themselves
// stored as, well, tensors in the tensorpipe::Message).
constexpr int kTpMessagePickleIdx = 3;

} // namespace

std::tuple<tensorpipe::Message, TensorpipeWriteBuffers> tensorpipeSerialize(
    Message&& rpcMessage,
    std::vector<c10::DeviceIndex> deviceIndices) {
  tensorpipe::Message tpMessage;
  TensorpipeWriteBuffers buffers;

  // Metadata
  buffers.type = std::make_unique<MessageType>(rpcMessage.type());
  buffers.id = std::make_unique<int64_t>(rpcMessage.id());
  // kTpMessageTypeIdx = 0
  tpMessage.payloads.push_back(
      tensorpipe::Message::Payload{buffers.type.get(), sizeof(MessageType)});
  // kTpMessageIdIdx = 1
  tpMessage.payloads.push_back(
      tensorpipe::Message::Payload{buffers.id.get(), sizeof(int64_t)});

  // Payload
  buffers.payload = std::move(rpcMessage.payload());
  // TensorPipe uses the same Message class for both reading and writing, thus
  // it uses non-const pointers even though it doesn't modify them when writing.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  char* payloadPtr = const_cast<char*>(buffers.payload.data());
  // kTpMessagePayloadIdx = 2
  tpMessage.payloads.push_back(
      tensorpipe::Message::Payload{payloadPtr, buffers.payload.size()});

  // Tensors
  if (deviceIndices.empty()) {
    buffers.tensors = cloneSparseTensors(rpcMessage.tensors()).vec();
  } else {
    std::vector<torch::Tensor> tensors;
    tensors.reserve(rpcMessage.tensors().size());
    for (const auto& tensor : rpcMessage.tensors()) {
      tensors.emplace_back(tensor.cpu());
    }
    buffers.tensors = cloneSparseTensors(tensors).vec();
  }

  torch::jit::Pickler pickler([&](const void* buf, size_t sz) -> size_t {
    buffers.pickle.insert(
        buffers.pickle.end(),
        static_cast<const char*>(buf),
        static_cast<const char*>(buf) + sz);
    return sz;
  });
  pickler.protocol();
  pickler.pushIValue(buffers.tensors);
  pickler.stop();
  // kTpMessagePickleIdx = 3
  tpMessage.payloads.push_back(tensorpipe::Message::Payload{
      buffers.pickle.data(), buffers.pickle.size()});
  for (size_t i = 0; i < pickler.tensorData().size(); ++i) {
    const auto& tensorData =
        jit::getWriteableTensorData(pickler.tensorData()[i]);
    // Enforce memory copy if tensor is created from torch::from_blob, means
    // that the tensor doesn't own the memory.
    std::string metadata =
        deviceIndices.empty() ? "" : std::to_string(deviceIndices[i]);

    if (!tensorData.storageHasDeleter()) {
      std::vector<char> storageData(
          tensorData.data(), tensorData.data() + tensorData.sizeInBytes());
      tpMessage.tensors.push_back(tensorpipe::Message::Tensor{
          storageData.data(), storageData.size(), std::move(metadata)});
      buffers.copiedTensors.push_back(std::move(storageData));
    } else {
      // TensorPipe uses the same Message class for both reading and writing, so
      // it uses non-const ptrs even though it doesn't modify them when writing.
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      char* tensorPtr = const_cast<char*>(tensorData.data());
      tpMessage.tensors.push_back(tensorpipe::Message::Tensor{
          tensorPtr, tensorData.sizeInBytes(), std::move(metadata)});
    }
  }

  return std::make_tuple(std::move(tpMessage), std::move(buffers));
}

TensorpipeReadBuffers tensorpipeAllocate(tensorpipe::Message& tpMessage) {
  TensorpipeReadBuffers buffers;

  TORCH_INTERNAL_ASSERT(
      tpMessage.payloads.size() == 4,
      "message expected to contain 4 payloads, whereas it contained ",
      tpMessage.payloads.size(),
      " payloads");

  TORCH_INTERNAL_ASSERT(
      tpMessage.payloads[kTpMessageTypeIdx].length == sizeof(MessageType),
      "first payload expected to contain ",
      sizeof(MessageType),
      " bytes, whereas it contained ",
      tpMessage.payloads[kTpMessageTypeIdx].length,
      " bytes");
  buffers.type = std::make_unique<MessageType>();
  tpMessage.payloads[kTpMessageTypeIdx].data = buffers.type.get();

  TORCH_INTERNAL_ASSERT(
      tpMessage.payloads[kTpMessageIdIdx].length == sizeof(int64_t),
      "second payload expected to contain ",
      sizeof(int64_t),
      " bytes, whereas it contained ",
      tpMessage.payloads[kTpMessageIdIdx].length,
      " bytes");
  buffers.id = std::make_unique<int64_t>();
  tpMessage.payloads[kTpMessageIdIdx].data = buffers.id.get();

  // FIXME The two resizes below zero out the vectors, which is not needed.
  buffers.payload.resize(tpMessage.payloads[kTpMessagePayloadIdx].length);
  tpMessage.payloads[kTpMessagePayloadIdx].data = buffers.payload.data();

  buffers.pickle.resize(tpMessage.payloads[kTpMessagePickleIdx].length);
  tpMessage.payloads[kTpMessagePickleIdx].data = buffers.pickle.data();

  for (auto& tensor : tpMessage.tensors) {
    buffers.tensors.emplace_back(
        at::getCPUAllocator()->allocate(tensor.length));
    tensor.data = buffers.tensors.back().get();
  }

  return buffers;
}

Message tensorpipeDeserialize(
    tensorpipe::Message&& message,
    TensorpipeReadBuffers&& buffers) {
  // Tensors
  std::vector<at::Tensor> tensors;
  const char* pickleData = buffers.pickle.data();
  size_t pickleLen = buffers.pickle.size();
  size_t picklePos = 0;
  auto pickleReadFunc = [&](char* buf, size_t n) -> size_t {
    if (picklePos >= pickleLen || n == 0) {
      return 0;
    }
    size_t toCopy = std::min(picklePos + n, pickleLen) - picklePos;
    memcpy(buf, pickleData + picklePos, toCopy);
    picklePos += toCopy;
    return toCopy;
  };
  auto tensorReadFunc = [&](const std::string& ename) -> at::DataPtr {
    unsigned long index = std::stoul(ename);
    return std::move(buffers.tensors.at(index));
  };

  // No need to pass typeResolver here, as it always processes string and
  // tensors only
  torch::jit::Unpickler unpickler(
      pickleReadFunc, nullptr, nullptr, tensorReadFunc, {});
  auto ival = unpickler.parse_ivalue();
  for (auto&& t : ival.toTensorList()) {
    tensors.emplace_back(std::move(t));
  }

  // NB: This is a temporary solution. When TensorPipe Tensor.data can point to
  // a CUDA memory address, we should directly use CUDACachingAllocator to
  // create CUDA buffers in tensorpipeAllocate.
  for (size_t i = 0; i < message.tensors.size(); ++i) {
    auto& tensor = message.tensors[i];
    if (!tensor.metadata.empty()) {
      TORCH_INTERNAL_ASSERT(
          message.tensors.size() == tensors.size(),
          "Number of device indices must match the number of tensors in the "
          "RPC message. But got ",
          tensors.size(),
          " tensors with ",
          message.tensors.size(),
          " device indices.");
      tensors[i] = tensors[i].to(indexToDevice(std::stoi(tensor.metadata)));
    }
  }

  return Message(
      std::move(buffers.payload),
      std::move(tensors),
      *buffers.type,
      *buffers.id);
}
#endif /* USE_TENSORPIPE */

void writeWrappedPayload(
    std::vector<char>& originalPayload,
    std::vector<char>& additionalPayload) {
  originalPayload.insert(
      originalPayload.end(),
      additionalPayload.begin(),
      additionalPayload.end());

  // Add size of the additional payload
  int64_t indexToWrite = originalPayload.size();
  originalPayload.resize(originalPayload.size() + sizeof(int64_t));
  const int64_t additionalPayloadSize = additionalPayload.size();
  torch::utils::THP_encodeInt64Buffer(
      reinterpret_cast<uint8_t*>(originalPayload.data()) + indexToWrite,
      &additionalPayloadSize,
      torch::utils::THPByteOrder::THP_BIG_ENDIAN,
      1);
}

std::vector<at::IValue> readWrappedPayload(
    std::vector<char>& payload,
    const rpc::Message& message) {
  // Read the additional payload remove it from the payload.
  int64_t additionalPayloadSize;
  size_t indexToRead = payload.size() - sizeof(int64_t);
  TORCH_INTERNAL_ASSERT(indexToRead >= 0);
  torch::utils::THP_decodeInt64Buffer(
      &additionalPayloadSize,
      reinterpret_cast<uint8_t*>(payload.data()) + indexToRead,
      torch::utils::THPByteOrder::THP_BIG_ENDIAN,
      1);
  payload.resize(indexToRead);

  TORCH_INTERNAL_ASSERT(
      payload.size() > additionalPayloadSize,
      "Wrong payload sizes: payload.size() is ",
      payload.size(),
      " but additional payload size is ",
      additionalPayloadSize);
  auto wrappedPayloadBegin =
      static_cast<const char*>(message.payload().data()) + payload.size() -
      additionalPayloadSize;
  std::vector<torch::Tensor> tensorTable;
  IValue tuple = jit::unpickle(
      wrappedPayloadBegin,
      additionalPayloadSize,
      *rpc::RpcAgent::getCurrentRpcAgent()->getTypeResolver(),
      &tensorTable);
  std::vector<at::IValue> tupleElements = tuple.toTuple()->elements();
  payload.resize(payload.size() - additionalPayloadSize);
  return tupleElements;
}
} // namespace rpc
} // namespace distributed
} // namespace torch
