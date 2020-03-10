#include <torch/csrc/distributed/rpc/utils.h>

#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_resp.h>
#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_resp.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/python_call.h>
#include <torch/csrc/distributed/rpc/python_remote_call.h>
#include <torch/csrc/distributed/rpc/python_resp.h>
#include <torch/csrc/distributed/rpc/rref_proto.h>
#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/distributed/rpc/script_remote_call.h>
#include <torch/csrc/distributed/rpc/script_resp.h>
#include <torch/csrc/jit/pickler.h>
#include <torch/csrc/jit/unpickler.h>

namespace torch {
namespace distributed {
namespace rpc {

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
    while (*ptr != ' ' && ptr != endp) {
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
    while (*ptr != '\n' && ptr != endp) {
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
    auto storageSize = t.storage().elementSize() * t.storage().numel();
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
  std::vector<jit::WriteableTensorData> tensorData;

  if (!payload.empty()) {
    entries.push_back({kPayload, payload.data(), payload.size()});
  }

  if (!tensors.empty()) {
    torch::jit::Pickler pickler(
        [&](const void* buf, size_t sz) -> size_t {
          metaEntry.append(static_cast<const char*>(buf), sz);
          return sz;
        },
        nullptr);
    pickler.protocol();
    pickler.pushIValue(cloneSparseTensors(tensors));
    pickler.stop();
    // tensorData is in function scope so that the data() pointers stay valid.
    tensorData = pickler.tensorData();
    entries.push_back({kMeta, metaEntry.data(), metaEntry.size()});
    for (size_t i = 0; i < tensorData.size(); i++) {
      entries.push_back({c10::to_string(i),
                         tensorData[i].data(),
                         tensorData[i].sizeInBytes()});
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

    torch::jit::Unpickler unpickler(
        metaDataReadFunc, nullptr, nullptr, sectionReadFunc, {});
    auto ival = unpickler.parse_ivalue();
    for (auto&& t : ival.toTensorList()) {
      tensors.emplace_back(std::move(t));
    }
  }
  return {std::move(payload), std::move(tensors)};
}

} // namespace rpc
} // namespace distributed
} // namespace torch
