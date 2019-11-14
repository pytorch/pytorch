#include <torch/csrc/distributed/rpc/utils.h>

#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/cleanup_autograd_context_resp.h>
#include <torch/csrc/distributed/autograd/rpc_messages/propagate_gradients_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h>
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

std::unique_ptr<RpcCommandBase> deserializeResponse(const Message& response) {
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
    case MessageType::EXCEPTION: {
      std::string err(response.payload().begin(), response.payload().end());
      throw std::runtime_error(err);
    }
    case MessageType::FORWARD_AUTOGRAD_RESP: {
      return autograd::RpcWithAutograd::fromMessage(response);
    }
    case MessageType::BACKWARD_AUTOGRAD_RESP: {
      return autograd::RpcWithAutograd::fromMessage(response);
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

  std::vector<std::pair<std::string, size_t>> header_ents;
  bool ok = false;
  while (ptr < endp) {
    if (*ptr == '\n') {
      ok = true;
      break;
    }
    const char* num_ptr = ptr;
    while (*ptr != ' ' && ptr < endp) {
      ptr++;
    }
    if (ptr >= endp) {
      break;
    }
    std::string name(num_ptr, ptr - num_ptr);
    if (++ptr >= endp) {
      break; // past the ' '
    }
    const char* size_ptr = ptr;
    while (*ptr != '\n' && ptr < endp) {
      ptr++;
    }
    if (ptr >= endp) {
      break;
    }
    size_t sz = c10::stoll(std::string(size_ptr, ptr - size_ptr));
    header_ents.push_back({name, sz});
    ++ptr; // past the '\n'
  }
  ++ptr; // past the final \n

  std::unordered_map<std::string, std::pair<const char*, size_t>> out;
  for (size_t i = 0; i < header_ents.size(); ++i) {
    out[header_ents[i].first] = {ptr, header_ents[i].second};
    ptr += header_ents[i].second;
  }
  if (!ok || ptr > endp) {
    throw std::runtime_error("failed bounds");
  }
  return out;
}

static const char* kMeta = "meta";
static const char* kPayload = "payload";
}; // namespace

std::string wireSerialize(
    const std::vector<char>& payload,
    const std::vector<at::Tensor>& tensors) {
  struct Ent {
    std::string name;
    const char* data;
    size_t size;
  };
  std::vector<Ent> entries;
  std::string metaEntry;
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
    pickler.pushIValue(tensors);
    pickler.stop();
    auto writeable_tensors = pickler.tensorData();
    entries.push_back({kMeta, metaEntry.data(), metaEntry.size()});
    for (size_t i = 0; i < writeable_tensors.size(); i++) {
      entries.push_back({c10::to_string(i),
                         writeable_tensors[i].data(),
                         writeable_tensors[i].sizeInBytes()});
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
  auto payload_it = sections.find(kPayload);
  if (payload_it != sections.end() && payload_it->second.second != 0) {
    payload.assign(
        payload_it->second.first,
        payload_it->second.first + payload_it->second.second);
  }

  std::vector<at::Tensor> tensors;
  auto meta_it = sections.find(kMeta);
  if (meta_it != sections.end()) {
    const auto& meta_data = meta_it->second;
    size_t meta_data_pos = 0;
    auto meta_data_read_func = [&](char* buf, size_t n) -> size_t {
      if (meta_data_pos >= meta_data.second) {
        return 0;
      }
      size_t to_copy =
          std::min(meta_data_pos + n, meta_data.second) - meta_data_pos;
      memcpy(buf, meta_data.first + meta_data_pos, to_copy);
      meta_data_pos += to_copy;
      return to_copy;
    };
    auto section_read_func = [&](const std::string& fname) -> at::DataPtr {
      auto it = sections.find(fname);
      if (it == sections.end()) {
        throw std::runtime_error("Couldn't find file " + fname);
      }
      const auto& idat = it->second;
      void* ptr = malloc(idat.second);
      DataPtr dptr(ptr, ptr, [](void* p) { free(p); }, DeviceType::CPU);
      memcpy(ptr, idat.first, idat.second);
      return dptr;
    };

    torch::jit::Unpickler unpickler(
        meta_data_read_func, nullptr, nullptr, section_read_func, {});
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
