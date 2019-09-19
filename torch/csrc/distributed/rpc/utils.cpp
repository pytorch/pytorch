#include <torch/csrc/distributed/rpc/utils.h>
#include <torch/csrc/distributed/rpc/python_udf_call.h>
#include <torch/csrc/distributed/rpc/python_udf_resp.h>
#include <torch/csrc/distributed/rpc/rpc_with_autograd.h>
#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/distributed/rpc/script_remote_call.h>
#include <torch/csrc/distributed/rpc/script_resp.h>
#include <torch/csrc/distributed/rpc/script_rref_proto.h>

namespace torch {
namespace distributed {
namespace rpc {

std::unique_ptr<RpcCommandBase> deserializeRequest(const Message& request) {
  switch (request.type()) {
    case MessageType::SCRIPT_CALL: {
      return ScriptCall::fromMessage(request);
    }
    case MessageType::PYTHON_CALL: {
      return PythonUDFCall::fromMessage(request);
    }
    case MessageType::REMOTE_CALL: {
      return ScriptRemoteCall::fromMessage(request);
    }
    case MessageType::RREF_FETCH_CALL: {
      return ScriptRRefFetchCall::fromMessage(request);
    }
    case MessageType::RREF_USER_CREATE: {
      return ScriptRRefCreate::fromMessage(request);
    }
    case MessageType::RREF_USER_DELETE: {
      return ScriptRRefDelete::fromMessage(request);
    }
    case MessageType::MESSAGE_WITH_AUTOGRAD_REQ: {
      return RpcWithAutograd::fromMessage(request);
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
      return PythonUDFResp::fromMessage(response);
    }
    case MessageType::EXCEPTION: {
      std::string err(response.payload().begin(), response.payload().end());
      throw std::runtime_error(err);
    }
    case MessageType::MESSAGE_WITH_AUTOGRAD_RESP: {
      return RpcWithAutograd::fromMessage(response);
    }
    default: {
      TORCH_INTERNAL_ASSERT(
          false, "Response type ", response.type(), " not supported.");
    }
  }
}

} // namespace rpc
} // namespace distributed
} // namespace torch
