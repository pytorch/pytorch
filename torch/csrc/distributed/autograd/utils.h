#pragma once

#include <torch/csrc/distributed/autograd/context/context.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_req.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_profiling_resp.h>

namespace torch {
namespace distributed {
namespace autograd {

// This method is used to attach the 'send' autograd function to the autograd
// graph when we use RPC. This method creates a new 'send' autograd function
// and attaches the provided tensors as next_edges to the 'send' function. In
// addition to this, it also registers the send function in the provided
// autograd context. Finally, the RPC message is updated with appropriate
// autograd information for the recipient.
TORCH_API void addSendRpcBackward(
    const ContextPtr& autogradContext,
    const AutogradMetadata& autogradMetadata,
    std::vector<torch::Tensor>& tensors);

// This method is used to attach the 'recv' autograd function to the autograd
// graph when we use RPC. This method creates a new 'recv' autograd function
// and attaches the provided tensors as inputs to the 'recv' function. It
// creates a new autograd context if needed and registers the 'recv' function
// with this context.
//
// Returns a pointer to the autograd context created.
TORCH_API ContextPtr addRecvRpcBackward(
    const AutogradMetadata& autogradMetadata,
    std::vector<torch::Tensor>& tensors,
    rpc::worker_id_t fromWorkerId,
    const std::unordered_map<c10::Device, c10::Device>& deviceMap);

// This method is a wrapper utility used internally to wrap autograd info
// and attach autograd function for each type of rpc call if it has valid
// context and tensors require grads or forceGradRecording is true, in this
// case, return RpcWithAutograd message; otherwise return original rpc message.
// NB: forceGradRecording is useful when the request does not contain any tensor
// but the corresponding response does.
TORCH_API rpc::Message getMessageWithAutograd(
    const rpc::worker_id_t dstId,
    rpc::Message&& wrappedRpcMsg,
    rpc::MessageType msgType,
    bool forceGradRecording = false,
    const std::unordered_map<c10::Device, c10::Device>& deviceMap =
        {});

// Send message after autograd checking
TORCH_API c10::intrusive_ptr<c10::ivalue::Future>
sendMessageWithAutograd(
    rpc::RpcAgent& agent,
    const rpc::WorkerInfo& dst,
    rpc::Message&& wrappedRpcMsg,
    bool forceGradRecording = false,
    const float rpcTimeoutSeconds = torch::distributed::rpc::kUnsetRpcTimeout,
    bool forceDisableProfiling = false);

} // namespace autograd
} // namespace distributed
} // namespace torch
