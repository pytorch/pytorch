#pragma once

#include <tensorpipe/core/message.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/utils/byte_order.h>

namespace torch {
namespace distributed {
namespace rpc {

// Given an RPC message received as a request over the wire, deserialize it into
// the appropriate 'RpcCommandBase' type.
TORCH_API std::unique_ptr<RpcCommandBase> deserializeRequest(
    const Message& request);

// Given an RPC message received as a response over the wire, deserialize it
// into the appropriate 'RpcCommandBase' type, if the response is
// FORWARD_AUTOGRAD_RESP type, unwrap it, attach recvBackward() functions
// to received tensors and set the wrappedMsgType to its wrapped message type.
TORCH_API std::unique_ptr<RpcCommandBase> deserializeResponse(
    const Message& response,
    MessageType& wrappedMsgType);

// Given an RPC message received as a response over the wire, deserialize it
// into the valid IValue if the message is for a script rpc result,
// otherwise deserialize it into dummy none ivalue that will never be used.
// In this deserialization, we also attach recv rpc backward functions if
// needed.
IValue deserializeResptoIValueInternal(
    RpcCommandBase& rpc,
    MessageType messageType);
TORCH_API IValue deserializeRespToIValue(const Message& message);

// Note: format is subject to change and intended for RPCs.
// For saving persistently to disk, use torch::save().
TORCH_API std::string wireSerialize(
    const std::vector<char>& payload,
    const std::vector<at::Tensor>& tensors);

TORCH_API std::pair<std::vector<char>, std::vector<at::Tensor>> wireDeserialize(
    const void* data,
    size_t data_size);

// TensorPipeEntry represents serialized tensorpipe message,
// plus reserved tensor datas to keep memory lifetime.
struct TensorPipeEntry {
  tensorpipe::Message message;
  // To keep original user tensors + cloned sparse tensors.
  std::vector<torch::Tensor> reservedTensors;
  // To keep memory of tensors who do not own underlying
  // memory, say created from torch::from_blob()
  std::vector<std::vector<uint8_t>> copiedTensors;
};

// TensorPipe doesn't own any underlying memory. Users are required to
// keep rpcMessage alive for the returned TensorPipeEntry to be valid,
// since TensorPipe message just keeps raw pointers to the memory.
TORCH_API TensorPipeEntry tensorpipeSerialize(const Message& rpcMessage);

// The passed-in tensorpipe message is partial, which just contains
// necessary information for memory allocation, like payload length
// and tensor metadata. The returned RPC message doesn't have any
// data, but would be valid after tensorpipe finishs data transfer.
TORCH_API Message tensorpipeAllocateMessage(tensorpipe::Message& tpMessage);

// Some Tensors are effectively views of larger Tensors, where only a small
// subset of the Storage data is referenced. This normally is good and avoids
// copies when kept locally, but if we naively push the whole Storage over the
// wire, we'll end up with excess network traffic. This change clones tensors if
// we'd save at least half the data, and over a minimum hurdle.
TORCH_API c10::List<at::Tensor> cloneSparseTensors(
    const std::vector<at::Tensor>& tensors);

// Combines an original payload and wrapped payload into the original payload.
// Used to generate the overall payload for the wrapped RPC.
TORCH_API void generateWrappedPayload(
    std::vector<char>& originalPayload,
    std::vector<char>& additionalPayload);

// Reads the additional payload from a wrapped RPC off of the input payload.
// After this, payload will contain the payload of the original, un-wrapped RPC.
TORCH_API std::vector<at::IValue> readPayload(
    std::vector<char>& payload,
    const rpc::Message& message);

} // namespace rpc
} // namespace distributed
} // namespace torch
