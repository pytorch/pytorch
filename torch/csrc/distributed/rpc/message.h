#pragma once

#include <torch/serialize.h>
#include <vector>

namespace torch {
namespace distributed {
namespace rpc {

enum MessageType {
  SCRIPT_CALL = 0,
  SCRIPT_RET,
  PYTHON_CALL,
  PYTHON_RET,
  SHUTDOWN,
  EXCEPTION,
  UNKNOWN
};

// A message to be sent/received by an RpcAgent.
//
// A Message object contains 4 fields:
//    payload (std::vector<char>): a binary chunk of data.
//    tensors (std::vector<torch::Tensor>): all tensors. Tensor data are not
//        included in the payload, and it is up to the RpcAgent implementation
//        to determine how to serialize them. This design is helpful for
//        communicating super large tensors where serializing all the data at
//        once leads to excessively large memory footprint. An implementation
//        can then serialize and send tensors chunck-by-chunk, in the streaming
//        fashion.
//    type (MessageType): type of the message.
//    id (int64_t): message id, this is used by ProcessGroupAgent to match
//                  request and response. Other implementation can ignore it
//                  if they have their own ways to do matching.
//
// Layers above ``RpcAgent`` only converts ScriptCall, ScriptRet, PythonCall,
// and PythonRet into a Message, and it is up to the RpcAgent
// implementation to determine how to serialize a message.
class TORCH_API Message final {
 public:
  // Holds AutogradMetadata that needs to be passed around via RPC.
  class AutogradMetadata {
   public:
    AutogradMetadata();

    AutogradMetadata(int64_t autograd_context_id, int64_t autograd_message_id);

    int64_t getAutogradContextId() const;

    int64_t getAutogradMessageId() const;

    void setAutogradContextId(int64_t autograd_context_id);

    void setAutogradMessageId(int64_t autograd_message_id);

   private:
    // autograd_context_id_ is a globally unique integer that identifies a
    // particular distributed autograd pass.
    int64_t autograd_context_id_;
    // autograd_message_id_ is a globally unique integer that identifies a pair
    // of send/recv autograd functions.
    int64_t autograd_message_id_;
  };

  Message();

  Message(std::vector<char>&& payload,
          std::vector<torch::Tensor>&& tensors,
          MessageType type);

  Message(
      std::vector<char>&& payload,
      std::vector<torch::Tensor>&& tensors,
      MessageType type,
      int64_t id,
      const AutogradMetadata& autograd_metadata = AutogradMetadata());

  Message(const Message& other);
  Message(Message&& other) noexcept;
  Message& operator=(Message const& rhs) &;
  Message& operator=(Message&& rhs) &;
  void swap(Message& rhs) noexcept;

  const std::vector<char>& payload() const;
  std::vector<torch::Tensor>& tensors();
  const std::vector<torch::Tensor>& tensors() const;
  const MessageType& type() const;

  bool isRequest() const;
  bool isResponse() const;
  bool isShutdown() const;

  // id is an optional field to match request/response. If an RpcAgent
  // implementation is able to do the matching without using this id, it can be
  // dropped during message serialization.
  int64_t id() const;
  void setId(int64_t id);

  int64_t getAutogradContextId() const;

  int64_t getAutogradMessageId() const;

  void setAutogradMetadata(const AutogradMetadata& autograd_metadata);

  bool hasAutogradMetadata() const;

 private:
  static constexpr int64_t kInvalidAutogradId = -1;

  std::vector<char> payload_;
  std::vector<torch::Tensor> tensors_;
  MessageType type_ = MessageType::UNKNOWN;
  int64_t id_ = -1;
  AutogradMetadata autograd_metadata_;
};

} // rpc
} // distributed
} // torch
