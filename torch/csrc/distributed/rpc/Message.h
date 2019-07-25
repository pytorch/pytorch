#pragma once

#include <torch/csrc/distributed/rpc/rpc_headers.h>
#include <torch/serialize.h>
#include <vector>

namespace torch {
namespace distributed {
namespace rpc {

enum MessageType {
  BUILTIN_OP = 0,
  BUILTIN_RET,
  PYTHON_UDF_OP,
  PYTHON_UDF_RET,
  SHUTDOWN,
  UNKNOWN
};

// A message to be send/recv by an RpcAgent
class TORCH_API Message final {
 public:

  Message();

  Message(std::vector<char>&& meta,
          std::vector<torch::Tensor>&& tensors,
          MessageType type);

  Message(std::vector<char>&& meta,
          std::vector<torch::Tensor>&& tensors,
          MessageType type,
          int64_t id);

  Message(const Message & other);
  Message& operator=(Message const & rhs) &;
  Message& operator=(Message && rhs) &;
  void swap(Message & rhs) noexcept;
  ~Message();

  const std::vector<char>& meta() const;
  const std::vector<torch::Tensor>& tensors() const;
  const MessageType& type() const;

  bool isOp() const;
  bool isRet() const;
  bool isShutdown() const;

  // id is an optional field to match request/response. If an RpcAgent
  // implementation is able to do the matching without using this id, it can be
  // dropped during message serialization.
  int64_t id() const;
  void setId(int64_t id);

 private:
  std::vector<char> meta_;
  std::vector<torch::Tensor> tensors_;
  MessageType type_;
  int64_t id_;
};

}
}
}
