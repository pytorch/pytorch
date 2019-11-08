#pragma once

#include <ATen/core/ivalue.h>
#include <atomic>

namespace torch {
namespace distributed {
namespace rpc {

using worker_id_t = int16_t;
using local_id_t = int64_t;

struct TORCH_API GloballyUniqueId final {
  GloballyUniqueId(worker_id_t createdOn, local_id_t localId);
  GloballyUniqueId(const GloballyUniqueId& other) = default;
  GloballyUniqueId& operator=(const GloballyUniqueId& other) = delete;

  bool operator==(const GloballyUniqueId& other) const;
  bool operator!=(const GloballyUniqueId& other) const;

  at::IValue toIValue() const;
  static GloballyUniqueId fromIValue(const at::IValue&);

  struct Hash {
    size_t operator()(const GloballyUniqueId& key) const {
      return (uint64_t(key.createdOn_) << kLocalIdBits) | key.localId_;
    }
  };

  static constexpr int kLocalIdBits = 48;

  const worker_id_t createdOn_;
  const local_id_t localId_;
};

TORCH_API std::ostream& operator<<(
    std::ostream& os,
    const GloballyUniqueId& globalId);

using RRefId = GloballyUniqueId;
using ForkId = GloballyUniqueId;

struct TORCH_API SerializedPyObj final {
  SerializedPyObj(std::string&& payload, std::vector<at::Tensor>&& tensors)
      : payload_(std::move(payload)), tensors_(std::move(tensors)) {}

  std::vector<at::IValue> toIValues() const;
  static SerializedPyObj fromIValues(std::vector<at::IValue> value);

  const std::string payload_;
  const std::vector<at::Tensor> tensors_;
};

struct RpcAgentOptions {
  RpcAgentOptions() {}
  std::chrono::milliseconds rpcTimeout;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
