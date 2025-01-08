#pragma once

#include <ATen/core/ivalue.h>

namespace torch::distributed::rpc {

using worker_id_t = int16_t;
using local_id_t = int64_t;

bool getAllowJitRRefPickle();
TORCH_API void enableJitRRefPickle();
TORCH_API void disableJitRRefPickle();

struct TORCH_API JitRRefPickleGuard {
  JitRRefPickleGuard();
  JitRRefPickleGuard(JitRRefPickleGuard&& other) = delete;
  JitRRefPickleGuard(const JitRRefPickleGuard&) = delete;
  JitRRefPickleGuard& operator=(const JitRRefPickleGuard&) = delete;
  JitRRefPickleGuard& operator=(JitRRefPickleGuard&&) = delete;
  ~JitRRefPickleGuard();
};

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

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const worker_id_t createdOn_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const local_id_t localId_;
};

TORCH_API std::ostream& operator<<(
    std::ostream& os,
    const GloballyUniqueId& globalId);

using RRefId = GloballyUniqueId;
using ForkId = GloballyUniqueId;
using ProfilingId = GloballyUniqueId;

struct TORCH_API SerializedPyObj final {
  SerializedPyObj(std::string&& payload, std::vector<at::Tensor>&& tensors)
      : payload_(std::move(payload)), tensors_(std::move(tensors)) {}

  std::vector<at::IValue> toIValues() &&;
  static SerializedPyObj fromIValues(std::vector<at::IValue> value);

  std::string payload_;
  std::vector<at::Tensor> tensors_;
};

} // namespace torch::distributed::rpc
