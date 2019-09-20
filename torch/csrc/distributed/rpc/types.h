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

} // namespace rpc
} // namespace distributed
} // namespace torch
