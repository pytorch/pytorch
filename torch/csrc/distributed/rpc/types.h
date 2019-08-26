#pragma once

#include <ATen/core/ivalue.h>
#include <atomic>

namespace torch {
namespace distributed {
namespace rpc {

using worker_id_t = int16_t;
using local_id_t = uint64_t;

struct RRefId final {
  RRefId(worker_id_t createdOn, local_id_t localId);
  RRefId(const RRefId& other);
  bool operator==(const RRefId& other) const;

  at::IValue toIValue() const;
  static RRefId fromIValue(const at::IValue&&);

  struct Hash {
    size_t operator()(const RRefId& rrefId) const {
      return (uint64_t(rrefId.createdOn_) << 48) | rrefId.localId_;
    }
  };

  const worker_id_t createdOn_;
  const local_id_t localId_;
};

std::ostream &operator<<(std::ostream &os, const RRefId &m);

using ForkId = RRefId;

} // namespace rpc
} // namespace distributed
} // namespace torch
