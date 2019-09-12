#include <torch/csrc/distributed/rpc/types.h>

namespace torch {
namespace distributed {
namespace rpc {

static_assert(
    std::numeric_limits<local_id_t>::max() <=
        std::numeric_limits<int64_t>::max(),
    "The max value of local_id_t must be within the range of int64_t");
static_assert(
    std::numeric_limits<worker_id_t>::max() <=
        std::numeric_limits<int64_t>::max(),
    "The max value of worker_id_t must be within the range of int64_t");

GloballyUniqueId::GloballyUniqueId(worker_id_t createdOn, local_id_t localId)
    : createdOn_(createdOn), localId_(localId) {}

bool GloballyUniqueId::operator==(const GloballyUniqueId& other) const {
  return createdOn_ == other.createdOn_ && localId_ == other.localId_;
}

bool GloballyUniqueId::operator!=(const GloballyUniqueId& other) const {
  return createdOn_ != other.createdOn_ || localId_ != other.localId_;
}

at::IValue GloballyUniqueId::toIValue() const {
  return c10::ivalue::Tuple::create(
      {static_cast<int64_t>(createdOn_), static_cast<int64_t>(localId_)});
}

GloballyUniqueId GloballyUniqueId::fromIValue(const at::IValue& ivalue) {
  auto ivalues = ivalue.toTuple()->elements();
  TORCH_CHECK(
      ivalues.size() == 2,
      "Constructing GloballyUniqueId from ivalue "
      "expects a GenericList of two elements, but got ",
      ivalues.size());

  TORCH_CHECK(
      ivalues[0].toInt() <= std::numeric_limits<worker_id_t>::max(),
      "GloballyUniqueId createdOn out of range, got ",
      ivalues[0].toInt());
  worker_id_t createdOn = ivalues[0].toInt();

  TORCH_CHECK(
      ivalues[1].toInt() <= std::numeric_limits<local_id_t>::max(),
      "GloballyUniqueId localId out of range, got ",
      ivalues[1].toInt());
  local_id_t localId = ivalues[1].toInt();

  return GloballyUniqueId(createdOn, localId);
}

std::ostream& operator<<(std::ostream& os, GloballyUniqueId const& globalId) {
  return os << "GloballyUniqueId(" << globalId.createdOn_ << ", "
            << globalId.localId_ << ")";
}

} // namespace rpc
} // namespace distributed
} // namespace torch
