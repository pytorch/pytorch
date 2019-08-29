#include <torch/csrc/distributed/rpc/types.h>


namespace torch {
namespace distributed {
namespace rpc {

GloballyUniqueId::GloballyUniqueId(worker_id_t createdOn, local_id_t localId)
    : createdOn_(createdOn), localId_(localId) {}

bool GloballyUniqueId::operator==(const GloballyUniqueId& other) const {
  return createdOn_ == other.createdOn_ && localId_ == other.localId_;
}

at::IValue GloballyUniqueId::toIValue() const {
  std::vector<at::IValue> ivalues = {(int64_t)createdOn_, (int64_t)localId_};
  return c10::ivalue::Tuple::create(std::move(ivalues));
}

GloballyUniqueId GloballyUniqueId::fromIValue(at::IValue&& ivalue) {
  auto ivalues = ivalue.toTuple()->elements();
  TORCH_CHECK(ivalues.size() == 2, "Constructing GloballyUniqueId from ivalue "
      "expects a GenericList of two elements, but got ", ivalues.size());

  worker_id_t createdOn = ivalues[0].toInt();
  local_id_t localId = ivalues[1].toInt();

  TORCH_CHECK(createdOn < std::numeric_limits<worker_id_t>::max(),
      "GloballyUniqueId createdOn out of range, got ", createdOn);

  TORCH_CHECK(localId < std::numeric_limits<local_id_t>::max(),
      "GloballyUniqueId localId out of range, got ", localId);

  return GloballyUniqueId(createdOn, localId);
}

std::ostream &operator<<(std::ostream &os, GloballyUniqueId const &globalId) {
    return os << "GloballyUniqueId(" << globalId.createdOn_ << ", "
              << globalId.localId_ << ")";
}

} // namespace rpc
} // namespace distributed
} // namespace torch
