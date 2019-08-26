#include <torch/csrc/distributed/rpc/types.h>


namespace torch {
namespace distributed {
namespace rpc {

RRefId::RRefId(worker_id_t createdOn, local_id_t localId)
    : createdOn_(createdOn), localId_(localId) {}

RRefId::RRefId(const RRefId& other)
    : createdOn_(other.createdOn_), localId_(other.localId_) {}

bool RRefId::operator==(const RRefId& other) const {
  return createdOn_ == other.createdOn_ && localId_ == other.localId_;
}

at::IValue RRefId::toIValue() const {
  std::vector<at::IValue> ivalues = {(int64_t)createdOn_, (int64_t)localId_};
  return c10::ivalue::Tuple::create(std::move(ivalues));
}

RRefId RRefId::fromIValue(const at::IValue&& ivalue) {
  auto ivalues = ivalue.toTuple()->elements();
  TORCH_CHECK(ivalues.size() == 2, "Constructing RRefId from ivalue expects "
      "a GenericList of two elements, but got ", ivalues.size());

  int64_t createdOn = ivalues[0].toInt();
  int64_t localId = ivalues[1].toInt();

  TORCH_CHECK(createdOn < std::numeric_limits<worker_id_t>::max(),
      "RRefId createdOn out of range, got ", createdOn);

  TORCH_CHECK(localId < std::numeric_limits<local_id_t>::max(),
      "RRefId localId out of range, got ", localId);

  return RRefId(createdOn, localId);
}

std::ostream &operator<<(std::ostream &os, RRefId const &rrefId) {
    return os << "RRefId(" << rrefId.createdOn_ << ", "
              << rrefId.localId_ << ")";
}

} // namespace rpc
} // namespace distributed
} // namespace torch
