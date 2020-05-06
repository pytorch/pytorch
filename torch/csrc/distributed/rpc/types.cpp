#include <torch/csrc/distributed/rpc/types.h>

namespace torch {
namespace distributed {
namespace rpc {

// Thread local flag to enforce rref JIT pickling to be allowed only
// in the scope of an rpc call. For other scopes like when model is
// saved by calling torch.save(), rref is not allowed to be pickled directly.
static thread_local bool allowJitRRefPickle = false;

bool getAllowJitRRefPickle() {
  return allowJitRRefPickle;
}

static_assert(
    std::numeric_limits<local_id_t>::max() <=
        std::numeric_limits<int64_t>::max(),
    "The max value of local_id_t must be within the range of int64_t");
static_assert(
    std::numeric_limits<worker_id_t>::max() <=
        std::numeric_limits<int64_t>::max(),
    "The max value of worker_id_t must be within the range of int64_t");

///////////////////////////  JitRRefPickleGuard   ///////////////////////////
JitRRefPickleGuard::JitRRefPickleGuard() {
  allowJitRRefPickle = true;
}
JitRRefPickleGuard::~JitRRefPickleGuard() {
  allowJitRRefPickle = false;
}

///////////////////////////  GloballyUniqueId   ///////////////////////////

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

///////////////////////////  SerializedPyObj   ///////////////////////////

std::vector<at::IValue> SerializedPyObj::toIValues() && {
  std::vector<at::IValue> ivalues;
  ivalues.reserve(tensors_.size() + 1);
  for (auto& tensor : tensors_) {
    ivalues.emplace_back(std::move(tensor));
  }
  ivalues.emplace_back(std::move(payload_));
  return ivalues;
}

SerializedPyObj SerializedPyObj::fromIValues(std::vector<at::IValue> values) {
  std::string payload = values.back().toStringRef();
  values.pop_back();
  std::vector<at::Tensor> tensors;
  tensors.reserve(values.size());
  for (auto& value : values) {
    tensors.emplace_back(value.toTensor());
  }
  return SerializedPyObj(std::move(payload), std::move(tensors));
}

} // namespace rpc
} // namespace distributed
} // namespace torch
