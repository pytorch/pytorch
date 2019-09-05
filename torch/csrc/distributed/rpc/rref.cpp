#include <torch/csrc/distributed/rpc/rref.h>

#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/script_rref_proto.h>

namespace torch {
namespace distributed {
namespace rpc {

std::atomic<local_id_t> RRefContext::nextLocalId_{0};

//////////////////////////  RRefForkData  /////////////////////////////////

RRefForkData::RRefForkData(
    worker_id_t ownerId,
    const RRefId& rrefId,
    const ForkId& forkId)
    : ownerId_(ownerId), rrefId_(rrefId), forkId_(forkId) {}

at::IValue RRefForkData::toIValue() const {
  std::vector<at::IValue> ivalues = {
      (int64_t)ownerId_, rrefId_.toIValue(), forkId_.toIValue()};

  return c10::ivalue::Tuple::create(std::move(ivalues));
}

RRefForkData RRefForkData::fromIValue(const at::IValue& ivalue) {
  auto ivalues = ivalue.toTuple()->elements();

  TORCH_CHECK(
      ivalues.size() == 3,
      "Constructing RRefForkData from ivalue "
      "expects a GenericList of 3 elements, but got ",
      ivalues.size());

  int64_t ownerId = ivalues[0].toInt();
  TORCH_CHECK(
      ownerId < std::numeric_limits<worker_id_t>::max(),
      "RRefId createdOn out of range, got ",
      ownerId);

  RRefId rrefId = RRefId::fromIValue(ivalues[1]);
  ForkId forkId = ForkId::fromIValue(ivalues[2]);

  return RRefForkData(ownerId, rrefId, forkId);
}

//////////////////////////////  RRef  /////////////////////////////////////

RRef::RRef(worker_id_t ownerId, const RRefId& rrefId)
    : ownerId_(ownerId), rrefId_(rrefId) {}

worker_id_t RRef::owner() const {
  return ownerId_;
}

const RRefId& RRef::id() const {
  return rrefId_;
}

RRefForkData RRef::fork() const {
  return RRefForkData(
      ownerId_, rrefId_, RRefContext::getInstance()->genRRefId());
  // NB: does not support sharing RRefs between users
  // TODO: notify the owner
}

//////////////////////////  UserRRef  /////////////////////////////////////

template <typename T>
UserRRef<T>::UserRRef(
    worker_id_t ownerId,
    const RRefId& rrefId,
    const ForkId& forkId)
    : RRef(ownerId, rrefId), forkId_(forkId) {
  AT_ASSERT(
      !(forkId_ == rrefId_),
      "User RRef's fork ID should not be the same as its rref Id");
  // Do nothing,
  // (1) If this UserRRef is shared from another UserRRef x, x should
  // notified the owner on my behalf.
  // (2) If this UserRRef is shared from the OwnerRRef, the OwnerRRef already
  // knows this UserRRef.
  // (3) If this the creator UserRRef, ScriptRemoteCall will properly notify
  // the owner.
}

template <typename T>
UserRRef<T>::~UserRRef() {
  auto& ctx = RRefContext::getInstance();
  if (ctx->getWorkerId() != ownerId_) {
    ctx->agent()->send(
        ctx->agent()->getWorkerId(ownerId_),
        ScriptUserDelete(RRefForkData(ownerId_, rrefId_, forkId_).toIValue())
            .toMessage());
  }
}

template <typename T>
const ForkId& UserRRef<T>::forkId() const {
  return forkId_;
}

template <typename T>
bool UserRRef<T>::isOwner() const {
  return false;
}

template <typename T>
bool UserRRef<T>::isPyObj() {
  return std::is_same<T, py::object>::value;
}

template <>
IValue UserRRef<IValue>::toHere() {
  auto& agent = RRefContext::getInstance()->agent();
  std::shared_ptr<FutureMessage> fm = agent->send(
      agent->getWorkerId(ownerId_),
      ScriptRRefFetchCall(id().toIValue()).toMessage());
  auto srv = ScriptRRefFetchRet::fromMessage(fm->wait());
  return srv.value();
}

template <>
py::object UserRRef<py::object>::toHere() {
  auto& agent = RRefContext::getInstance()->agent();
  std::shared_ptr<FutureMessage> fm = agent->send(
      agent->getWorkerId(ownerId_),
      PythonRRefFetchCall(id().toIValue()).toMessage());
  auto srv = ScriptRRefFetchRet::fromMessage(fm->wait());
  return PythonRpcHandler::deserialize(srv.value().toStringRef());
}

template class UserRRef<IValue>;
template class UserRRef<py::object>;

//////////////////////////  OwnerRRef  /////////////////////////////////////

template <typename T>
bool OwnerRRef<T>::isOwner() const {
  return true;
}

template <typename T>
T OwnerRRef<T>::getValue() const {
  // TODO: use callback to make this non-blocking
  std::unique_lock<std::mutex> lock(mutex_);
  valueCV_.wait(lock, [this] { return value_.has_value(); });
  return value_.value();
}

template <typename T>
void OwnerRRef<T>::setValue(T&& value) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    value_ = std::move(value);
  }
  valueCV_.notify_all();
}

template <typename T>
bool OwnerRRef<T>::isPyObj() {
  return std::is_same<T, py::object>::value;
}

template class OwnerRRef<IValue>;
template class OwnerRRef<py::object>;

} // namespace rpc
} // namespace distributed
} // namespace torch
