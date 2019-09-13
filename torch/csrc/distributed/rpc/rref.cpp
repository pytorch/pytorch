#include <torch/csrc/distributed/rpc/rref.h>

#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/rref_proto.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

constexpr int RFD_TUPLE_SIZE = 5; // number of RRefForkData fields in py::tuple
constexpr int OWNER_IDX = 0; // index of ownerId in the tuple
constexpr int RREFID_ON_IDX = 1; // index of RRefId.createdOn_ in the tuple
constexpr int RREFID_ID_IDX = 2; // index of RRefId.localId_ in the tuple
constexpr int FORKID_ON_IDX = 3; // index of ForkId.createdOn_ in the tuple
constexpr int FORKID_ID_IDX = 4; // index of ForkId.localId_ in the tuple

} // namespace

std::atomic<local_id_t> RRefContext::nextLocalId_{0};

//////////////////////////  RRefForkData  /////////////////////////////////

RRefForkData::RRefForkData(
    worker_id_t ownerId,
    const RRefId& rrefId,
    const ForkId& forkId)
    : ownerId_(ownerId), rrefId_(rrefId), forkId_(forkId) {}

at::IValue RRefForkData::toIValue() const {
  return c10::ivalue::Tuple::create(
      {(int64_t)ownerId_, rrefId_.toIValue(), forkId_.toIValue()});
}

py::tuple RRefForkData::toPyTuple() const {
  return py::make_tuple(
      ownerId_,
      rrefId_.createdOn_,
      rrefId_.localId_,
      forkId_.createdOn_,
      forkId_.localId_);
}

RRefForkData RRefForkData::fromPyTuple(py::tuple t) {
  TORCH_INTERNAL_ASSERT(
      t.size() == RFD_TUPLE_SIZE,
      "Pickled RRefForkData must contain 5 numbers.");
  worker_id_t ownerId = t[OWNER_IDX].cast<worker_id_t>();
  RRefId rrefId = RRefId(
      t[RREFID_ON_IDX].cast<worker_id_t>(),
      t[RREFID_ID_IDX].cast<local_id_t>());
  RRefId forkId = RRefId(
      t[FORKID_ON_IDX].cast<worker_id_t>(),
      t[FORKID_ID_IDX].cast<local_id_t>());
  return RRefForkData(ownerId, rrefId, forkId);
}

RRefForkData RRefForkData::fromIValue(const at::IValue& ivalue) {
  auto ivalues = ivalue.toTuple()->elements();

  TORCH_INTERNAL_ASSERT(
      ivalues.size() == 3,
      "Constructing RRefForkData from ivalue "
      "expects a GenericList of 3 elements, but got ",
      ivalues.size());

  int64_t ownerId = ivalues[0].toInt();
  TORCH_INTERNAL_ASSERT(
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

RRefForkData RRef::fork() const {
  return RRefForkData(
      ownerId_, rrefId_, RRefContext::getInstance()->genGloballyUniqueId());
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
  TORCH_INTERNAL_ASSERT(
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
  // TODO: queue this in RRefContext instead of doing it here.
  auto& ctx = RRefContext::getInstance();
  if (ctx->getWorkerId() != ownerId_) {
    auto fm = ctx->agent()->send(
        ctx->agent()->getWorkerInfo(ownerId_),
        RRefUserDelete(rrefId_, forkId_).toMessage());

    fm->addCallback(
        [](const Message& message) { RRefContext::handleException(message); });
  }
}

template <typename T>
const ForkId& UserRRef<T>::forkId() const {
  return forkId_;
}

template <>
IValue UserRRef<IValue>::toHere() {
  auto& agent = RRefContext::getInstance()->agent();
  std::shared_ptr<FutureMessage> fm = agent->send(
      agent->getWorkerInfo(ownerId_),
      ScriptRRefFetchCall(rrefId()).toMessage());
  const Message& message = fm->wait();
  RRefContext::handleException(message);
  auto srv = RRefFetchRet::fromMessage(message);
  return srv.value();
}

template <>
py::object UserRRef<py::object>::toHere() {
  auto& agent = RRefContext::getInstance()->agent();
  std::shared_ptr<FutureMessage> fm = agent->send(
      agent->getWorkerInfo(ownerId_),
      PythonRRefFetchCall(rrefId()).toMessage());
  const Message& message = fm->wait();
  RRefContext::handleException(message);
  auto srv = RRefFetchRet::fromMessage(message);
  return PythonRpcHandler::deserialize(srv.value().toStringRef());
}

template class UserRRef<IValue>;
template class UserRRef<py::object>;

//////////////////////////  OwnerRRef  /////////////////////////////////////

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

template class OwnerRRef<IValue>;
template class OwnerRRef<py::object>;

} // namespace rpc
} // namespace distributed
} // namespace torch
