#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/rref.h>
#include <torch/csrc/distributed/rpc/script_rref_proto.h>


namespace torch {
namespace distributed {
namespace rpc {

std::atomic<local_id_t> RRefContext::nextLocalId_ {0};

//////////////////////////  RRefForkData  /////////////////////////////////

RRefForkData::RRefForkData(
    worker_id_t ownerId, const RRefId& rrefId, const ForkId& forkId)
    : ownerId_(ownerId), rrefId_(rrefId), forkId_(forkId) {}

at::IValue RRefForkData::toIValue() const {
  std::vector<at::IValue> ivalues = {
      (int64_t) ownerId_,
      rrefId_.toIValue(),
      forkId_.toIValue()
  };

  return c10::ivalue::Tuple::create(std::move(ivalues));
}

RRefForkData RRefForkData::fromIValue(const at::IValue&& ivalue) {
  auto ivalues = ivalue.toTuple()->elements();

  TORCH_CHECK(ivalues.size() == 3, "Constructing RRefForkData from ivalue "
      "expects a GenericList of two elements, but got ", ivalues.size());

  int64_t ownerId = ivalues[0].toInt();
  TORCH_CHECK(ownerId < std::numeric_limits<worker_id_t>::max(),
      "RRefId createdOn out of range, got ", ownerId);

  RRefId rrefId = RRefId::fromIValue(std::move(ivalues[1]));
  ForkId forkId = ForkId::fromIValue(std::move(ivalues[2]));

  return RRefForkData(ownerId, rrefId, forkId);
}

//////////////////////////////  RRef  /////////////////////////////////////

RRef::RRef(worker_id_t ownerId, const RRefId& rrefId, const ForkId& forkId)
    : ownerId_(ownerId), rrefId_(rrefId), forkId_(forkId) {}

worker_id_t RRef::owner() const {
  return ownerId_;
}

const RRefId& RRef::id() const {
  return rrefId_;
}

const ForkId& RRef::forkId() const {
  return forkId_;
}

at::IValue RRef::fork() const {
  return RRefForkData(
      ownerId_, rrefId_, RRefContext::getInstance()->genRRefId()
  ).toIValue();
  // NB: does not support sharing RRefs between users
  // TODO: notify the owner
}

//////////////////////////  UserRRef  /////////////////////////////////////


UserRRef::UserRRef(
    worker_id_t ownerId, const RRefId& rrefId, const ForkId& forkId)
    : RRef(ownerId, rrefId, forkId) {
  AT_ASSERT(!(forkId_ == rrefId_),
      "User RRef's fork ID should not be the same as its rref Id");
  if (RRefContext::getInstance()->getWorkerId() == rrefId_.createdOn_) {
    // creator user, notify owner.
    auto& agent = RRefContext::getInstance()->agent();
    agent->send(
        agent->getWorkerId(ownerId_),
        ScriptRRefAdd(
            RRefForkData(ownerId_, rrefId_, forkId_).toIValue()
        ).toMessage());
  } else {
    AT_ERROR("Does not support sharing RRefs between users yet");
  }
}

UserRRef::~UserRRef() {
  auto& ctx = RRefContext::getInstance();
  if (ctx->getWorkerId() != ownerId_) {
    ctx->agent()->send(
        ctx->agent()->getWorkerId(ownerId_),
        ScriptRRefDel(
            RRefForkData(ownerId_, rrefId_, forkId_).toIValue()
        ).toMessage());
  }
}

bool UserRRef::isOwner() const {
  return false;
}

IValue UserRRef::getValue() const {
  AT_ERROR("UserRRef does not support getValue(), use toHere() instead.");
}

void UserRRef::setValue(IValue&& value) {
  AT_ERROR("UserRRef does not support setValue.");
}

IValue UserRRef::toHere() {
  auto& agent = RRefContext::getInstance()->agent();
  std::shared_ptr<FutureMessage> fm =
      agent->send(
          agent->getWorkerId(ownerId_),
          ScriptRRefFetch(id().toIValue()).toMessage()
      );
  auto srv = ScriptRRefValue::fromMessage(fm->wait());
  return srv.value();
}

} // namespace rpc
} // namespace distributed
} // namespace torch
