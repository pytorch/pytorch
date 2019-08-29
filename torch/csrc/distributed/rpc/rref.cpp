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
    : ownerId_(ownerId), rrefId_(rrefId), forkId_(forkId) {
  auto& ctx = RRefContext::getInstance();

  if (ownerId == ctx->getWorkerId()) {
    // This is the owner RRef
    AT_ASSERT(forkId_ == rrefId_,
        "Owner RRef's fork ID should be the same as its rref Id");
    // only owner RRef keeps track of forks
    children_fork_ids = std::unordered_set<ForkId, ForkId::Hash>();
  } else {
    // This is the user RRef
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
}

RRef::~RRef() {
  auto& ctx = RRefContext::getInstance();
  if (ctx->getWorkerId() != ownerId_) {
    ctx->agent()->send(
        ctx->agent()->getWorkerId(ownerId_),
        ScriptRRefDel(
            RRefForkData(ownerId_, rrefId_, forkId_).toIValue()
        ).toMessage());
  }
}

worker_id_t RRef::owner() const {
  return ownerId_;
}

const RRefId& RRef::id() const {
  return rrefId_;
}

const ForkId& RRef::forkId() const {
  return forkId_;
}

bool RRef::isOwner() const {
  return RRefContext::getInstance()->getWorkerId() == ownerId_;
}

IValue RRef::toHere() {
  auto& ctx = RRefContext::getInstance();
  if (owner() == ctx->getWorkerId()) {
    return getValue();
  } else {
    auto& agent = ctx->agent();
    std::shared_ptr<FutureMessage> fm =
        agent->send(
            agent->getWorkerId(ownerId_),
            ScriptRRefFetch(id().toIValue()).toMessage()
        );
    auto srv = ScriptRRefValue::fromMessage(fm->wait());
    return srv.value();
  }
}

at::IValue RRef::fork() const {
  return RRefForkData(
      ownerId_, rrefId_, RRefContext::getInstance()->genRRefId()
  ).toIValue();
  // NB: does not support sharing RRefs between users
  // TODO: notify the owner
}

//////////////////////////  RRefImpl  /////////////////////////////////////

} // namespace rpc
} // namespace distributed
} // namespace torch
