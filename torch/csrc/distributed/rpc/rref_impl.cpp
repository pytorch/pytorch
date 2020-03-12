#include <torch/csrc/distributed/rpc/rref_impl.h>

#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/rref_proto.h>
#include <torch/csrc/distributed/rpc/utils.h>

namespace {
// If the type is subtype of named type, return its qualifiedname, otherwise
// return its type str.
std::string getTypeStr(const c10::TypePtr& type) {
  switch (type->kind()) {
    case c10::TypeKind::FunctionType:
      return type->cast<c10::FunctionType>()->name()->qualifiedName();
    case c10::TypeKind::TupleType:
      return type->cast<c10::TupleType>()->name()->qualifiedName();
    case c10::TypeKind::ClassType:
      return type->cast<c10::ClassType>()->name()->qualifiedName();
    case c10::TypeKind::InterfaceType:
      return type->cast<c10::InterfaceType>()->name()->qualifiedName();
    default:
      return type->str();
  }
}
} // namespace

namespace torch {
namespace distributed {
namespace rpc {

std::atomic<local_id_t> RRefContext::nextLocalId_{0};

//////////////////////////  RRefForkData  /////////////////////////////////

RRefForkData::RRefForkData(
    worker_id_t ownerId,
    const RRefId& rrefId,
    const ForkId& forkId,
    worker_id_t parent,
    std::string typeStr)
    : ownerId_(ownerId),
      rrefId_(rrefId),
      forkId_(forkId),
      parent_(parent),
      typeStr_(std::move(typeStr)) {}

//////////////////////////////  RRef  /////////////////////////////////////

RRef::RRef(worker_id_t ownerId, const RRefId& rrefId, TypePtr type)
    : RRefInterface(),
      ownerId_(ownerId),
      rrefId_(rrefId),
      type_(std::move(type)) {}

RRefForkData RRef::fork() const {
  auto& ctx = RRefContext::getInstance();
  return RRefForkData(
      ownerId_,
      rrefId_,
      ctx.genGloballyUniqueId(),
      ctx.getWorkerId(),
      getTypeStr(type_));
}

//////////////////////////  UserRRef  /////////////////////////////////////

UserRRef::UserRRef(
    worker_id_t ownerId,
    const RRefId& rrefId,
    const ForkId& forkId,
    TypePtr type)
    : RRef(ownerId, rrefId, std::move(type)),
      forkId_(forkId),
      confirmed_(false) {
  // Do nothing,
  // (1) If this UserRRef is a fork of an existing RRef, RRefContext will send
  //     a RREF_FORK_REQUEST message to the owner.
  // (2) If this the creator UserRRef, ScriptRemoteCall or PythonRemoteCall will
  //     properly notify the owner.
}

void UserRRef::tryDel() {
  std::lock_guard<std::mutex> lockGuard(deletedOnOwnerMutex_);
  if (!deletedOnOwner_) {
    try {
      RRefContext::getInstance().delUser(ownerId_, rrefId_, forkId_);
      deletedOnOwner_ = true;
    } catch (const std::exception& ex) {
      LOG(ERROR) << "Error occurred when deleting UserRRef instance, "
                 << "RRefId = " << rrefId_ << ", ForkId = " << forkId_ << " : "
                 << ex.what();
    } catch (...) {
      LOG(ERROR) << "Error occurred when deleting UserRRef instance, "
                 << "RRefId = " << rrefId_ << ", ForkId = " << forkId_ << " : "
                 << "unknown error";
    }
  }
}

void UserRRef::release_resources() {
  tryDel();
}

const ForkId& UserRRef::forkId() const {
  return forkId_;
}

IValue UserRRef::toHere() {
  auto agent = RpcAgent::getCurrentRpcAgent();

  // ScriptRRefFetchCall message always carries autograd context id even if
  // the message itself does not contain any tensor, because the response would
  // potentially contain tensors.
  Message msgToSend;

  if (isPyObj()) {
    msgToSend = PythonRRefFetchCall(ownerId_, rrefId()).toMessage();
  } else {
    msgToSend = ScriptRRefFetchCall(ownerId_, rrefId()).toMessage();
  }

  auto futureResponse = autograd::sendMessageWithAutograd(
      *agent,
      agent->getWorkerInfo(ownerId_),
      std::move(msgToSend),
      true /* forceGradRecording */);

  const Message& message = futureResponse->wait();
  MessageType msgType = message.type();
  auto response = deserializeResponse(message, msgType);
  TORCH_INTERNAL_ASSERT(
      msgType == MessageType::SCRIPT_RREF_FETCH_RET ||
          msgType == MessageType::PYTHON_RREF_FETCH_RET,
      "Message type should either be SCRIPT_RREF_FETCH_RET "
      "or PYTHON_RREF_FETCH_RET");
  RpcCommandBase& rpc = *response;
  auto& rrefFetchRet = static_cast<RRefFetchRet&>(rpc);
  if (isPyObj()) {
    // wrap python serialized vector of ivalues into tuple, this
    // made the C++ toHere interface to return single IValue
    return ivalue::Tuple::create(rrefFetchRet.values());
  } else {
    return rrefFetchRet.values().front();
  }
}

//////////////////////////  OwnerRRef  /////////////////////////////////////

const IValue& OwnerRRef::getValue() const {
  std::unique_lock<std::mutex> lock(mutex_);
  valueCV_.wait(lock, [this] { return value_.has_value(); });
  return value_.value();
}

bool OwnerRRef::hasValue() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return value_.has_value();
}

std::shared_ptr<FutureMessage> OwnerRRef::getFuture() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (future_.get()) {
    return future_;
  }
  future_ = std::make_shared<FutureMessage>();
  std::shared_ptr<FutureMessage> ret = future_;
  if (value_.has_value()) {
    lock.unlock();
    ret->markCompleted(Message());
  }
  return ret;
}

void OwnerRRef::setValue(IValue&& value) {
  std::unique_lock<std::mutex> lock(mutex_);
  value_ = std::move(value);
  std::shared_ptr<FutureMessage> future;
  future.swap(future_);
  lock.unlock();
  valueCV_.notify_all();
  if (future.get() && !future->completed()) {
    future->markCompleted(Message());
  }
}

} // namespace rpc
} // namespace distributed
} // namespace torch
