#include <torch/csrc/distributed/rpc/rref_impl.h>
#include <ATen/record_function.h>

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
      return type->annotation_str();
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

void RRef::handleError(
    RPCErrorType errorType,
    const FutureMessage& futMessage) {
  static std::unordered_map<
      RPCErrorType,
      std::function<void(const FutureMessage& fm)>,
      std::hash<int>>
      errorHandlers = {
          {RPCErrorType::TIMEOUT,
           [this](const FutureMessage& /* unused */) { setTimedOut(); }},
          {RPCErrorType::INTENTIONAL_FAILURE,
           [this](const FutureMessage& /* unused */) { setTimedOut(); }},
          {RPCErrorType::UNKNOWN_ERROR, [](const FutureMessage& fm) {
             // Default error handler
             RRefContext::handleException(fm);
           }}};
  errorHandlers.find(errorType)->second(futMessage);
}

//////////////////////////  UserRRef  /////////////////////////////////////

UserRRef::UserRRef(
    worker_id_t ownerId,
    const RRefId& rrefId,
    const ForkId& forkId,
    TypePtr type)
    : RRef(ownerId, rrefId, std::move(type)),
      forkId_(forkId),
      confirmedByOwner_(false) {
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
      LOG(ERROR) << "Error occurred when deleting" << *this << " : "
                 << ex.what();
    } catch (...) {
      LOG(ERROR) << "Error occurred when deleting" << *this << " : "
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

IValue UserRRef::toHere(const float timeoutSeconds) const {
  TORCH_CHECK(
      !getTimedOut(),
      "RRef creation via rpc.remote() timed out, and it "
      "is possible that the RRef on the owner node does not exist.");
  // see Note [Best-Effort Check on Deleted UserRRefs]
  RECORD_USER_SCOPE("to_here");
  TORCH_CHECK(
      !deletedOnOwner_,
      *this,
      " has been deleted. Cannot call to_here() on it after deletion.");
  TORCH_CHECK(
      !type_->is_module(),
      *this,
      " is an RRef to a ScriptModule. "
      "It can't be sent through RPC "
      "from owner, ",
      ownerWorkerInfo(),
      ", to user, ",
      RpcAgent::getCurrentRpcAgent()->getWorkerInfo(),
      ".");

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
      true /* forceGradRecording */,
      timeoutSeconds);

  // TODO: we should ideally be able to interrupt this blocking wait if we check
  // getTimedOut() and it is true
  // (https://github.com/pytorch/pytorch/issues/39411).
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

RRefForkData UserRRef::fork() const {
  // Note [Best-Effort Check on Deleted UserRRefs]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // This check does not guarantee correctness, as there could be another thread
  // trying to delete this UserRRef concurrently. Passing this check does not
  // mean this RRef will be alive throughout this function. This is just our
  // best-effort attempt to raise proper error messages. The behavior of using
  // deleted UserRRefs is undefined.
  //
  // The reason for not implementing strict checks are:
  // 1. This would need to acquire lock on deletedOnOwnerMutex_, which would
  //    introduce unnecessary overhead for most normal use cases.
  // 2. This would introduce a lot of complexities to get the behavior correct.
  //    Assume we acquired the lock here, and there is another thread X block
  //    waiting in tryDel() on the lock. Exiting this fork function would
  //    unblock thread X. However, while X proceeds with deleting this UserRRef,
  //    the call site of fork() might have added the UserRRef to
  //    pendingChildren_ map, but up to this point, nothing prevents X from
  //    deleting this RRef even if it shouldn't do so due to the state change
  //    in pendingChildren_. We might be able to get it right for now by locking
  //    and checking pendingChildren_ in X, but the gain does not seem to
  //    worth the complexity.
  TORCH_CHECK(
      !deletedOnOwner_,
      *this,
      " has been deleted. Cannot call fork an UserRRef after deletion.");
  return RRef::fork();
}

//////////////////////////  OwnerRRef  /////////////////////////////////////

const IValue& OwnerRRef::getValue() const {
  TORCH_CHECK(
      !getTimedOut(),
      "RRef creation via rpc.remote() timed out, and it "
      "is possible that the RRef on the owner node does not exist.");
  future_->wait();
  if (future_->hasError()) {
    (void)future_->value(); // Throws the error.
  }
  return future_->constValue();
}

bool OwnerRRef::hasValue() const {
  return future_->completed();
}

std::shared_ptr<JitFuture> OwnerRRef::getFuture() {
  return future_;
}

void OwnerRRef::setValue(IValue&& value) {
  future_->markCompleted(value);
}

void OwnerRRef::setError(const std::string& error) {
  future_->setErrorIfNeeded(error);
}

std::ostream& operator<<(std::ostream& os, const RRef& rref) {
  if (rref.isOwner()) {
    return os << "OwnerRRef("
              << "rref_id=" << rref.rrefId() << ")";
  } else {
    return os << "UserRRef("
              << "rref_id=" << rref.rrefId()
              << ", fork_id=" << static_cast<const UserRRef*>(&rref)->forkId()
              << ")";
  }
}

} // namespace rpc
} // namespace distributed
} // namespace torch
