#include <torch/csrc/distributed/rpc/py_rref.h>

#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/jit/pybind_utils.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

// Constants below are used in PyRRef pickling and unpickling. PyRRef is
// converted into a py::tuple in pickling, and reconstructed from the py::tuple
// in pickling.
constexpr int RREF_TUPLE_SIZE = 6;  // number of data fields in the py::tuple
constexpr int OWNER_IDX = 0;        // index of ownerId in the tuple
constexpr int RREFID_ON_IDX = 1;    // index of RRefId.createdOn_ in the tuple
constexpr int RREFID_ID_IDX = 2;    // index of RRefId.localId_ in the tuple
constexpr int FORKID_ON_IDX = 3;    // index of ForkId.createdOn_ in the tuple
constexpr int FORKID_ID_IDX = 4;    // index of ForkId.localId_ in the tuple
constexpr int TYPE_IDX = 5;         // index of type (py::object or IValue)

} // namespace

thread_local worker_id_t PyRRef::currentDst = -1;

PyRRef::PyRRef(std::shared_ptr<RRef> rref) : rref_(std::move(rref)) {
  TORCH_CHECK(rref_, "PyRRef must not wrap nullptr");
}

bool PyRRef::isOwner() const {
  return rref_->isOwner();
}

worker_id_t PyRRef::owner() const {
  return rref_->owner();
}

py::object PyRRef::toHere() {
  if (rref_->isPyObj()) {
    if (rref_->isOwner()) {
      return std::static_pointer_cast<OwnerRRef<py::object>>(rref_)->getValue();
    } else {
      return std::static_pointer_cast<UserRRef<py::object>>(rref_)->toHere();
    }
  } else {
    IValue value;
    if (rref_->isOwner()) {
      value = std::static_pointer_cast<OwnerRRef<IValue>>(rref_)->getValue();
    } else {
      value = std::static_pointer_cast<UserRRef<IValue>>(rref_)->toHere();
    }
    AutoGIL ag;
    return torch::jit::toPyObject(std::move(value));
  }
}

py::object PyRRef::localValue() {
  if (!rref_->isOwner()) {
    auto& ctx = RRefContext::getInstance();
    AT_ERROR(
        "Cannot call localValue() on a non-local reference. Call it on ",
        ctx->getWorkerName()
    );
  }

  if (rref_->isPyObj()) {
    return std::dynamic_pointer_cast<OwnerRRef<py::object>>(rref_)->getValue();
  } else {
    auto value =
        std::dynamic_pointer_cast<OwnerRRef<IValue>>(rref_)->getValue();
    AutoGIL ag;
    return torch::jit::toPyObject(std::move(value));
  }
}

py::tuple PyRRef::pickle() const {
  auto& ctx = RRefContext::getInstance();
  auto rfd = ctx->forkTo(rref_, currentDst);
  auto& ownerId = rfd.ownerId_;
  auto& rrefId = rfd.rrefId_;
  auto& forkId = rfd.forkId_;
  bool isPyObj = rref_->isPyObj();

  auto t = py::make_tuple(
      ownerId,
      rrefId.createdOn_,
      rrefId.localId_,
      forkId.createdOn_,
      forkId.localId_,
      isPyObj);
  return t;
}

PyRRef PyRRef::unpickle(const py::tuple& t) {
  TORCH_INTERNAL_ASSERT(
      t.size() == RREF_TUPLE_SIZE, "Pickled RRef must contain 6 numbers.");
  auto& ctx = RRefContext::getInstance();
  worker_id_t ownerId = t[OWNER_IDX].cast<worker_id_t>();
  RRefId rrefId = RRefId(
      t[RREFID_ON_IDX].cast<worker_id_t>(),
      t[RREFID_ID_IDX].cast<local_id_t>());
  RRefId forkId = RRefId(
      t[FORKID_ON_IDX].cast<worker_id_t>(),
      t[FORKID_ID_IDX].cast<local_id_t>());
  bool isPyObj = t[TYPE_IDX].cast<bool>();
  if (isPyObj) {
    return PyRRef(ctx->getOrCreateRRef<py::object>(ownerId, rrefId, forkId));
  } else {
    return PyRRef(ctx->getOrCreateRRef<IValue>(ownerId, rrefId, forkId));
  }
}

void PyRRef::setCurrentDst(worker_id_t dst) {
  currentDst = dst;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
