#include <torch/csrc/distributed/rpc/py_rref.h>

#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/distributed/rpc/rref_context.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

py::object toPyObj(const IValue&) {

}

IValue fromPyObj(const py::object& pyObj) {

}

}

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
  if (rref_->isOwner()) {
    AT_ERROR("Cannot call toHere() on OwnerRRef, use localValue() instead.");
  }
  if (rref_->isPyObj()) {
    return std::dynamic_pointer_cast<UserRRef<py::object>>(rref_)->toHere();
  } else {
    return torch::jit::toPyObject(
        std::dynamic_pointer_cast<UserRRef<IValue>>(rref_)->toHere());
  }
}

py::object PyRRef::localValue() {
  if (!rref_->isOwner()) {
    AT_ERROR("Cannot call localValue() on UserRRef");
  }

  if (rref_->isPyObj()) {
    return std::dynamic_pointer_cast<OwnerRRef<py::object>>(rref_)->getValue();
  } else {
    return torch::jit::toPyObject(
        std::dynamic_pointer_cast<OwnerRRef<IValue>>(rref_)->getValue());
  }
}

py::tuple PyRRef::pickle() const {
  auto& ctx = RRefContext::getInstance();
  auto rfd = ctx->forkTo(rref_, currentDst);
  auto& ownerId = rfd.ownerId_;
  auto& rrefId = rfd.rrefId_;
  auto& forkId = rfd.forkId_;

  auto t = py::make_tuple(
      ownerId,
      rrefId.createdOn_, rrefId.localId_,
      forkId.createdOn_, forkId.localId_
  );
  return t;
}

PyRRef PyRRef::unpickle(const py::tuple& t) {
  TORCH_INTERNAL_ASSERT(t.size() == 5, "Pickled RRef must contain 5 numbers.");
  auto& ctx = RRefContext::getInstance();
  worker_id_t ownerId = t[0].cast<worker_id_t>();
  RRefId rrefId = RRefId(t[1].cast<worker_id_t>(), t[2].cast<local_id_t>());
  RRefId forkId = RRefId(t[3].cast<worker_id_t>(), t[4].cast<local_id_t>());
  auto rref = ctx->getOrCreateRRef<IValue>(ownerId, rrefId, forkId);
  return PyRRef(rref);
}

} // namespace rpc
} // namespace distributed
} // namespace torch
