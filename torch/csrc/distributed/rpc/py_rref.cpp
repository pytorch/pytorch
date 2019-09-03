#include <torch/csrc/distributed/rpc/py_rref.h>

#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/distributed/rpc/rref_context.h>

namespace torch {
namespace distributed {
namespace rpc {

thread_local worker_id_t PyRRef::currentDst = -1;

PyRRef::PyRRef(std::shared_ptr<RRef> rref) : rref_(std::move(rref)) {
  TORCH_CHECK(rref_, "PyRRef must not wrap nullptr");
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
  return py::make_tuple(torch::jit::toPyObject(ctx->forkTo(rref_, currentDst)));
}

PyRRef PyRRef::unpickle(py::tuple t) {
  auto& ctx = RRefContext::getInstance();
  auto rref = ctx->getOrCreateRRef<IValue>(t[0].cast<IValue>());
  return PyRRef(rref);
}

} // namespace rpc
} // namespace distributed
} // namespace torch
