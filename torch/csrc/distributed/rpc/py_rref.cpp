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
constexpr int RREF_TUPLE_SIZE = 2; // number of data fields in the py::tuple
constexpr int RFD_IDX = 0; // index of RRefForkData
constexpr int TYPE_IDX = 1; // index of type (py::object or IValue)

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
        ctx->getWorkerName());
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
  return py::make_tuple(rfd.toPyTuple(), rref_->isPyObj());
}

PyRRef PyRRef::unpickle(const py::tuple& t) {
  TORCH_INTERNAL_ASSERT(
      t.size() == RREF_TUPLE_SIZE, "Pickled RRef must contain 6 numbers.");
  auto& ctx = RRefContext::getInstance();
  auto rfd = RRefForkData::fromPyTuple(t[RFD_IDX].cast<py::tuple>());
  bool isPyObj = t[TYPE_IDX].cast<bool>();
  if (isPyObj) {
    return PyRRef(ctx->getOrCreateRRef<py::object>(rfd));
  } else {
    return PyRRef(ctx->getOrCreateRRef<IValue>(rfd));
  }
}

worker_id_t PyRRef::setCurrentDst(worker_id_t dst) {
  auto previousDst = currentDst;
  currentDst = dst;
  return previousDst;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
