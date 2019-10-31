#include <torch/csrc/distributed/rpc/py_rref.h>

#include <torch/csrc/distributed/rpc/python_functions.h>
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/jit/pybind_utils.h>

namespace torch {
namespace distributed {
namespace rpc {

namespace {

// Constants below are used in PyRRef pickling and unpickling. PyRRef is
// converted into a py::tuple in pickling, and reconstructed from the py::tuple
// in unpickling.
constexpr int RFD_IDX = 0; // index of RRefForkData
constexpr int TYPE_IDX = 1; // index of type (py::object or IValue)

// number of data fields in the py::tuple.
// NB: if more fields are added, make sure this field is also bumped
constexpr int RREF_TUPLE_SIZE = 2;

} // namespace

//////////////////////////  PyFutureToHere  /////////////////////////////////

template <typename T>
PyFutureToHere<T>::PyFutureToHere(std::shared_ptr<ivalue::Future> future)
    : future_(std::move(future)) {}

template <>
py::object PyFutureToHere<IValue>::wait() const {
  future_->wait();
  {
    // acquiring GIL as torch::jit::toPyObject creates new py::object
    // without grabbing the GIL.
    AutoGIL ag;
    return torch::jit::toPyObject(future_->value());
  }
}

template <>
py::object PyFutureToHere<py::object>::wait() const {
  future_->wait();
  // PythonRpcHandler acquires the GIL on creating the py::object
  return PythonRpcHandler::getInstance().deserialize(
      SerializedPyObj::fromIValues(future_->value().toTuple()->elements()));
}

template class PyFutureToHere<IValue>;
template class PyFutureToHere<py::object>;

//////////////////////////  PyFutureLocalValue  ////////////////////////////////

template <typename T>
PyFutureLocalValue<T>::PyFutureLocalValue(std::shared_ptr<OwnerRRef<T>> rref)
    : rref_(std::move(rref)) {}

template <>
py::object PyFutureLocalValue<IValue>::wait() const {
  auto value = rref_->getValue();
  {
    // acquiring GIL as torch::jit::toPyObject creates new py::object without
    // grabbing the GIL.
    AutoGIL ag;
    return torch::jit::toPyObject(std::move(value));
  }
}

template <>
py::object PyFutureLocalValue<py::object>::wait() const {
  const py::object& value = rref_->getValue();
  {
    // acquiring GIL as the return statement construct a new py::object from
    // a const reference.
    AutoGIL ag;
    return value;
  }
}

template class PyFutureLocalValue<IValue>;
template class PyFutureLocalValue<py::object>;

///////////////////////////  PyRRef  //////////////////////////////////

PyRRef::PyRRef(std::shared_ptr<RRef> rref) : rref_(std::move(rref)) {
  TORCH_CHECK(rref_, "PyRRef must not wrap nullptr");
}

bool PyRRef::isOwner() const {
  return rref_->isOwner();
}

WorkerInfo PyRRef::owner() const {
  return RRefContext::getInstance().agent()->getWorkerInfo(rref_->owner());
}

std::shared_ptr<PyFuture> PyRRef::toHere() {
  if (rref_->isOwner()) {
    return localValue();
  } else {
    if (rref_->isPyObj()) {
      auto userRRef = std::static_pointer_cast<UserRRef<py::object>>(rref_);
      return std::make_shared<PyFutureToHere<py::object>>(userRRef->toHere());
    } else {
      auto userRRef = std::static_pointer_cast<UserRRef<IValue>>(rref_);
      return std::make_shared<PyFutureToHere<IValue>>(userRRef->toHere());
    }
  }
}

std::shared_ptr<PyFuture> PyRRef::localValue() {
  TORCH_CHECK(
      rref_->isOwner(),
      "Cannot call localValue() on a non-local reference. Call it on ",
      RRefContext::getInstance().getWorkerName());

  if (rref_->isPyObj()) {
    return std::make_shared<PyFutureLocalValue<py::object>>(
        std::dynamic_pointer_cast<OwnerRRef<py::object>>(rref_));
  } else {
    return std::make_shared<PyFutureLocalValue<IValue>>(
        std::dynamic_pointer_cast<OwnerRRef<IValue>>(rref_));
  }
}

py::tuple PyRRef::pickle() const {
  auto& ctx = RRefContext::getInstance();
  // TODO: use a dispatch table to pickle/unpickle an RRef, and only only
  // install the dispatch table only when there are indeed RPC activities. As
  // a counter example, checkpointing a model with RRefs should not trigger
  // forks to be added as a fork or a child.
  auto rfd = ctx.prepareChildFork(rref_);
  return py::make_tuple(rfd.toPyTuple(), rref_->isPyObj());
}

PyRRef PyRRef::unpickle(const py::tuple& t) {
  TORCH_INTERNAL_ASSERT(
      t.size() == RREF_TUPLE_SIZE, "Pickled RRef must contain 2 numbers.");
  auto& ctx = RRefContext::getInstance();
  auto rfd = RRefForkData::fromPyTuple(t[RFD_IDX].cast<py::tuple>());
  std::shared_ptr<RRef> rref = nullptr;
  bool isPyObj = t[TYPE_IDX].cast<bool>();
  if (isPyObj) {
    rref = ctx.getOrCreateRRef<py::object>(rfd);
  } else {
    rref = ctx.getOrCreateRRef<IValue>(rfd);
  }

  ctx.notifyOwnerAndParentOfFork(rfd.forkId_, rfd.parent_, rref);
  return PyRRef(std::move(rref));
}

PyRRef PyRRef::local(const py::object& value) {
  auto rref = RRefContext::getInstance().createOwnerRRef<py::object>();
  py::object copy(value); // increases refcount
  rref->setValue(std::move(copy));
  return PyRRef(std::move(rref));
}

} // namespace rpc
} // namespace distributed
} // namespace torch
