#include <torch/csrc/distributed/rpc/py_rref.h>

#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/distributed/rpc/rref_context.h>

namespace torch {
namespace distributed {
namespace rpc {

PyRRef::PyRRef(std::shared_ptr<RRef>& rref) : rref_(rref) {
  TORCH_CHECK(rref_, "PyRRef must not wrap nullptr");
}

worker_id_t PyRRef::owner() const {
  return rref_->owner();
}

py::object PyRRef::toHere() {
  return torch::jit::toPyObject(rref_->toHere());
}

py::tuple PyRRef::pickle() const {
  //auto& ctx = RRefContext::getInstance();
  AT_ERROR("Does not support pickle RRef");
  //return py::make_tuple(torch::jit::toPyObject(rref_->fork()));
}

PyRRef PyRRef::unpickle(py::tuple t) {
  auto& ctx = RRefContext::getInstance();
  auto rref = ctx->getOrCreateRRef<IValue>(t[0].cast<IValue>());
  return PyRRef(rref);
}

} // namespace rpc
} // namespace distributed
} // namespace torch
