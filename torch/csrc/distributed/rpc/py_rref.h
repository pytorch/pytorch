#pragma once

#include <torch/csrc/distributed/rpc/rref_impl.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace distributed {
namespace rpc {

enum RRefProxyType { RPC_SYNC, RPC_ASYNC, REMOTE };

// Python wrapper of an RRef shared_ptr that supports Python
// pickle and unpickle.
class PyRRef {
 public:
  explicit PyRRef(const py::object& value, const py::object& type_hint);
  explicit PyRRef(c10::intrusive_ptr<RRef> rref);

  bool isOwner() const;
  bool confirmedByOwner() const;
  WorkerInfo owner() const;
  std::string ownerName() const;
  py::object toHere();
  py::object localValue();
  std::string str() const;
  py::tuple pickle() const;
  static PyRRef unpickle(const py::tuple& t);
  c10::IValue toIValue();
  // Future that is associated with the creation of this RRef on the remote end.
  // This is only used to get the future corresponding to the rref for profiling
  // use cases.
  const std::shared_ptr<FutureIValue> getFuture() const;

  // create a proxy on this RRef, which can be used to launch RPC on the owner
  // of this RRef to run functions on the object referenced by this RRef.
  py::object createRRefProxy(PyRRef& self, const RRefProxyType& mode) const;

 private:
  c10::intrusive_ptr<RRef> rref_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
