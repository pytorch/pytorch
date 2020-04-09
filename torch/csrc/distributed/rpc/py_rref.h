#pragma once

#include <torch/csrc/distributed/rpc/rref_impl.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace distributed {
namespace rpc {

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

 private:
  c10::intrusive_ptr<RRef> rref_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
