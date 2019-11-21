#pragma once

#include <torch/csrc/distributed/rpc/rref.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace distributed {
namespace rpc {

// Python wrapper of an RRef shared_ptr that supports Python
// pickle and unpickle.
class PyRRef {
 public:
  explicit PyRRef(std::shared_ptr<RRef> rref);
  // creates a local RRef with the given object as value
  explicit PyRRef(const py::object& value);

  bool isOwner() const;
  WorkerInfo owner() const;
  py::object toHere();
  py::object localValue();
  py::tuple pickle() const;
  static PyRRef unpickle(const py::tuple& t);

 private:
  std::shared_ptr<RRef> rref_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
