#pragma once

#include <torch/csrc/distributed/rpc/rref.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace distributed {
namespace rpc {


class PyRRef {
 public:
  PyRRef(std::shared_ptr<RRef>& rref);

  worker_id_t owner() const;
  py::object toHere();
  py::tuple pickle() const;
  static PyRRef unpickle(py::tuple t);

 private:
  std::shared_ptr<RRef> rref_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
