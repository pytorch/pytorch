#pragma once

#include <torch/csrc/distributed/rpc/rref.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace distributed {
namespace rpc {

struct PyFuture {
  virtual py::object wait() const = 0;
  virtual ~PyFuture() = default;
};

template <typename T>
class PyFutureToHere final : public PyFuture {
 public:
  explicit PyFutureToHere(std::shared_ptr<ivalue::Future> future);
  py::object wait() const override;
  ~PyFutureToHere() override = default;

 private:
  const std::shared_ptr<ivalue::Future> future_;
};

template <typename T>
class PyFutureLocalValue final : public PyFuture {
 public:
  explicit PyFutureLocalValue(std::shared_ptr<OwnerRRef<T>> rref);
  py::object wait() const override;
  ~PyFutureLocalValue() override = default;

 private:
  const std::shared_ptr<OwnerRRef<T>> rref_;
};

// Python wrapper of an RRef shared_ptr that supports Python
// pickle and unpickle.
class PyRRef {
 public:
  explicit PyRRef(std::shared_ptr<RRef> rref);

  bool isOwner() const;
  worker_id_t owner() const;
  std::shared_ptr<PyFuture> toHere();
  std::shared_ptr<PyFuture> localValue();
  py::tuple pickle() const;
  static PyRRef unpickle(const py::tuple& t);

 private:
  std::shared_ptr<RRef> rref_;
};

} // namespace rpc
} // namespace distributed
} // namespace torch
