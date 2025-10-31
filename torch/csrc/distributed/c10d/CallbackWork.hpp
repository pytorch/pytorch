#pragma once

#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils.h>

namespace c10d {

class CallbackWork : public Work {
 public:
  explicit CallbackWork(py::function callback);

  ~CallbackWork() override;

  bool wait(std::chrono::milliseconds timeout) override;

  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

 private:
  py::function callback_;
  c10::intrusive_ptr<c10::ivalue::Future> future_;
};

} // namespace c10d
