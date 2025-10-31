#pragma once

#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils.h>

namespace c10d {

class CallbackWork : public Work {
 public:
  explicit CallbackWork(py::function callback)
      : callback_(std::move(callback)) {}

  ~CallbackWork() override;

  bool isCompleted() override {
    return false;
  }

  bool isSuccess() const override {
    return false;
  }

  std::exception_ptr exception() const override {
    return nullptr;
  }

  int sourceRank() const override {
    TORCH_CHECK(false, "CallbackWork::sourceRank() not implemented");
  }

  std::vector<at::Tensor> result() override {
    return {};
  }

  bool wait(std::chrono::milliseconds timeout) override {
    return true;
  }

  void abort() override {
    TORCH_CHECK(false, "CallbackWork::abort() not implemented");
  }

  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
    TORCH_CHECK(false, "CallbackWork::getFuture() not implemented");
  }

 private:
  py::function callback_;
};

} // namespace c10d
