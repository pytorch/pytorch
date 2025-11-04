#pragma once

#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils.h>

namespace c10d {

// PythonCallbackWork is a subclass of Work that wraps a Python callback
// function that implements wait(). This allows asynchronous work to
// be integrated with Python code, enabling custom completion logic or
// post-processing in Python.
class PythonCallbackWork : public Work {
 public:
  explicit PythonCallbackWork(py::function callback);

  ~PythonCallbackWork() override;

  bool wait(std::chrono::milliseconds timeout) override;

  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

 private:
  py::function callback_;
  c10::intrusive_ptr<c10::ivalue::Future> future_;
};

} // namespace c10d
