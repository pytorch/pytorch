#pragma once

#include <pybind11/pybind11.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/profiler/unwind/unwind.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

namespace torch {

struct CapturedTraceback : public c10::GatheredContext {
  struct PyFrame {
    PyCodeObject* code;
    int lasti;
  };

  static std::shared_ptr<CapturedTraceback> gather(
      bool python,
      bool script,
      bool cpp);
  CapturedTraceback() = default;
  CapturedTraceback(const CapturedTraceback&) = delete;
  CapturedTraceback& operator=(const CapturedTraceback&) = delete;
  ~CapturedTraceback();

 private:
  std::vector<PyFrame> frames_;
  std::vector<void*> cpp_frames_;
  std::vector<jit::StackEntry> script_frames_;
  friend std::vector<py::object> symbolize(
      std::vector<CapturedTraceback*> to_symbolize);
  friend class std::shared_ptr<CapturedTraceback>;
};

std::vector<py::object> symbolize(std::vector<CapturedTraceback*> to_symbolize);

} // namespace torch
