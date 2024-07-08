#include <torch/csrc/lazy/python/python_util.h>

#include <Python.h>
#include <frameobject.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/lazy/core/debug_util.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_compat.h>
#include <torch/csrc/utils/python_strings.h>

namespace torch {
namespace lazy {

std::optional<SourceLocation> GetPythonFrameTop() {
  if (!Py_IsInitialized()) {
    return c10::nullopt;
  }
  pybind11::gil_scoped_acquire gil;
  PyFrameObject* frame = PyEval_GetFrame();
  if (frame == nullptr) {
    return c10::nullopt;
  }
  SourceLocation loc;
  auto code = THPCodeObjectPtr(PyFrame_GetCode(frame));
  loc.line = PyFrame_GetLineNumber(frame);
  loc.file = THPUtils_unpackString(code->co_filename);
  loc.function = THPUtils_unpackString(code->co_name);
  return loc;
}

std::vector<SourceLocation> GetPythonFrames() {
  std::vector<SourceLocation> frames;
  if (Py_IsInitialized()) {
    pybind11::gil_scoped_acquire gil;
    PyFrameObject* frame = PyEval_GetFrame();
    if (frame != nullptr) {
      Py_INCREF(frame);
    }
    while (frame != nullptr) {
      SourceLocation loc;
      auto code = THPCodeObjectPtr(PyFrame_GetCode(frame));
      loc.line = PyFrame_GetLineNumber(frame);
      loc.file = THPUtils_unpackString(code->co_filename);
      loc.function = THPUtils_unpackString(code->co_name);
      frames.push_back(std::move(loc));
      auto new_frame = PyFrame_GetBack(frame);
      Py_DECREF(frame);
      frame = new_frame;
    }
  }
  return frames;
}

} // namespace lazy
} // namespace torch
