#include <torch/csrc/lazy/python/python_util.h>

#include <Python.h>
#include <frameobject.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/lazy/core/debug_util.h>

namespace torch {
namespace lazy {

c10::optional<SourceLocation> GetPythonFrameTop() {
  if (!Py_IsInitialized()) {
    return c10::nullopt;
  }
  pybind11::gil_scoped_acquire gil;
  PyFrameObject* frame = PyEval_GetFrame();
  if (frame == nullptr) {
    return c10::nullopt;
  }
  SourceLocation loc;
  loc.line = PyCode_Addr2Line(frame->f_code, frame->f_lasti);
  loc.file = THPUtils_unpackString(frame->f_code->co_filename);
  loc.function = THPUtils_unpackString(frame->f_code->co_name);
  return loc;
}

std::vector<SourceLocation> GetPythonFrames() {
  std::vector<SourceLocation> frames;
  if (Py_IsInitialized()) {
    pybind11::gil_scoped_acquire gil;
    PyFrameObject* frame = PyEval_GetFrame();
    while (frame != nullptr) {
      SourceLocation loc;
      loc.line = PyCode_Addr2Line(frame->f_code, frame->f_lasti);
      loc.file = THPUtils_unpackString(frame->f_code->co_filename);
      loc.function = THPUtils_unpackString(frame->f_code->co_name);
      frames.push_back(std::move(loc));
      frame = frame->f_back;
    }
  }
  return frames;
}

}  // namespace lazy
}  // namespace torch
