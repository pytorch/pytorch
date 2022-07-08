#include <torch/csrc/lazy/python/python_util.h>

#include <Python.h>
#include <frameobject.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/lazy/core/debug_util.h>
#include <torch/csrc/utils/python_strings.h>

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
#if PY_VERSION_HEX >= 0x030B0000
  loc.line = PyCode_Addr2Line(PyFrame_GetCode(frame), PyFrame_GetLasti(frame));
  loc.file = THPUtils_unpackString(PyFrame_GetCode(frame)->co_filename);
  loc.function = THPUtils_unpackString(PyFrame_GetCode(frame)->co_name);
#else
  loc.line = PyCode_Addr2Line(frame->f_code, frame->f_lasti);
  loc.file = THPUtils_unpackString(frame->f_code->co_filename);
  loc.function = THPUtils_unpackString(frame->f_code->co_name);
#endif
  return loc;
}

std::vector<SourceLocation> GetPythonFrames() {
  std::vector<SourceLocation> frames;
  if (Py_IsInitialized()) {
    pybind11::gil_scoped_acquire gil;
    PyFrameObject* frame = PyEval_GetFrame();
    while (frame != nullptr) {
      SourceLocation loc;
#if PY_VERSION_HEX >= 0x030B0000
      loc.line =
          PyCode_Addr2Line(PyFrame_GetCode(frame), PyFrame_GetLasti(frame));
      loc.file = THPUtils_unpackString(PyFrame_GetCode(frame)->co_filename);
      loc.function = THPUtils_unpackString(PyFrame_GetCode(frame)->co_name);
#else
      loc.line = PyCode_Addr2Line(frame->f_code, frame->f_lasti);
      loc.file = THPUtils_unpackString(frame->f_code->co_filename);
      loc.function = THPUtils_unpackString(frame->f_code->co_name);
#endif
      frames.push_back(std::move(loc));
#if PY_VERSION_HEX >= 0x030B0000
      frame = PyFrame_GetBack(frame);
#else
      frame = frame->f_back;
#endif
    }
  }
  return frames;
}

} // namespace lazy
} // namespace torch
