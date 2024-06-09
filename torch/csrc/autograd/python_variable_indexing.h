#pragma once

#include <c10/core/SymInt.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_symnode.h>

namespace torch::autograd {

struct UnpackedSlice {
  c10::SymInt start;
  c10::SymInt stop;
  c10::SymInt step;
};

// This mirrors Cpython's PySlice_Unpack method
inline UnpackedSlice __PySlice_Unpack(PyObject* _r) {
  PySliceObject* r = (PySliceObject*)_r;
  /* this is harder to get right than you might think */

  c10::SymInt start_sym, stop_sym, step_sym;

  auto clip_val = [](Py_ssize_t val) {
    if (val < c10::SymInt::min_representable_int()) {
      auto r = PyErr_WarnEx(
          PyExc_UserWarning,
          "Truncating the start/stop/step "
          "of slice. This is likely because of "
          "saved old models when the start/stop/step were larger.",
          1);
      if (r != 0) {
        throw python_error();
      }
      return (Py_ssize_t)(c10::SymInt::min_representable_int());
    }
    return val;
  };

  if (r->step == Py_None) {
    step_sym = c10::SymInt(1);
  } else {
    if (torch::is_symint(r->step)) {
      auto step_sym = py::handle(r->step).cast<c10::SymInt>();
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      Py_ssize_t step;
      if (!_PyEval_SliceIndex(r->step, &step)) {
        throw python_error();
      }
      if (step == 0) {
        PyErr_SetString(PyExc_ValueError, "slice step cannot be zero");
      }

      step = clip_val(step);
      step_sym = c10::SymInt(step);
    }
  }

  if (torch::is_symint(r->start)) {
    start_sym = py::handle(r->start).cast<c10::SymInt>();
  } else if (r->start == Py_None) {
    start_sym = c10::SymInt(step_sym < 0 ? PY_SSIZE_T_MAX : 0);
  } else {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Py_ssize_t start;
    if (!_PyEval_SliceIndex(r->start, &start)) {
      throw python_error();
    }
    start = clip_val(start);
    start_sym = c10::SymInt(start);
  }

  if (torch::is_symint(r->stop)) {
    stop_sym = py::handle(r->stop).cast<c10::SymInt>();
  } else if (r->stop == Py_None) {
    stop_sym = c10::SymInt(
        step_sym < 0 ? c10::SymInt::min_representable_int() : PY_SSIZE_T_MAX);
  } else {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Py_ssize_t stop;
    if (!_PyEval_SliceIndex(r->stop, &stop)) {
      throw python_error();
    }
    stop = clip_val(stop);
    stop_sym = c10::SymInt(stop);
  }

  return UnpackedSlice{
      std::move(start_sym), std::move(stop_sym), std::move(step_sym)};
}

Py_ssize_t THPVariable_length(PyObject* self);
PyObject* THPVariable_getitem(PyObject* self, PyObject* index);
int THPVariable_setitem(PyObject* self, PyObject* index, PyObject* value);

Variable valueToTensor(
    c10::TensorOptions options,
    PyObject* value,
    const at::Device& device);

} // namespace torch::autograd
