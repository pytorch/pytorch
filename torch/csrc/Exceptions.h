#pragma once

#include <exception>
#include <stdexcept>
#include <string>

#ifndef NO_PYTHON

#include "ATen/Error.h"
#include "THP_export.h"
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/utils/auto_gil.h"

#define HANDLE_TH_ERRORS                                                       \
  try {

#define END_HANDLE_TH_ERRORS_RET(retval)                                       \
  } catch (python_error &e) {                                                  \
    return retval;                                                             \
  } catch (const at::Error &e) {                                               \
    auto msg = torch::processErrorMsg(e.what_without_backtrace());              \
    PyErr_SetString(PyExc_RuntimeError, msg.c_str());                          \
    return retval;                                                             \
  } catch (torch::PyTorchError &e) {                                           \
    auto msg = torch::processErrorMsg(e.what());                               \
    PyErr_SetString(e.python_type(), msg.c_str());                             \
    return retval;                                                             \
  } catch (const std::exception &e) {                                          \
    auto msg = torch::processErrorMsg(e.what());                               \
    PyErr_SetString(PyExc_RuntimeError, msg.c_str());                          \
    return retval;                                                             \
  }

#define END_HANDLE_TH_ERRORS END_HANDLE_TH_ERRORS_RET(NULL)

extern PyObject *THPException_FatalError;

// Throwing this exception means that the python error flags have been already
// set and control should be immediately returned to the interpreter.
struct python_error : public std::exception {
  python_error() : type(nullptr), value(nullptr), traceback(nullptr) {}

  python_error(const python_error &other) : type(other.type), value(other.value), traceback(other.traceback) {
    AutoGIL gil;
    Py_XINCREF(type);
    Py_XINCREF(value);
    Py_XINCREF(traceback);
  }

  python_error(python_error&& other) {
    type = std::move(other.type);
    value = std::move(other.value);
    traceback = std::move(other.traceback);
    other.type = nullptr;
    other.value = nullptr;
    other.traceback = nullptr;
  }

  ~python_error() {
    if (type || value || traceback) {
      AutoGIL gil;
      Py_XDECREF(type);
      Py_XDECREF(value);
      Py_XDECREF(traceback);
    }
  }

  /** Saves the exception so that it can be re-thrown on a different thread */
  inline void persist() {
    if (type) return; // Don't overwrite exceptions
    // PyErr_Fetch overwrites the pointers
    AutoGIL gil;
    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(traceback);
    PyErr_Fetch(&type, &value, &traceback);
  }

  /** Sets the current Python error from this exception */
  inline void restore() {
    if (!type) return;
    // PyErr_Restore steals references
    AutoGIL gil;
    Py_XINCREF(type);
    Py_XINCREF(value);
    Py_XINCREF(traceback);
    PyErr_Restore(type, value, traceback);
  }

  PyObject* type;
  PyObject* value;
  PyObject* traceback;
};

#ifdef _THP_CORE

bool THPException_init(PyObject *module);
#endif

namespace torch {

THP_CLASS std::string processErrorMsg(std::string str);

// Abstract base class for exceptions which translate to specific Python types
struct PyTorchError : public std::exception {
  virtual PyObject* python_type() = 0;
  virtual const char* what() const noexcept override {
    return msg.c_str();
  }
  std::string msg;
};

// Translates to Python IndexError
struct IndexError : public PyTorchError {
  IndexError(const char *format, ...);
  virtual PyObject* python_type() override {
    return PyExc_IndexError;
  }
};

// Translates to Python TypeError
struct TypeError : public PyTorchError {
  TypeError(const char *format, ...);
  virtual PyObject* python_type() override {
    return PyExc_TypeError;
  }
};

// Translates to Python ValueError
struct ValueError : public PyTorchError {
  ValueError(const char *format, ...);
  virtual PyObject* python_type() override {
    return PyExc_ValueError;
  }
};

} // namespace torch

#else

namespace torch {

struct PyTorchError : public std::exception {
  virtual const char* what() const noexcept override {
    return msg.c_str();
  }
  std::string msg;
};

struct ValueError : public PyTorchError {
  ValueError(const char *format, ...);
};

} // namespace torch

#endif
