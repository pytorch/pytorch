#pragma once

#include <exception>
#include <string>

#include <c10/util/Exception.h>
#include <torch/csrc/THP_export.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/utils/auto_gil.h>

#define HANDLE_TH_ERRORS                                                       \
  try {
#define END_HANDLE_TH_ERRORS_RET(retval)                           \
  }                                                                \
  catch (python_error & e) {                                       \
    return retval;                                                 \
  }                                                                \
  catch (const c10::IndexError& e) {                               \
    auto msg = torch::processErrorMsg(e.what_without_backtrace()); \
    PyErr_SetString(PyExc_IndexError, msg.c_str());                \
    return retval;                                                 \
  }                                                                \
  catch (const c10::Error& e) {                                    \
    auto msg = torch::processErrorMsg(e.what_without_backtrace()); \
    PyErr_SetString(PyExc_RuntimeError, msg.c_str());              \
    return retval;                                                 \
  }                                                                \
  catch (torch::PyTorchError & e) {                                \
    auto msg = torch::processErrorMsg(e.what());                   \
    PyErr_SetString(e.python_type(), msg.c_str());                 \
    return retval;                                                 \
  }                                                                \
  catch (const std::exception& e) {                                \
    auto msg = torch::processErrorMsg(e.what());                   \
    PyErr_SetString(PyExc_RuntimeError, msg.c_str());              \
    return retval;                                                 \
  }

#define END_HANDLE_TH_ERRORS END_HANDLE_TH_ERRORS_RET(nullptr)

extern PyObject *THPException_FatalError;

// Throwing this exception means that the python error flags have been already
// set and control should be immediately returned to the interpreter.
struct python_error : public std::exception {
  python_error() : type(nullptr), value(nullptr), traceback(nullptr) {}

  python_error(const python_error& other)
      : type(other.type),
        value(other.value),
        traceback(other.traceback),
        message(other.message) {
    AutoGIL gil;
    Py_XINCREF(type);
    Py_XINCREF(value);
    Py_XINCREF(traceback);
  }

  python_error(python_error&& other) {
    type = other.type;
    value = other.value;
    traceback = other.traceback;
    message = std::move(other.message);
    other.type = nullptr;
    other.value = nullptr;
    other.traceback = nullptr;
  }

  ~python_error() override {
    if (type || value || traceback) {
      AutoGIL gil;
      Py_XDECREF(type);
      Py_XDECREF(value);
      Py_XDECREF(traceback);
    }
  }

  virtual const char* what() const noexcept override {
    return message.c_str();
  }

  void build_message() {
    // Ensure we have the GIL.
    AutoGIL gil;

    // No errors should be set when we enter the function since PyErr_Fetch
    // clears the error indicator.
    TORCH_INTERNAL_ASSERT(!PyErr_Occurred());

    // Default message.
    message = "python_error";

    // Try to retrieve the error message from the value.
    if (value != nullptr) {
      // Reference count should not be zero.
      TORCH_INTERNAL_ASSERT(value->ob_refcnt > 0);

      PyObject* pyStr = PyObject_Str(value);
      if (pyStr != nullptr) {
        PyObject* encodedString =
            PyUnicode_AsEncodedString(pyStr, "utf-8", "strict");
        if (encodedString != nullptr) {
          char* bytes = PyBytes_AS_STRING(encodedString);
          if (bytes != nullptr) {
            // Set the message.
            message = std::string(bytes);
          }
          Py_XDECREF(encodedString);
        }
        Py_XDECREF(pyStr);
      }
    }

    // Clear any errors since we don't want to propagate errors for functions
    // that are trying to build a string for the error message.
    PyErr_Clear();
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
    build_message();
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

  // Message to return to the user when 'what()' is invoked.
  std::string message;
};

bool THPException_init(PyObject *module);

namespace torch {

THP_CLASS std::string processErrorMsg(std::string str);

// Abstract base class for exceptions which translate to specific Python types
struct PyTorchError : public std::exception {
  virtual PyObject* python_type() = 0;
  const char* what() const noexcept override {
    return msg.c_str();
  }
  std::string msg;
};

// Translates to Python IndexError
struct IndexError : public PyTorchError {
  IndexError(const char *format, ...);
  PyObject* python_type() override {
    return PyExc_IndexError;
  }
};

// Translates to Python TypeError
struct TypeError : public PyTorchError {
  TORCH_API TypeError(const char *format, ...);
  PyObject* python_type() override {
    return PyExc_TypeError;
  }
};

// Translates to Python ValueError
struct ValueError : public PyTorchError {
  ValueError(const char *format, ...);
  PyObject* python_type() override {
    return PyExc_ValueError;
  }
};

} // namespace torch
