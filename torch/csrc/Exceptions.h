#pragma once

#include <exception>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <system_error>

#include <ATen/detail/FunctionTraits.h>
#include <c10/util/C++17.h>
#include <c10/util/Exception.h>
#include <c10/util/StringUtil.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/runtime/jit_exception.h>
#include <torch/csrc/utils/cpp_stacktraces.h>
#include <torch/csrc/utils/pybind.h>

#if defined(USE_DISTRIBUTED) && defined(USE_C10D)
#include <torch/csrc/distributed/c10d/exception.h>
#endif

static inline void PyErr_SetString(PyObject* type, const std::string& message) {
  PyErr_SetString(type, message.c_str());
}
/// NOTE [ Conversion Cpp Python Warning ]
/// The warning handler cannot set python warnings immediately
/// as it requires acquiring the GIL (potential deadlock)
/// and would need to cleanly exit if the warning raised a
/// python error. To solve this, we buffer the warnings and
/// process them when we go back to python.
/// This requires the two try/catch blocks below to handle the
/// following cases:
///   - If there is no Error raised in the inner try/catch, the
///     buffered warnings are processed as python warnings.
///     - If they don't raise an error, the function process with the
///       original return code.
///     - If any of them raise an error, the error is set (PyErr_*) and
///       the destructor will raise a cpp exception python_error() that
///       will be caught by the outer try/catch that will be able to change
///       the return value of the function to reflect the error.
///   - If an Error was raised in the inner try/catch, the inner try/catch
///     must set the python error. The buffered warnings are then
///     processed as cpp warnings as we cannot predict before hand
///     whether a python warning will raise an error or not and we
///     cannot handle two errors at the same time.
/// This advanced handler will only be used in the current thread.
/// If any other thread is used, warnings will be processed as
/// cpp warnings.
#define HANDLE_TH_ERRORS                              \
  try {                                               \
    torch::PyWarningHandler __enforce_warning_buffer; \
    try {
#define _CATCH_GENERIC_ERROR(ErrorType, PythonErrorType, retstmnt) \
  catch (const c10::ErrorType& e) {                                \
    auto msg = torch::get_cpp_stacktraces_enabled()                \
        ? e.what()                                                 \
        : e.what_without_backtrace();                              \
    PyErr_SetString(PythonErrorType, torch::processErrorMsg(msg)); \
    retstmnt;                                                      \
  }

// Only catch torch-specific exceptions
#define CATCH_CORE_ERRORS(retstmnt)                                           \
  catch (python_error & e) {                                                  \
    e.restore();                                                              \
    retstmnt;                                                                 \
  }                                                                           \
  catch (py::error_already_set & e) {                                         \
    e.restore();                                                              \
    retstmnt;                                                                 \
  }                                                                           \
  _CATCH_GENERIC_ERROR(IndexError, PyExc_IndexError, retstmnt)                \
  _CATCH_GENERIC_ERROR(ValueError, PyExc_ValueError, retstmnt)                \
  _CATCH_GENERIC_ERROR(TypeError, PyExc_TypeError, retstmnt)                  \
  _CATCH_GENERIC_ERROR(                                                       \
      NotImplementedError, PyExc_NotImplementedError, retstmnt)               \
  _CATCH_GENERIC_ERROR(LinAlgError, THPException_LinAlgError, retstmnt)       \
  _CATCH_GENERIC_ERROR(                                                       \
      OutOfMemoryError, THPException_OutOfMemoryError, retstmnt)              \
  _CATCH_GENERIC_ERROR(                                                       \
      DistBackendError, THPException_DistBackendError, retstmnt)              \
  _CATCH_GENERIC_ERROR(                                                       \
      DistNetworkError, THPException_DistNetworkError, retstmnt)              \
  _CATCH_GENERIC_ERROR(DistStoreError, THPException_DistStoreError, retstmnt) \
  _CATCH_GENERIC_ERROR(DistError, THPException_DistError, retstmnt)           \
  _CATCH_GENERIC_ERROR(Error, PyExc_RuntimeError, retstmnt)                   \
  catch (torch::PyTorchError & e) {                                           \
    auto msg = torch::processErrorMsg(e.what());                              \
    PyErr_SetString(e.python_type(), msg);                                    \
    retstmnt;                                                                 \
  }

#define CATCH_TH_ERRORS(retstmnt) CATCH_CORE_ERRORS(retstmnt)

#define CATCH_ALL_ERRORS(retstmnt)               \
  CATCH_TH_ERRORS(retstmnt)                      \
  catch (const std::exception& e) {              \
    auto msg = torch::processErrorMsg(e.what()); \
    PyErr_SetString(PyExc_RuntimeError, msg);    \
    retstmnt;                                    \
  }

#define END_HANDLE_TH_ERRORS_PYBIND                                 \
  }                                                                 \
  catch (...) {                                                     \
    __enforce_warning_buffer.set_in_exception();                    \
    throw;                                                          \
  }                                                                 \
  }                                                                 \
  catch (py::error_already_set & e) {                               \
    throw;                                                          \
  }                                                                 \
  catch (py::builtin_exception & e) {                               \
    throw;                                                          \
  }                                                                 \
  catch (torch::jit::JITException & e) {                            \
    throw;                                                          \
  }                                                                 \
  catch (const std::exception& e) {                                 \
    torch::translate_exception_to_python(std::current_exception()); \
    throw py::error_already_set();                                  \
  }

#define END_HANDLE_TH_ERRORS_RET(retval)                            \
  }                                                                 \
  catch (...) {                                                     \
    __enforce_warning_buffer.set_in_exception();                    \
    throw;                                                          \
  }                                                                 \
  }                                                                 \
  catch (const std::exception& e) {                                 \
    torch::translate_exception_to_python(std::current_exception()); \
    return retval;                                                  \
  }

#define END_HANDLE_TH_ERRORS END_HANDLE_TH_ERRORS_RET(nullptr)

extern PyObject *THPException_FatalError, *THPException_LinAlgError,
    *THPException_OutOfMemoryError, *THPException_DistError,
    *THPException_DistBackendError, *THPException_DistNetworkError,
    *THPException_DistStoreError;

// Throwing this exception means that the python error flags have been already
// set and control should be immediately returned to the interpreter.
struct python_error : public std::exception {
  python_error() : type(nullptr), value(nullptr), traceback(nullptr) {}

  python_error(const python_error& other)
      : type(other.type),
        value(other.value),
        traceback(other.traceback),
        message(other.message) {
    pybind11::gil_scoped_acquire gil;
    Py_XINCREF(type);
    Py_XINCREF(value);
    Py_XINCREF(traceback);
  }

  python_error(python_error&& other) noexcept
      : type(other.type),
        value(other.value),
        traceback(other.traceback),
        message(std::move(other.message)) {
    other.type = nullptr;
    other.value = nullptr;
    other.traceback = nullptr;
  }

  ~python_error() override {
    if (type || value || traceback) {
      pybind11::gil_scoped_acquire gil;
      Py_XDECREF(type);
      Py_XDECREF(value);
      Py_XDECREF(traceback);
    }
  }

  const char* what() const noexcept override {
    return message.c_str();
  }

  void build_message() {
    // Ensure we have the GIL.
    pybind11::gil_scoped_acquire gil;

    // No errors should be set when we enter the function since PyErr_Fetch
    // clears the error indicator.
    TORCH_INTERNAL_ASSERT(!PyErr_Occurred());

    // Default message.
    message = "python_error";

    // Try to retrieve the error message from the value.
    if (value != nullptr) {
      // Reference count should not be zero.
      TORCH_INTERNAL_ASSERT(Py_REFCNT(value) > 0);

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
    if (type)
      return; // Don't overwrite exceptions
    // PyErr_Fetch overwrites the pointers
    pybind11::gil_scoped_acquire gil;
    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(traceback);
    PyErr_Fetch(&type, &value, &traceback);
    build_message();
  }

  /** Sets the current Python error from this exception */
  inline void restore() {
    if (!type)
      return;
    // PyErr_Restore steals references
    pybind11::gil_scoped_acquire gil;
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

bool THPException_init(PyObject* module);

namespace torch {

// Set python current exception from a C++ exception
TORCH_PYTHON_API void translate_exception_to_python(const std::exception_ptr&);

TORCH_PYTHON_API std::string processErrorMsg(std::string str);

// Abstract base class for exceptions which translate to specific Python types
struct PyTorchError : public std::exception {
  PyTorchError() = default;
  PyTorchError(std::string msg_) : msg(std::move(msg_)) {}
  virtual PyObject* python_type() = 0;
  const char* what() const noexcept override {
    return msg.c_str();
  }
  std::string msg;
};

// Declare a printf-like function on gcc & clang
// The compiler can then warn on invalid format specifiers
#ifdef __GNUC__
#define TORCH_FORMAT_FUNC(FORMAT_INDEX, VA_ARGS_INDEX) \
  __attribute__((format(printf, FORMAT_INDEX, VA_ARGS_INDEX)))
#else
#define TORCH_FORMAT_FUNC(FORMAT_INDEX, VA_ARGS_INDEX)
#endif

// Translates to Python IndexError
struct IndexError : public PyTorchError {
  using PyTorchError::PyTorchError;
  IndexError(const char* format, ...) TORCH_FORMAT_FUNC(2, 3);
  PyObject* python_type() override {
    return PyExc_IndexError;
  }
};

// Translates to Python TypeError
struct TypeError : public PyTorchError {
  using PyTorchError::PyTorchError;
  TORCH_PYTHON_API TypeError(const char* format, ...) TORCH_FORMAT_FUNC(2, 3);
  PyObject* python_type() override {
    return PyExc_TypeError;
  }
};

// Translates to Python ValueError
struct ValueError : public PyTorchError {
  using PyTorchError::PyTorchError;
  TORCH_PYTHON_API ValueError(const char* format, ...) TORCH_FORMAT_FUNC(2, 3);
  PyObject* python_type() override {
    return PyExc_ValueError;
  }
};

// Translates to Python NotImplementedError
struct NotImplementedError : public PyTorchError {
  NotImplementedError(const char* format, ...) TORCH_FORMAT_FUNC(2, 3);
  NotImplementedError() = default;
  PyObject* python_type() override {
    return PyExc_NotImplementedError;
  }
};

// Translates to Python AttributeError
struct AttributeError : public PyTorchError {
  AttributeError(const char* format, ...) TORCH_FORMAT_FUNC(2, 3);
  PyObject* python_type() override {
    return PyExc_AttributeError;
  }
};

// Translates to Python LinAlgError
struct LinAlgError : public PyTorchError {
  LinAlgError(const char* format, ...) TORCH_FORMAT_FUNC(2, 3);
  PyObject* python_type() override {
    return THPException_LinAlgError;
  }
};

// ATen warning handler for Python
struct PyWarningHandler {
  // Move actual handler into a separate class with a noexcept
  // destructor. Otherwise, we need to force all WarningHandler
  // subclasses to have a noexcept(false) destructor.
  struct InternalHandler : at::WarningHandler {
    ~InternalHandler() override = default;
    void process(const c10::Warning& warning) override;

    std::vector<c10::Warning> warning_buffer_;
  };

 public:
  /// See NOTE [ Conversion Cpp Python Warning ] for noexcept justification
  TORCH_PYTHON_API PyWarningHandler() noexcept(true);
  // NOLINTNEXTLINE(bugprone-exception-escape)
  TORCH_PYTHON_API ~PyWarningHandler() noexcept(false);

  /** Call if an exception has been thrown

   *  Necessary to determine if it is safe to throw from the desctructor since
   *  std::uncaught_exception is buggy on some platforms and generally
   *  unreliable across dynamic library calls.
   */
  void set_in_exception() {
    in_exception_ = true;
  }

 private:
  InternalHandler internal_handler_;
  at::WarningHandler* prev_handler_;
  bool in_exception_;
};

namespace detail {
template <typename Func, size_t i>
using Arg = typename invoke_traits<Func>::template arg<i>::type;

template <typename Func, size_t... Is>
auto wrap_pybind_function_impl_(
    Func&& f,
    std::index_sequence<Is...>,
    bool release_gil) {
  using result_type = typename invoke_traits<Func>::result_type;
  namespace py = pybind11;

  // f=f is needed to handle function references on older compilers
  return [f = std::forward<Func>(f),
          release_gil](Arg<Func, Is>... args) -> result_type {
    HANDLE_TH_ERRORS
    if (release_gil) {
      py::gil_scoped_release no_gil;
      return c10::guts::invoke(f, std::forward<Arg<Func, Is>>(args)...);
    } else {
      return c10::guts::invoke(f, std::forward<Arg<Func, Is>>(args)...);
    }
    END_HANDLE_TH_ERRORS_PYBIND
  };
}
} // namespace detail

// Wrap a function with TH error and warning handling.
// Returns a function object suitable for registering with pybind11.
template <typename Func>
auto wrap_pybind_function(Func&& f) {
  using traits = invoke_traits<Func>;
  return torch::detail::wrap_pybind_function_impl_(
      std::forward<Func>(f), std::make_index_sequence<traits::arity>{}, false);
}

// Wrap a function with TH error, warning handling and releases the GIL.
// Returns a function object suitable for registering with pybind11.
template <typename Func>
auto wrap_pybind_function_no_gil(Func&& f) {
  using traits = invoke_traits<Func>;
  return torch::detail::wrap_pybind_function_impl_(
      std::forward<Func>(f), std::make_index_sequence<traits::arity>{}, true);
}

} // namespace torch
