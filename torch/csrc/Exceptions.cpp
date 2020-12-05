#include <torch/csrc/Exceptions.h>
#include <torch/csrc/python_headers.h>

#include <utility>
#include <vector>
#include <cstdarg>
#include <exception>
#include <sstream>

#include <torch/csrc/THP.h>

PyObject *THPException_FatalError;

#define ASSERT_TRUE(cond) if (!(cond)) return false
bool THPException_init(PyObject *module)
{
  ASSERT_TRUE(THPException_FatalError = PyErr_NewException("torch.FatalError", nullptr, nullptr));
  ASSERT_TRUE(PyModule_AddObject(module, "FatalError", THPException_FatalError) == 0);
  return true;
}

namespace torch {

static bool compute_cpp_stack_traces_enabled() {
  auto envar = std::getenv("TORCH_SHOW_CPP_STACKTRACES");
  if (envar) {
    if (strcmp(envar, "0") == 0) {
      return false;
    }
    if (strcmp(envar, "1") == 0) {
      return true;
    }
    TORCH_WARN("ignoring invalid value for TORCH_SHOW_CPP_STACKTRACES: ", envar,
               " valid values are 0 or 1.");
  }
  return false;
}

bool get_cpp_stacktraces_enabled() {
  static bool enabled = compute_cpp_stack_traces_enabled();
  return enabled;
}

void replaceAll(std::string & str,
    const std::string & old_str,
    const std::string & new_str) {
  std::string::size_type pos = 0u;
  while ((pos = str.find(old_str, pos)) != std::string::npos) {
    str.replace(pos, old_str.length(), new_str);
  }
}

std::string processErrorMsg(std::string str) {

  // Translate Aten types to their respective pytorch ones
  std::vector<std::pair<std::string, std::string>> changes {
    {"Variable[SparseCOO_CUDAByteType]", "torch.cuda.sparse.ByteTensor"},
    {"Variable[SparseCOO_CUDACharType]", "torch.cuda.sparse.CharTensor"},
    {"Variable[SparseCOO_CUDADoubleType]", "torch.cuda.sparse.DoubleTensor"},
    {"Variable[SparseCOO_CUDAFloatType]", "torch.cuda.sparse.FloatTensor"},
    {"Variable[SparseCOO_CUDAIntType]", "torch.cuda.sparse.IntTensor"},
    {"Variable[SparseCOO_CUDALongType]", "torch.cuda.sparse.LongTensor"},
    {"Variable[SparseCOO_CUDAShortType]", "torch.cuda.sparse.ShortTensor"},
    {"Variable[SparseCOO_CUDAHalfType]", "torch.cuda.sparse.HalfTensor"},
    {"Variable[SparseCOO_CPUByteType]", "torch.sparse.ByteTensor"},
    {"Variable[SparseCOO_CPUCharType]", "torch.sparse.CharTensor"},
    {"Variable[SparseCOO_CPUDoubleType]", "torch.sparse.DoubleTensor"},
    {"Variable[SparseCOO_CPUFloatType]", "torch.sparse.FloatTensor"},
    {"Variable[SparseCOO_CPUIntType]", "torch.sparse.IntTensor"},
    {"Variable[SparseCOO_CPULongType]", "torch.sparse.LongTensor"},
    {"Variable[SparseCOO_CPUShortType]", "torch.sparse.ShortTensor"},
    {"Variable[SparseCOO_CPUHalfType]", "torch.sparse.HalfTensor"},
    {"Variable[CUDAByteType]", "torch.cuda.ByteTensor"},
    {"Variable[CUDACharType]", "torch.cuda.CharTensor"},
    {"Variable[CUDADoubleType]", "torch.cuda.DoubleTensor"},
    {"Variable[CUDAFloatType]", "torch.cuda.FloatTensor"},
    {"Variable[CUDAIntType]", "torch.cuda.IntTensor"},
    {"Variable[CUDALongType]", "torch.cuda.LongTensor"},
    {"Variable[CUDAShortType]", "torch.cuda.ShortTensor"},
    {"Variable[CUDAHalfType]", "torch.cuda.HalfTensor"},
    {"Variable[CPUByteType]", "torch.ByteTensor"},
    {"Variable[CPUCharType]", "torch.CharTensor"},
    {"Variable[CPUDoubleType]", "torch.DoubleTensor"},
    {"Variable[CPUFloatType]", "torch.FloatTensor"},
    {"Variable[CPUIntType]", "torch.IntTensor"},
    {"Variable[CPULongType]", "torch.LongTensor"},
    {"Variable[CPUShortType]", "torch.ShortTensor"},
    {"Variable[CPUHalfType]", "torch.HalfTensor"},
    {"SparseCOO_CUDAByteType", "torch.cuda.sparse.ByteTensor"},
    {"SparseCOO_CUDACharType", "torch.cuda.sparse.CharTensor"},
    {"SparseCOO_CUDADoubleType", "torch.cuda.sparse.DoubleTensor"},
    {"SparseCOO_CUDAFloatType", "torch.cuda.sparse.FloatTensor"},
    {"SparseCOO_CUDAIntType", "torch.cuda.sparse.IntTensor"},
    {"SparseCOO_CUDALongType", "torch.cuda.sparse.LongTensor"},
    {"SparseCOO_CUDAShortType", "torch.cuda.sparse.ShortTensor"},
    {"SparseCOO_CUDAHalfType", "torch.cuda.sparse.HalfTensor"},
    {"SparseCOO_CPUByteType", "torch.sparse.ByteTensor"},
    {"SparseCOO_CPUCharType", "torch.sparse.CharTensor"},
    {"SparseCOO_CPUDoubleType", "torch.sparse.DoubleTensor"},
    {"SparseCOO_CPUFloatType", "torch.sparse.FloatTensor"},
    {"SparseCOO_CPUIntType", "torch.sparse.IntTensor"},
    {"SparseCOO_CPULongType", "torch.sparse.LongTensor"},
    {"SparseCOO_CPUShortType", "torch.sparse.ShortTensor"},
    {"SparseCOO_CPUHalfType", "torch.sparse.HalfTensor"},
    {"CUDAByteType", "torch.cuda.ByteTensor"},
    {"CUDACharType", "torch.cuda.CharTensor"},
    {"CUDADoubleType", "torch.cuda.DoubleTensor"},
    {"CUDAFloatType", "torch.cuda.FloatTensor"},
    {"CUDAIntType", "torch.cuda.IntTensor"},
    {"CUDALongType", "torch.cuda.LongTensor"},
    {"CUDAShortType", "torch.cuda.ShortTensor"},
    {"CUDAHalfType", "torch.cuda.HalfTensor"},
    {"CPUByteType", "torch.ByteTensor"},
    {"CPUCharType", "torch.CharTensor"},
    {"CPUDoubleType", "torch.DoubleTensor"},
    {"CPUFloatType", "torch.FloatTensor"},
    {"CPUIntType", "torch.IntTensor"},
    {"CPULongType", "torch.LongTensor"},
    {"CPUShortType", "torch.ShortTensor"},
    {"CPUHalfType", "torch.HalfTensor"},
  };

  for (const auto & it : changes) {
    replaceAll(str, it.first, it.second);
  }

  return str;
}

static std::string formatMessage(const char *format, va_list fmt_args) {
  static const size_t ERROR_BUF_SIZE = 1024;
  char error_buf[ERROR_BUF_SIZE];
  vsnprintf(error_buf, ERROR_BUF_SIZE, format, fmt_args);

  // Ensure that the string is null terminated
  error_buf[sizeof(error_buf) / sizeof(*error_buf) - 1] = 0;

  return std::string(error_buf);
}

IndexError::IndexError(const char *format, ...) {
  va_list fmt_args;
  va_start(fmt_args, format);
  msg = formatMessage(format, fmt_args);
  va_end(fmt_args);
}

TypeError::TypeError(const char *format, ...) {
  va_list fmt_args;
  va_start(fmt_args, format);
  msg = formatMessage(format, fmt_args);
  va_end(fmt_args);
}

ValueError::ValueError(const char *format, ...) {
  va_list fmt_args;
  va_start(fmt_args, format);
  msg = formatMessage(format, fmt_args);
  va_end(fmt_args);
}

AttributeError::AttributeError(const char* format, ...) {
  va_list fmt_args;
  va_start(fmt_args, format);
  msg = formatMessage(format, fmt_args);
  va_end(fmt_args);
}

void PyWarningHandler::process(
    const c10::SourceLocation& source_location,
    const std::string& msg,
    const bool verbatim) {
  warning_buffer_.push_back({source_location, msg, verbatim});
};

PyWarningHandler::PyWarningHandler() noexcept(true):
      prev_handler_(c10::Warning::get_warning_handler()),
      in_exception_(false) {
  c10::Warning::set_warning_handler(this);
}

/// See NOTE [ Conversion Cpp Python Warning ] for noexcept justification
/// NOLINTNEXTLINE(bugprone-exception-escape)
PyWarningHandler::~PyWarningHandler() noexcept(false) {
  c10::Warning::set_warning_handler(prev_handler_);

  if (warning_buffer_.size() > 0) {
    PyObject *type, *value, *traceback;
    pybind11::gil_scoped_acquire gil;
    auto result = 0;
    if (in_exception_) {
      // This (combined with PyErr_Restore below) also works when no python
      // error has been set yet
      PyErr_Fetch(&type, &value, &traceback);
    }
    for (const auto& warning : warning_buffer_) {
      auto source_location = warning.source_location_;
      const auto& msg = processErrorMsg(warning.msg_);
      if (source_location.file == nullptr) {
        result = PyErr_WarnEx(PyExc_RuntimeWarning, msg.c_str(), 1);
      } else if (warning.verbatim_) {
        // Sets the source location from the warning
        // Note: PyErr_WarnExplicit will disregard Python's warning filter
        // and always appear. This is in contrast to PyErr_WarnEx,
        // which respects the warning filter.
        result = PyErr_WarnExplicit(
            /*category=*/PyExc_UserWarning,
            /*message=*/msg.c_str(),
            /*filename=*/source_location.file,
            /*lineno=*/source_location.line,
            /*module=*/nullptr,
            /*registry=*/nullptr);
      } else {
        // Lets Python set the source location and puts the C++ warning
        // location into the message.
        std::ostringstream os;
        os << msg << " (Triggered internally at  " << source_location.file;
        os << ":" << source_location.line << ".)";
        result = PyErr_WarnEx(PyExc_UserWarning, os.str().c_str(), 1);
      }
      if (result < 0) {
        if (in_exception_) {
          // PyErr_Print prints the traceback to sys.stderr and
          // clears the error indicator
          PyErr_Print();
        } else {
          break;
        }
      }
    }
    warning_buffer_.clear();
    if ((result < 0) && (!in_exception_)) {
      /// A warning raised an error, we need to force the parent
      /// function to return an error code.
      throw python_error();
    }
    if (in_exception_) {
      PyErr_Restore(type, value, traceback);
    }
  }
}



} // namespace torch
