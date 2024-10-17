#include <torch/csrc/Exceptions.h>
#include <torch/csrc/dynamo/python_logger.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>

namespace torch::dynamo {

namespace {
// see https://github.com/pytorch/pytorch/pull/34845
void throw_python_error() {
  python_error err;
  err.persist();
  // NOLINTNEXTLINE(misc-throw-by-value-catch-by-reference)
  throw err;
}
} // namespace

PythonLogger::PythonLogger(PyObject* logger) : logger_(logger) {
  TORCH_INTERNAL_ASSERT(logger_ != nullptr);
}

// must be called while GIL is held
void PythonLogger::log(PythonLogger::Level level, std::string_view msg) const {
  THPObjectPtr pymethod(PyUnicode_FromString(levelNames_[level].data()));
  TORCH_INTERNAL_ASSERT(pymethod != nullptr);
  THPObjectPtr pyfunc(PyObject_GetAttr(logger_, pymethod.get()));
  if (pyfunc == nullptr) {
    throw_python_error();
  }
  PyObject* result = PyObject_CallFunction(pyfunc.get(), "s", msg.data());
  if (result == nullptr) {
    throw_python_error();
  }
}

} // namespace torch::dynamo
