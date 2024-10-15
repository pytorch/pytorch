#include <torch/csrc/Exceptions.h>
#include <torch/csrc/dynamo/python_logger.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>

namespace torch::dynamo {

PythonLogger::PythonLogger(PyObject* logger) : logger_(logger) {
  TORCH_INTERNAL_ASSERT(logger_ != nullptr);
}

// must be called while GIL is held
void PythonLogger::log(PythonLogger::Level level, std::string_view msg) const {
  THPObjectPtr pymethod(PyUnicode_FromString(levelNames_[level].data()));
  THPObjectPtr pyfunc(PyObject_GetAttr(logger_, pymethod.get()));
  PyObject* result = PyObject_CallFunction(pyfunc.get(), "s", msg.data());
  if (result == nullptr) {
    python_error err;
    err.persist();
    // NOLINTNEXTLINE(misc-throw-by-value-catch-by-reference)
    throw err;
  }
}

} // namespace torch::dynamo
