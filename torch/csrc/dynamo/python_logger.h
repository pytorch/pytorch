#include <torch/csrc/utils/python_stub.h>
#include <string_view>
#include <unordered_map>

namespace torch::dynamo {

struct PythonLogger {
  PythonLogger() = delete;
  explicit PythonLogger(PyObject* logger);

  enum Level : unsigned int {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    CRITICAL = 4,
    COUNT // Keep this as the last enum
  };

  // must be called while GIL is held
  void log(Level level, std::string_view msg) const;

 private:
  static constexpr std::array<std::string_view, COUNT> levelNames_ = {
      "debug", // Level::DEBUG
      "info", // Level::INFO
      "warning", // Level::WARNING
      "error", // Level::ERROR
      "critical" // Level::CRITICAL
  };

  // Note: logger_ must stay valid for the lifetime of this object
  PyObject* logger_;
};

} // namespace torch::dynamo
