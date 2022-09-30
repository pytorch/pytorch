#pragma once
#include <torch/csrc/onnx/diagnostics/generated/rules.h>
#include <torch/csrc/utils/pybind.h>
#include <string>

namespace torch {
namespace onnx {
namespace diagnostics {

/**
 * @brief Level of a diagnostic.
 * @details The levels are defined by the SARIF specification, and are not
 * modifiable. For alternative categories, please use Tag instead.
 * @todo Introduce Tag to C++ api.
 */
enum class Level : uint8_t {
  kNone,
  kNote,
  kWarning,
  kError,
};

static constexpr const char* const kPyLevelNames[] = {
    "NONE",
    "NOTE",
    "WARNING",
    "ERROR",
};

// Wrappers around Python diagnostics.
// TODO: Move to .cpp file in following PR.

inline py::object _PyDiagnostics() {
  return py::module::import("torch.onnx._internal.diagnostics");
}

inline py::object _PyEngine() {
  return _PyDiagnostics().attr("engine");
}

inline py::object _PyContext() {
  return _PyDiagnostics().attr("context");
}

inline py::object _PyRule(Rule rule) {
  return _PyDiagnostics().attr("rules").attr(
      kPyRuleNames[static_cast<uint32_t>(rule)]);
}

inline py::object _PyLevel(Level level) {
  return _PyDiagnostics().attr("levels").attr(
      kPyLevelNames[static_cast<uint32_t>(level)]);
}

inline void Diagnose(
    Rule rule,
    Level level,
    std::vector<std::string> messageArgs = {}) {
  py::object py_rule = _PyRule(rule);
  py::object py_level = _PyLevel(level);
  py::object py_context = _PyContext();

  py::dict kwargs = py::dict();
  // TODO: statically check that size of messageArgs matches with rule.
  kwargs["message_args"] = messageArgs;
  py_context.attr("diagnose")(py_rule, py_level, **kwargs);
}

} // namespace diagnostics
} // namespace onnx
} // namespace torch
