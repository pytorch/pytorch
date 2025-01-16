#pragma once
#include <torch/csrc/onnx/diagnostics/generated/rules.h>
#include <torch/csrc/utils/pybind.h>
#include <string>

namespace torch::onnx::diagnostics {

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

// NOLINTNEXTLINE(*array*)
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
    std::unordered_map<std::string, std::string> messageArgs = {}) {
  py::object py_rule = _PyRule(rule);
  py::object py_level = _PyLevel(level);

  // TODO: statically check that size of messageArgs matches with rule.
  py::object py_message =
      py_rule.attr("format_message")(**py::cast(messageArgs));

  // to use the `_a` literal for arguments
  using namespace pybind11::literals;
  _PyDiagnostics().attr("diagnose")(
      py_rule, py_level, py_message, "cpp_stack"_a = true);
}

} // namespace torch::onnx::diagnostics
