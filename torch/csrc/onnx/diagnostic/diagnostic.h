#pragma once
#include <torch/csrc/onnx/diagnostic/generated/rules.h>
#include <torch/csrc/utils/pybind.h>
#include <string>

namespace torch {
namespace onnx {
namespace diagnostic {

/**
 * @brief Level of a diagnostic.
 */
enum class Level : uint32_t {
  None,
  Note,
  Warning,
  Error,
};

static constexpr const char* const LevelNames[] = {
    "NONE",
    "NOTE",
    "WARNING",
    "ERROR",
};

/**
 * @brief Wrapper class around Python diagnostics.
 */
class DiagnosticPythonWrapper {
 public:
  static DiagnosticPythonWrapper& get() {
    static DiagnosticPythonWrapper instance;
    return instance;
  }

  DiagnosticPythonWrapper(DiagnosticPythonWrapper const&) = delete;
  void operator=(DiagnosticPythonWrapper const&) = delete;

  py::object py_engine;
  py::object py_rules_cls;
  py::object py_level_cls;

 private:
  DiagnosticPythonWrapper() {
    py::object diagnostic_module = py::module::import("torch.onnx.diagnostic");
    py_engine = diagnostic_module.attr("engine");
    py_rules_cls = diagnostic_module.attr("rules");
    py_level_cls = diagnostic_module.attr("Level");
  }
};

// TODO: more arguments:
//  locations,
//  related_locations,
//  IR for related node, value.
inline void Diagnose(
    Rule rule,
    Level level,
    std::vector<std::string> messageArgs = {}) {
  auto& python_wrapper = DiagnosticPythonWrapper::get();
  py::object py_rule =
      python_wrapper.py_rules_cls.attr(RuleNames[(unsigned)rule]);
  py::object py_level =
      python_wrapper.py_level_cls.attr(LevelNames[(unsigned)level]);
  py::dict kwargs = py::dict();
  // TODO: statically check that size of messageArgs matches with rule.
  kwargs["message_args"] = messageArgs;
  python_wrapper.py_engine.attr("diagnose")(py_rule, py_level, **kwargs);
}

} // namespace diagnostic
} // namespace onnx
} // namespace torch
