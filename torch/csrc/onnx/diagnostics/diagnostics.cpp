// #include <torch/csrc/onnx/diagnostics/diagnostics.h>

// namespace torch {
// namespace onnx {
// namespace diagnostics {

// py::object _PyDiagnostics() {
//   return py::module::import("torch.onnx._internal.diagnostics");
// }

// py::object _PyEngine() {
//   return _PyDiagnostics().attr("engine");
// }

// py::object _PyRule(Rule rule) {
//   return _PyDiagnostics().attr("rules").attr(
//       kPyRuleNames[static_cast<uint32_t>(rule)]);
// }

// py::object _PyLevel(Level level) {
//   return _PyDiagnostics().attr("levels").attr(
//       kPyLevelNames[static_cast<uint32_t>(level)]);
// }

// // TODO: more arguments:
// //  locations,
// //  related_locations,
// //  IR for related node, value.
// void Diagnose(Rule rule, Level level, std::vector<std::string> messageArgs) {
//   py::object py_rule = _PyRule(rule);
//   py::object py_level = _PyLevel(level);
//   py::object py_engine = _PyEngine();

//   py::dict kwargs = py::dict();
//   // TODO: statically check that size of messageArgs matches with rule.
//   kwargs["message_args"] = messageArgs;
//   py_engine.attr("diagnose")(py_rule, py_level, **kwargs);
// }

// } // namespace diagnostics
// } // namespace onnx
// } // namespace torch
