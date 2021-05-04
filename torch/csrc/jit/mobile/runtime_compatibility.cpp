#include <ATen/core/dispatch/Dispatcher.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/mobile/runtime_compatibility.h>
#include <torch/csrc/jit/runtime/operator.h>

namespace torch {
namespace jit {

uint64_t _get_runtime_bytecode_version() {
  return caffe2::serialize::kProducedBytecodeVersion;
}

/*
 * Returns all registered PyTorch ops and their versioning
 */
std::unordered_map<std::string, OperatorInfo> _get_runtime_ops_and_info() {
  std::unordered_map<std::string, OperatorInfo> result;

  // Grab the jit operators
  auto nonDispatcherOperators = torch::jit::getAllOperators();
  for (const auto& full_op : nonDispatcherOperators) {
    auto op = full_op->schema();
    int num_schema_args = op.arguments().size();
    if (op.overload_name() != "") {
      result.emplace(
          op.name() + "." + op.overload_name(), OperatorInfo{num_schema_args});
    } else {
      result.emplace(op.name(), OperatorInfo{num_schema_args});
    }
  }

  // Grab the dispatcher operators
  auto dispatcherOperators = c10::Dispatcher::singleton().getAllOpNames();
  for (auto& op : dispatcherOperators) {
    // grab schema
    const auto op_handle = c10::Dispatcher::singleton().findOp(op);
    int num_schema_args = NO_SCHEMA;
    if (op_handle->hasSchema()) {
      num_schema_args = op_handle->schema().arguments().size();
    }

    if (op.overload_name != "") {
      result.emplace(
          op.name + "." + op.overload_name, OperatorInfo{num_schema_args});
    } else {
      result.emplace(op.name, OperatorInfo{num_schema_args});
    }
  }

  return result;
}

} // namespace jit
} // namespace torch
