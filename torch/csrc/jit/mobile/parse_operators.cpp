#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/mobile/parse_operators.h>

namespace torch {
namespace jit {
namespace mobile {

std::string operator_str(
    const std::string& name,
    const std::string& overloadname) {
  std::string result = name;
  if (!overloadname.empty()) {
    result += "." + overloadname;
  }
  return result;
}

/**
 * Loads operators by looking them up in the Dispatcher and returns
 * the set of operator names (with overload) that are not supported
 * by the current runtime.
 */
std::unordered_set<std::string> load_and_find_unsupported_operator_names(
    const std::vector<IValue>& ops_list,
    mobile::Function* function,
    int64_t model_version) {
  std::unordered_set<std::string> unsupported_op_names;
  // ops_list is the list of operator names that were read in from
  // bytecode.plk for the method that is currently being processed.
  for (const auto& op : ops_list) {
    auto op_item = op.toTupleRef().elements();
    TORCH_CHECK(
        op_item.size() >= 2,
        "There should be either two parts (name and overload name), ",
        "or three parts (name, overload name and number of specified args) ",
        "for an operator");
    c10::optional<int> num_args;
    if (op_item.size() > 2) {
      num_args = op_item[2].toInt();
    }
    auto op_found = function->append_operator(
        op_item[0].toString()->string(),
        op_item[1].toString()->string(),
        num_args,
        model_version);
    if (!op_found) {
      unsupported_op_names.emplace(operator_str(
          op_item[0].toString()->string(), op_item[1].toString()->string()));
    }
  }
  return unsupported_op_names;
}

void print_unsupported_ops_and_throw(
    const std::unordered_set<std::string>& unsupported_ops) {
  std::string error_message("{");
  for (const auto& op_name : unsupported_ops) {
    error_message += op_name + ", ";
  }
  error_message += "}";
  TORCH_CHECK(
      false,
      "Following ops cannot be found. ",
      "Check fburl.com/missing_ops for the fix.",
      error_message);
}

void parseOperators(
    const std::vector<IValue>& ops_list,
    const int64_t& model_version,
    const uint64_t& module_load_options,
    mobile::Function* function) {
  std::unordered_set<std::string> unsupported_op_names =
      load_and_find_unsupported_operator_names(
          ops_list, function, model_version);
  if ((module_load_options & MobileModuleLoadOptions::OPERATOR_CHECK) &&
      !unsupported_op_names.empty()) {
    print_unsupported_ops_and_throw(unsupported_op_names);
  }
}

} // namespace mobile
} // namespace jit
} // namespace torch
