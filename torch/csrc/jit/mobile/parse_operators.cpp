#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/mobile/parse_operators.h>

namespace torch {
namespace jit {
namespace mobile {

void parseOperators(
    c10::ivalue::TupleElements&& ops_list,
    const uint64_t& module_load_options,
    mobile::Function* function) {
  for (auto& op : std::move(ops_list)) {
    auto op_item = std::move(*std::move(op).toTuple()).elements();
    TORCH_CHECK(
        op_item.size() >= 2,
        "There should be either two parts (name and overload name), ",
        "or three parts (name, overload name and number of specified args) ",
        "for an operator");
    c10::optional<int> num_args;
    if (op_item.size() > 2) {
      num_args = op_item[2].toInt();
    }
    function->append_operator(
        op_item[0].toStringRef(), op_item[1].toStringRef(), num_args);
  }
  function->initialize_operators(
      (module_load_options & MobileModuleLoadOptions::OPERATOR_CHECK));
}

} // namespace mobile
} // namespace jit
} // namespace torch
