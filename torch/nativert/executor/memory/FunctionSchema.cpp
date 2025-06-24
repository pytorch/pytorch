#include <torch/nativert/executor/memory/FunctionSchema.h>

namespace torch::nativert {

bool FunctionSchema::alias(size_t input_idx, size_t output_idx) const {
  // probably quicker than using a map since
  // overridden inputs/outputs should be small
  for (const auto& [i, o] : aliasing_spec_) {
    if (i == input_idx && o == output_idx) {
      return true;
    }
  }

  if (!aliasing_spec_.empty()) {
    VLOG(1) << "aliasing spec is not empty but no entry found for ("
            << input_idx << "-->" << output_idx
            << ") -- falling back to schema->may_contain_alias()";
  }

  return c10_fn_schema_.may_contain_alias(
      {c10::SchemaArgType::output, output_idx},
      {c10::SchemaArgType::input, input_idx},
      /* bidirectional = */ false);
}

} // namespace torch::nativert
