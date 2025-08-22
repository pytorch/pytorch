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

  VLOG(1) << "checking aliasing spec for " << c10_fn_schema_.name() << " "
          << (c10_fn_schema_.is_varret() ? "varret" : "non-varret") << " "
          << (c10_fn_schema_.is_vararg() ? "vararg" : "non-vararg");

  if (!aliasing_spec_.empty()) {
    VLOG(1) << "aliasing spec is not empty but no entry found for ("
            << input_idx << "-->" << output_idx
            << ") -- falling back to schema->may_contain_alias()";
  }

  /*
    varret and vararg will contribute to the input/output idx's
    but because we don't know how many inputs/outputs there are,
    the schema will consider these indices to be out of bounds.

    e.g., op(a, b, c, d) where c and d are variadic will result in
    may_contain_alias(x, idx_of(c)) and may_contain_alias(x, idx_of(d)) to throw
    an out-of-bounds exception

    in this case, we can apply the worst-case aliasing to the varidic
    inputs/outputs i.e., all outputs might alias all varargs and all inputs
    might be aliased by all varrets
  */

  if (c10_fn_schema_.is_vararg() &&
      input_idx >= c10_fn_schema_.arguments().size()) {
    VLOG(1) << "applying worst-case aliasing for " << c10_fn_schema_.name()
            << "'s variadic input " << input_idx;
    return true;
  }

  if (c10_fn_schema_.is_varret() &&
      output_idx >= c10_fn_schema_.returns().size()) {
    VLOG(1) << "applying worst-case aliasing for " << c10_fn_schema_.name()
            << "'s variadic output " << output_idx;
    return true;
  }

  return c10_fn_schema_.may_contain_alias(
      {c10::SchemaArgType::output, output_idx},
      {c10::SchemaArgType::input, input_idx},
      /* bidirectional = */ false);
}

} // namespace torch::nativert
