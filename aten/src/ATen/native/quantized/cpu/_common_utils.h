#pragma once

namespace _internal {
// In Conv1d and ConvTranspose1d we use 2d equivalents to implement the 1d
// versions. This variable and utility make sure the arguments are correct.
constexpr int64_t kConv1dSqueezeDim = 0;
static torch::List<int64_t> MakeArgForConv1d(const torch::List<int64_t>& arg,
                                             int64_t base_value) {
  TORCH_CHECK(arg.size() > 0, "Argument must have elements.");
  torch::List<int64_t> result({arg.get(0), base_value});
  if (arg.size() == 1) {
    result[1] = arg.get(0);
  } else {
    result[1] = arg.get(1);
  }
  result[kConv1dSqueezeDim] = base_value;
  return result;
}
} // namespace _internal
