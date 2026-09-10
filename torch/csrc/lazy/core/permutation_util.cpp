#include <c10/util/irange.h>
#include <torch/csrc/lazy/core/permutation_util.h>

#include <algorithm>
#include <numeric>

namespace torch::lazy {

std::vector<int64_t> InversePermutation(
    c10::ArrayRef<int64_t> input_permutation) {
  TORCH_CHECK(IsPermutation(input_permutation));
  std::vector<int64_t> output_permutation(input_permutation.size(), -1);
  for (const auto i : c10::irange(input_permutation.size())) {
    output_permutation.at(input_permutation.at(i)) = static_cast<int64_t>(i);
  }
  return output_permutation;
}

bool IsPermutation(c10::ArrayRef<int64_t> permutation) {
  std::vector<int64_t> trivial_permutation(permutation.size());
  std::iota(trivial_permutation.begin(), trivial_permutation.end(), 0);
  return std::is_permutation(
      permutation.begin(), permutation.end(), trivial_permutation.begin());
}

} // namespace torch::lazy
