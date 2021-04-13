#include "lazy_tensors/permutation_util.h"

#include <algorithm>
#include <numeric>

namespace lazy_tensors {

std::vector<int64> InversePermutation(
    lazy_tensors::Span<const int64> input_permutation) {
  DCHECK(IsPermutation(input_permutation));
  std::vector<int64> output_permutation(input_permutation.size(), -1);
  for (size_t i = 0; i < input_permutation.size(); ++i) {
    output_permutation.at(input_permutation.at(i)) = i;
  }
  return output_permutation;
}

bool IsPermutation(lazy_tensors::Span<const int64> permutation) {
  std::vector<int64> trivial_permutation(permutation.size());
  std::iota(trivial_permutation.begin(), trivial_permutation.end(), 0);
  return std::is_permutation(permutation.begin(), permutation.end(),
                             trivial_permutation.begin());
}

bool IsIdentityPermutation(lazy_tensors::Span<const int64> permutation) {
  for (int64 i = 0; i < permutation.size(); ++i) {
    if (permutation[i] != i) {
      return false;
    }
  }
  return true;
}

}  // namespace lazy_tensors
