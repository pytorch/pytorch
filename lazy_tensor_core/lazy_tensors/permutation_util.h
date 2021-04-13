#pragma once

#include <vector>

#include "absl/strings/str_join.h"
#include "lazy_tensors/span.h"
#include "lazy_tensors/types.h"

namespace lazy_tensors {

std::vector<int64> InversePermutation(
    lazy_tensors::Span<const int64> input_permutation);

bool IsPermutation(lazy_tensors::Span<const int64> permutation);

bool IsIdentityPermutation(lazy_tensors::Span<const int64> permutation);

template <typename Container>
inline std::vector<typename Container::value_type> PermuteInverse(
    const Container& input, lazy_tensors::Span<const int64> permutation) {
  using T = typename Container::value_type;
  lazy_tensors::Span<const T> data(input);
  CHECK(IsPermutation(permutation));
  std::vector<T> output(data.size());
  for (size_t i = 0; i < permutation.size(); ++i) {
    output[permutation[i]] = data[i];
  }
  return output;
}

}  // namespace lazy_tensors
