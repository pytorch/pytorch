#pragma once

#include <c10/util/Logging.h>

#include <vector>

#include "lazy_tensors/str_join.h"

namespace lazy_tensors {

std::vector<int64_t> InversePermutation(
    c10::ArrayRef<int64_t> input_permutation);

bool IsPermutation(c10::ArrayRef<int64_t> permutation);

bool IsIdentityPermutation(c10::ArrayRef<int64_t> permutation);

template <typename Container>
inline std::vector<typename Container::value_type> PermuteInverse(
    const Container& input, c10::ArrayRef<int64_t> permutation) {
  using T = typename Container::value_type;
  c10::ArrayRef<T> data(input);
  CHECK(IsPermutation(permutation));
  std::vector<T> output(data.size());
  for (size_t i = 0; i < permutation.size(); ++i) {
    output[permutation[i]] = data[i];
  }
  return output;
}

}  // namespace lazy_tensors
