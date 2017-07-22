#ifndef CAFFE2_OPERATORS_CONV_OP_CACHE_H_
#define CAFFE2_OPERATORS_CONV_OP_CACHE_H_

#include <functional>
#include <unordered_map>
#include <vector>

#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {
template <typename T>
class AlgorithmsCache {
 public:
  T getAlgorithm(
      const std::vector<TIndex>& bottom,
      const std::vector<TIndex>& desc,
      std::function<T()> generatingFunc);

 private:
  std::unordered_map<int64_t, T> hash_;
};

template <typename T>
T AlgorithmsCache<T>::getAlgorithm(
    const std::vector<TIndex>& vec1,
    const std::vector<TIndex>& vec2,
    std::function<T()> generatingFunc) {
  int64_t seed = 0;
  std::hash<TIndex> hashFn;
  for (const auto num : vec1) {
    // Copied from boost::hash_combine.
    // Adding 1 to differentiate between first and second vector.
    seed ^= hashFn(num) + 0x9e3779b9 + (seed << 6) + (seed >> 2) + 1;
  }

  for (const auto num : vec2) {
    // Copied from boost::hash_combine.
    seed ^= hashFn(num) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }

  if (seed == 0) {
    return generatingFunc();
  }

  if (hash_.find(seed) == hash_.end()) {
    T value = generatingFunc();
    hash_[seed] = value;
  }

  return hash_[seed];
}
}
#endif
