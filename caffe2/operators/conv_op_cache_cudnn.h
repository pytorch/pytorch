#ifndef CAFFE2_OPERATORS_CONV_OP_CACHE_H_
#define CAFFE2_OPERATORS_CONV_OP_CACHE_H_

#include <functional>
#include <unordered_map>
#include <vector>

#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {
template <typename TAlgorithm>
class AlgorithmsCache {
 public:
  // Caches the best algorithm for a given
  // combination of tensor dimensions & compute data type.
  //
  TAlgorithm getAlgorithm(
      at::IntArrayRef tensorDimensions1,
      at::IntArrayRef tensorDimensions2,
      int algorithmFlags, // Differentiate between algorithms with different
                          // parameters in a generic way
      std::function<TAlgorithm()> generatingFunc);

 private:
  std::unordered_map<int64_t, TAlgorithm> hash_;
};

template <typename TAlgorithm>
TAlgorithm AlgorithmsCache<TAlgorithm>::getAlgorithm(
    at::IntArrayRef tensorDimensions1,
    at::IntArrayRef tensorDimensions2,
    int algorithmFlags,
    std::function<TAlgorithm()> generatingFunc) {
  int64_t seed = 0;
  // Hash all of the inputs, which we wiill then use to try and look up
  // a previously discovered algorithm, or fall back to generating a new one.
  std::hash<int64_t> hashFn;
  for (const auto num : tensorDimensions1) {
    // Copied from boost::hash_combine.
    // Adding 1 to differentiate between first and second vector.
    seed ^= hashFn(num) + 0x9e3779b9 + (seed << 6) + (seed >> 2) + 1;
  }

  for (const auto num : tensorDimensions2) {
    // Copied from boost::hash_combine.
    seed ^= hashFn(num) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }

  // Adding 2 to differentiate from previous vectors
  seed ^= hashFn(algorithmFlags) + 0x9e3779b9 + (seed << 6) + (seed >> 2) + 2;

  if (seed == 0) {
    return generatingFunc();
  }

  if (hash_.find(seed) == hash_.end()) {
    TAlgorithm value = generatingFunc();
    hash_[seed] = value;
  }

  return hash_[seed];
}
} // namespace caffe2

#endif
