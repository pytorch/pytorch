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
}
#endif
