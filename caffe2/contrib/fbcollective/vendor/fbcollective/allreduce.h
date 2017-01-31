#pragma once

#include "fbcollective/algorithm.h"

namespace fbcollective {

template <typename T>
class Allreduce : public Algorithm {
 public:
  using ReduceFunction = void(T*, const T*, size_t n);

  Allreduce(const std::shared_ptr<Context>& context, ReduceFunction fn)
      : Algorithm(context), fn_(fn) {
    if (fn_ == nullptr) {
      // Default to addition
      fn_ = [](T* dst, const T* src, size_t n) {
        for (int i = 0; i < n; i++) {
          dst[i] += src[i];
        }
      };
    }
  }

  virtual ~Allreduce(){};

 protected:
  ReduceFunction* fn_;
};

} // namespace fbcollective
