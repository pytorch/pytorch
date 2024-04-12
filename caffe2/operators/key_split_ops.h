#pragma once

#include <c10/util/irange.h>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

#include <vector>

namespace caffe2 {
template <typename T, class Context>
class KeySplitOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit KeySplitOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        categorical_limit_(
            this->template GetSingleArgument<int>("categorical_limit", 0)) {
    CAFFE_ENFORCE_GT(categorical_limit_, 0);
  }

  bool RunOnDevice() override {
    auto& keys = Input(0);
    const auto N = keys.numel();
    const T *const keys_data = keys.template data<T>();
    std::vector<int> counts(categorical_limit_);
    std::vector<int*> eids(categorical_limit_);
    for (const auto k : c10::irange(categorical_limit_)) {
      counts[k] = 0;
    }
    for (const auto i : c10::irange(N)) {
      const auto k = keys_data[i];
      CAFFE_ENFORCE_GT(categorical_limit_, k);
      CAFFE_ENFORCE_GE(k, 0);
      counts[k]++;
    }
    for (const auto k : c10::irange(categorical_limit_)) {
      auto *const eid = Output(k, {counts[k]}, at::dtype<int>());
      eids[k] = eid->template mutable_data<int>();
      counts[k] = 0;
    }
    for (const auto i : c10::irange(N)) {
      const auto k = keys_data[i];
      eids[k][counts[k]++] = i;
    }
    return true;
  }

 private:
  int categorical_limit_;
};
} // namespace caffe2
