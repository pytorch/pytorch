#pragma once

#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

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
    int N = keys.numel();
    const T* keys_data = keys.template data<T>();
    std::vector<int> counts(categorical_limit_);
    std::vector<int*> eids(categorical_limit_);
    for (int k = 0; k < categorical_limit_; k++) {
      counts[k] = 0;
    }
    for (int i = 0; i < N; i++) {
      int k = keys_data[i];
      CAFFE_ENFORCE_GT(categorical_limit_, k);
      CAFFE_ENFORCE_GE(k, 0);
      counts[k]++;
    }
    for (int k = 0; k < categorical_limit_; k++) {
      auto* eid = Output(k, {counts[k]}, at::dtype<int>());
      eids[k] = eid->template mutable_data<int>();
      counts[k] = 0;
    }
    for (int i = 0; i < N; i++) {
      int k = keys_data[i];
      eids[k][counts[k]++] = i;
    }
    return true;
  }

 private:
  int categorical_limit_;
};
} // namespace caffe2
