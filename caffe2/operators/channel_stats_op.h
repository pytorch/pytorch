#ifndef CAFFE2_OPERATORS_CHANNEL_STATS_OP_H_
#define CAFFE2_OPERATORS_CHANNEL_STATS_OP_H_

#include <string>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class ChannelStatsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit ChannelStatsOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<std::string>("order", "NCHW"))) {
    CAFFE_ENFORCE_NE(order_, StorageOrder::UNKNOWN);
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(0);
    const int ndim = X.dim();
    const int N = X.dim32(0);
    const int C = order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1);
    const int HxW = X.numel() / (N * C);
    auto* sum = Output(0, {C}, at::dtype<T>());
    auto* sumsq = Output(1, {C}, at::dtype<T>());
    const T* X_data = X.template data<T>();
    T* sum_data = sum->template mutable_data<T>();
    T* sumsq_data = sumsq->template mutable_data<T>();
    return order_ == StorageOrder::NCHW
        ? ComputeChannelStatsNCHW<T>(N, C, HxW, X_data, sum_data, sumsq_data)
        : ComputeChannelStatsNHWC<T>(N, C, HxW, X_data, sum_data, sumsq_data);
  }

 private:
  template <typename T>
  bool
  ComputeChannelStatsNCHW(int N, int C, int HxW, const T* X, T* sum, T* sumsq);

  template <typename T>
  bool
  ComputeChannelStatsNHWC(int N, int C, int HxW, const T* X, T* sum, T* sumsq);

  const StorageOrder order_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_CHANNEL_STATS_OP_H_
