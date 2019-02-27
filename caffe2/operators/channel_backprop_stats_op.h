#ifndef CAFFE2_OPERATORS_CHANNEL_BACKPROP_STATS_OP_H_
#define CAFFE2_OPERATORS_CHANNEL_BACKPROP_STATS_OP_H_

#include <string>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class ChannelBackpropStatsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ChannelBackpropStatsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<std::string>("order", "NCHW"))) {
    CAFFE_ENFORCE_NE(order_, StorageOrder::UNKNOWN);
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(INPUT);
    const auto& dY = Input(OUTPUT_GRAD);
    const auto& mean = Input(SAVED_MEAN);
    const auto& rstd = Input(SAVED_RSTD);
    const int ndim = X.dim();
    const int N = X.dim32(0);
    const int C = order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1);
    const int HxW = X.numel() / (N * C);
    auto* dscale = Output(SCALE_GRAD, {C}, at::dtype<T>());
    auto* dbias = Output(BIAS_GRAD, {C}, at::dtype<T>());
    const T* dY_data = dY.template data<T>();
    const T* X_data = X.template data<T>();
    const T* mean_data = mean.template data<T>();
    const T* rstd_data = rstd.template data<T>();
    T* dscale_data = dscale->template mutable_data<T>();
    T* dbias_data = dbias->template mutable_data<T>();
    if (order_ == StorageOrder::NCHW) {
      return ChannelStatsBackwardNCHW<T>(
          N,
          C,
          HxW,
          dY_data,
          X_data,
          mean_data,
          rstd_data,
          dscale_data,
          dbias_data);
    } else {
      return ChannelStatsBackwardNHWC<T>(
          N,
          C,
          HxW,
          dY_data,
          X_data,
          mean_data,
          rstd_data,
          dscale_data,
          dbias_data);
    }
  }

 private:
  template <typename T>
  bool ChannelStatsBackwardNCHW(
      int N,
      int C,
      int HxW,
      const T* dY,
      const T* X,
      const T* mean,
      const T* rstd,
      T* dscale,
      T* dbias);

  template <typename T>
  bool ChannelStatsBackwardNHWC(
      int N,
      int C,
      int HxW,
      const T* dY,
      const T* X,
      const T* mean,
      const T* rstd,
      T* dscale,
      T* dbias);

  const StorageOrder order_;

  INPUT_TAGS(INPUT, SAVED_MEAN, SAVED_RSTD, OUTPUT_GRAD);
  OUTPUT_TAGS(SCALE_GRAD, BIAS_GRAD);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_CHANNEL_BACKPROP_STATS_OP_H_
