#ifndef CAFFE2_OPERATORS_ROI_ALIGN_OP_H_
#define CAFFE2_OPERATORS_ROI_ALIGN_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/export_caffe2_op_to_c10.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(RoIAlign)

namespace caffe2 {

template <typename T, class Context>
class RoIAlignOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit RoIAlignOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<string>("order", "NCHW"))),
        OP_SINGLE_ARG(float, "spatial_scale", spatial_scale_, 1.0f),
        OP_SINGLE_ARG(int, "pooled_h", pooled_h_, 1),
        OP_SINGLE_ARG(int, "pooled_w", pooled_w_, 1),
        OP_SINGLE_ARG(int, "sampling_ratio", sampling_ratio_, -1),
        OP_SINGLE_ARG(bool, "aligned", aligned_, false) {
    TORCH_DCHECK_GT(spatial_scale_, 0.0f);
    TORCH_DCHECK_GT(pooled_h_, 0);
    TORCH_DCHECK_GT(pooled_w_, 0);
    DCHECK(order_ == StorageOrder::NCHW || order_ == StorageOrder::NHWC);
  }

  bool RunOnDevice() override {
    const auto& X = Input(0);
    const auto& R = Input(1);

    CAFFE_ENFORCE_EQ(X.dim(), 4);
    CAFFE_ENFORCE_EQ(R.dim(), 2);
    const int64_t roi_cols = R.size(1);
    CAFFE_ENFORCE(roi_cols == 4 || roi_cols == 5);
    const int64_t N = R.size(0);
    const int64_t C = X.size(order_ == StorageOrder::NCHW ? 1 : 3);
    const int64_t H = X.size(order_ == StorageOrder::NCHW ? 2 : 1);
    const int64_t W = X.size(order_ == StorageOrder::NCHW ? 3 : 2);
    const std::vector<int64_t> Y_sizes = order_ == StorageOrder::NCHW
        ? std::vector<int64_t>{N, C, pooled_h_, pooled_w_}
        : std::vector<int64_t>{N, pooled_h_, pooled_w_, C};

    auto* Y = Output(0, Y_sizes, at::dtype<T>());
    if (N == 0) {
      return true;
    }
    const T* X_data = X.template data<T>();
    const T* R_data = R.template data<T>();
    T* Y_data = Y->template mutable_data<T>();
    return order_ == StorageOrder::NCHW
        ? RunOnDeviceWithOrderNCHW(N, C, H, W, roi_cols, X_data, R_data, Y_data)
        : RunOnDeviceWithOrderNHWC(
              N, C, H, W, roi_cols, X_data, R_data, Y_data);
  }

 private:
  bool RunOnDeviceWithOrderNCHW(
      int64_t N,
      int64_t C,
      int64_t H,
      int64_t W,
      int64_t roi_cols,
      const T* X,
      const T* R,
      T* Y);

  bool RunOnDeviceWithOrderNHWC(
      int64_t N,
      int64_t C,
      int64_t H,
      int64_t W,
      int64_t roi_cols,
      const T* X,
      const T* R,
      T* Y);

  const StorageOrder order_;
  const float spatial_scale_;
  const int pooled_h_;
  const int pooled_w_;
  const int sampling_ratio_;
  const bool aligned_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_ROI_ALIGN_OP_H_
