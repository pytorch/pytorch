#ifndef CAFFE2_OPERATORS_IM2COL_OP_H_
#define CAFFE2_OPERATORS_IM2COL_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "c10/util/irange.h"

namespace caffe2 {

template <typename T, class Context>
class Im2ColOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit Im2ColOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        pad_(this->template GetSingleArgument<int>("pad", 0)),
        kernel_h_(this->template GetSingleArgument<int>(
            "kernel_h",
            this->template GetSingleArgument<int>("kernel", 0))),
        kernel_w_(this->template GetSingleArgument<int>(
            "kernel_w",
            this->template GetSingleArgument<int>("kernel", 0))),
        dilation_h_(this->template GetSingleArgument<int>(
            "dilation_h",
            this->template GetSingleArgument<int>("dilation", 1))),
        dilation_w_(this->template GetSingleArgument<int>(
            "dilation_w",
            this->template GetSingleArgument<int>("dilation", 1))),
        stride_h_(this->template GetSingleArgument<int>(
            "stride_h",
            this->template GetSingleArgument<int>("stride", 1))),
        stride_w_(this->template GetSingleArgument<int>(
            "stride_w",
            this->template GetSingleArgument<int>("stride", 1))),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_ENFORCE(kernel_h_ > 0);
    CAFFE_ENFORCE(kernel_w_ > 0);
    CAFFE_ENFORCE(dilation_h_ > 0);
    CAFFE_ENFORCE(dilation_w_ > 0);
    CAFFE_ENFORCE(stride_h_ > 0);
    CAFFE_ENFORCE(stride_w_ > 0);
    CAFFE_ENFORCE(pad_ >= 0);
  }

  bool RunOnDevice() override {
    auto& X = Input(0);

    CAFFE_ENFORCE(4 == X.dim());

    int N = 0, C = 0, H = 0, W = 0;
    switch (order_) {
      case StorageOrder::NCHW:
        N = X.dim32(0);
        C = X.dim32(1);
        H = X.dim32(2);
        W = X.dim32(3);
        break;
      case StorageOrder::NHWC:
        N = X.dim32(0);
        H = X.dim32(1);
        W = X.dim32(2);
        C = X.dim32(3);
        break;
      default:
        CAFFE_THROW("Unknown storage order: ", order_);
    }

    const int dkernel_h = dilation_h_ * (kernel_h_ - 1) + 1;
    const int dkernel_w = dilation_w_ * (kernel_w_ - 1) + 1;
    CAFFE_ENFORCE(H >= dkernel_h);
    CAFFE_ENFORCE(W >= dkernel_w);
    const int out_h = (H + 2 * pad_ - dkernel_h) / stride_h_ + 1;
    const int out_w = (W + 2 * pad_ - dkernel_w) / stride_w_ + 1;

    switch (order_) {
      case StorageOrder::NCHW: {
        auto* Y = Output(
            0,
            std::vector<int64_t>{N, C * kernel_h_ * kernel_w_, out_h, out_w},
            at::dtype<T>());

        const size_t dx = X.numel() / N;
        const size_t dy = Y->numel() / N;
        for (const auto n : c10::irange(N)) {
          const auto* xdata = X.template data<T>() + (n * dx);
          auto* ydata = Y->template mutable_data<T>() + (n * dy);
          math::Im2Col<T, Context, StorageOrder::NCHW>(
              C,
              H,
              W,
              kernel_h_,
              kernel_w_,
              dilation_h_,
              dilation_w_,
              pad_,
              pad_,
              pad_,
              pad_,
              stride_h_,
              stride_w_,
              xdata,
              ydata,
              &context_);
        }
      }; break;
      case StorageOrder::NHWC: {
        auto* Y = Output(
            0,
            std::vector<int64_t>{N, out_h, out_w, kernel_h_ * kernel_w_ * C},
            at::dtype<T>());

        const size_t dx = X.numel() / N;
        const size_t dy = Y->numel() / N;
        for (const auto n : c10::irange(N)) {
          const auto* xdata = X.template data<T>() + (n * dx);
          auto* ydata = Y->template mutable_data<T>() + (n * dy);
          math::Im2Col<T, Context, StorageOrder::NHWC>(
              C,
              H,
              W,
              kernel_h_,
              kernel_w_,
              dilation_h_,
              dilation_w_,
              pad_,
              pad_,
              pad_,
              pad_,
              stride_h_,
              stride_w_,
              xdata,
              ydata,
              &context_);
        }
      }; break;
      default:
        CAFFE_THROW("Unknown storage order: ", order_);
    }

    return true;
  }

 private:
  int pad_;
  int kernel_h_;
  int kernel_w_;
  int dilation_h_;
  int dilation_w_;
  int stride_h_;
  int stride_w_;
  StorageOrder order_;
};

template <typename T, class Context>
class Col2ImOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit Col2ImOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        pad_(this->template GetSingleArgument<int>("pad", 0)),
        kernel_h_(this->template GetSingleArgument<int>(
            "kernel_h",
            this->template GetSingleArgument<int>("kernel", 0))),
        kernel_w_(this->template GetSingleArgument<int>(
            "kernel_w",
            this->template GetSingleArgument<int>("kernel", 0))),
        dilation_h_(this->template GetSingleArgument<int>(
            "dilation_h",
            this->template GetSingleArgument<int>("dilation", 1))),
        dilation_w_(this->template GetSingleArgument<int>(
            "dilation_w",
            this->template GetSingleArgument<int>("dilation", 1))),
        stride_h_(this->template GetSingleArgument<int>(
            "stride_h",
            this->template GetSingleArgument<int>("stride", 1))),
        stride_w_(this->template GetSingleArgument<int>(
            "stride_w",
            this->template GetSingleArgument<int>("stride", 1))),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_ENFORCE(kernel_h_ > 0);
    CAFFE_ENFORCE(kernel_w_ > 0);
    CAFFE_ENFORCE(dilation_h_ > 0);
    CAFFE_ENFORCE(dilation_w_ > 0);
    CAFFE_ENFORCE(stride_h_ > 0);
    CAFFE_ENFORCE(stride_w_ > 0);
    CAFFE_ENFORCE(pad_ >= 0);
  }

  bool RunOnDevice() override {
    auto& X = Input(0);
    auto& Z = Input(1);

    auto* Y = Output(0, Z.sizes(), at::dtype<T>());
    CAFFE_ENFORCE(4 == Y->dim());

    int N = 0, C = 0, H = 0, W = 0;
    switch (order_) {
      case StorageOrder::NCHW:
        N = Y->dim32(0);
        C = Y->dim32(1);
        H = Y->dim32(2);
        W = Y->dim32(3);
        break;
      case StorageOrder::NHWC:
        N = Y->dim32(0);
        H = Y->dim32(1);
        W = Y->dim32(2);
        C = Y->dim32(3);
        break;
      default:
        CAFFE_THROW("Unknown storage order: ", order_);
    }

    const int dkernel_h = dilation_h_ * (kernel_h_ - 1) + 1;
    const int dkernel_w = dilation_w_ * (kernel_w_ - 1) + 1;
    CAFFE_ENFORCE(H >= dkernel_h);
    CAFFE_ENFORCE(W >= dkernel_w);
    const int out_h = (H + 2 * pad_ - dkernel_h) / stride_h_ + 1;
    const int out_w = (W + 2 * pad_ - dkernel_w) / stride_w_ + 1;
    CAFFE_ENFORCE(X.numel() == N * kernel_h_ * kernel_w_ * C * out_h * out_w);

    const size_t dx = X.numel() / N;
    const size_t dy = Y->numel() / N;

    // could template-specialize this, but it's test code...
    switch (order_) {
      case StorageOrder::NCHW: {
        for (const auto n : c10::irange(N)) {
          const auto* xdata = X.template data<T>() + (n * dx);
          auto* ydata = Y->template mutable_data<T>() + (n * dy);
          math::Col2Im<T, Context, StorageOrder::NCHW>(
              C,
              H,
              W,
              kernel_h_,
              kernel_w_,
              dilation_h_,
              dilation_w_,
              pad_,
              pad_,
              pad_,
              pad_,
              stride_h_,
              stride_w_,
              xdata,
              ydata,
              &context_);
        }
      }; break;
      case StorageOrder::NHWC: {
        for (const auto n : c10::irange(N)) {
          const auto* xdata = X.template data<T>() + (n * dx);
          auto* ydata = Y->template mutable_data<T>() + (n * dy);
          math::Col2Im<T, Context, StorageOrder::NHWC>(
              C,
              H,
              W,
              kernel_h_,
              kernel_w_,
              dilation_h_,
              dilation_w_,
              pad_,
              pad_,
              pad_,
              pad_,
              stride_h_,
              stride_w_,
              xdata,
              ydata,
              &context_);
        }
      }; break;
      default:
        CAFFE_THROW("Unknown storage order: ", order_);
    }

    return true;
  }

 private:
  int pad_;
  int kernel_h_;
  int kernel_w_;
  int dilation_h_;
  int dilation_w_;
  int stride_h_;
  int stride_w_;
  StorageOrder order_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_IM2COL_OP_H_
