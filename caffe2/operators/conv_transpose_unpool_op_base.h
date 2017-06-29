#ifndef CAFFE2_OPERATORS_CONV_TRANSPOSE_UNPOOL_OP_BASE_H_
#define CAFFE2_OPERATORS_CONV_TRANSPOSE_UNPOOL_OP_BASE_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_op_shared.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/proto/caffe2_legacy.pb.h"
#include "caffe2/utils/math.h"

CAFFE2_DECLARE_bool(caffe2_force_shared_col_buffer);

namespace caffe2 {

template <class Context>
class ConvTransposeUnpoolBase : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ConvTransposeUnpoolBase(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        legacy_pad_(
            static_cast<LegacyPadding>(OperatorBase::GetSingleArgument<int>(
                "legacy_pad",
                LegacyPadding::NOTSET))),
        pad_(OperatorBase::GetSingleArgument<int>("pad", 0)),
        pad_t_(OperatorBase::GetSingleArgument<int>("pad_t", pad_)),
        pad_l_(OperatorBase::GetSingleArgument<int>("pad_l", pad_)),
        pad_b_(OperatorBase::GetSingleArgument<int>("pad_b", pad_)),
        pad_r_(OperatorBase::GetSingleArgument<int>("pad_r", pad_)),
        kernel_h_(OperatorBase::GetSingleArgument<int>(
            "kernel_h",
            OperatorBase::GetSingleArgument<int>("kernel", 0))),
        kernel_w_(OperatorBase::GetSingleArgument<int>(
            "kernel_w",
            OperatorBase::GetSingleArgument<int>("kernel", 0))),
        stride_h_(OperatorBase::GetSingleArgument<int>(
            "stride_h",
            OperatorBase::GetSingleArgument<int>("stride", 1))),
        stride_w_(OperatorBase::GetSingleArgument<int>(
            "stride_w",
            OperatorBase::GetSingleArgument<int>("stride", 1))),
        adj_h_(OperatorBase::GetSingleArgument<int>(
            "adj_h",
            OperatorBase::GetSingleArgument<int>("adj", 0))),
        adj_w_(OperatorBase::GetSingleArgument<int>(
            "adj_w",
            OperatorBase::GetSingleArgument<int>("adj", 0))),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))),
        shared_buffer_(
            OperatorBase::GetSingleArgument<int>("shared_buffer", 0)),
        ws_(ws) {
    CAFFE_ENFORCE(kernel_h_ > 0);
    CAFFE_ENFORCE(kernel_w_ > 0);
    // For the padding, they should either be the legacy padding strategy
    // (VALID or SAME), or an explicit, non-negative value.
    if (legacy_pad_ == LegacyPadding::VALID ||
        legacy_pad_ == LegacyPadding::SAME) {
      CAFFE_ENFORCE(
          !OperatorBase::HasArgument("pad") &&
              !OperatorBase::HasArgument("pad_t") &&
              !OperatorBase::HasArgument("pad_l") &&
              !OperatorBase::HasArgument("pad_b") &&
              !OperatorBase::HasArgument("pad_r"),
          "If you use legacy padding VALID or SAME, you should not specify "
          "any specific padding values.");
    }
    CAFFE_ENFORCE(pad_ >= 0);
    CAFFE_ENFORCE(pad_t_ >= 0);
    CAFFE_ENFORCE(pad_l_ >= 0);
    CAFFE_ENFORCE(pad_b_ >= 0);
    CAFFE_ENFORCE(pad_r_ >= 0);
    CAFFE_ENFORCE(stride_h_ > 0);
    CAFFE_ENFORCE(stride_w_ > 0);
    CAFFE_ENFORCE(adj_h_ < stride_h_);
    CAFFE_ENFORCE(adj_w_ < stride_w_);

    // Create shared buffer mutex in the constructor
    // to avoid race-condition in DAGNet.
    if (FLAGS_caffe2_force_shared_col_buffer || shared_buffer_) {
      createSharedBuffer<Context>(ws_);
    }
  }
  // Sets the output size. The output channel is manually specified.
  void SetOutputSize(
      const Tensor<Context>& input,
      Tensor<Context>* output,
      int output_channel) {
    CAFFE_ENFORCE(4 == input.ndim());
    CAFFE_ENFORCE(input.size() > 0);
    int N = input.dim32(0);
    bool channel_first = false; // initialized to suppress compiler warning.
    int H = 0, W = 0; // initialized to suppress compiler warning.
    int M = 0;
    switch (order_) {
      case StorageOrder::NHWC:
        channel_first = false;
        H = input.dim32(1);
        W = input.dim32(2);
        M = input.dim32(3);
        break;
      case StorageOrder::NCHW:
        channel_first = true;
        M = input.dim32(1);
        H = input.dim32(2);
        W = input.dim32(3);
        break;
      default:
        LOG(FATAL) << "Unknown Storage order: " << order_;
    }
    int output_height = 0, output_width = 0;
    ComputeSizeAndPad(
        H, stride_h_, kernel_h_, adj_h_, &pad_t_, &pad_b_, &output_height);
    ComputeSizeAndPad(
        W, stride_w_, kernel_w_, adj_w_, &pad_l_, &pad_r_, &output_width);
    if (channel_first) {
      output->Resize(N, output_channel, output_height, output_width);
    } else {
      output->Resize(N, output_height, output_width, output_channel);
    }
    VLOG(2) << "In: N " << N << " M " << M << " H " << H << " W " << W;
    VLOG(2) << "Out: output_channel " << output_channel << " H "
            << output_height << " W " << output_width;
  }

  bool RunOnDevice() override {
    switch (order_) {
      case StorageOrder::NHWC:
        return RunOnDeviceWithOrderNHWC();
      case StorageOrder::NCHW:
        return RunOnDeviceWithOrderNCHW();
      default:
        LOG(FATAL) << "Unknown storage order: " << order_;
    }
    // To suppress old compiler warnings
    return true;
  }

  virtual bool RunOnDeviceWithOrderNCHW() {
    CAFFE_THROW("Not implemented");
  }

  virtual bool RunOnDeviceWithOrderNHWC() {
    CAFFE_THROW("Not implemented");
  }

  virtual ~ConvTransposeUnpoolBase() {}

 private:
  LegacyPadding legacy_pad_;
  int pad_;

 protected:
  int pad_t_;
  int pad_l_;
  int pad_b_;
  int pad_r_;
  int kernel_h_;
  int kernel_w_;
  int stride_h_;
  int stride_w_;
  int adj_h_;
  int adj_w_;
  StorageOrder order_;
  bool shared_buffer_;
  Workspace* ws_;

  inline void ComputeSizeAndPad(
      const int in_size,
      const int stride,
      const int kernel,
      const int adj,
      int* pad_head,
      int* pad_tail,
      int* out_size) {
    switch (legacy_pad_) {
      case LegacyPadding::NOTSET:
        CAFFE_ENFORCE(*pad_head >= 0);
        CAFFE_ENFORCE(*pad_tail >= 0);
        *out_size =
            (in_size - 1) * stride + kernel + adj - *pad_head - *pad_tail;
        break;
      // We handle cases of LegacyPadding::VALID and LegacyPadding::SAME
      // the same way
      case LegacyPadding::VALID:
      case LegacyPadding::SAME:
        *pad_head = 0;
        *pad_tail = 0;
        *out_size = (in_size - 1) * stride + kernel + adj;
        break;
      case LegacyPadding::CAFFE_LEGACY_POOLING:
        LOG(FATAL) << "CAFFE_LEGACY_POOLING is no longer supported.";
        break;
    }
  }
};

#define USE_CONV_TRANSPOSE_UNPOOL_BASE_FUNCTIONS(Context) \
  USE_OPERATOR_FUNCTIONS(Context);                        \
  using ConvTransposeUnpoolBase<Context>::pad_t_;         \
  using ConvTransposeUnpoolBase<Context>::pad_b_;         \
  using ConvTransposeUnpoolBase<Context>::pad_l_;         \
  using ConvTransposeUnpoolBase<Context>::pad_r_;         \
  using ConvTransposeUnpoolBase<Context>::kernel_h_;      \
  using ConvTransposeUnpoolBase<Context>::kernel_w_;      \
  using ConvTransposeUnpoolBase<Context>::stride_h_;      \
  using ConvTransposeUnpoolBase<Context>::stride_w_;      \
  using ConvTransposeUnpoolBase<Context>::order_;         \
  using ConvTransposeUnpoolBase<Context>::shared_buffer_; \
  using ConvTransposeUnpoolBase<Context>::ws_

} // namespace caffe2

#endif // CAFFE2_OPERATORS_CONV_TRANSPOSE_UNPOOL_OP_BASE_H_
