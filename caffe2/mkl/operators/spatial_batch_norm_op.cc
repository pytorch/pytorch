#include "caffe2/operators/spatial_batch_norm_op.h"
#include <math.h>

#include "caffe2/mkl/mkl_utils.h"

#ifdef CAFFE2_HAS_MKL_DNN

namespace caffe2 {
namespace mkl {

template <typename T>
class MKLBNOp final : public SpatialBNOp<MKLContext> {
 public:
  MKLBNOp(const OperatorDef& operator_def, Workspace* ws)
      : SpatialBNOp<MKLContext>(operator_def, ws) {
    OPERATOR_NEEDS_FEATURE(
        order_ == StorageOrder::NCHW, "Only NCHW order supported.");
    OPERATOR_NEEDS_FEATURE(
        operator_def.input(0) != operator_def.output(0),
        "Inplace BN not supported");
  }

  bool RunOnDevice() {
    auto& X = OperatorBase::Input<MKLMemory<float>>(INPUT);
    auto& scale = OperatorBase::Input<MKLMemory<float>>(SCALE);
    auto& bias = OperatorBase::Input<MKLMemory<float>>(BIAS);

    MKLMemory<float>* Y = OperatorBase::Output<MKLMemory<float>>(OUTPUT);
    // anded with is_test_-1 to avoid uninitialized access in case of testing
    MKLMemory<float>* running_mean =
        OperatorBase::Output<MKLMemory<float>>(RUNNING_MEAN & (is_test_ - 1));
    MKLMemory<float>* running_var =
        OperatorBase::Output<MKLMemory<float>>(RUNNING_VAR & (is_test_ - 1));
    MKLMemory<float>* saved_mean =
        OperatorBase::Output<MKLMemory<float>>(SAVED_MEAN & (is_test_ - 1));
    MKLMemory<float>* saved_var =
        OperatorBase::Output<MKLMemory<float>>(SAVED_INV_VAR & (is_test_ - 1));

    // current code supports only NCHW -
    // have to look for MKL related changes for NHWC later
    const int N = X.dim32(0);
    const int C = (order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(3));
    const int H = (order_ == StorageOrder::NCHW ? X.dim32(2) : X.dim32(1));
    const int W = (order_ == StorageOrder::NCHW ? X.dim32(3) : X.dim32(2));

    DCHECK_EQ(scale.ndim(), 1);
    DCHECK_EQ(bias.ndim(), 1);
    DCHECK_EQ(scale.dim32(0), C);
    DCHECK_EQ(bias.dim32(0), C);

    bool dims_changed;
    CHECK_INPUT_DIMS(X, dims_changed);
    if (dims_changed || FLAGS_caffe2_mkl_memonger_in_use) {
      // Create main primitive.
      if (is_test_) {
        primitive_.Reset(
            dnnBatchNormalizationCreateForward_v2<T>,
            nullptr,
            X.layout(),
            epsilon_,
            dnnUseInputMeanVariance | dnnUseScaleShift);
      } else {
        primitive_.Reset(
            dnnBatchNormalizationCreateForward_v2<T>,
            nullptr,
            X.layout(),
            epsilon_,
            dnnUseScaleShift);

        // using scale dims as it is also of size C
        saved_mean->Reset(scale.dims(), primitive_, dnnResourceMean);
        saved_var->Reset(scale.dims(), primitive_, dnnResourceVariance);
        running_mean->Reset(scale.dims(), primitive_, dnnResourceMean);
        running_var->Reset(scale.dims(), primitive_, dnnResourceVariance);

        running_mean_buf = (T*)running_mean->buffer();
        running_var_buf = (T*)running_var->buffer();
      }
      Y->Reset(X.dims(), primitive_, dnnResourceDst);
      buffer_.Reset(X.dims(), primitive_, dnnResourceDst, true);

      scale_bias_layout_.Reset(primitive_, dnnResourceScaleShift);
      scale_bias_buffer_ =
          caffe2::make_unique<MKLWorkspace<float>>(scale_bias_layout_);

      // fill scale and bias into a single buffer
      scale_buf = (T*)scale.buffer();
      bias_buf = (T*)bias.buffer();
      for (int i = 0; i < C; i++) {
        scale_bias_buffer_->buffer()[i] = scale_buf[i];
        scale_bias_buffer_->buffer()[C + i] = bias_buf[i];
      }
    }

    // Try to share from the output: this allows us to avoid unnecessary copy
    // operations, if the output is already allocated and is having the same
    // layout as the buffer has.
    bool shared = buffer_.ShareFrom(*Y);
    resources_[dnnResourceSrc] = X.buffer();
    resources_[dnnResourceDst] = buffer_.buffer();
    resources_[dnnResourceScaleShift] = scale_bias_buffer_->buffer();

    if (is_test_) {
      auto& est_mean = OperatorBase::Input<MKLMemory<float>>(EST_MEAN);
      auto& est_var = OperatorBase::Input<MKLMemory<float>>(EST_VAR);

      resources_[dnnResourceMean] = est_mean.buffer();
      resources_[dnnResourceVariance] = est_var.buffer();
    } else {
      resources_[dnnResourceMean] = saved_mean->buffer();
      resources_[dnnResourceVariance] = saved_var->buffer();
    }

    MKLDNN_SAFE_CALL(mkl::dnnExecute<float>(primitive_, resources_));

    if (!is_test_) {
      // compute running mean and variance
      saved_mean_buf = (T*)saved_mean->buffer();
      saved_var_buf = (T*)saved_var->buffer();

      for (int i = 0; i < C; i++) {
        running_mean_buf[i] = running_mean_buf[i] * momentum_ +
            saved_mean_buf[i] * (1. - momentum_);
        running_var_buf[i] = running_var_buf[i] * momentum_ +
            saved_var_buf[i] * (1. - momentum_);
        saved_var_buf[i] = (1 / sqrt(saved_var_buf[i] + epsilon_));
      }
    }
    buffer_.CopyTo(Y, primitive_, dnnResourceDst);
    if (FLAGS_caffe2_mkl_memonger_in_use && !shared) {
      buffer_.Reset();
    }
    return true;
  }

 private:
  vector<TIndex> cached_input_dims_;
  LayoutWrapper<T> scale_bias_layout_;
  LayoutWrapper<T> saved_mean_layout_;
  LayoutWrapper<T> saved_var_layout_;
  LayoutWrapper<T> running_mean_layout_;
  LayoutWrapper<T> running_var_layout_;
  std::unique_ptr<MKLWorkspace<T>> scale_bias_buffer_;
  T* scale_buf = nullptr;
  T* bias_buf = nullptr;
  T* saved_mean_buf = nullptr;
  T* saved_var_buf = nullptr;
  T* running_mean_buf = nullptr;
  T* running_var_buf = nullptr;
  PrimitiveWrapper<T> primitive_;
  MKLMemory<T> buffer_;
  void* resources_[dnnResourceNumber] = {0};
};
} // namespace mkl

REGISTER_MKL_OPERATOR(SpatialBN, mkl::MKLBNOp<float>);
} // namespace caffe2

#endif // CAFFE2_HAS_MKL_DNN
