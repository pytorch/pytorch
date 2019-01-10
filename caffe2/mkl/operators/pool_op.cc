#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"

#include "caffe2/mkl/mkl_utils.h"

#ifdef CAFFE2_HAS_MKL_DNN

namespace caffe2 {

namespace mkl {

template <typename T>
class MKLPoolOp final : public ConvPoolOpBase<MKLContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(MKLContext);
  MKLPoolOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<MKLContext>(operator_def, ws) {
    CAFFE_ENFORCE(
        (dilation_h() == 1) && (dilation_w() == 1),
        "Pooling op does not support dilation right now.");
    if (!global_pooling_) {
      CAFFE_ENFORCE(
          pad_t() < kernel_h() && pad_b() < kernel_h() &&
              pad_l() < kernel_w() && pad_r() < kernel_w(),
          "Pad should be smaller than kernel.");
    }
    // Figure out the pooling descriptor.
    if (operator_def.type().substr(0, 7) == "MaxPool") {
      algo = dnnAlgorithmPoolingMax;
    } else if (operator_def.type().substr(0, 11) == "AveragePool") {
      algo = dnnAlgorithmPoolingAvg;
    } else {
      LOG(FATAL) << "Unsupported pooling method: " << operator_def.type();
    }
  }

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

  // Input: X
  // Output: Y
 private:
  vector<TIndex> cached_input_dims_;
  // vector<TIndex> cached_avgpool_input_dims_;
  LayoutWrapper<T> workspace_layout_;
  std::unique_ptr<MKLWorkspace<T>> workspace_buffer_;
  PrimitiveWrapper<T> primitive_;
  MKLMemory<T> buffer_;
  void* resources_[dnnResourceNumber] = {0};
  dnnAlgorithm_t algo;
};

template <>
bool MKLPoolOp<float>::RunOnDeviceWithOrderNCHW() {
  auto& X = OperatorBase::Input<MKLMemory<float>>(0);
  MKLMemory<float>* Y = OperatorBase::Output<MKLMemory<float>>(0);

  bool dims_changed;
  CHECK_INPUT_DIMS(X, dims_changed);
  if (dims_changed || FLAGS_caffe2_mkl_memonger_in_use) {
    // We will utilize the SetOutputSize() function in the base class
    // with dummy TensorCPU input and output to calculate the sizes.
    TensorCPU dummy_input(X.dims());
    TensorCPU dummy_output;

    ConvPoolOpBase<MKLContext>::SetOutputSize(
        dummy_input, &dummy_output, X.dim32(1));
    size_t dim = X.ndim();
    CAFFE_ENFORCE(4 == dim);

    int paddings[4] = {-pad_l(), -pad_t(), -pad_r(), -pad_b()};
    size_t strides[2] = {stride_w(), stride_h()};
    size_t kernel_size[2] = {kernel_w(), kernel_h()};

    // Create main primitive.
    primitive_.Reset(
        dnnPoolingCreateForward_F32,
        nullptr,
        algo,
        X.layout(),
        kernel_size,
        strides,
        paddings,
        dnnBorderZerosAsymm);

    Y->Reset(dummy_output.dims(), primitive_, dnnResourceDst);
    buffer_.Reset(dummy_output.dims(), primitive_, dnnResourceDst, true);

    workspace_layout_.Reset(primitive_, dnnResourceWorkspace);
    workspace_buffer_ =
        caffe2::make_unique<MKLWorkspace<float>>(workspace_layout_);
  }

  // Try to share from the output: this allows us to avoid unnecessary copy
  // operations, if the output is already allocated and is having the same
  // layout as the buffer has.
  bool shared = buffer_.ShareFrom(*Y);
  resources_[dnnResourceSrc] = X.buffer();
  resources_[dnnResourceDst] = buffer_.buffer();
  resources_[dnnResourceWorkspace] = workspace_buffer_->buffer();
  MKLDNN_SAFE_CALL(mkl::dnnExecute<float>(primitive_, resources_));
  buffer_.CopyTo(Y, primitive_, dnnResourceDst);
  if (FLAGS_caffe2_mkl_memonger_in_use && !shared) {
    buffer_.Reset();
  }
  return true;
}

template <>
bool MKLPoolOp<float>::RunOnDeviceWithOrderNHWC() {
  CAFFE_NOT_IMPLEMENTED;
}

} // namespace mkl

REGISTER_MKL_OPERATOR(AveragePool, mkl::MKLPoolOp<float>);
REGISTER_MKL_OPERATOR(MaxPool, mkl::MKLPoolOp<float>);

} // namespace caffe2

#endif // CAFFE2_HAS_MKL_DNN
