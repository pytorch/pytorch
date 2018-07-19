#include "caffe2/operators/local_response_normalization_op.h"
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

#include "caffe2/mkl/mkl_utils.h"

#ifdef CAFFE2_HAS_MKL_DNN

namespace caffe2 {
namespace mkl {

template <typename T>
class MKLLRNOp final : public LRNOpBase<T, MKLContext> {
 public:
  MKLLRNOp(const OperatorDef& operator_def, Workspace* ws)
      : LRNOpBase<T, MKLContext>(operator_def, ws) {}

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

 private:
  vector<TIndex> cached_input_dims_;
  LayoutWrapper<T> workspace_layout_;
  std::unique_ptr<MKLWorkspace<T>> workspace_buffer_;
  PrimitiveWrapper<T> primitive_;
  MKLMemory<T> buffer_;
  void* resources_[dnnResourceNumber] = {0};
};

template <>
bool MKLLRNOp<float>::RunOnDeviceWithOrderNCHW() {
  auto& X = OperatorBase::Input<MKLMemory<float>>(0);
  MKLMemory<float>* Y = OperatorBase::Output<MKLMemory<float>>(0);

  bool dims_changed;
  CHECK_INPUT_DIMS(X, dims_changed);
  if (dims_changed || FLAGS_caffe2_mkl_memonger_in_use) {
    size_t dim = X.ndim();
    CAFFE_ENFORCE(4 == dim);

    // Create main primitive.
    primitive_.Reset(
        dnnLRNCreateForward_F32,
        nullptr,
        X.layout(),
        size_,
        alpha_,
        beta_,
        bias_);

    Y->Reset(X.dims(), primitive_, dnnResourceDst);
    buffer_.Reset(X.dims(), primitive_, dnnResourceDst, true);

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
bool MKLLRNOp<float>::RunOnDeviceWithOrderNHWC() {
  CAFFE_NOT_IMPLEMENTED;
}

} // namespace mkl

REGISTER_MKL_OPERATOR(LRN, mkl::MKLLRNOp<float>);

} // namespace caffe2

#endif // CAFFE2_HAS_MKL_DNN
