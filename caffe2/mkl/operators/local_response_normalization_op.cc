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

  ~MKLLRNOp() {
    if (workspace_buffer_ != NULL) {
      dnnReleaseBuffer<T>(workspace_buffer_);
      workspace_buffer_ = NULL;
    }
  }

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

 private:
  vector<TIndex> cached_input_dims_;
  LayoutWrapper<T> workspace_layout_;
  T* workspace_buffer_ = nullptr;
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
  if (dims_changed) {
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
    MKLDNN_SAFE_CALL(mkl::dnnAllocateBuffer<float>(
        (void**)(&workspace_buffer_), workspace_layout_));
  }

  // Try to share from the output: this allows us to avoid unnecessary copy
  // operations, if the output is already allocated and is having the same
  // layout as the buffer has.
  buffer_.ShareFrom(*Y);
  resources_[dnnResourceSrc] = X.buffer();
  resources_[dnnResourceDst] = buffer_.buffer();
  resources_[dnnResourceWorkspace] = workspace_buffer_;
  MKLDNN_SAFE_CALL(mkl::dnnExecute<float>(primitive_, resources_));
  buffer_.CopyTo(Y, primitive_, dnnResourceDst);
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
