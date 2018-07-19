#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/mkl/mkl_utils.h"

#ifdef CAFFE2_HAS_MKL_DNN

namespace caffe2 {
namespace mkl {

template <typename T>
class MKLFullyConnectedOp final : public MKLOperator<T> {
 public:
  MKLFullyConnectedOp(const OperatorDef& operator_def, Workspace* ws)
      : MKLOperator<T>(operator_def, ws),
        axis_(OperatorBase::GetSingleArgument<int32_t>("axis", 1)) {}
  ~MKLFullyConnectedOp() {}

  bool RunOnDevice() override {
    auto& X = OperatorBase::Input<MKLMemory<float>>(INPUT);
    auto& filter = OperatorBase::Input<MKLMemory<float>>(FILTER);
    auto& bias = OperatorBase::Input<MKLMemory<float>>(BIAS);
    MKLMemory<float>* Y = OperatorBase::Output<MKLMemory<float>>(0);

    CAFFE_ENFORCE(filter.ndim() == 2, filter.ndim());
    CAFFE_ENFORCE(bias.ndim() == 1, bias.ndim());

    bool dims_changed;
    CHECK_INPUT_FILTER_DIMS(X, filter, dims_changed);
    if (dims_changed || FLAGS_caffe2_mkl_memonger_in_use) {
      const int N = filter.dim32(0);
      CAFFE_ENFORCE(N == bias.dim32(0));

      auto Y_shape = X.dims();
      Y_shape[1] = N;
      Y_shape.resize(2);

      size_t inputSizes[4];
      if (X.ndim() == 2) {
        inputSizes[0] = X.dim32(1);
        inputSizes[1] = X.dim32(0);
      } else {
        CAFFE_ENFORCE(X.ndim(), 4);
        inputSizes[0] = X.dim32(3);
        inputSizes[1] = X.dim32(2);
        inputSizes[2] = X.dim32(1);
        inputSizes[3] = X.dim32(0);
      }

      size_t outputSizes[2] = {Y_shape[1], Y_shape[0]};

      primitive_.Reset(
          dnnInnerProductCreateForwardBias<float>,
          nullptr,
          X.ndim(),
          inputSizes,
          outputSizes[0]);

      Y->Reset(Y_shape, primitive_, dnnResourceDst);
      buffer_.Reset(Y_shape, primitive_, dnnResourceDst, true);

      input_layout_.Reset(primitive_, dnnResourceSrc);
      filter_layout_.Reset(primitive_, dnnResourceFilter);
    }

    // Try to share from the output: this allows us to avoid unnecessary copy
    // operations, if the output is already allocated and is having the same
    // layout as the buffer has.
    bool shared = buffer_.ShareFrom(*Y);

    std::shared_ptr<void> X_view =
        X.View(input_layout_, primitive_, dnnResourceSrc);
    std::shared_ptr<void> filter_view =
        filter.View(filter_layout_, primitive_, dnnResourceFilter);

    resources_[dnnResourceSrc] = X_view.get();
    resources_[dnnResourceFilter] = filter_view.get();

    resources_[dnnResourceBias] = bias.buffer();
    resources_[dnnResourceDst] = buffer_.buffer();

    MKLDNN_SAFE_CALL(mkl::dnnExecute<T>(primitive_, resources_));
    buffer_.CopyTo(Y, primitive_, dnnResourceDst);
    if (FLAGS_caffe2_mkl_memonger_in_use && !shared) {
      buffer_.Reset();
    }
    return true;
  }

 private:
  // Input: X, W, b
  // Output: Y
  size_t axis_{1};
  vector<TIndex> cached_input_dims_;
  vector<TIndex> cached_filter_dims_;
  PrimitiveWrapper<T> primitive_;
  LayoutWrapper<T> input_layout_;
  LayoutWrapper<T> filter_layout_;
  LayoutWrapper<T> bias_layout_;
  MKLMemory<T> buffer_;
  void* resources_[dnnResourceNumber] = {0};
  INPUT_TAGS(INPUT, FILTER, BIAS);
};

} // namespace mkl

REGISTER_MKL_OPERATOR(FC, mkl::MKLFullyConnectedOp<float>);

} // namespace caffe2

#endif // CAFFE2_HAS_MKL_DNN
