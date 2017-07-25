#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/mkl_utils.h"

#ifdef CAFFE2_HAS_MKL_DNN

namespace caffe2 {
namespace mkl {

template <typename T>
class MKLSumOp final : public MKLOperator<T> {
 public:
  MKLSumOp(const OperatorDef& operator_def, Workspace* ws)
      : MKLOperator<T>(operator_def, ws) {
    coefficients_.resize(this->InputSize(), 1);
  }

  bool RunOnDevice() override {
    auto X_ = [&](size_t i) -> const MKLMemory<float>& {
      return OperatorBase::Input<MKLMemory<float>>(i);
    };
    MKLMemory<float>* Y = OperatorBase::Output<MKLMemory<float>>(0);
    bool dims_changed;
    {
      const auto& X = X_(0);
      CHECK_INPUT_DIMS(dims_changed);
    }
    if (dims_changed) {
      primitive_.Reset(
          dnnSumCreate<float>,
          nullptr,
          this->InputSize(),
          X_(0).layout(),
          coefficients_.data());
      if (Y != &X_(0)) {
        Y->Reset(X_(0).dims(), primitive_, dnnResourceDst);
      }
      buffer_.Reset(X_(0).dims(), primitive_, dnnResourceDst, true);
    }
    for (auto i = 0; i < this->InputSize(); ++i) {
      CAFFE_ENFORCE(dnnLayoutCompare_F32(X_(0).layout(), X_(i).layout()));
    }
    for (auto i = 0; i < this->InputSize(); ++i) {
      resources_[dnnResourceMultipleSrc + i] = X_(i).buffer();
    }
    if (Y != &X_(0)) {
      // TODO: MKLDNN seems broken in the in-place case.
      buffer_.ShareFrom(*Y);
    }
    resources_[dnnResourceDst] = buffer_.buffer();
    MKLDNN_SAFE_CALL(mkl::dnnExecute<T>(primitive_, resources_));
    buffer_.CopyTo(Y, primitive_, dnnResourceDst);
    return true;
  }

 private:
  // Input: X, W, b
  // Output: Y
  std::vector<float> coefficients_;
  vector<TIndex> cached_input_dims_;
  PrimitiveWrapper<T> primitive_;
  MKLMemory<T> buffer_;
  void* resources_[dnnResourceNumber] = {0};
  INPUT_TAGS(INPUT, FILTER, BIAS);
};

} // namespace mkl

REGISTER_MKL_OPERATOR(Sum, mkl::MKLSumOp<float>);

} // namespace caffe2

#endif // CAFFE2_HAS_MKL_DNN
