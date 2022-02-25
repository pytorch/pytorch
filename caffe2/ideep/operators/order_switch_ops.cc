#include <caffe2/ideep/ideep_utils.h>

using namespace caffe2;

namespace {

class IDEEPNHWC2NCHWOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_SIMPLE_IDEEP_CTOR_DTOR(IDEEPNHWC2NCHWOp);
  USE_IDEEP_OPERATOR_FUNCTIONS();

  bool RunOnDevice() override {
    const auto& X = Input(0);
    CAFFE_ENFORCE_EQ(X.ndims(), 4);
    CAFFE_ENFORCE(X.get_desc().is_nhwc());

    auto *Y = Output(OUTPUT);
    CAFFE_ENFORCE(Y != &X);

    // NOTE: NHWC changes the shape in framework, but not in MKL-DNN
    // Thus, for iDEEP tensor, the shapes of NCHW and NHWC are identical.
    Y->init({X.get_dims(), X.get_data_type(), iformat::nchw});
    Y->feed_from(X);
    return true;
  }

 private:
  INPUT_TAGS(INPUT);
  OUTPUT_TAGS(OUTPUT);
};

class IDEEPNCHW2NHWCOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_SIMPLE_IDEEP_CTOR_DTOR(IDEEPNCHW2NHWCOp);
  USE_IDEEP_OPERATOR_FUNCTIONS();

  bool RunOnDevice() override {
    const auto& X = Input(0);
    CAFFE_ENFORCE_EQ(X.ndims(), 4);
    CAFFE_ENFORCE(X.get_desc().is_nchw());

    auto *Y = Output(OUTPUT);
    CAFFE_ENFORCE(Y != &X);

    // NOTE: NHWC changes the shape in framework, but not in MKL-DNN
    // Thus, for iDEEP tensor, the shapes of NCHW and NHWC are identical.
    Y->init({X.get_dims(), X.get_data_type(), iformat::nhwc});
    Y->feed_from(X);
    return true;
  }

 private:
  INPUT_TAGS(INPUT);
  OUTPUT_TAGS(OUTPUT);
};

REGISTER_IDEEP_OPERATOR(NHWC2NCHW, IDEEPNHWC2NCHWOp);
REGISTER_IDEEP_OPERATOR(NCHW2NHWC, IDEEPNCHW2NHWCOp);

} // namespace
