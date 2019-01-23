#include <caffe2/ideep/ideep_utils.h>

namespace caffe2 {

class IDEEPInt8DequantizeOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();
  USE_SIMPLE_IDEEP_CTOR_DTOR(IDEEPInt8DequantizeOp);

  bool RunOnDevice() override {
    const auto& X = Input(0);
    auto* Y = Output(0);

    Y->init({X.get_dims(), idtype::f32, X.get_public_format()});
    Y->feed_from(X);

    return true;
  }
};

REGISTER_IDEEP_OPERATOR_WITH_ENGINE(Int8Dequantize, DNNLOWP, IDEEPInt8DequantizeOp);

} // namespace caffe2
