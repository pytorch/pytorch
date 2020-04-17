#include <caffe2/ideep/ideep_utils.h>

using namespace caffe2;

namespace {

class IDEEPInt8DequantizeOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPInt8DequantizeOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws) {

    if (HasArgument("output_order")) {
      Y_fmt_ = static_cast<iformat>(
        this->template GetSingleArgument<int>("output_order",
                                              static_cast<int>(iformat::nchw)));
    }
  }
  virtual ~IDEEPInt8DequantizeOp() {}

  bool RunOnDevice() override {
    const auto& X = Input(0);
    auto* Y = Output(0);
    if (Y_fmt_ != iformat::undef) {
      Y->init(X.get_desc().to_type(idtype::f32).to_format(Y_fmt_));
    } else {
      Y->init(X.get_desc().to_type(idtype::f32));
    }
    Y->feed_from(X);

    return true;
  }

 private:
  iformat Y_fmt_ {iformat::undef};
};

REGISTER_IDEEP_OPERATOR_WITH_ENGINE(Int8Dequantize, DNNLOWP, IDEEPInt8DequantizeOp);

} // namespace
