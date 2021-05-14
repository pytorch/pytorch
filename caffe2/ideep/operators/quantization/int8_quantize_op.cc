#include <caffe2/ideep/ideep_utils.h>

using namespace caffe2;

namespace {

class IDEEPInt8QuantizeOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPInt8QuantizeOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        scale_(this->template GetSingleArgument<float>("Y_scale", 1.0)),
        zero_point_(
            this->template GetSingleArgument<int32_t>("Y_zero_point", 0)) {

    if (HasArgument("output_order")) {
      Y_fmt_ = static_cast<iformat>(
        this->template GetSingleArgument<int>("output_order",
                                              static_cast<int>(iformat::nchw)));
    }

    CAFFE_ENFORCE(zero_point_ == 0 || zero_point_ == 128,
        "Not support this zero point");
    Y_data_type_ = zero_point_ == 0 ? idtype::u8 : idtype::s8;
    Y_scales_ = ConvertScales({scale_});
  }
  // NOLINTNEXTLINE(modernize-use-override,modernize-use-equals-default)
  virtual ~IDEEPInt8QuantizeOp() {}

  bool RunOnDevice() override {
    const auto& X = Input(0);
    CAFFE_ENFORCE(X.get_data_type() == idtype::f32, "Not support data type");

    auto* Y = Output(0);
    if (Y_fmt_ != iformat::undef) {
      Y->init(X.get_desc().to_type(Y_data_type_).to_format(Y_fmt_));
    } else {
      Y->init(X.get_desc().to_type(Y_data_type_));
    }
    Y->set_scale(Y_scales_);
    Y->feed_from(X);

    return true;
  }

 private:
  float scale_;
  int32_t zero_point_;
  iscale Y_scales_;
  idtype Y_data_type_;
  iformat Y_fmt_ {iformat::undef};

  INPUT_TAGS(INPUT0);
  OUTPUT_TAGS(OUTPUT);
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_IDEEP_OPERATOR_WITH_ENGINE(Int8Quantize, DNNLOWP, IDEEPInt8QuantizeOp);

} // namespace
