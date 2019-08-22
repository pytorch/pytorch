#include <caffe2/ideep/ideep_utils.h>

namespace caffe2 {

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
        this->template GetSingleArgument<int>("output_order", iformat::nchw));
    }

    if (HasArgument("dst_type")) {
      dst_type_ = OperatorBase::GetSingleArgument<string>("dst_type", "s8");
    }

    CAFFE_ENFORCE(zero_point_ == 0 || zero_point_ == 128,
        "Not support this zero point");
    Y_data_type_ = zero_point_ == 0 ? idtype::u8 : idtype::s8;
    Y_scales_ = ConvertScales({scale_});
  }
  virtual ~IDEEPInt8QuantizeOp() {}

  bool RunOnDevice() override {
    const auto& X = Input(0);
    CAFFE_ENFORCE(X.get_data_type() == idtype::f32, "Not support data type");

    auto* Y = Output(0);
    Y->reinit({X.get_dims(), Y_data_type_,
        Y_fmt_ != iformat::format_undef
        ? Y_fmt_ : X.get_public_format()});
    Y->set_scale(Y_scales_);
    Y->feed_from(X);

    if (dst_type_ == "u8") {
      auto* dst_ptr = static_cast<char*>(Y->get_data_handle());
      int dims = X.get_dim(0)*X.get_dim(1)*X.get_dim(2)*X.get_dim(3);
      for (int i = 0; i < dims; ++i) {
        dst_ptr[i] += char(128); 
      }
      
      Y->set_descriptor({X.get_dims(), idtype::u8,
        Y_fmt_ != iformat::format_undef
        ? Y_fmt_ : X.get_public_format()});
    }

    return true;
  }

 private:
  float scale_;
  int32_t zero_point_;
  iscale Y_scales_;
  idtype Y_data_type_;
  iformat Y_fmt_ {iformat::format_undef};
  std::string dst_type_;

  INPUT_TAGS(INPUT0);
  OUTPUT_TAGS(OUTPUT);
};

REGISTER_IDEEP_OPERATOR_WITH_ENGINE(Int8Quantize, DNNLOWP, IDEEPInt8QuantizeOp);

} // namespace caffe2
