#include <caffe2/ideep/ideep_utils.h>

using namespace caffe2;

namespace {

template <bool ReluFused>
class IDEEPInt8SumReluOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPInt8SumReluOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        scale_(this->template GetSingleArgument<float>("Y_scale", 1.0)),
        zero_point_(
            this->template GetSingleArgument<int32_t>("Y_zero_point", 0)) {
    if (ReluFused || zero_point_ == 0) {
      Y_data_type_ = idtype::u8;
      CAFFE_ENFORCE_EQ(zero_point_, 0, "Wrong zero point");
    } else {
      Y_data_type_ = idtype::s8;
      CAFFE_ENFORCE_EQ(zero_point_, 128, "Wrong zero point");
    }

    Y_scales_ = ConvertScales({scale_});
  }
  virtual ~IDEEPInt8SumReluOp() {}

  bool RunOnDevice() override {
    itensor temp_ten;
    itensor::dims input_dims;
    vector<itensor> inputs_itensor;

    CAFFE_ENFORCE_GT(InputSize(), 1, "Wrong input size (must > 1)");
    for (int i = 0; i < InputSize(); ++i) {
      CAFFE_ENFORCE(OperatorBase::InputBlob(i).template IsType<itensor>());
      auto& Xi = Input(i);
      if (input_dims.empty())
        input_dims = Xi.get_dims();
      CAFFE_ENFORCE(input_dims == Xi.get_dims());
      inputs_itensor.emplace_back(
          Xi.get_data_type() != idtype::f32 ? Xi.dequantize() : Xi);
    }

    temp_ten.init({input_dims, idtype::f32});
    const vector<float> scales(InputSize(), 1.0);
    ideep::sum::compute(scales, inputs_itensor, temp_ten);
    if (ReluFused) {
      ideep::eltwise_forward::compute(temp_ten, temp_ten);
    }

    auto* Y = Output(OUTPUT);
    Y->init({temp_ten.get_dims(), Y_data_type_, iformat::nhwc});
    Y->set_scale(Y_scales_);
    Y->feed_from(temp_ten);
    return true;
  }

 private:
  float scale_;
  int32_t zero_point_;
  iscale Y_scales_;
  idtype Y_data_type_;

  INPUT_TAGS(INPUT0);
  OUTPUT_TAGS(OUTPUT);
};

REGISTER_IDEEP_OPERATOR_WITH_ENGINE(Int8Sum, DNNLOWP, IDEEPInt8SumReluOp<false>);
REGISTER_IDEEP_OPERATOR_WITH_ENGINE(Int8Add, DNNLOWP, IDEEPInt8SumReluOp<false>);

REGISTER_IDEEP_OPERATOR_WITH_ENGINE(Int8SumRelu, DNNLOWP, IDEEPInt8SumReluOp<true>);
REGISTER_IDEEP_OPERATOR_WITH_ENGINE(Int8AddRelu, DNNLOWP, IDEEPInt8SumReluOp<true>);

} // namespace
