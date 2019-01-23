#include <caffe2/ideep/ideep_utils.h>

namespace caffe2 {

class IDEEPInt8SumReluOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPInt8SumReluOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        scale_(this->template GetSingleArgument<float>("Y_scale", 1.0)),
        zero_point_(
            this->template GetSingleArgument<int32_t>("Y_zero_point", 0)) {
    CAFFE_ENFORCE(zero_point_ == 0);
    Y_scales_ = ConvertScales({scale_});
  }
  virtual ~IDEEPInt8SumReluOp() {}

  bool RunOnDevice() override {
    itensor temp_ten;
    if (InputSize() == 1) {
      CAFFE_ENFORCE(OperatorBase::InputBlob(0).template IsType<itensor>());
      const auto& X = Input(INPUT0);
      temp_ten.init(X.get_descriptor());
      ideep::eltwise_forward::compute(X, temp_ten);
    } else {
      itensor::dims input_dims;
      vector<itensor> inputs_itensor;
      for (int i = 0; i < InputSize(); ++i) {
        CAFFE_ENFORCE(OperatorBase::InputBlob(i).template IsType<itensor>());
        auto& Xi = Input(i);
        if (input_dims.empty())
          input_dims = Xi.get_dims();
        CAFFE_ENFORCE(input_dims == Xi.get_dims());
        inputs_itensor.emplace_back(Xi);
      }

      temp_ten.init({input_dims, idtype::f32});
      const vector<float> scales(InputSize(), 1.0);
      ideep::sum::compute(scales, inputs_itensor, temp_ten);
      ideep::eltwise_forward::compute(temp_ten, temp_ten);
    }

    auto* Y = Output(OUTPUT);
    Y->init({temp_ten.get_dims(), idtype::u8, iformat::nhwc});
    Y->set_scale(Y_scales_);
    Y->feed_from(temp_ten);
    return true;
  }

 private:
  float scale_;
  int32_t zero_point_;
  iscale Y_scales_;

  INPUT_TAGS(INPUT0);
  OUTPUT_TAGS(OUTPUT);
};

REGISTER_IDEEP_OPERATOR_WITH_ENGINE(Int8SumRelu, DNNLOWP, IDEEPInt8SumReluOp);
REGISTER_IDEEP_OPERATOR_WITH_ENGINE(Int8AddRelu, DNNLOWP, IDEEPInt8SumReluOp);

} // namespace caffe2
