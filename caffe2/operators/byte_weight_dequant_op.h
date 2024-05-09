#ifndef CAFFE2_OPERATORS_BYTE_WEIGHT_DEQUANT_OP_H_
#define CAFFE2_OPERATORS_BYTE_WEIGHT_DEQUANT_OP_H_

#include "caffe2/core/operator.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename Context>
class ByteWeightDequantOp : public Operator<Context> {
 public:
  ByteWeightDequantOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        min_(this->template GetSingleArgument<float>("min", -3)),
        max_(this->template GetSingleArgument<float>("max", 3)),
        shape_(this->template GetRepeatedArgument<int64_t>("shape")) {}

  USE_OPERATOR_FUNCTIONS(Context);
  using Operator<Context>::Operator;

  bool RunOnDevice() override {
    const auto& WI = Input(0);

    auto* Y = Output(0, shape_, at::dtype<float>());
    float bin_interval = (max_ - min_) / 255.0;
    int total = 1;
    for (const auto i : c10::irange(0U, shape_.size())) {
      total *= Y->size(i);
    }
    const uint8_t* Xdata;
    if (WI.template IsType<uint8_t>()) {
      CAFFE_ENFORCE(total, WI.nbytes());
      Xdata = WI.template data<uint8_t>();
    } else {
      CAFFE_ENFORCE(total, WI.template data<std::string>()[0].size());
      Xdata = reinterpret_cast<const uint8_t*>(
          WI.template data<std::string>()[0].c_str());
    }
    auto* Ydata = Y->template mutable_data<float>();
    ConstEigenVectorMap<uint8_t> index(&Xdata[0], total);
    EigenVectorMap<float> weights(&Ydata[0], total);
    weights = (index.cast<float>().array() * bin_interval) + min_;
    return true;
  }

 private:
  float min_;
  float max_;
  std::vector<int64_t> shape_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_BYTE_WEIGHT_DEQUANT_OP_H_
