#include "caffe2/operators/conv_op.h"
#include "caffe2/operators/conv_op_impl.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(Conv, ConvOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(ConvGradient, ConvGradientOp<float, CPUContext>);

struct GetConvGradient : public GetGradientDefBase {
  static vector<OperatorDef>* Create(const OperatorDef& def) {
    CAFFE_CHECK_EQ(def.input_size(), 3);
    return SingleGradientDef(
        "ConvGradient", "",
        vector<string>{I(def, 0), I(def, 1), GO(def, 0)},
        vector<string>{GI(def, 1), GI(def, 2), GI(def, 0)});
  }
};
REGISTER_GRADIENT(Conv, GetConvGradient);

}  // namespace
}  // namespace caffe2
