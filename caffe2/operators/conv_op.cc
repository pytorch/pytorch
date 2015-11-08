#include "caffe2/operators/conv_op.h"
#include "caffe2/operators/conv_op_impl.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(Conv, ConvOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(ConvGradient, ConvGradientOp<float, CPUContext>);

struct GetConvGradient : public GetGradientDefBase {
  static vector<OperatorDef>* Create(const OperatorDef& def) {
    CAFFE_CHECK_EQ(def.input_size(), 3);
    return new vector<OperatorDef>{
        CreateOperatorDef(
            "ConvGradient", "",
            std::vector<string>{def.input(0), def.input(1),
                                GradientName(def.output(0))},
            std::vector<string>{GradientName(def.input(1)),
                                GradientName(def.input(2)),
                                GradientName(def.input(0))})};
  }
};
REGISTER_GRADIENT(Conv, GetConvGradient);

}  // namespace
}  // namespace caffe2
