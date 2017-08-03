#ifndef CAFFE2_OPERATORS_SINUSOID_POSITION_ENCODING_OP_H_
#define CAFFE2_OPERATORS_SINUSOID_POSITION_ENCODING_OP_H_

#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class SinusoidPositionEncodingOp : public Operator<Context> {
 public:
  SinusoidPositionEncodingOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        embedding_size_(OperatorBase::template GetSingleArgument<int>(
            "embedding_size",
            100)),
        alpha_(
            OperatorBase::template GetSingleArgument<float>("alpha", 10000)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, OperatorBase::Input<TensorCPU>(0));
  }

  template <typename Index>
  bool DoRunWithType() {
    auto& positions = Input(0);
    auto* output = Output(0);

    CAFFE_ENFORCE_EQ(positions.ndim(), 2, "POSITIONS should be a 2-D tensor");

    auto shape = positions.dims();
    shape.push_back(embedding_size_);
    output->Resize(shape);

    int N = positions.size();
    const Index* idxs = positions.template data<Index>();
    float* out = output->template mutable_data<float>();

    for (int i = 0; i < N; ++i) {
      float pos = (float)idxs[i];

      for (int j = 0; j < embedding_size_; ++j) {
        float exponent = (float)j / ((float)embedding_size_);
        float dim_scale = std::pow(alpha_, exponent);

        int loc = i * embedding_size_ + j;
        if (j % 2 == 0) {
          out[loc] = std::sin(pos / dim_scale);
        } else {
          out[loc] = std::cos(pos / dim_scale);
        }
      }
    }
    return true;
  }

 protected:
  int embedding_size_;
  float alpha_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_SINUSOID_POSITION_ENCODING_OP_H_
