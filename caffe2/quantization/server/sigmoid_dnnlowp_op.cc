#include "caffe2/quantization/server/elementwise_dnnlowp_op.h"
#include "caffe2/quantization/server/sigmoid.h"

namespace caffe2 {

using namespace dnnlowp;

template <typename T>
class SigmoidFunctor {
 public:
  explicit SigmoidFunctor() : sigmoid_() {}

  inline void operator()(const int n, const T* x, T* y) {
    for (int i = 0; i < n; ++i) {
      y[i] = sigmoid_.Compute(x[i]);
    }
  }

  TensorQuantizationParams GetOutputQuantizationParams() const {
    return sigmoid_.GetOutputQuantizationParams();
  }

 private:
  Sigmoid<T> sigmoid_;
};

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Sigmoid,
    DNNLOWP,
    UnaryElementwiseWithArgsDNNLowPOp<
        std::uint8_t,
        SigmoidFunctor<std::uint8_t>>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    Int8Sigmoid,
    DNNLOWP,
    UnaryElementwiseWithArgsDNNLowPOp<
        std::uint8_t,
        SigmoidFunctor<std::uint8_t>>);

} // namespace caffe2
